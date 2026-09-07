from __future__ import annotations

import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr

from furniture_ai.api_entry import app
from furniture_ai.config import Settings, get_settings
from furniture_ai.contracts import FloorPlanAnalysis, Opening, OpeningKind, Point, Room
from furniture_ai.layout import furnish_floor_plan
from furniture_ai.rendering import (
    PromptCompiler,
    RenderBackendError,
    RendererKind,
    RenderingService,
    RenderPreviewRequest,
    SceneCompiler,
)


def _design():
    floor_plan = FloorPlanAnalysis(
        source_width=800,
        source_height=600,
        rooms=[
            Room(
                id="room-1",
                room_type="living_room",
                polygon=[
                    Point(x=20, y=20),
                    Point(x=780, y=20),
                    Point(x=780, y=580),
                    Point(x=20, y=580),
                ],
                area=425_600,
            )
        ],
        openings=[
            Opening(
                id="door-1",
                kind=OpeningKind.DOOR,
                start=Point(x=20, y=240),
                end=Point(x=20, y=330),
            )
        ],
    )
    return furnish_floor_plan(floor_plan)


def _png_bytes(width: int = 64, height: int = 48) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeImages:
    def __init__(self, *, image_bytes: bytes | None = None, error: Exception | None = None) -> None:
        self.image_bytes = image_bytes or _png_bytes()
        self.error = error
        self.calls: list[dict[str, object]] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        encoded = base64.b64encode(self.image_bytes).decode("ascii")
        return SimpleNamespace(data=[SimpleNamespace(b64_json=encoded)])


class FakeImageClient:
    def __init__(self, images: FakeImages | None = None) -> None:
        self.images = images or FakeImages()


def _image_settings() -> Settings:
    return Settings(
        environment="test",
        openai_api_key=SecretStr("fake-render-key"),
        openai_image_model="gpt-image-2",
        openai_image_quality="medium",
        openai_image_size="1536x1024",
    )


def test_scene_compiler_grounds_catalog_products_openings_and_style() -> None:
    scene = SceneCompiler().compile(
        _design(),
        style="  warm   modern   minimal  ",
        room_id="room-1",
    )

    assert scene.style == "warm modern minimal"
    assert len(scene.rooms) == 1
    assert scene.rooms[0].furniture
    product_ids = {item.product_id for item in scene.rooms[0].furniture}
    assert {"sofa-3-seat", "coffee-table", "tv-unit"}.issubset(product_ids)
    assert all(item.product_name for item in scene.rooms[0].furniture)
    assert len(scene.openings) == 1
    assert scene.openings[0].id == "door-1"
    assert scene.openings[0].kind is OpeningKind.DOOR


def test_prompt_compiler_is_deterministic_and_grounding_first() -> None:
    scene = SceneCompiler().compile(_design(), style="warm modern")
    compiler = PromptCompiler()

    first = compiler.compile(scene)
    second = compiler.compile(scene)

    assert first == second
    assert len(first.scene_fingerprint) == 64
    assert "Preserve the supplied room geometry" in first.positive_prompt
    assert "Three-seat sofa" in first.positive_prompt
    assert "door door-1" in first.positive_prompt
    assert "do not move, remove, or duplicate grounded furniture" in first.negative_prompt


def test_mock_renderer_returns_visual_svg_without_claiming_photorealism() -> None:
    result = RenderingService().preview(
        RenderPreviewRequest(design=_design(), style="warm modern", seed=7)
    )

    assert result.status == "preview"
    assert result.photorealistic is False
    assert result.artifact.media_type == "image/svg+xml"
    prefix = "data:image/svg+xml;base64,"
    assert result.artifact.data_uri.startswith(prefix)
    svg = base64.b64decode(result.artifact.data_uri[len(prefix) :]).decode("utf-8")
    assert svg.startswith("<svg")
    assert "Three-seat sofa" in svg
    assert 'data-kind="door"' in svg
    assert result.warnings


def test_gpt_image_2_renderer_returns_verified_photoreal_png() -> None:
    client = FakeImageClient()
    result = RenderingService(
        settings=_image_settings(),
        renderer_client=client,
    ).preview(
        RenderPreviewRequest(
            design=_design(),
            style="luxury warm modern",
            backend=RendererKind.OPENAI_GPT_IMAGE_2,
            seed=123,
        )
    )

    assert result.photorealistic is True
    assert result.artifact.backend is RendererKind.OPENAI_GPT_IMAGE_2
    assert result.artifact.media_type == "image/png"
    assert result.artifact.width == 64
    assert result.artifact.height == 48
    assert result.artifact.data_uri.startswith("data:image/png;base64,")
    assert result.artifact.metadata["model"] == "gpt-image-2"
    assert result.artifact.metadata["seed_requested"] == 123
    assert result.artifact.metadata["seed_applied"] is False

    call = client.images.calls[0]
    assert call["model"] == "gpt-image-2"
    assert call["quality"] == "medium"
    assert call["size"] == "1536x1024"
    assert call["background"] == "opaque"
    assert "Three-seat sofa" in call["prompt"]
    assert "Hard constraints:" in call["prompt"]


def test_gpt_image_2_renderer_sanitizes_external_client_failures() -> None:
    client = FakeImageClient(FakeImages(error=RuntimeError("token=private-render-secret")))
    service = RenderingService(settings=_image_settings(), renderer_client=client)
    request = RenderPreviewRequest(
        design=_design(),
        style="modern",
        backend=RendererKind.OPENAI_GPT_IMAGE_2,
    )

    with pytest.raises(RenderBackendError, match="Photorealistic image generation failed") as exc:
        service.preview(request)

    assert "private-render-secret" not in str(exc.value)


def test_gpt_image_2_renderer_rejects_non_image_output() -> None:
    client = FakeImageClient(FakeImages(image_bytes=b"not-a-png"))
    service = RenderingService(settings=_image_settings(), renderer_client=client)

    with pytest.raises(RenderBackendError, match="returned an invalid image"):
        service.preview(
            RenderPreviewRequest(
                design=_design(),
                style="modern",
                backend=RendererKind.OPENAI_GPT_IMAGE_2,
            )
        )


def test_v2_render_preview_endpoint_returns_grounded_scene_and_artifact() -> None:
    request = RenderPreviewRequest(
        design=_design(),
        style="japandi natural",
        room_id="room-1",
        seed=11,
    )
    response = TestClient(app).post(
        "/api/v2/render/preview",
        json=request.model_dump(mode="json"),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["photorealistic"] is False
    assert payload["scene"]["style"] == "japandi natural"
    assert payload["scene"]["rooms"][0]["furniture"]
    assert payload["scene"]["openings"][0]["kind"] == "door"
    assert payload["artifact"]["backend"] == "mock"
    assert payload["artifact"]["data_uri"].startswith("data:image/svg+xml;base64,")


def test_v2_photoreal_preview_fails_closed_when_openai_is_not_configured() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        environment="test",
        openai_api_key=None,
    )
    request = RenderPreviewRequest(
        design=_design(),
        style="modern",
        backend=RendererKind.OPENAI_GPT_IMAGE_2,
    )
    try:
        response = TestClient(app).post(
            "/api/v2/render/preview",
            json=request.model_dump(mode="json"),
        )
    finally:
        app.dependency_overrides.pop(get_settings, None)

    assert response.status_code == 503
    assert response.json()["detail"] == "Photorealistic renderer is unavailable"


def test_v2_render_preview_rejects_unknown_room() -> None:
    request = RenderPreviewRequest(
        design=_design(),
        style="modern",
        room_id="missing-room",
    )
    response = TestClient(app).post(
        "/api/v2/render/preview",
        json=request.model_dump(mode="json"),
    )

    assert response.status_code == 422
    assert "Unknown render room_id" in response.json()["detail"]
