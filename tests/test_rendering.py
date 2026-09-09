from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import SecretStr, ValidationError

from furniture_ai.api_entry import app
from furniture_ai.config import Settings, get_settings
from furniture_ai.contracts import FloorPlanAnalysis, Opening, OpeningKind, Point, Room
from furniture_ai.layout import furnish_floor_plan
from furniture_ai.rendering import (
    PromptCompiler,
    RenderBackendError,
    RenderBackendUnavailable,
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


class FakeOpenArtRunner:
    def __init__(
        self,
        *,
        image_bytes: bytes | None = None,
        error: Exception | None = None,
    ) -> None:
        self.image_bytes = image_bytes or _png_bytes()
        self.error = error
        self.calls: list[tuple[list[str], float]] = []

    def run(self, args: list[str], *, timeout: float) -> None:
        self.calls.append((list(args), timeout))
        if self.error is not None:
            raise self.error
        output_dir = Path(args[args.index("-o") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "generated.png").write_bytes(self.image_bytes)


def _image_settings() -> Settings:
    return Settings(
        environment="test",
        openai_api_key=SecretStr("fake-render-key"),
        openai_image_model="gpt-image-2",
        openai_image_quality="medium",
        openai_image_size="1536x1024",
    )


def _openart_settings(*, enabled: bool = True) -> Settings:
    return Settings(
        environment="test",
        openart_enabled=enabled,
        openart_cli_path="openart-test",
        openart_image_timeout_seconds=90,
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


def test_openart_style_flare_request_accepts_generation_controls() -> None:
    request = RenderPreviewRequest(
        design=_design(),
        style="luxury warm modern",
        backend="openart_gpt_image_25_flare",
        generation={
            "mode": "text2image",
            "aspect_ratio": "4:3",
            "resolution_tier": "2k",
            "quality": "medium",
            "image_count": 1,
            "lock_aspect_ratio": True,
            "auto_enhance_prompt": False,
            "visual_references": [],
        },
    )

    assert request.backend.value == "openart_gpt_image_25_flare"
    assert request.generation.mode == "text2image"
    assert request.generation.aspect_ratio == "4:3"
    assert request.generation.resolution_tier == "2k"
    assert request.generation.quality == "medium"


def test_openart_flare_builds_grounded_text2image_cli_request() -> None:
    runner = FakeOpenArtRunner()
    result = RenderingService(
        settings=_openart_settings(),
        renderer_client=runner,
    ).preview(
        RenderPreviewRequest(
            design=_design(),
            style="luxury warm modern",
            backend=RendererKind.OPENART_GPT_IMAGE_25_FLARE,
            generation={
                "mode": "text2image",
                "aspect_ratio": "4:3",
                "resolution_tier": "2k",
                "quality": "medium",
            },
        )
    )

    assert result.photorealistic is True
    assert result.artifact.backend is RendererKind.OPENART_GPT_IMAGE_25_FLARE
    assert result.artifact.media_type == "image/png"
    assert result.artifact.metadata["provider"] == "openart_cli"
    assert result.artifact.metadata["model"] == "gpt-image-2-5-flare"
    assert result.artifact.metadata["mode"] == "text2image"
    assert result.artifact.metadata["reference_images_applied"] == 0

    args, timeout = runner.calls[0]
    assert args[:3] == ["openart-test", "generate", "image"]
    assert args[args.index("--model") + 1] == "gpt-image-2-5-flare"
    assert args[args.index("--aspect-ratio") + 1] == "4:3"
    assert args[args.index("--resolution") + 1] == "2k"
    assert "Three-seat sofa" in args[3]
    assert "Hard constraints:" in args[3]
    assert "--image" not in args
    assert timeout == 90


def test_openart_sunburst_routes_to_quality_focused_model() -> None:
    runner = FakeOpenArtRunner()
    result = RenderingService(
        settings=_openart_settings(),
        renderer_client=runner,
    ).preview(
        RenderPreviewRequest(
            design=_design(),
            backend=RendererKind.OPENART_GPT_IMAGE_25_SUNBURST,
        )
    )

    args, _ = runner.calls[0]
    assert args[args.index("--model") + 1] == "gpt-image-2-5-sunburst"
    assert result.artifact.metadata["model"] == "gpt-image-2-5-sunburst"


def test_openart_image2image_passes_only_trusted_visual_references() -> None:
    runner = FakeOpenArtRunner()
    references = [
        {
            "url": "https://cdn.openart.ai/furniture/room-reference.png",
            "label": "room",
        },
        {
            "url": "https://cdn.openart.ai/furniture/sofa-reference.png",
            "label": "sofa",
        },
    ]
    result = RenderingService(
        settings=_openart_settings(),
        renderer_client=runner,
    ).preview(
        RenderPreviewRequest(
            design=_design(),
            backend=RendererKind.OPENART_GPT_IMAGE_25_FLARE,
            generation={
                "mode": "image2image",
                "visual_references": references,
            },
        )
    )

    args, _ = runner.calls[0]
    image_args = [args[index + 1] for index, value in enumerate(args) if value == "--image"]
    assert image_args == [reference["url"] for reference in references]
    assert result.artifact.metadata["reference_images_applied"] == 2
    assert not any("were not sent" in warning for warning in result.warnings)


def test_openart_image2image_requires_reference() -> None:
    with pytest.raises(ValidationError, match="requires at least one visual reference"):
        RenderPreviewRequest(
            design=_design(),
            backend=RendererKind.OPENART_GPT_IMAGE_25_FLARE,
            generation={"mode": "image2image"},
        )


def test_openart_visual_reference_rejects_arbitrary_remote_url() -> None:
    with pytest.raises(ValidationError, match="cdn.openart.ai"):
        RenderPreviewRequest(
            design=_design(),
            backend=RendererKind.OPENART_GPT_IMAGE_25_FLARE,
            generation={
                "mode": "image2image",
                "visual_references": [
                    {
                        "url": "https://example.com/internal-room.png",
                        "label": "unsafe",
                    }
                ],
            },
        )


def test_openart_rejects_unverified_quality_flag_instead_of_guessing_cli_option() -> None:
    runner = FakeOpenArtRunner()
    service = RenderingService(
        settings=_openart_settings(),
        renderer_client=runner,
    )

    with pytest.raises(RenderBackendUnavailable, match="medium quality tier only"):
        service.preview(
            RenderPreviewRequest(
                design=_design(),
                backend=RendererKind.OPENART_GPT_IMAGE_25_FLARE,
                generation={"quality": "high"},
            )
        )

    assert runner.calls == []


def test_openart_renderer_fails_closed_when_not_enabled() -> None:
    service = RenderingService(
        settings=_openart_settings(enabled=False),
        renderer_client=FakeOpenArtRunner(),
    )

    with pytest.raises(RenderBackendUnavailable, match="not configured"):
        service.preview(
            RenderPreviewRequest(
                design=_design(),
                backend=RendererKind.OPENART_GPT_IMAGE_25_FLARE,
            )
        )


def test_openart_renderer_rejects_invalid_image_output() -> None:
    runner = FakeOpenArtRunner(image_bytes=b"not-an-image")
    service = RenderingService(
        settings=_openart_settings(),
        renderer_client=runner,
    )

    with pytest.raises(RenderBackendError, match="returned an invalid image"):
        service.preview(
            RenderPreviewRequest(
                design=_design(),
                backend=RendererKind.OPENART_GPT_IMAGE_25_FLARE,
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
