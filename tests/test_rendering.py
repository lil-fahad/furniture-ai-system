from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from furniture_ai.api_entry import app
from furniture_ai.contracts import FloorPlanAnalysis, Point, Room
from furniture_ai.layout import furnish_floor_plan
from furniture_ai.rendering import PromptCompiler, RenderPreviewRequest, RenderingService, SceneCompiler


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
    )
    return furnish_floor_plan(floor_plan)


def test_scene_compiler_grounds_catalog_products_and_normalizes_style() -> None:
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


def test_prompt_compiler_is_deterministic_and_grounding_first() -> None:
    scene = SceneCompiler().compile(_design(), style="warm modern")
    compiler = PromptCompiler()

    first = compiler.compile(scene)
    second = compiler.compile(scene)

    assert first == second
    assert len(first.scene_fingerprint) == 64
    assert "Preserve the supplied room geometry" in first.positive_prompt
    assert "Three-seat sofa" in first.positive_prompt
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
    assert result.warnings


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
    assert payload["artifact"]["backend"] == "mock"
    assert payload["artifact"]["data_uri"].startswith("data:image/svg+xml;base64,")


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
