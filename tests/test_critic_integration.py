from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from furniture_ai import api, pipeline as pipeline_module
from furniture_ai.api import app
from furniture_ai.config import Settings
from furniture_ai.contracts import (
    DesignResult,
    FloorPlanAnalysis,
    FurniturePlacement,
    Point,
    Room,
)
from furniture_ai.critic import DesignCriticRejected, SpatialDesignCritic
from furniture_ai.pipeline import DesignPipeline


def empty_plan() -> FloorPlanAnalysis:
    return FloorPlanAnalysis(source_width=100, source_height=100)


def invalid_result() -> DesignResult:
    return DesignResult(
        floor_plan=FloorPlanAnalysis(
            source_width=100,
            source_height=100,
            rooms=[
                Room(
                    id="room-1",
                    room_type="living_room",
                    polygon=[
                        Point(x=0, y=0),
                        Point(x=100, y=0),
                        Point(x=100, y=100),
                        Point(x=0, y=100),
                    ],
                    area=10_000,
                    furniture=[
                        FurniturePlacement(
                            id="outside",
                            category="chair",
                            center=Point(x=95, y=50),
                            width=20,
                            depth=20,
                        )
                    ],
                )
            ],
        ),
        placed_items=1,
    )


class FakeAnalyzer:
    def analyze(self, image, *, pixels_per_cm=None):
        return empty_plan()


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_pipeline_rejects_invalid_generator_output(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = DesignPipeline(Settings(environment="test"))
    pipeline.analyzer = FakeAnalyzer()
    monkeypatch.setattr(pipeline_module, "furnish_floor_plan", lambda floor_plan: invalid_result())

    with pytest.raises(DesignCriticRejected):
        pipeline.run(Image.new("RGB", (64, 64), "white"))


def test_analyze_api_maps_critic_rejection_to_503(monkeypatch: pytest.MonkeyPatch) -> None:
    report = SpatialDesignCritic().inspect(invalid_result())

    class RejectingPipeline:
        def __init__(self, settings) -> None:
            self.settings = settings

        def run(self, image, **kwargs):
            raise DesignCriticRejected(report)

    monkeypatch.setattr(api, "DesignPipeline", RejectingPipeline)
    response = TestClient(app).post(
        "/api/v1/analyze",
        files={"image": ("plan.png", png_bytes(), "image/png")},
        data={"use_openai": "false", "preferences": ""},
    )
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Generated layout failed deterministic spatial validation"
    }
