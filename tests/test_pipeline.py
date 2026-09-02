from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from pydantic import SecretStr

from furniture_ai import pipeline as pipeline_module
from furniture_ai.config import Settings
from furniture_ai.openai_service import OpenAIDesignService, OpenAIUnavailable
from furniture_ai.pipeline import DesignPipeline


def synthetic_plan() -> Image.Image:
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 570, 370), outline="black", width=12)
    draw.line((300, 30, 300, 370), fill="black", width=12)
    return image


@pytest.fixture()
def pipeline() -> DesignPipeline:
    return DesignPipeline(Settings(environment="test"))


class FakeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def refine_room_types(self, image, floor_plan):
        return {"room-1": ("office", 0.93)}

    def create_design_brief(self, floor_plan, preferences: str) -> str:
        return f"Brief: {preferences}"


class NullRoomsService(OpenAIDesignService):
    """Drives the real refine_room_types with a malformed {"rooms": null} payload."""

    def __init__(self, settings: Settings) -> None:
        keyed = settings.model_copy(update={"openai_api_key": SecretStr("fake-test-key")})
        client = SimpleNamespace(responses=_Responses('{"rooms": null}'))
        super().__init__(keyed, client=client)


def test_pipeline_without_openai(pipeline: DesignPipeline) -> None:
    result = pipeline.run(synthetic_plan(), use_openai=False)
    assert result.placed_items > 0
    assert "+openai-vision" not in result.floor_plan.analysis_method
    assert result.design_brief is None


def test_pipeline_openai_happy_path(
    pipeline: DesignPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_module, "OpenAIDesignService", FakeService)
    result = pipeline.run(synthetic_plan(), use_openai=True, preferences="cozy")
    assert result.floor_plan.analysis_method.endswith("+openai-vision")
    room_one = next(room for room in result.floor_plan.rooms if room.id == "room-1")
    assert room_one.room_type == "office"
    assert room_one.confidence == pytest.approx(0.93)
    assert result.design_brief == "Brief: cozy"


def test_pipeline_openai_unavailable_falls_back(
    pipeline: DesignPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(settings: Settings):
        raise OpenAIUnavailable("OPENAI_API_KEY is not configured")

    monkeypatch.setattr(pipeline_module, "OpenAIDesignService", unavailable)
    result = pipeline.run(synthetic_plan(), use_openai=True, preferences="cozy")
    assert any("OpenAI refinement unavailable" in warning for warning in result.warnings)
    assert "+openai-vision" not in result.floor_plan.analysis_method
    assert result.placed_items > 0
    assert result.design_brief is None


def test_pipeline_openai_malformed_payload_falls_back(
    pipeline: DesignPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_module, "OpenAIDesignService", NullRoomsService)
    result = pipeline.run(synthetic_plan(), use_openai=True)
    assert any("OpenAI refinement unavailable" in warning for warning in result.warnings)
    assert result.placed_items > 0


def test_pipeline_design_brief_failure_is_a_warning(
    pipeline: DesignPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BriefFails(FakeService):
        def create_design_brief(self, floor_plan, preferences: str) -> str:
            raise RuntimeError("upstream timeout")

    monkeypatch.setattr(pipeline_module, "OpenAIDesignService", BriefFails)
    result = pipeline.run(synthetic_plan(), use_openai=True, preferences="cozy")
    assert result.design_brief is None
    assert any("OpenAI design brief unavailable" in warning for warning in result.warnings)


def test_pipeline_runs_without_model_bundle(pipeline: DesignPipeline) -> None:
    # No professional checkpoints are installed in the test environment; the
    # OpenCV fallback analyzer must still produce a furnished result.
    result = pipeline.run(synthetic_plan())
    assert result.floor_plan.rooms
    assert result.placed_items > 0


class _Responses:
    def __init__(self, text: str) -> None:
        self.text = text

    def create(self, **kwargs):
        return SimpleNamespace(output_text=self.text)


def _service_with_payload(text: str):
    from pydantic import SecretStr

    from furniture_ai.openai_service import OpenAIDesignService

    settings = Settings(environment="test", openai_api_key=SecretStr("fake-test-key"))
    client = SimpleNamespace(responses=_Responses(text))
    return OpenAIDesignService(settings, client=client)


def _plan():
    from furniture_ai.contracts import FloorPlanAnalysis, Point, Room

    return FloorPlanAnalysis(
        source_width=100,
        source_height=100,
        rooms=[
            Room(
                id="room-1",
                room_type="room",
                polygon=[Point(x=0, y=0), Point(x=90, y=0), Point(x=90, y=90)],
                area=4050,
            )
        ],
    )


def test_refine_room_types_rejects_null_rooms() -> None:
    service = _service_with_payload('{"rooms": null}')
    with pytest.raises(ValueError, match="'rooms' field must be a list"):
        service.refine_room_types(Image.new("RGB", (64, 64), "white"), _plan())


def test_refine_room_types_skips_junk_and_unknown_room_ids() -> None:
    payload = (
        '{"rooms": ["junk", {"id": "room-1", "room_type": "office", "confidence": "high"},'
        ' {"id": "room-2", "room_type": "kitchen", "confidence": 0.7}]}'
    )
    service = _service_with_payload(payload)
    result = service.refine_room_types(Image.new("RGB", (64, 64), "white"), _plan())
    assert result == {}
