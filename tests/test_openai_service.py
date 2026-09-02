from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import SecretStr

from furniture_ai.config import Settings
from furniture_ai.contracts import FloorPlanAnalysis, Point, Room
from furniture_ai.openai_service import OpenAIDesignService


class FakeResponses:
    def create(self, **kwargs):
        return SimpleNamespace(
            output_text='{"rooms":[{"id":"room-1","room_type":"office","confidence":0.91}]}'
        )


class FakeClient:
    responses = FakeResponses()


def test_openai_refinement_parser_without_network() -> None:
    settings = Settings(environment="test", openai_api_key=SecretStr("fake-test-key"))
    service = OpenAIDesignService(settings, client=FakeClient())
    plan = FloorPlanAnalysis(
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
    result = service.refine_room_types(Image.new("RGB", (100, 100), "white"), plan)
    assert result["room-1"] == ("office", 0.91)


def make_service(output_text: str) -> OpenAIDesignService:
    class Responses:
        def create(self, **kwargs):
            return SimpleNamespace(output_text=output_text)

    class Client:
        responses = Responses()

    settings = Settings(environment="test", openai_api_key=SecretStr("fake-test-key"))
    return OpenAIDesignService(settings, client=Client())


def sample_plan() -> FloorPlanAnalysis:
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


def image() -> Image.Image:
    return Image.new("RGB", (100, 100), "white")


def test_null_rooms_payload_raises_value_error() -> None:
    service = make_service('{"rooms": null}')
    with pytest.raises(ValueError, match="rooms"):
        service.refine_room_types(image(), sample_plan())


def test_non_list_rooms_payload_raises_value_error() -> None:
    service = make_service('{"rooms": {"id": "room-1"}}')
    with pytest.raises(ValueError, match="rooms"):
        service.refine_room_types(image(), sample_plan())


def test_malformed_items_and_unknown_room_ids_are_skipped() -> None:
    service = make_service(
        '{"rooms": ['
        '"not-a-dict", '
        '{"id": "room-1", "room_type": "kitchen", "confidence": "high"}, '
        '{"id": "room-1", "room_type": "office"}, '
        '{"id": "", "room_type": "bedroom", "confidence": 0.9}, '
        '{"id": "room-2", "room_type": "Living Room", "confidence": 1.7}'
        "]}"
    )
    result = service.refine_room_types(image(), sample_plan())
    assert result == {"room-1": ("office", 0.5)}


def test_nan_confidence_is_skipped() -> None:
    service = make_service('{"rooms": [{"id": "room-1", "confidence": NaN}]}')
    assert service.refine_room_types(image(), sample_plan()) == {}


def test_unconfigured_key_raises_unavailable() -> None:
    from furniture_ai.openai_service import OpenAIUnavailable

    with pytest.raises(OpenAIUnavailable, match="OPENAI_API_KEY"):
        OpenAIDesignService(Settings(environment="test", openai_api_key=None))


def test_json_object_rejects_missing_or_non_object_json() -> None:
    from furniture_ai.openai_service import _json_object

    with pytest.raises(ValueError, match="did not contain a JSON object"):
        _json_object("no json here")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        _json_object("[1, 2, 3]")
    assert _json_object('```json\n{"rooms": []}\n```') == {"rooms": []}


def test_create_design_brief() -> None:
    service = make_service("  A calm, functional layout.  ")
    brief = service.create_design_brief(sample_plan(), "minimalist")
    assert brief == "A calm, functional layout."


def test_pipeline_treats_malformed_payload_as_service_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furniture_ai.pipeline as pipeline_module
    from furniture_ai.pipeline import DesignPipeline

    class MalformedService:
        def __init__(self, settings) -> None:
            pass

        def refine_room_types(self, image, floor_plan):
            raise ValueError("The model response 'rooms' field must be a list")

    monkeypatch.setattr(pipeline_module, "OpenAIDesignService", MalformedService)
    settings = Settings(environment="test", openai_api_key=SecretStr("fake-test-key"))
    plan_image = Image.new("RGB", (300, 200), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(plan_image)
    draw.rectangle((10, 10, 290, 190), outline="black", width=8)
    result = DesignPipeline(settings).run(plan_image, use_openai=True)
    assert any("OpenAI refinement unavailable" in warning for warning in result.warnings)
