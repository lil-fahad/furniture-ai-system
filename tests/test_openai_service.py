from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import SecretStr, ValidationError

from furniture_ai.config import Settings
from furniture_ai.contracts import FloorPlanAnalysis, Point, Room
from furniture_ai.openai_service import OpenAIDesignService


class FakeResponses:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict[str, object]] = []
        self.usage = SimpleNamespace(input_tokens=101, output_tokens=17, total_tokens=118)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id="resp_test_123",
            output_text=self.output_text,
            usage=self.usage,
        )


class FakeClient:
    def __init__(self, output_text: str) -> None:
        self.responses = FakeResponses(output_text)


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


def make_service(output_text: str) -> tuple[OpenAIDesignService, FakeClient]:
    client = FakeClient(output_text)
    settings = Settings(environment="test", openai_api_key=SecretStr("fake-test-key"))
    return OpenAIDesignService(settings, client=client), client


def test_openai_refinement_uses_strict_structured_outputs() -> None:
    service, client = make_service(
        '{"rooms":[{"id":"room-1","room_type":"office","confidence":0.91}]}'
    )
    result = service.refine_room_types(image(), sample_plan())
    assert result["room-1"] == ("office", 0.91)

    call = client.responses.calls[0]
    format_config = call["text"]["format"]
    assert format_config["type"] == "json_schema"
    assert format_config["name"] == "room_refinements"
    assert format_config["strict"] is True
    assert format_config["schema"]["additionalProperties"] is False


def test_openai_telemetry_contains_no_prompt_content() -> None:
    service, _ = make_service(
        '{"rooms":[{"id":"room-1","room_type":"office","confidence":0.91}]}'
    )
    service.refine_room_types(image(), sample_plan())
    telemetry = service.last_telemetry
    assert telemetry is not None
    assert telemetry.operation == "refine_room_types"
    assert telemetry.model == service.model
    assert telemetry.response_id == "resp_test_123"
    assert telemetry.input_tokens == 101
    assert telemetry.output_tokens == 17
    assert telemetry.total_tokens == 118
    assert telemetry.latency_ms >= 0
    assert "prompt" not in telemetry.__dict__
    assert "input" not in telemetry.__dict__


def test_structured_output_rejects_null_or_non_list_rooms() -> None:
    service, _ = make_service('{"rooms": null}')
    with pytest.raises(ValidationError):
        service.refine_room_types(image(), sample_plan())

    service, _ = make_service('{"rooms": {"id": "room-1"}}')
    with pytest.raises(ValidationError):
        service.refine_room_types(image(), sample_plan())


def test_structured_output_rejects_malformed_items() -> None:
    service, _ = make_service(
        '{"rooms":[{"id":"room-1","room_type":"garage","confidence":0.7}]}'
    )
    with pytest.raises(ValidationError):
        service.refine_room_types(image(), sample_plan())

    service, _ = make_service(
        '{"rooms":[{"id":"room-1","room_type":"office","confidence":1.7}]}'
    )
    with pytest.raises(ValidationError):
        service.refine_room_types(image(), sample_plan())


def test_unknown_room_ids_are_rejected_locally_after_schema_validation() -> None:
    service, _ = make_service(
        '{"rooms":[{"id":"room-2","room_type":"kitchen","confidence":0.7}]}'
    )
    assert service.refine_room_types(image(), sample_plan()) == {}


def test_unconfigured_key_raises_unavailable() -> None:
    from furniture_ai.openai_service import OpenAIUnavailable

    with pytest.raises(OpenAIUnavailable, match="OPENAI_API_KEY"):
        OpenAIDesignService(Settings(environment="test", openai_api_key=None))


def test_json_object_legacy_helper_rejects_missing_or_non_object_json() -> None:
    from furniture_ai.openai_service import _json_object

    with pytest.raises(ValueError, match="did not contain a JSON object"):
        _json_object("no json here")
    with pytest.raises(ValueError, match="did not contain a JSON object"):
        _json_object("[1, 2, 3]")
    assert _json_object('```json\n{"rooms": []}\n```') == {"rooms": []}


def test_create_design_brief() -> None:
    service, _ = make_service("  A calm, functional layout.  ")
    brief = service.create_design_brief(sample_plan(), "minimalist")
    assert brief == "A calm, functional layout."
    assert service.last_telemetry is not None
    assert service.last_telemetry.operation == "create_design_brief"


def test_pipeline_treats_malformed_payload_as_service_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import furniture_ai.pipeline as pipeline_module
    from furniture_ai.pipeline import DesignPipeline

    class MalformedService:
        def __init__(self, settings) -> None:
            pass

        def refine_room_types(self, image, floor_plan):
            raise ValueError("Structured room response was invalid")

    monkeypatch.setattr(pipeline_module, "OpenAIDesignService", MalformedService)
    settings = Settings(environment="test", openai_api_key=SecretStr("fake-test-key"))
    plan_image = Image.new("RGB", (300, 200), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(plan_image)
    draw.rectangle((10, 10, 290, 190), outline="black", width=8)
    result = DesignPipeline(settings).run(plan_image, use_openai=True)
    assert any("OpenAI refinement unavailable" in warning for warning in result.warnings)
