from __future__ import annotations

from types import SimpleNamespace

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
