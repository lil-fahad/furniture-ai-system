from __future__ import annotations

import base64
import json
import math
import re
from io import BytesIO
from typing import Any, Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from furniture_ai.config import Settings
from furniture_ai.contracts import FloorPlanAnalysis

RoomType = Literal[
    "living_room",
    "bedroom",
    "kitchen",
    "bathroom",
    "dining_room",
    "office",
    "hallway",
    "storage",
    "balcony",
    "room",
]
ROOM_TYPE_LABELS = frozenset(RoomType.__args__)


class OpenAIUnavailable(RuntimeError):
    pass


class RoomRefinement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    room_type: RoomType
    confidence: float = Field(ge=0, le=1)


class RoomRefinementResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rooms: list[RoomRefinement]


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _json_object(text: str) -> dict[str, object]:
    """Parse a JSON object from a legacy text response.

    Modern OpenAI SDKs use Structured Outputs through ``responses.parse``.
    This parser remains as a compatibility path for older clients and simple
    deterministic test doubles.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The model response did not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("The model response must be a JSON object")
    return value


def _legacy_refinements(
    payload: dict[str, object], valid_room_ids: set[str]
) -> dict[str, tuple[str, float]]:
    """Lenient compatibility parser with strict labels and room-id scoping."""
    rooms = payload.get("rooms", [])
    if not isinstance(rooms, list):
        raise ValueError("The model response 'rooms' field must be a list")

    result: dict[str, tuple[str, float]] = {}
    for item in rooms:
        if not isinstance(item, dict):
            continue
        room_id = str(item.get("id", ""))
        if room_id not in valid_room_ids:
            continue
        room_type = str(item.get("room_type", "room")).strip().lower().replace(" ", "_")
        if room_type not in ROOM_TYPE_LABELS:
            continue
        try:
            confidence = float(item.get("confidence", 0.5))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence):
            continue
        result[room_id] = (room_type, min(max(confidence, 0.0), 1.0))
    return result


class OpenAIDesignService:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        if not settings.openai_configured:
            raise OpenAIUnavailable("OPENAI_API_KEY is not configured")
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise OpenAIUnavailable("Install the openai package to use AI refinement") from exc
            client = OpenAI(api_key=key)
        self.client = client
        self.model = settings.openai_model

    def refine_room_types(
        self,
        image: Image.Image,
        floor_plan: FloorPlanAnalysis,
    ) -> dict[str, tuple[str, float]]:
        room_summary = [
            {"id": room.id, "area": room.area, "current_type": room.room_type}
            for room in floor_plan.rooms
        ]
        valid_room_ids = {room.id for room in floor_plan.rooms}
        prompt = (
            "Analyze this architectural floor plan and classify only the supplied room IDs. "
            "Never invent room IDs. Use only these room types: living_room, bedroom, kitchen, "
            "bathroom, dining_room, office, hallway, storage, balcony, room. "
            f"Geometric candidates: {json.dumps(room_summary)}"
        )
        request_input = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(image),
                        "detail": "high",
                    },
                ],
            }
        ]

        parse = getattr(self.client.responses, "parse", None)
        if callable(parse):
            response = parse(
                model=self.model,
                input=request_input,
                text_format=RoomRefinementResponse,
            )
            parsed = getattr(response, "output_parsed", None)
            if parsed is None:
                raise ValueError("The model did not return a structured room refinement")
            if not isinstance(parsed, RoomRefinementResponse):
                parsed = RoomRefinementResponse.model_validate(parsed)
            return {
                item.id: (item.room_type, item.confidence)
                for item in parsed.rooms
                if item.id in valid_room_ids
            }

        response = self.client.responses.create(model=self.model, input=request_input)
        return _legacy_refinements(_json_object(response.output_text), valid_room_ids)

    def create_design_brief(self, floor_plan: FloorPlanAnalysis, preferences: str) -> str:
        prompt = (
            "Create a concise professional interior-design brief. Respect circulation, doors, room "
            "dimensions, and the furniture placements in the supplied JSON. Do not claim exact "
            "physical dimensions when pixels_per_cm is absent. Preferences: "
            f"{preferences}\nFloor plan JSON: {floor_plan.model_dump_json()}"
        )
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text.strip()
