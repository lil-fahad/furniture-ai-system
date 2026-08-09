from __future__ import annotations

import base64
import json
import math
import re
from io import BytesIO
from typing import Any

from PIL import Image

from furniture_ai.config import Settings
from furniture_ai.contracts import FloorPlanAnalysis


class OpenAIUnavailable(RuntimeError):
    pass


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _json_object(text: str) -> dict[str, object]:
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
        prompt = (
            "Analyze this architectural floor plan. Return JSON only with this shape: "
            '{"rooms":[{"id":"room-1","room_type":"living_room",'
            '"confidence":0.85}]}. Use the supplied room ids exactly. Prefer these labels: '
            "living_room, bedroom, kitchen, bathroom, dining_room, office, hallway, storage, "
            f"balcony, room. Geometric candidates: {json.dumps(room_summary)}"
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
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
            ],
        )
        payload = _json_object(response.output_text)
        rooms = payload.get("rooms", [])
        if not isinstance(rooms, list):
            raise ValueError("The model response 'rooms' field must be a list")
        result: dict[str, tuple[str, float]] = {}
        for item in rooms:
            if not isinstance(item, dict):
                continue
            room_id = str(item.get("id", ""))
            room_type = str(item.get("room_type", "room")).strip().lower().replace(" ", "_")
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(confidence):
                continue
            confidence = min(max(confidence, 0.0), 1.0)
            if room_id:
                result[room_id] = (room_type, confidence)
        return result

    def create_design_brief(self, floor_plan: FloorPlanAnalysis, preferences: str) -> str:
        prompt = (
            "Create a concise professional interior-design brief. Respect circulation, doors, room "
            "dimensions, and the furniture placements in the supplied JSON. Do not claim "
            "exact physical "
            "dimensions when pixels_per_cm is absent. Preferences: "
            f"{preferences}\nFloor plan JSON: {floor_plan.model_dump_json()}"
        )
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text.strip()
