from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image

from furniture_ai.ai_contracts import RoomRefinementResponse, room_refinement_json_schema
from furniture_ai.config import Settings
from furniture_ai.contracts import FloorPlanAnalysis


class OpenAIUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class OpenAICallTelemetry:
    operation: str
    model: str
    latency_ms: float
    response_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _json_object(text: str) -> dict[str, object]:
    """Parse a JSON object from text for backward-compatible utility callers.

    Room refinement no longer relies on this tolerant parser; it uses strict
    Structured Outputs and a typed Pydantic contract instead.
    """

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE
        )
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("The model response did not contain a JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("The model response must be a JSON object")
    return value


def _usage_value(usage: object | None, name: str) -> int | None:
    value = getattr(usage, name, None)
    return value if isinstance(value, int) and value >= 0 else None


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
            client = OpenAI(
                api_key=key,
                timeout=settings.openai_timeout_seconds,
                max_retries=1,
            )
        self.client = client
        self.model = settings.openai_model
        self.last_telemetry: OpenAICallTelemetry | None = None

    def _create_response(self, operation: str, **kwargs: object) -> Any:
        started = time.perf_counter()
        response = self.client.responses.create(**kwargs)
        latency_ms = max((time.perf_counter() - started) * 1000.0, 0.0)
        usage = getattr(response, "usage", None)
        response_id = getattr(response, "id", None)
        self.last_telemetry = OpenAICallTelemetry(
            operation=operation,
            model=self.model,
            latency_ms=latency_ms,
            response_id=response_id if isinstance(response_id, str) else None,
            input_tokens=_usage_value(usage, "input_tokens"),
            output_tokens=_usage_value(usage, "output_tokens"),
            total_tokens=_usage_value(usage, "total_tokens"),
        )
        return response

    def refine_room_types(
        self,
        image: Image.Image,
        floor_plan: FloorPlanAnalysis,
    ) -> dict[str, tuple[str, float]]:
        room_summary = [
            {"id": room.id, "area": room.area, "current_type": room.room_type}
            for room in floor_plan.rooms
        ]
        valid_ids = {room.id for room in floor_plan.rooms}
        prompt = (
            "Analyze the architectural floor plan and refine only the semantic room labels for "
            "the supplied geometric candidates. Use the supplied room ids exactly. Do not create "
            "rooms, coordinates, dimensions, openings, or furniture. If a room is visually "
            "ambiguous, lower the confidence rather than inventing detail. Geometric candidates: "
            f"{json.dumps(room_summary)}"
        )
        response = self._create_response(
            "refine_room_types",
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
            text={
                "format": {
                    "type": "json_schema",
                    "name": "room_refinements",
                    "schema": room_refinement_json_schema(),
                    "strict": True,
                }
            },
        )
        structured = RoomRefinementResponse.model_validate_json(response.output_text)
        result: dict[str, tuple[str, float]] = {}
        for item in structured.rooms:
            if item.id not in valid_ids:
                continue
            result[item.id] = (item.room_type, item.confidence)
        return result

    def create_design_brief(self, floor_plan: FloorPlanAnalysis, preferences: str) -> str:
        prompt = (
            "Create a concise professional interior-design brief. Respect circulation, doors, room "
            "dimensions, and the furniture placements in the supplied JSON. "
            "Do not claim exact physical dimensions when pixels_per_cm is absent. "
            f"Preferences: {preferences}\nFloor plan JSON: {floor_plan.model_dump_json()}"
        )
        response = self._create_response(
            "create_design_brief",
            model=self.model,
            input=prompt,
        )
        return response.output_text.strip()
