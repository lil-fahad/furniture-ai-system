from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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


class RoomRefinement(BaseModel):
    """One AI-proposed semantic refinement for an existing local room id."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    room_type: RoomType
    confidence: float = Field(ge=0.0, le=1.0)


class RoomRefinementResponse(BaseModel):
    """Strict structured-output contract for room semantic refinement."""

    model_config = ConfigDict(extra="forbid")

    rooms: list[RoomRefinement] = Field(max_length=100)


def room_refinement_json_schema() -> dict[str, object]:
    """Return the JSON Schema supplied to OpenAI Structured Outputs."""

    return RoomRefinementResponse.model_json_schema()
