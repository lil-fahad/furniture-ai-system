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
    """Schema for one AI-proposed semantic refinement.

    Geometry and identity remain local concerns. The model may only propose a
    supported semantic label and bounded confidence for a supplied room id.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=120)
    room_type: RoomType
    confidence: float = Field(ge=0.0, le=1.0)


class RoomRefinementResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rooms: list[RoomRefinement] = Field(default_factory=list, max_length=100)


def room_refinement_json_schema() -> dict[str, object]:
    """Return the strict JSON Schema used by the OpenAI Responses API.

    Keeping schema generation local lets tests validate the contract without
    making network calls or depending on a specific model.
    """

    return RoomRefinementResponse.model_json_schema()
