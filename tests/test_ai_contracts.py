from __future__ import annotations

import pytest
from pydantic import ValidationError

from furniture_ai.ai_contracts import RoomRefinementResponse, room_refinement_json_schema


def test_room_refinement_contract_accepts_supported_values() -> None:
    response = RoomRefinementResponse.model_validate(
        {
            "rooms": [
                {"id": "room-1", "room_type": "office", "confidence": 0.91},
                {"id": "room-2", "room_type": "living_room", "confidence": 0.62},
            ]
        }
    )
    assert response.rooms[0].id == "room-1"
    assert response.rooms[1].room_type == "living_room"


def test_room_refinement_contract_rejects_unknown_type_and_extra_fields() -> None:
    with pytest.raises(ValidationError):
        RoomRefinementResponse.model_validate(
            {"rooms": [{"id": "room-1", "room_type": "garage", "confidence": 0.7}]}
        )
    with pytest.raises(ValidationError):
        RoomRefinementResponse.model_validate(
            {
                "rooms": [
                    {
                        "id": "room-1",
                        "room_type": "office",
                        "confidence": 0.7,
                        "invented_geometry": [1, 2, 3],
                    }
                ]
            }
        )


def test_room_refinement_contract_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        RoomRefinementResponse.model_validate(
            {"rooms": [{"id": "room-1", "room_type": "office", "confidence": 1.1}]}
        )


def test_json_schema_is_strict_enough_for_structured_outputs() -> None:
    schema = room_refinement_json_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    rooms_schema = schema["properties"]["rooms"]
    assert rooms_schema["type"] == "array"
