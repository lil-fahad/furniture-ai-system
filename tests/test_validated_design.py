from __future__ import annotations

import furniture_ai.api as api_module
from furniture_ai.contracts import (
    DesignResult,
    FloorPlanAnalysis,
    LayoutValidationReport,
    Point,
    Room,
    ValidatedDesignRequest,
)


def _plan() -> FloorPlanAnalysis:
    return FloorPlanAnalysis(
        source_width=500,
        source_height=400,
        rooms=[
            Room(
                id="room-1",
                room_type="living_room",
                polygon=[
                    Point(x=0, y=0),
                    Point(x=400, y=0),
                    Point(x=400, y=300),
                    Point(x=0, y=300),
                ],
                area=120000,
            )
        ],
    )


def test_validated_design_gates_execution_on_constraints(monkeypatch) -> None:
    plan = _plan()
    design = DesignResult(floor_plan=plan, placed_items=2)
    validation = LayoutValidationReport(
        valid=False,
        checked_rooms=1,
        checked_items=2,
        issues=[],
    )
    captured: dict[str, object] = {}

    def fake_furnish(floor_plan, *, room_type_overrides):
        captured["room_types"] = room_type_overrides
        return design

    def fake_validate(floor_plan, *, minimum_clearance):
        captured["minimum_clearance"] = minimum_clearance
        return validation

    monkeypatch.setattr(api_module, "furnish_floor_plan", fake_furnish)
    monkeypatch.setattr(api_module, "validate_layout_constraints", fake_validate)

    result = api_module.create_validated_design(
        ValidatedDesignRequest(
            floor_plan=plan,
            room_types={"room-1": "office"},
            minimum_clearance=12.5,
        )
    )
    assert result.design is design
    assert result.validation is validation
    assert result.execution_ready is False
    assert captured["room_types"] == {"room-1": "office"}
    assert captured["minimum_clearance"] == 12.5


def test_validated_design_marks_only_valid_layout_execution_ready(monkeypatch) -> None:
    plan = _plan()
    design = DesignResult(floor_plan=plan, placed_items=0)
    validation = LayoutValidationReport(
        valid=True,
        checked_rooms=1,
        checked_items=0,
        issues=[],
    )
    monkeypatch.setattr(
        api_module,
        "furnish_floor_plan",
        lambda floor_plan, *, room_type_overrides: design,
    )
    monkeypatch.setattr(
        api_module,
        "validate_layout_constraints",
        lambda floor_plan, *, minimum_clearance: validation,
    )

    result = api_module.create_validated_design(ValidatedDesignRequest(floor_plan=plan))
    assert result.execution_ready is True
