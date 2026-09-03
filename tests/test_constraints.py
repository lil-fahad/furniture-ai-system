from __future__ import annotations

import math

import pytest

from furniture_ai.constraints import validate_layout_constraints
from furniture_ai.contracts import (
    FloorPlanAnalysis,
    FurniturePlacement,
    Opening,
    OpeningKind,
    Point,
    Room,
)


def placement(
    item_id: str,
    x: float,
    y: float,
    width: float = 20,
    depth: float = 20,
    rotation: float = 0,
) -> FurniturePlacement:
    return FurniturePlacement(
        id=item_id,
        category="test",
        center=Point(x=x, y=y),
        width=width,
        depth=depth,
        rotation_degrees=rotation,
    )


def plan_with(
    *items: FurniturePlacement,
    openings: list[Opening] | None = None,
) -> FloorPlanAnalysis:
    return FloorPlanAnalysis(
        source_width=100,
        source_height=100,
        rooms=[
            Room(
                id="room-1",
                room_type="room",
                polygon=[
                    Point(x=0, y=0),
                    Point(x=100, y=0),
                    Point(x=100, y=100),
                    Point(x=0, y=100),
                ],
                area=10_000,
                furniture=list(items),
            )
        ],
        openings=openings or [],
    )


def issue_codes(report) -> set[str]:
    return {issue.code for issue in report.issues}


def test_valid_layout_has_no_issues() -> None:
    report = validate_layout_constraints(
        plan_with(placement("a", 20, 20), placement("b", 70, 70))
    )
    assert report.valid is True
    assert report.checked_rooms == 1
    assert report.checked_items == 2
    assert report.issues == []


def test_collision_is_rejected() -> None:
    report = validate_layout_constraints(
        plan_with(placement("a", 30, 30), placement("b", 35, 30))
    )
    assert report.valid is False
    assert "collision" in issue_codes(report)


def test_furniture_outside_room_is_rejected() -> None:
    report = validate_layout_constraints(plan_with(placement("outside", 95, 50)))
    assert report.valid is False
    assert "outside_room" in issue_codes(report)


def test_rotated_furniture_outside_room_is_rejected() -> None:
    report = validate_layout_constraints(
        plan_with(placement("rotated", 88, 88, width=20, depth=40, rotation=45))
    )
    assert report.valid is False
    assert "outside_room" in issue_codes(report)


def test_door_intersection_is_rejected() -> None:
    door = Opening(
        id="door-1",
        kind=OpeningKind.DOOR,
        start=Point(x=0, y=45),
        end=Point(x=0, y=55),
    )
    report = validate_layout_constraints(
        plan_with(placement("blocking", 5, 50, width=10, depth=10), openings=[door])
    )
    assert report.valid is False
    issue = next(issue for issue in report.issues if issue.code == "door_blocked")
    assert issue.opening_id == "door-1"
    assert issue.item_ids == ["blocking"]


def test_requested_clearance_is_enforced_without_assuming_a_default() -> None:
    plan = plan_with(placement("a", 20, 20), placement("b", 50, 20))
    default_report = validate_layout_constraints(plan)
    strict_report = validate_layout_constraints(plan, minimum_clearance=15)

    assert default_report.valid is True
    assert strict_report.valid is False
    assert "clearance_violation" in issue_codes(strict_report)


def test_zero_area_furniture_footprint_is_rejected() -> None:
    report = validate_layout_constraints(plan_with(placement("zero", 50, 50, width=0)))
    assert report.valid is False
    assert "invalid_footprint" in issue_codes(report)


def test_non_finite_furniture_footprint_is_rejected_before_shapely() -> None:
    report = validate_layout_constraints(
        plan_with(placement("infinite", 50, 50, width=math.inf))
    )
    assert report.valid is False
    assert "invalid_footprint" in issue_codes(report)


def test_negative_clearance_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_layout_constraints(plan_with(), minimum_clearance=-1)


def test_non_finite_clearance_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_layout_constraints(plan_with(), minimum_clearance=math.inf)
