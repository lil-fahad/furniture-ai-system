from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from furniture_ai.contracts import FloorPlanAnalysis, Point, Room
from furniture_ai.layout import furnish_floor_plan, rectangle, room_polygon


def test_layout_is_deterministic_and_non_overlapping() -> None:
    floor_plan = FloorPlanAnalysis(
        source_width=800,
        source_height=600,
        rooms=[
            Room(
                id="room-1",
                room_type="living_room",
                polygon=[
                    Point(x=20, y=20),
                    Point(x=780, y=20),
                    Point(x=780, y=580),
                    Point(x=20, y=580),
                ],
                area=425_600,
            )
        ],
    )
    first = furnish_floor_plan(floor_plan.model_copy(deep=True))
    second = furnish_floor_plan(floor_plan.model_copy(deep=True))
    assert first.model_dump() == second.model_dump()
    placements = first.floor_plan.rooms[0].furniture
    assert placements
    shapes = [
        rectangle(item.center.x, item.center.y, item.width, item.depth, item.rotation_degrees)
        for item in placements
    ]
    room = Polygon([(20, 20), (780, 20), (780, 580), (20, 580)])
    assert all(room.covers(shape) for shape in shapes)
    for index, shape in enumerate(shapes):
        assert not any(shape.intersects(other) for other in shapes[index + 1 :])


def _room(room_id: str, room_type: str, size: float = 560.0) -> Room:
    return Room(
        id=room_id,
        room_type=room_type,
        polygon=[
            Point(x=20, y=20),
            Point(x=20 + size, y=20),
            Point(x=20 + size, y=20 + size),
            Point(x=20, y=20 + size),
        ],
        area=size * size,
    )


def test_room_polygon_rejects_collinear_points() -> None:
    with pytest.raises(ValueError, match="degenerate|invalid"):
        room_polygon([Point(x=0, y=0), Point(x=1, y=1), Point(x=2, y=2)])


def test_room_polygon_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least three"):
        room_polygon([Point(x=0, y=0), Point(x=1, y=1)])


def test_room_polygon_accepts_square() -> None:
    polygon = room_polygon(
        [Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10), Point(x=0, y=10)]
    )
    assert polygon.area == pytest.approx(100.0)


def test_furnish_does_not_mutate_input() -> None:
    floor_plan = FloorPlanAnalysis(
        source_width=600,
        source_height=600,
        rooms=[_room("room-1", "living_room")],
    )
    snapshot = floor_plan.model_dump()
    furnish_floor_plan(floor_plan, room_type_overrides={"room-1": "office"})
    assert floor_plan.model_dump() == snapshot
    assert floor_plan.rooms[0].room_type == "living_room"
    assert floor_plan.rooms[0].furniture == []


def test_furnish_handles_plan_without_rooms() -> None:
    floor_plan = FloorPlanAnalysis(source_width=100, source_height=100, rooms=[])
    result = furnish_floor_plan(floor_plan)
    assert result.placed_items == 0
    assert result.floor_plan.rooms == []


def test_furnish_handles_tiny_scale() -> None:
    floor_plan = FloorPlanAnalysis(
        source_width=600,
        source_height=600,
        rooms=[_room("room-1", "living_room")],
        pixels_per_cm=1e-9,
    )
    result = furnish_floor_plan(floor_plan)
    assert result.placed_items > 0
    assert all(
        item.dimension_source == "physical" for item in result.floor_plan.rooms[0].furniture
    )


def test_unknown_room_type_warning_is_distinct() -> None:
    floor_plan = FloorPlanAnalysis(
        source_width=600,
        source_height=600,
        rooms=[_room("room-1", "unknown_type")],
    )
    result = furnish_floor_plan(floor_plan)
    assert result.placed_items == 0
    assert any("No catalog products match room type" in warning for warning in result.warnings)
    assert not any("fit inside" in warning for warning in result.warnings)
