from __future__ import annotations

from shapely.geometry import Polygon

from furniture_ai.contracts import FloorPlanAnalysis, Point, Room
from furniture_ai.layout import furnish_floor_plan, rectangle


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
