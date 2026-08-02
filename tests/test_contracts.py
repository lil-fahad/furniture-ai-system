import pytest
from pydantic import ValidationError

from furniture_system.contracts import (
    BoundingBox,
    FloorPlanAnalysis,
    FurniturePlacement,
    Point,
    Room,
    Unit,
)


def test_bounding_box_area() -> None:
    box = BoundingBox(x=10, y=20, width=30, height=40)
    assert box.area == 1200


def test_floor_plan_requires_scale_for_physical_units() -> None:
    with pytest.raises(ValidationError):
        FloorPlanAnalysis(source_width=1000, source_height=800, unit=Unit.METER)


def test_floor_plan_contract_accepts_furniture_layout() -> None:
    analysis = FloorPlanAnalysis(
        source_width=1000,
        source_height=800,
        rooms=[
            Room(
                id="room-1",
                room_type="living_room",
                polygon=[
                    Point(x=0, y=0),
                    Point(x=500, y=0),
                    Point(x=500, y=400),
                    Point(x=0, y=400),
                ],
                furniture=[
                    FurniturePlacement(
                        id="sofa-1",
                        category="sofa",
                        center=Point(x=250, y=300),
                        width=200,
                        depth=90,
                    )
                ],
            )
        ],
    )
    assert analysis.rooms[0].furniture[0].category == "sofa"
    assert analysis.schema_version == "1.0"
