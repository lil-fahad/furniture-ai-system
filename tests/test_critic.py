from __future__ import annotations

import pytest

from furniture_ai.contracts import (
    DesignResult,
    FloorPlanAnalysis,
    FurniturePlacement,
    Point,
    Room,
)
from furniture_ai.critic import DesignCriticRejected, SpatialDesignCritic


def design_with_item(*, x: float, y: float, width: float = 20, depth: float = 20) -> DesignResult:
    floor_plan = FloorPlanAnalysis(
        source_width=100,
        source_height=100,
        rooms=[
            Room(
                id="room-1",
                room_type="living_room",
                polygon=[
                    Point(x=0, y=0),
                    Point(x=100, y=0),
                    Point(x=100, y=100),
                    Point(x=0, y=100),
                ],
                area=10_000,
                furniture=[
                    FurniturePlacement(
                        id="item-1",
                        category="chair",
                        center=Point(x=x, y=y),
                        width=width,
                        depth=depth,
                    )
                ],
            )
        ],
    )
    return DesignResult(floor_plan=floor_plan, placed_items=1)


def test_spatial_design_critic_accepts_valid_generated_layout() -> None:
    result = design_with_item(x=50, y=50)
    assert SpatialDesignCritic().require_valid(result) is result


def test_spatial_design_critic_rejects_invalid_generated_layout() -> None:
    result = design_with_item(x=95, y=50)
    with pytest.raises(DesignCriticRejected) as captured:
        SpatialDesignCritic().require_valid(result)
    assert captured.value.report.valid is False
    assert {issue.code for issue in captured.value.report.issues} == {"outside_room"}
    assert "outside_room" in str(captured.value)


def test_spatial_design_critic_clearance_is_explicit() -> None:
    result = DesignResult(
        floor_plan=FloorPlanAnalysis(
            source_width=100,
            source_height=100,
            rooms=[
                Room(
                    id="room-1",
                    room_type="living_room",
                    polygon=[
                        Point(x=0, y=0),
                        Point(x=100, y=0),
                        Point(x=100, y=100),
                        Point(x=0, y=100),
                    ],
                    area=10_000,
                    furniture=[
                        FurniturePlacement(
                            id="a",
                            category="chair",
                            center=Point(x=25, y=50),
                            width=20,
                            depth=20,
                        ),
                        FurniturePlacement(
                            id="b",
                            category="chair",
                            center=Point(x=55, y=50),
                            width=20,
                            depth=20,
                        ),
                    ],
                )
            ],
        ),
        placed_items=2,
    )
    assert SpatialDesignCritic().inspect(result).valid is True
    strict = SpatialDesignCritic(minimum_clearance=15).inspect(result)
    assert strict.valid is False
    assert {issue.code for issue in strict.issues} == {"clearance_violation"}
