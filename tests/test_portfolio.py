from __future__ import annotations

from furniture_ai.contracts import FloorPlanAnalysis, Point, Product, Room
from furniture_ai.portfolio import DesignPortfolioEngine, DesignPortfolioRequest


def _floor_plan() -> FloorPlanAnalysis:
    return FloorPlanAnalysis(
        source_width=120,
        source_height=120,
        rooms=[
            Room(
                id="living-1",
                room_type="living_room",
                polygon=[
                    Point(x=10, y=10),
                    Point(x=110, y=10),
                    Point(x=110, y=110),
                    Point(x=10, y=110),
                ],
                area=10_000,
                confidence=None,
            )
        ],
    )


def _catalog() -> list[Product]:
    return [
        Product(
            id="chair-a",
            name="Chair A",
            category="chair",
            width_cm=20,
            depth_cm=20,
            room_types=["living_room"],
        ),
        Product(
            id="sofa-a",
            name="Sofa A",
            category="sofa",
            width_cm=40,
            depth_cm=20,
            room_types=["living_room"],
        ),
    ]


def test_portfolio_generates_validated_ranked_candidates() -> None:
    request = DesignPortfolioRequest(floor_plan=_floor_plan())
    result = DesignPortfolioEngine(catalog=_catalog()).compose(request)

    assert len(result.candidates) == 3
    assert [candidate.rank for candidate in result.candidates] == [1, 2, 3]
    assert result.selected_candidate_id == result.candidates[0].id
    assert set(result.decision_graph.topological_order()) == {
        node.id for node in result.decision_graph.nodes
    }
    assert all(candidate.validation.checked_rooms == 1 for candidate in result.candidates)


def test_portfolio_never_invents_placement_confidence() -> None:
    result = DesignPortfolioEngine(catalog=_catalog()).compose(
        DesignPortfolioRequest(floor_plan=_floor_plan())
    )

    placements = [
        placement
        for candidate in result.candidates
        for room in candidate.design.floor_plan.rooms
        for placement in room.furniture
    ]
    assert placements
    assert all(placement.confidence is None for placement in placements)
