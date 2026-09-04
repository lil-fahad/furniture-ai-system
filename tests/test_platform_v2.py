from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from furniture_ai.api_entry import app
from furniture_ai.contracts import FloorPlanAnalysis, Point, Room
from furniture_ai.design_graph import DesignDecisionGraph, DesignGraphBuilder, DesignGraphNode
from furniture_ai.portfolio import DesignPortfolioEngine, DesignPortfolioRequest


def _png_bytes(width: int = 128, height: int = 128) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _floor_plan() -> FloorPlanAnalysis:
    return FloorPlanAnalysis(
        source_width=600,
        source_height=600,
        rooms=[
            Room(
                id="room-1",
                room_type="living_room",
                polygon=[
                    Point(x=20, y=20),
                    Point(x=580, y=20),
                    Point(x=580, y=580),
                    Point(x=20, y=580),
                ],
                area=313_600,
            )
        ],
    )


def test_v2_capabilities_are_explicit_about_evidence_and_ranking() -> None:
    response = TestClient(app).get("/api/v2/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_version"] == "2.0"
    assert payload["semantic_labels_default"] == "withheld_without_explicit_evidence"
    assert payload["placement_policies"] == ["balanced", "wall_first", "fit_first"]
    assert payload["ranking_is_confidence"] is False


def test_v2_analyze_withholds_heuristic_semantics_by_default() -> None:
    response = TestClient(app).post(
        "/api/v2/analyze",
        files={"image": ("plan.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["rooms"]
    assert all(room["room_type"] == "room" for room in payload["rooms"])
    assert all(room["confidence"] is None for room in payload["rooms"])
    assert payload["analysis_method"].endswith("+semantic-labels-redacted")
    assert any("Semantic room labels were withheld" in item for item in payload["warnings"])


def test_design_graph_rejects_duplicate_node_ids() -> None:
    duplicate = DesignGraphNode(id="same", kind="input", label="same")

    with pytest.raises(ValueError, match="node IDs must be unique"):
        DesignDecisionGraph(nodes=[duplicate, duplicate])


def test_design_graph_rejects_cycles() -> None:
    builder = DesignGraphBuilder()
    builder.add_node("a", "input", "A")
    builder.add_node("b", "decision", "B")
    builder.add_edge("a", "b", "feeds")
    builder.add_edge("b", "a", "cycles")

    with pytest.raises(ValueError, match="acyclic"):
        builder.build()


def test_design_graph_topological_order_is_deterministic() -> None:
    builder = DesignGraphBuilder()
    builder.add_node("root", "input", "Root")
    builder.add_node("zeta", "candidate", "Zeta")
    builder.add_node("alpha", "candidate", "Alpha")
    builder.add_edge("root", "zeta", "generates")
    builder.add_edge("root", "alpha", "generates")

    graph = builder.build()

    assert graph.topological_order() == ["root", "alpha", "zeta"]


def test_portfolio_respects_explicit_empty_catalog_and_stable_policy_order() -> None:
    request = DesignPortfolioRequest(floor_plan=_floor_plan())

    result = DesignPortfolioEngine(catalog=[]).compose(request)

    assert len(result.candidates) == 3
    assert [candidate.rank for candidate in result.candidates] == [1, 2, 3]
    assert [candidate.policy.value for candidate in result.candidates] == [
        "balanced",
        "wall_first",
        "fit_first",
    ]
    assert all(candidate.design.placed_items == 0 for candidate in result.candidates)
    assert result.selected_candidate_id == "candidate-balanced"
    assert result.execution_ready is True
    assert "not an aesthetic quality score" in result.ranking_basis
    assert len(result.decision_graph.topological_order()) == len(result.decision_graph.nodes)
