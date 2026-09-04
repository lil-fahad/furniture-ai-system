from __future__ import annotations

import pytest

from furniture_ai.design_graph import (
    DesignDecisionGraph,
    DesignGraphBuilder,
    DesignGraphEdge,
    DesignGraphNode,
)


def test_graph_builder_is_deterministic_and_acyclic() -> None:
    builder = DesignGraphBuilder()
    builder.add_node("input", "input", "Floor plan")
    builder.add_node("candidate", "candidate", "Balanced")
    builder.add_node("decision", "decision", "Selected")
    builder.add_edge("input", "candidate", "generates")
    builder.add_edge("candidate", "decision", "selected")

    graph = builder.build()

    assert graph.topological_order() == ["input", "candidate", "decision"]
    assert [node.id for node in graph.nodes] == ["candidate", "decision", "input"]


def test_graph_rejects_cycles() -> None:
    graph = DesignDecisionGraph(
        nodes=[
            DesignGraphNode(id="a", kind="input", label="A"),
            DesignGraphNode(id="b", kind="decision", label="B"),
        ],
        edges=[
            DesignGraphEdge(source="a", target="b", relation="next"),
            DesignGraphEdge(source="b", target="a", relation="back"),
        ],
    )

    with pytest.raises(ValueError, match="acyclic"):
        graph.topological_order()


def test_graph_builder_rejects_missing_nodes() -> None:
    builder = DesignGraphBuilder()
    builder.add_node("input", "input", "Floor plan")
    with pytest.raises(ValueError, match="reference nodes"):
        builder.add_edge("input", "missing", "generates")
