from __future__ import annotations

from collections import defaultdict
from heapq import heapify, heappop, heappush
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

GraphValue = str | int | float | bool | None
NodeKind = Literal[
    "input",
    "room",
    "candidate",
    "validation",
    "product",
    "decision",
    "evidence",
]


class DesignGraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    kind: NodeKind
    label: str = Field(min_length=1, max_length=240)
    attributes: dict[str, GraphValue] = Field(default_factory=dict)


class DesignGraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1, max_length=160)
    target: str = Field(min_length=1, max_length=160)
    relation: str = Field(min_length=1, max_length=120)


class DesignDecisionGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "2.0"
    nodes: list[DesignGraphNode] = Field(default_factory=list)
    edges: list[DesignGraphEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_graph(self) -> DesignDecisionGraph:
        node_ids = [node.id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Design graph node IDs must be unique")

        node_id_set = set(node_ids)
        edge_keys: set[tuple[str, str, str]] = set()
        for edge in self.edges:
            if edge.source not in node_id_set or edge.target not in node_id_set:
                raise ValueError("Design graph edges must reference existing nodes")
            edge_key = (edge.source, edge.target, edge.relation)
            if edge_key in edge_keys:
                raise ValueError("Design graph edges must be unique")
            edge_keys.add(edge_key)

        self.topological_order()
        return self

    def topological_order(self) -> list[str]:
        """Return a deterministic topological order and reject cycles."""
        node_ids = {node.id for node in self.nodes}
        outgoing: dict[str, list[str]] = defaultdict(list)
        indegree = {node_id: 0 for node_id in node_ids}

        for edge in self.edges:
            if edge.source not in node_ids or edge.target not in node_ids:
                raise ValueError("Design graph edges must reference existing nodes")
            outgoing[edge.source].append(edge.target)
            indegree[edge.target] += 1

        queue = [node_id for node_id, degree in indegree.items() if degree == 0]
        heapify(queue)
        ordered: list[str] = []
        while queue:
            node_id = heappop(queue)
            ordered.append(node_id)
            for target in sorted(outgoing[node_id]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    heappush(queue, target)

        if len(ordered) != len(node_ids):
            raise ValueError("Design decision graph must be acyclic")
        return ordered


class DesignGraphBuilder:
    """Deterministic DAG builder for auditable design decisions."""

    def __init__(self) -> None:
        self._nodes: dict[str, DesignGraphNode] = {}
        self._edges: set[tuple[str, str, str]] = set()

    def add_node(
        self,
        node_id: str,
        kind: NodeKind,
        label: str,
        **attributes: GraphValue,
    ) -> str:
        candidate = DesignGraphNode(
            id=node_id,
            kind=kind,
            label=label,
            attributes=attributes,
        )
        existing = self._nodes.get(node_id)
        if existing is not None and existing != candidate:
            raise ValueError(f"Conflicting design graph node: {node_id}")
        self._nodes[node_id] = candidate
        return node_id

    def add_edge(self, source: str, target: str, relation: str) -> None:
        if source not in self._nodes or target not in self._nodes:
            raise ValueError("Design graph edges must reference nodes before they are linked")
        self._edges.add((source, target, relation))

    def build(self) -> DesignDecisionGraph:
        return DesignDecisionGraph(
            nodes=[self._nodes[key] for key in sorted(self._nodes)],
            edges=[
                DesignGraphEdge(source=source, target=target, relation=relation)
                for source, target, relation in sorted(self._edges)
            ],
        )
