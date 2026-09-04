from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from furniture_ai.constraints import validate_layout_constraints
from furniture_ai.contracts import (
    DesignResult,
    FloorPlanAnalysis,
    LayoutValidationReport,
    Product,
)
from furniture_ai.design_graph import DesignDecisionGraph, DesignGraphBuilder
from furniture_ai.layout import furnish_floor_plan, load_catalog


class PlacementPolicy(StrEnum):
    BALANCED = "balanced"
    WALL_FIRST = "wall_first"
    FIT_FIRST = "fit_first"


class DesignPortfolioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    floor_plan: FloorPlanAnalysis
    room_types: dict[str, str] = Field(default_factory=dict)
    minimum_clearance: float = Field(
        default=0,
        ge=0,
        description="Explicit clearance in the same geometry units as the floor plan.",
    )
    policies: list[PlacementPolicy] = Field(
        default_factory=lambda: [
            PlacementPolicy.BALANCED,
            PlacementPolicy.WALL_FIRST,
            PlacementPolicy.FIT_FIRST,
        ],
        min_length=1,
        max_length=3,
    )

    @field_validator("policies")
    @classmethod
    def unique_policies(cls, value: list[PlacementPolicy]) -> list[PlacementPolicy]:
        if len(set(value)) != len(value):
            raise ValueError("Design portfolio policies must be unique")
        return value


class DesignCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    rank: int = Field(ge=1)
    policy: PlacementPolicy
    design: DesignResult
    validation: LayoutValidationReport
    execution_ready: bool


class DesignPortfolioResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "2.0"
    selected_candidate_id: str
    candidates: list[DesignCandidate]
    decision_graph: DesignDecisionGraph
    ranking_basis: str = (
        "Deterministic lexicographic ranking: valid layouts first, then more placed catalog "
        "items, then fewer validation issues, then stable policy order. This is not an "
        "aesthetic quality score or model confidence."
    )


class DesignPortfolioEngine:
    """Generate and gate multiple deterministic layouts without inventing confidence."""

    def __init__(self, *, catalog: list[Product] | None = None) -> None:
        self.catalog = catalog

    def compose(self, request: DesignPortfolioRequest) -> DesignPortfolioResult:
        catalog = self.catalog if self.catalog is not None else load_catalog()
        candidates: list[DesignCandidate] = []

        for policy_index, policy in enumerate(request.policies):
            design = furnish_floor_plan(
                request.floor_plan,
                room_type_overrides=request.room_types,
                catalog=catalog,
                placement_policy=policy.value,
            )
            validation = validate_layout_constraints(
                design.floor_plan,
                minimum_clearance=request.minimum_clearance,
            )
            candidates.append(
                DesignCandidate(
                    id=f"candidate-{policy.value}",
                    rank=policy_index + 1,
                    policy=policy,
                    design=design,
                    validation=validation,
                    execution_ready=validation.valid,
                )
            )

        policy_order = {policy: index for index, policy in enumerate(request.policies)}
        candidates.sort(
            key=lambda candidate: (
                not candidate.execution_ready,
                -candidate.design.placed_items,
                len(candidate.validation.issues),
                policy_order[candidate.policy],
            )
        )
        ranked = [
            candidate.model_copy(update={"rank": rank})
            for rank, candidate in enumerate(candidates, start=1)
        ]
        selected = ranked[0]
        graph = self._decision_graph(request, ranked, selected)
        return DesignPortfolioResult(
            selected_candidate_id=selected.id,
            candidates=ranked,
            decision_graph=graph,
        )

    @staticmethod
    def _decision_graph(
        request: DesignPortfolioRequest,
        candidates: list[DesignCandidate],
        selected: DesignCandidate,
    ) -> DesignDecisionGraph:
        builder = DesignGraphBuilder()
        input_id = builder.add_node(
            "input-floor-plan",
            "input",
            "Validated floor-plan input",
            rooms=len(request.floor_plan.rooms),
            openings=len(request.floor_plan.openings),
            unit=request.floor_plan.unit.value,
        )

        room_nodes: list[str] = []
        for room in request.floor_plan.rooms:
            room_node = builder.add_node(
                f"room:{room.id}",
                "room",
                room.room_type,
                room_id=room.id,
                area=float(room.area),
            )
            builder.add_edge(input_id, room_node, "contains")
            room_nodes.append(room_node)

        for candidate in candidates:
            candidate_node = builder.add_node(
                f"portfolio:{candidate.id}",
                "candidate",
                candidate.policy.value,
                rank=candidate.rank,
                placed_items=candidate.design.placed_items,
                execution_ready=candidate.execution_ready,
            )
            builder.add_edge(input_id, candidate_node, "generates")
            for room_node in room_nodes:
                builder.add_edge(room_node, candidate_node, "constrains")

            product_ids = sorted(
                {
                    placement.source_product_id
                    for room in candidate.design.floor_plan.rooms
                    for placement in room.furniture
                    if placement.source_product_id
                }
            )
            for product_id in product_ids:
                product_node = builder.add_node(
                    f"product:{product_id}",
                    "product",
                    product_id,
                    product_id=product_id,
                )
                builder.add_edge(candidate_node, product_node, "uses")

            for issue_index, issue in enumerate(candidate.validation.issues):
                issue_node = builder.add_node(
                    f"validation:{candidate.id}:{issue_index}",
                    "validation",
                    issue.code,
                    severity=issue.severity.value,
                    room_id=issue.room_id,
                    opening_id=issue.opening_id,
                )
                builder.add_edge(candidate_node, issue_node, "evaluated_by")

        decision_node = builder.add_node(
            "decision:selected",
            "decision",
            "Selected deterministic candidate",
            candidate_id=selected.id,
            policy=selected.policy.value,
            execution_ready=selected.execution_ready,
        )
        for candidate in candidates:
            builder.add_edge(
                f"portfolio:{candidate.id}",
                decision_node,
                "selected" if candidate.id == selected.id else "ranked_below",
            )
        return builder.build()
