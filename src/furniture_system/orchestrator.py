from __future__ import annotations

from furniture_system.contracts import ExecutionPlan, PlanRequest, PlanStep
from furniture_system.registry import Source, SourceRegistry

_TIER_PRIORITY = {"core": 0, "experimental": 1, "private": 2, "legacy": 3, "blocked": 99}


class PlanningError(ValueError):
    """Raised when an execution plan cannot be created."""


def _eligible(source: Source, request: PlanRequest) -> bool:
    if source.tier in {"blocked", "legacy"}:
        return False
    if source.tier == "experimental" and not request.allow_experimental:
        return False
    if source.visibility == "private" and not request.include_private:
        return False
    return source.path is not None


def _provider_priority(source: Source, coverage_count: int) -> tuple[int, int, bool, str]:
    """Rank providers by coverage first, then governance quality and stable id."""
    return (
        -coverage_count,
        _TIER_PRIORITY.get(source.tier, 98),
        source.visibility == "private",
        source.id,
    )


def build_execution_plan(registry: SourceRegistry, request: PlanRequest) -> ExecutionPlan:
    requested = list(dict.fromkeys(value.strip() for value in request.capabilities if value.strip()))
    if not requested:
        raise PlanningError("At least one non-empty capability is required")

    eligible = [source for source in registry.sources if _eligible(source, request)]
    remaining = requested.copy()
    selected: list[tuple[Source, list[str]]] = []

    # Deterministic greedy set cover: choose the provider that satisfies the most
    # unresolved capabilities, then prefer core/public providers. This avoids
    # splitting one coherent request across services when a single reviewed
    # component already covers the complete capability set.
    while remaining:
        candidates: list[tuple[Source, list[str]]] = []
        for source in eligible:
            covered = [capability for capability in remaining if capability in source.capabilities]
            if covered:
                candidates.append((source, covered))

        if not candidates:
            break

        source, covered = min(
            candidates,
            key=lambda candidate: _provider_priority(candidate[0], len(candidate[1])),
        )
        selected.append((source, covered))
        covered_set = set(covered)
        remaining = [capability for capability in remaining if capability not in covered_set]
        eligible = [candidate for candidate in eligible if candidate.id != source.id]

    steps = [
        PlanStep(
            order=index,
            source_id=source.id,
            repository=source.repository,
            capabilities=capabilities,
            tier=source.tier,
            path=source.path or "",
        )
        for index, (source, capabilities) in enumerate(selected, start=1)
    ]
    return ExecutionPlan(
        requested_capabilities=requested,
        steps=steps,
        unresolved_capabilities=remaining,
    )
