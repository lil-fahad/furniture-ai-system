from __future__ import annotations

from collections import OrderedDict

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


def _select_provider(
    registry: SourceRegistry,
    capability: str,
    request: PlanRequest,
) -> Source | None:
    candidates = [
        source
        for source in registry.sources
        if capability in source.capabilities and _eligible(source, request)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda source: (
            _TIER_PRIORITY.get(source.tier, 98),
            source.visibility == "private",
            source.id,
        ),
    )


def build_execution_plan(registry: SourceRegistry, request: PlanRequest) -> ExecutionPlan:
    requested = list(dict.fromkeys(value.strip() for value in request.capabilities if value.strip()))
    if not requested:
        raise PlanningError("At least one non-empty capability is required")

    assignments: OrderedDict[str, tuple[Source, list[str]]] = OrderedDict()
    unresolved: list[str] = []

    for capability in requested:
        provider = _select_provider(registry, capability, request)
        if provider is None:
            unresolved.append(capability)
            continue
        if provider.id not in assignments:
            assignments[provider.id] = (provider, [])
        assignments[provider.id][1].append(capability)

    steps = [
        PlanStep(
            order=index,
            source_id=source.id,
            repository=source.repository,
            capabilities=capabilities,
            tier=source.tier,
            path=source.path or "",
        )
        for index, (source, capabilities) in enumerate(assignments.values(), start=1)
    ]
    return ExecutionPlan(
        requested_capabilities=requested,
        steps=steps,
        unresolved_capabilities=unresolved,
    )
