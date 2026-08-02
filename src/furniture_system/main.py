from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from furniture_system import __version__
from furniture_system.registry import RegistryError, SourceRegistry


class SourceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    repository: str
    commit: str
    path: str | None
    visibility: str
    tier: str
    capabilities: list[str]
    review: str


@lru_cache(maxsize=1)
def get_registry() -> SourceRegistry:
    return SourceRegistry.load()


app = FastAPI(
    title="Furniture AI System",
    version=__version__,
    description=(
        "Unified discovery and governance API for the furniture and interior-design "
        "repositories in this monorepo."
    ),
)


@app.get("/health", tags=["system"])
def health() -> dict[str, object]:
    try:
        summary = get_registry().summary()
    except RegistryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "version": __version__, "registry": summary}


@app.get("/api/v1/components", response_model=list[SourceResponse], tags=["registry"])
def components(
    tier: Literal["core", "experimental", "legacy", "private", "blocked"] | None = Query(
        default=None
    ),
    visibility: Literal["public", "private"] | None = Query(default=None),
    include_blocked: bool = Query(default=False),
) -> list[SourceResponse]:
    sources = get_registry().list(
        tier=tier,
        visibility=visibility,
        include_blocked=include_blocked,
    )
    return [SourceResponse.model_validate(source.as_dict()) for source in sources]


@app.get("/api/v1/components/{source_id}", response_model=SourceResponse, tags=["registry"])
def component(source_id: str) -> SourceResponse:
    for source in get_registry().sources:
        if source.id == source_id:
            return SourceResponse.model_validate(source.as_dict())
    raise HTTPException(status_code=404, detail="Unknown component")


@app.get("/api/v1/capabilities", tags=["registry"])
def capabilities() -> dict[str, list[str]]:
    return get_registry().capabilities()


@app.get("/api/v1/security", tags=["governance"])
def security_status() -> dict[str, object]:
    registry = get_registry()
    blocked = [source.as_dict() for source in registry.sources if source.tier == "blocked"]
    return {
        "blocked_sources": blocked,
        "policy": (
            "Blocked sources are not imported as submodules. Rotate exposed credentials and "
            "rewrite contaminated history before reconsideration."
        ),
    }
