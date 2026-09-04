from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from furniture_ai import __version__
from furniture_ai.portfolio import (
    DesignPortfolioEngine,
    DesignPortfolioRequest,
    DesignPortfolioResult,
)
from furniture_ai.security import require_service_key

router = APIRouter(
    prefix="/api/v2",
    tags=["platform-v2"],
    dependencies=[Depends(require_service_key)],
)


@router.get("/capabilities")
def capabilities() -> dict[str, object]:
    return {
        "api_version": "2.0",
        "application_version": __version__,
        "design_portfolio": True,
        "decision_graph": True,
        "deterministic_validation": True,
        "placement_policies": ["balanced", "wall_first", "fit_first"],
        "ranking_is_confidence": False,
    }


@router.post("/design/portfolio", response_model=DesignPortfolioResult)
def design_portfolio(request: DesignPortfolioRequest) -> DesignPortfolioResult:
    try:
        return DesignPortfolioEngine().compose(request)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
