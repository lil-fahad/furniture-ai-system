from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from furniture_ai import __version__
from furniture_ai.config import Settings, get_settings
from furniture_ai.contracts import FloorPlanAnalysis
from furniture_ai.floorplan import FloorPlanAnalyzer
from furniture_ai.image_io import ImageValidationError, load_validated_image
from furniture_ai.openai_service import OpenAIDesignService, OpenAIUnavailable
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
        "geometry_only_analysis": True,
        "semantic_labels_require_evidence": True,
        "design_portfolio": True,
        "decision_graph": True,
        "deterministic_validation": True,
        "placement_policies": ["balanced", "wall_first", "fit_first"],
        "ranking_is_confidence": False,
    }


@router.post("/analyze", response_model=FloorPlanAnalysis)
async def analyze_floor_plan(
    image: Annotated[UploadFile, File(...)],
    active_settings: Annotated[Settings, Depends(get_settings)],
    pixels_per_cm: Annotated[float | None, Form(gt=0)] = None,
    use_openai: Annotated[bool, Form()] = False,
) -> FloorPlanAnalysis:
    """Extract geometry without assigning unsupported semantic room labels."""
    data = await image.read(active_settings.max_upload_bytes + 1)
    if len(data) > active_settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail="The uploaded image exceeds the configured byte limit",
        )
    try:
        loaded = await run_in_threadpool(
            load_validated_image,
            data,
            image.content_type,
            active_settings,
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    analyzer = FloorPlanAnalyzer()
    floor_plan = await run_in_threadpool(
        analyzer.analyze,
        loaded,
        pixels_per_cm=pixels_per_cm,
        infer_semantics=False,
    )
    if not use_openai:
        return floor_plan

    try:
        service = OpenAIDesignService(active_settings)
        refinements = await run_in_threadpool(
            service.refine_room_types,
            loaded,
            floor_plan,
        )
    except (OpenAIUnavailable, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=503,
            detail="OpenAI semantic refinement is unavailable",
        ) from exc

    unresolved: list[str] = []
    for room in floor_plan.rooms:
        refinement = refinements.get(room.id)
        if refinement is None:
            unresolved.append(room.id)
            continue
        room.room_type, room.confidence = refinement
    floor_plan.warnings = [
        warning
        for warning in floor_plan.warnings
        if not warning.startswith("Room semantic labels were intentionally left unresolved")
    ]
    if unresolved:
        floor_plan.warnings.append(
            "Semantic labels remain unresolved for rooms: " + ", ".join(unresolved)
        )
    floor_plan.analysis_method += "+openai-vision"
    return floor_plan


@router.post("/design/portfolio", response_model=DesignPortfolioResult)
def design_portfolio(request: DesignPortfolioRequest) -> DesignPortfolioResult:
    try:
        return DesignPortfolioEngine().compose(request)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
