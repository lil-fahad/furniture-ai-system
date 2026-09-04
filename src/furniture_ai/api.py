from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from furniture_ai import __version__
from furniture_ai.config import Settings, get_settings
from furniture_ai.constraints import validate_layout_constraints
from furniture_ai.contracts import (
    Booking,
    BookingCreate,
    DesignResult,
    LayoutRequest,
    LayoutValidationReport,
    LayoutValidationRequest,
    Product,
    SceneAnalysis,
    ValidatedDesignRequest,
    ValidatedDesignResult,
)
from furniture_ai.image_io import ImageValidationError, load_validated_image
from furniture_ai.layout import furnish_floor_plan, load_catalog
from furniture_ai.models import ModelRegistry
from furniture_ai.observability import elapsed_ms, normalize_request_id
from furniture_ai.pipeline import DesignPipeline
from furniture_ai.professional_vision import (
    ProfessionalVisionService,
    ProfessionalVisionUnavailable,
)
from furniture_ai.security import require_service_key
from furniture_ai.storage import BookingStore

app = FastAPI(
    title="Furniture AI System",
    version=__version__,
    description=(
        "Unified floor-plan analysis, furniture layout, AI guidance, catalog, "
        "and bookings API."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    request_id = normalize_request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["Server-Timing"] = f"app;dur={elapsed_ms(started):.2f}"
    return response


@lru_cache(maxsize=1)
def get_booking_store() -> BookingStore:
    return BookingStore(get_settings().database_path)


def _model_registry(active_settings: Settings) -> ModelRegistry:
    if active_settings.model_manifest_path.is_file():
        return ModelRegistry(active_settings.model_manifest_path)
    return ModelRegistry(
        active_settings.model_manifest_path,
        allow_packaged_default=not active_settings.model_manifest_path_overridden,
    )


@app.get("/health", tags=["system"])
def health() -> dict[str, object]:
    active = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "openai_configured": active.openai_configured,
        "professional_vision_available": active.professional_vision_available,
        "service_auth_enabled": active.service_auth_enabled,
    }


@app.get("/ready", tags=["system"])
def ready() -> dict[str, object]:
    active = get_settings()
    try:
        model_statuses = _model_registry(active).statuses()
        get_booking_store()
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Application dependencies are not ready",
        ) from exc
    return {
        "status": "ready",
        "models_present": sum(status.present for status in model_statuses),
        "models_registered": len(model_statuses),
    }


@app.post(
    "/api/v1/analyze",
    response_model=DesignResult,
    dependencies=[Depends(require_service_key)],
    tags=["design"],
)
async def analyze_and_design(
    image: Annotated[UploadFile, File(...)],
    active_settings: Annotated[Settings, Depends(get_settings)],
    pixels_per_cm: Annotated[float | None, Form(gt=0)] = None,
    use_openai: Annotated[bool, Form()] = False,
    preferences: Annotated[str, Form(max_length=3000)] = "",
) -> DesignResult:
    data = await image.read(active_settings.max_upload_bytes + 1)
    if len(data) > active_settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail="The uploaded image exceeds the configured byte limit",
        )
    try:
        loaded = await run_in_threadpool(
            load_validated_image, data, image.content_type, active_settings
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    pipeline = DesignPipeline(active_settings)
    return await run_in_threadpool(
        pipeline.run,
        loaded,
        pixels_per_cm=pixels_per_cm,
        use_openai=use_openai,
        preferences=preferences,
    )


@app.post(
    "/api/v1/scene",
    response_model=SceneAnalysis,
    dependencies=[Depends(require_service_key)],
    tags=["vision"],
)
async def analyze_room_scene(
    image: Annotated[UploadFile, File(...)],
    active_settings: Annotated[Settings, Depends(get_settings)],
    detection_threshold: Annotated[float, Form(ge=0, le=1)] = 0.55,
    include_depth: Annotated[bool, Form()] = True,
) -> SceneAnalysis:
    """Analyze a room photo with verified local Hugging Face model artifacts."""
    data = await image.read(active_settings.max_upload_bytes + 1)
    if len(data) > active_settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail="The uploaded image exceeds the configured byte limit",
        )
    try:
        loaded = await run_in_threadpool(
            load_validated_image, data, image.content_type, active_settings
        )
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        configured_device = (
            None
            if active_settings.professional_vision_device == "auto"
            else active_settings.professional_vision_device
        )
        service = ProfessionalVisionService(
            active_settings.professional_models_root,
            device=configured_device,
            precision=active_settings.professional_vision_precision,
            enable_torch_compile=active_settings.professional_vision_torch_compile,
        )
        return await run_in_threadpool(
            service.analyze,
            loaded,
            detection_threshold=detection_threshold,
            include_depth=include_depth,
        )
    except ProfessionalVisionUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Professional vision inference failed") from exc


@app.post(
    "/api/v1/layout",
    response_model=DesignResult,
    dependencies=[Depends(require_service_key)],
    tags=["design"],
)
def layout(request: LayoutRequest) -> DesignResult:
    try:
        return furnish_floor_plan(request.floor_plan, room_type_overrides=request.room_types)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid floor-plan geometry: {exc}",
        ) from exc


@app.post(
    "/api/v1/layout/validate",
    response_model=LayoutValidationReport,
    dependencies=[Depends(require_service_key)],
    tags=["design"],
)
def validate_layout(request: LayoutValidationRequest) -> LayoutValidationReport:
    """Validate spatial layout constraints independently from model confidence."""
    try:
        return validate_layout_constraints(
            request.floor_plan,
            minimum_clearance=request.minimum_clearance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/api/v1/design/validated",
    response_model=ValidatedDesignResult,
    dependencies=[Depends(require_service_key)],
    tags=["design"],
)
def create_validated_design(request: ValidatedDesignRequest) -> ValidatedDesignResult:
    """Generate a deterministic layout and immediately gate it through spatial constraints."""
    try:
        design = furnish_floor_plan(
            request.floor_plan,
            room_type_overrides=request.room_types,
        )
        validation = validate_layout_constraints(
            design.floor_plan,
            minimum_clearance=request.minimum_clearance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ValidatedDesignResult(
        design=design,
        validation=validation,
        execution_ready=validation.valid,
    )


@app.get(
    "/api/v1/catalog",
    response_model=list[Product],
    dependencies=[Depends(require_service_key)],
    tags=["catalog"],
)
def catalog(room_type: str | None = Query(default=None)) -> list[Product]:
    products = load_catalog()
    if room_type:
        products = [product for product in products if room_type in product.room_types]
    return products


@app.get("/api/v1/models", dependencies=[Depends(require_service_key)], tags=["models"])
def models(active_settings: Annotated[Settings, Depends(get_settings)]) -> list[dict[str, object]]:
    registry = _model_registry(active_settings)
    return [status.__dict__ for status in registry.statuses()]


@app.post(
    "/api/v1/bookings",
    response_model=Booking,
    dependencies=[Depends(require_service_key)],
    tags=["bookings"],
)
def create_booking(
    request: BookingCreate,
    store: Annotated[BookingStore, Depends(get_booking_store)],
) -> Booking:
    return store.create(request)


@app.get(
    "/api/v1/bookings",
    response_model=list[Booking],
    dependencies=[Depends(require_service_key)],
    tags=["bookings"],
)
def list_bookings(
    store: Annotated[BookingStore, Depends(get_booking_store)],
    limit: int = Query(default=100, ge=1, le=500),
) -> list[Booking]:
    return store.list(limit=limit)
