from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from furniture_ai import __version__
from furniture_ai.config import Settings, get_settings
from furniture_ai.contracts import (
    Booking,
    BookingCreate,
    DesignResult,
    LayoutRequest,
    Product,
    SupplierRecommendation,
)
from furniture_ai.image_io import ImageValidationError, load_validated_image
from furniture_ai.layout import furnish_floor_plan, load_catalog
from furniture_ai.models import ModelRegistry
from furniture_ai.pipeline import DesignPipeline
from furniture_ai.security import require_service_key
from furniture_ai.storage import BookingStore

settings = get_settings()
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
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


@lru_cache(maxsize=1)
def get_booking_store() -> BookingStore:
    return BookingStore(get_settings().database_path)


@lru_cache(maxsize=1)
def get_supplier_rows() -> list[dict[str, str]]:
    from furniture_ai.supplier_ranker import load_supplier_rows

    return load_supplier_rows(get_settings().supplier_data_path)


@lru_cache(maxsize=1)
def get_supplier_model():
    from furniture_ai.supplier_ranker import load_supplier_model

    return load_supplier_model(get_settings().supplier_model_path)


@app.get("/health", tags=["system"])
def health() -> dict[str, object]:
    active = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "openai_configured": active.openai_configured,
        "service_auth_enabled": active.service_auth_enabled,
    }


@app.get("/ready", tags=["system"])
def ready() -> dict[str, object]:
    active = get_settings()
    try:
        model_statuses = ModelRegistry(active.model_manifest_path).statuses()
        get_booking_store()
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Application dependencies are not ready",
        ) from exc
    active = get_settings()
    return {
        "status": "ready",
        "models_present": sum(status.present for status in model_statuses),
        "models_registered": len(model_statuses),
        "supplier_data_present": active.supplier_data_path.is_file(),
        "supplier_ranker_present": active.supplier_model_path.is_file(),
    }


@app.post(
    "/api/v1/analyze",
    response_model=DesignResult,
    dependencies=[Depends(require_service_key)],
    tags=["design"],
)
async def analyze_and_design(
    image: Annotated[UploadFile, File(...)],
    pixels_per_cm: Annotated[float | None, Form(gt=0)] = None,
    use_openai: Annotated[bool, Form()] = False,
    preferences: Annotated[str, Form(max_length=3000)] = "",
    active_settings: Annotated[Settings, Depends(get_settings)] = settings,
) -> DesignResult:
    data = await image.read(active_settings.max_upload_bytes + 1)
    try:
        loaded = load_validated_image(data, image.content_type, active_settings)
    except ImageValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return DesignPipeline(active_settings).run(
        loaded,
        pixels_per_cm=pixels_per_cm,
        use_openai=use_openai,
        preferences=preferences,
    )


@app.post(
    "/api/v1/layout",
    response_model=DesignResult,
    dependencies=[Depends(require_service_key)],
    tags=["design"],
)
def layout(request: LayoutRequest) -> DesignResult:
    return furnish_floor_plan(request.floor_plan, room_type_overrides=request.room_types)


@app.get("/api/v1/catalog", response_model=list[Product], tags=["catalog"])
def catalog(room_type: str | None = Query(default=None)) -> list[Product]:
    products = load_catalog()
    if room_type:
        products = [product for product in products if room_type in product.room_types]
    return products


@app.get("/api/v1/models", tags=["models"])
def models(active_settings: Annotated[Settings, Depends(get_settings)]) -> list[dict[str, object]]:
    registry = ModelRegistry(active_settings.model_manifest_path)
    return [status.__dict__ for status in registry.statuses()]


@app.get(
    "/api/v1/suppliers/recommend",
    response_model=list[SupplierRecommendation],
    tags=["suppliers"],
)
def recommend_suppliers(
    category: str | None = Query(default=None, max_length=100),
    requires_dropshipping: bool = Query(default=False),
    requires_3d_models: bool = Query(default=False),
    requires_direct_fulfillment: bool = Query(default=False),
    max_lead_days: float | None = Query(default=None, gt=0),
    max_moq: float | None = Query(default=None, gt=0),
    max_price: float | None = Query(default=None, gt=0),
    top_k: int = Query(default=10, ge=1, le=41),
) -> list[SupplierRecommendation]:
    try:
        from furniture_ai.supplier_ranker import SupplierPreference, rank_suppliers

        preference = SupplierPreference(
            category=category,
            requires_dropshipping=requires_dropshipping,
            requires_3d_models=requires_3d_models,
            requires_direct_fulfillment=requires_direct_fulfillment,
            max_lead_days=max_lead_days,
            max_moq=max_moq,
            max_price=max_price,
        )
        results = rank_suppliers(
            get_supplier_rows(),
            get_supplier_model(),
            preference=preference,
            top_k=top_k,
        )
    except (FileNotFoundError, ImportError, OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Supplier ranker is unavailable") from exc
    return [SupplierRecommendation.model_validate(item) for item in results]


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
