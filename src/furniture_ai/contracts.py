from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from shapely.geometry import Polygon

NonNegativeFloat = Annotated[float, Field(ge=0)]
Confidence = Annotated[float, Field(ge=0, le=1)]

MIN_POLYGON_AREA = 1e-6


class Unit(StrEnum):
    PIXEL = "px"
    CENTIMETER = "cm"
    METER = "m"


class Point(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float


class OpeningKind(StrEnum):
    DOOR = "door"
    WINDOW = "window"


class Opening(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: OpeningKind
    start: Point
    end: Point
    confidence: Confidence | None = None


class FurniturePlacement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    category: str
    center: Point
    width: NonNegativeFloat
    depth: NonNegativeFloat
    rotation_degrees: float = Field(default=0, ge=0, lt=360)
    dimension_source: str = "room-relative"
    confidence: Confidence | None = None
    source_product_id: str | None = None


class Room(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    room_type: str
    polygon: list[Point] = Field(min_length=3)
    area: NonNegativeFloat
    confidence: Confidence | None = None
    furniture: list[FurniturePlacement] = Field(default_factory=list)

    @field_validator("polygon")
    @classmethod
    def validate_polygon_geometry(cls, points: list[Point]) -> list[Point]:
        if not all(math.isfinite(point.x) and math.isfinite(point.y) for point in points):
            raise ValueError("Room polygon coordinates must be finite")
        polygon = Polygon([(point.x, point.y) for point in points])
        if polygon.buffer(0).is_empty:
            raise ValueError("Room polygon is degenerate (collinear or near-zero area)")
        if not polygon.is_valid:
            raise ValueError("Room polygon must be a simple (non-self-intersecting) polygon")
        if polygon.area < MIN_POLYGON_AREA:
            raise ValueError("Room polygon is degenerate (collinear or near-zero area)")
        return points


class FloorPlanAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    unit: Unit = Unit.PIXEL
    pixels_per_cm: float | None = Field(default=None, gt=0)
    rooms: list[Room] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    analysis_method: str = "opencv"

    @model_validator(mode="after")
    def validate_scale(self) -> FloorPlanAnalysis:
        if self.unit is not Unit.PIXEL and self.pixels_per_cm is None:
            raise ValueError("pixels_per_cm is required for physical units")
        return self


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x_min: NonNegativeFloat
    y_min: NonNegativeFloat
    x_max: NonNegativeFloat
    y_max: NonNegativeFloat

    @model_validator(mode="after")
    def validate_bounds(self) -> BoundingBox:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("Bounding-box maximums must not be smaller than minimums")
        return self


class SceneObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(min_length=1, max_length=120)
    confidence: Confidence
    box: BoundingBox


class RelativeDepthSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    p10: Confidence
    median: Confidence
    p90: Confidence
    note: str = (
        "Per-image normalized relative depth only; values are not physical dimensions "
        "and must not be compared across images without calibration."
    )


class SceneAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "1.0"
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    objects: list[SceneObject] = Field(default_factory=list)
    relative_depth: RelativeDepthSummary | None = None
    model_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LayoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    floor_plan: FloorPlanAnalysis
    room_types: dict[str, str] = Field(default_factory=dict)


class ConstraintSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ConstraintIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=1, max_length=80)
    severity: ConstraintSeverity
    message: str = Field(min_length=1, max_length=500)
    room_id: str
    item_ids: list[str] = Field(default_factory=list)
    opening_id: str | None = None


class LayoutValidationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    floor_plan: FloorPlanAnalysis
    minimum_clearance: NonNegativeFloat = Field(
        default=0,
        description="Optional minimum separation in the floor-plan geometry coordinate units.",
    )


class LayoutValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: bool
    checked_rooms: int = Field(ge=0)
    checked_items: int = Field(ge=0)
    issues: list[ConstraintIssue] = Field(default_factory=list)


class DesignResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    floor_plan: FloorPlanAnalysis
    placed_items: int
    warnings: list[str] = Field(default_factory=list)
    design_brief: str | None = None


class ValidatedDesignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    floor_plan: FloorPlanAnalysis
    room_types: dict[str, str] = Field(default_factory=dict)
    minimum_clearance: NonNegativeFloat = Field(
        default=0,
        description="Explicit minimum separation in floor-plan geometry coordinate units.",
    )


class ValidatedDesignResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    design: DesignResult
    validation: LayoutValidationReport
    execution_ready: bool


class Product(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    category: str
    width_cm: float = Field(gt=0)
    depth_cm: float = Field(gt=0)
    room_types: list[str]
    price_sar: float | None = Field(default=None, ge=0)
    source_url: str | None = None


class BookingCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_name: str = Field(min_length=2, max_length=120)
    contact: str = Field(min_length=3, max_length=200)
    requested_at: str = Field(min_length=8, max_length=64)
    notes: str = Field(default="", max_length=2000)


class Booking(BookingCreate):
    id: int
    status: str
    created_at: str
