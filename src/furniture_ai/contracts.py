from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonNegativeFloat = Annotated[float, Field(ge=0)]
Confidence = Annotated[float, Field(ge=0, le=1)]


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


class LayoutRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    floor_plan: FloorPlanAnalysis
    room_types: dict[str, str] = Field(default_factory=dict)


class DesignResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    floor_plan: FloorPlanAnalysis
    placed_items: int
    warnings: list[str] = Field(default_factory=list)
    design_brief: str | None = None


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
