# ruff: noqa: I001
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


class BoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float
    width: NonNegativeFloat
    height: NonNegativeFloat

    @property
    def area(self) -> float:
        return self.width * self.height


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
    confidence: Confidence | None = None
    source_product_id: str | None = None


class Room(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    room_type: str
    polygon: list[Point] = Field(min_length=3)
    area: NonNegativeFloat | None = None
    confidence: Confidence | None = None
    furniture: list[FurniturePlacement] = Field(default_factory=list)


class FloorPlanAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    unit: Unit = Unit.PIXEL
    scale: float | None = Field(default=None, gt=0)
    rooms: list[Room] = Field(default_factory=list)
    openings: list[Opening] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scale(self) -> FloorPlanAnalysis:
        if self.unit is not Unit.PIXEL and self.scale is None:
            raise ValueError("scale is required when unit is not pixels")
        return self


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capabilities: list[str] = Field(min_length=1)
    allow_experimental: bool = False
    include_private: bool = False


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1)
    source_id: str
    repository: str
    capabilities: list[str] = Field(min_length=1)
    tier: str
    path: str


class ExecutionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_capabilities: list[str]
    steps: list[PlanStep]
    unresolved_capabilities: list[str] = Field(default_factory=list)
