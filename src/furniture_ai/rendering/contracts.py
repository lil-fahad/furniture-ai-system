from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from furniture_ai.contracts import DesignResult, Point, Unit


class RendererKind(StrEnum):
    MOCK = "mock"


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: Literal["wide_room", "eye_level", "corner"] = "wide_room"
    lens_mm: float = Field(default=24.0, ge=14.0, le=120.0)
    height_cm: float = Field(default=150.0, ge=40.0, le=300.0)


class SceneFurnitureItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    product_id: str | None = Field(default=None, max_length=160)
    product_name: str = Field(min_length=1, max_length=240)
    category: str = Field(min_length=1, max_length=120)
    room_id: str = Field(min_length=1, max_length=160)
    center: Point
    width: float = Field(gt=0)
    depth: float = Field(gt=0)
    rotation_degrees: float = Field(ge=0, lt=360)
    dimension_source: str = Field(min_length=1, max_length=120)
    reference_url: str | None = Field(default=None, max_length=2000)


class SceneRoom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=160)
    room_type: str = Field(min_length=1, max_length=120)
    polygon: list[Point] = Field(min_length=3)
    area: float = Field(ge=0)
    furniture: list[SceneFurnitureItem] = Field(default_factory=list)


class SceneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    source_width: int = Field(gt=0)
    source_height: int = Field(gt=0)
    unit: Unit = Unit.PIXEL
    pixels_per_cm: float | None = Field(default=None, gt=0)
    style: str = Field(min_length=1, max_length=120)
    camera: CameraSpec = Field(default_factory=CameraSpec)
    rooms: list[SceneRoom] = Field(min_length=1)
    negative_constraints: list[str] = Field(default_factory=list)


class RenderPromptPackage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    positive_prompt: str = Field(min_length=1, max_length=12_000)
    negative_prompt: str = Field(min_length=1, max_length=6000)
    reference_urls: list[str] = Field(default_factory=list)
    scene_fingerprint: str = Field(min_length=64, max_length=64)


class RenderArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: RendererKind
    media_type: str = Field(min_length=1, max_length=120)
    data_uri: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class RenderPreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    design: DesignResult
    style: str = Field(default="warm modern", min_length=1, max_length=120)
    room_id: str | None = Field(default=None, max_length=160)
    backend: RendererKind = RendererKind.MOCK
    seed: int = Field(default=0, ge=0, le=2_147_483_647)


class RenderPreviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["preview"] = "preview"
    photorealistic: bool
    scene: SceneSpec
    prompt: RenderPromptPackage
    artifact: RenderArtifact
    warnings: list[str] = Field(default_factory=list)
