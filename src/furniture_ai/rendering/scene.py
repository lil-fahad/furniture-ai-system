from __future__ import annotations

from furniture_ai.contracts import DesignResult, Product
from furniture_ai.layout import load_catalog
from furniture_ai.rendering.contracts import CameraSpec, SceneFurnitureItem, SceneRoom, SceneSpec


DEFAULT_NEGATIVE_CONSTRAINTS = [
    "do not change the room boundary geometry",
    "do not move, remove, or duplicate grounded furniture",
    "do not invent doors or windows",
    "do not distort walls, floors, or furniture proportions",
    "do not add text, logos, watermarks, or people",
]


class SceneCompiler:
    """Convert an execution-ready design into a renderer-neutral scene specification."""

    def __init__(self, *, catalog: list[Product] | None = None) -> None:
        products = catalog if catalog is not None else load_catalog()
        self._catalog = {product.id: product for product in products}

    def compile(
        self,
        design: DesignResult,
        *,
        style: str,
        room_id: str | None = None,
    ) -> SceneSpec:
        normalized_style = " ".join(style.split())
        if not normalized_style:
            raise ValueError("Render style must not be empty")

        source_rooms = design.floor_plan.rooms
        if room_id is not None:
            source_rooms = [room for room in source_rooms if room.id == room_id]
            if not source_rooms:
                raise ValueError(f"Unknown render room_id: {room_id}")
        if not source_rooms:
            raise ValueError("Render scene requires at least one room")

        rooms: list[SceneRoom] = []
        for room in source_rooms:
            furniture: list[SceneFurnitureItem] = []
            for placement in room.furniture:
                product = (
                    self._catalog.get(placement.source_product_id)
                    if placement.source_product_id
                    else None
                )
                furniture.append(
                    SceneFurnitureItem(
                        id=placement.id,
                        product_id=placement.source_product_id,
                        product_name=product.name if product else placement.category.replace("_", " "),
                        category=placement.category,
                        room_id=room.id,
                        center=placement.center,
                        width=placement.width,
                        depth=placement.depth,
                        rotation_degrees=placement.rotation_degrees,
                        dimension_source=placement.dimension_source,
                        reference_url=product.source_url if product else None,
                    )
                )
            rooms.append(
                SceneRoom(
                    id=room.id,
                    room_type=room.room_type,
                    polygon=[point.model_copy() for point in room.polygon],
                    area=room.area,
                    furniture=furniture,
                )
            )

        return SceneSpec(
            source_width=design.floor_plan.source_width,
            source_height=design.floor_plan.source_height,
            unit=design.floor_plan.unit,
            pixels_per_cm=design.floor_plan.pixels_per_cm,
            style=normalized_style,
            camera=CameraSpec(),
            rooms=rooms,
            negative_constraints=list(DEFAULT_NEGATIVE_CONSTRAINTS),
        )
