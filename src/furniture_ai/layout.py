from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from shapely.affinity import rotate
from shapely.geometry import LineString, Polygon, box
from shapely.geometry import Point as ShapelyPoint

from furniture_ai.contracts import (
    DesignResult,
    FloorPlanAnalysis,
    FurniturePlacement,
    Point,
    Product,
)

WALL_CATEGORIES = {"sofa", "bed", "wardrobe", "tv_unit", "desk", "cabinet"}


@lru_cache(maxsize=1)
def load_catalog(path: str = "data/furniture_catalog.json") -> list[Product]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Product.model_validate(item) for item in payload]


def room_polygon(points: list[Point]) -> Polygon:
    polygon = Polygon([(point.x, point.y) for point in points]).buffer(0)
    if polygon.is_empty or not isinstance(polygon, Polygon):
        raise ValueError("Room polygon is invalid")
    return polygon


def rectangle(cx: float, cy: float, width: float, depth: float, angle: float) -> Polygon:
    candidate = box(cx - width / 2, cy - depth / 2, cx + width / 2, cy + depth / 2)
    return rotate(candidate, angle, origin=(cx, cy), use_radians=False) if angle else candidate


def _dimensions(
    room: Polygon,
    product: Product,
    pixels_per_cm: float | None,
) -> tuple[float, float, str]:
    if pixels_per_cm:
        return product.width_cm * pixels_per_cm, product.depth_cm * pixels_per_cm, "physical"
    min_x, min_y, max_x, max_y = room.bounds
    short_side = max(min(max_x - min_x, max_y - min_y), 1.0)
    scale = short_side * 0.36 / max(product.width_cm, product.depth_cm)
    return product.width_cm * scale, product.depth_cm * scale, "room-relative"


def _candidate_centers(room: Polygon, wall_preferred: bool) -> list[tuple[float, float]]:
    min_x, min_y, max_x, max_y = room.bounds
    fractions = (0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88)
    candidates = [
        ShapelyPoint(
            min_x + (max_x - min_x) * x_fraction,
            min_y + (max_y - min_y) * y_fraction,
        )
        for x_fraction in fractions
        for y_fraction in fractions
    ]
    candidates = [candidate for candidate in candidates if room.covers(candidate)]
    if wall_preferred:
        candidates.sort(key=lambda point: (point.distance(room.boundary), point.y, point.x))
    else:
        centroid = room.centroid
        candidates.sort(key=lambda point: (point.distance(centroid), point.y, point.x))
    return [(point.x, point.y) for point in candidates]


def _valid(
    room: Polygon,
    candidate: Polygon,
    placed: list[Polygon],
    gates: list[LineString],
    clearance: float,
    wall_margin: float,
) -> bool:
    inner = room.buffer(-wall_margin)
    if inner.is_empty:
        inner = room
    if not inner.covers(candidate):
        return False
    if any(candidate.buffer(clearance).intersects(gate) for gate in gates):
        return False
    return not any(candidate.intersects(existing) for existing in placed)


def furnish_floor_plan(
    floor_plan: FloorPlanAnalysis,
    *,
    room_type_overrides: dict[str, str] | None = None,
    catalog: list[Product] | None = None,
) -> DesignResult:
    active_catalog = catalog or load_catalog()
    override = room_type_overrides or {}
    placed_total = 0
    warnings = list(floor_plan.warnings)

    for room in floor_plan.rooms:
        room.room_type = override.get(room.id, room.room_type)
        polygon = room_polygon(room.polygon)
        min_x, min_y, max_x, max_y = polygon.bounds
        short_side = max(min(max_x - min_x, max_y - min_y), 1.0)
        clearance = short_side * 0.025
        wall_margin = short_side * 0.008
        gates = [
            LineString([(opening.start.x, opening.start.y), (opening.end.x, opening.end.y)])
            for opening in floor_plan.openings
        ]
        products = [product for product in active_catalog if room.room_type in product.room_types]
        products.sort(key=lambda product: (product.category, product.id))
        placed_shapes: list[Polygon] = []
        placements: list[FurniturePlacement] = []

        for product in products:
            width, depth, source = _dimensions(polygon, product, floor_plan.pixels_per_cm)
            accepted: tuple[float, float, float, Polygon] | None = None
            for cx, cy in _candidate_centers(polygon, product.category in WALL_CATEGORIES):
                for angle in (0.0, 90.0):
                    candidate = rectangle(cx, cy, width, depth, angle)
                    if _valid(polygon, candidate, placed_shapes, gates, clearance, wall_margin):
                        accepted = (cx, cy, angle, candidate)
                        break
                if accepted:
                    break
            if accepted is None:
                continue
            cx, cy, angle, shape = accepted
            placed_shapes.append(shape)
            placements.append(
                FurniturePlacement(
                    id=f"{room.id}-{product.id}",
                    category=product.category,
                    center=Point(x=cx, y=cy),
                    width=width,
                    depth=depth,
                    rotation_degrees=angle,
                    dimension_source=source,
                    confidence=0.82,
                    source_product_id=product.id,
                )
            )
        room.furniture = placements
        placed_total += len(placements)
        if not placements:
            warnings.append(f"No catalog items fit inside {room.id}")

    return DesignResult(floor_plan=floor_plan, placed_items=placed_total, warnings=warnings)
