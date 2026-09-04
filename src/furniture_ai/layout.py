from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Literal

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
from furniture_ai.geometry import room_polygon_from_coordinates

WALL_CATEGORIES = {"sofa", "bed", "wardrobe", "tv_unit", "desk", "cabinet"}
PlacementPolicyName = Literal["balanced", "wall_first", "fit_first"]
VALID_PLACEMENT_POLICIES: frozenset[str] = frozenset(
    {"balanced", "wall_first", "fit_first"}
)


def _catalog_text(path: str | Path | None) -> str:
    if path is not None:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Furniture catalog not found: {source}")
        return source.read_text(encoding="utf-8")

    from furniture_ai.config import get_settings

    settings = get_settings()
    source = Path(settings.catalog_path)
    if source.is_file():
        return source.read_text(encoding="utf-8")
    if settings.catalog_path_overridden:
        raise FileNotFoundError(f"Furniture catalog not found: {source}")
    resource = resources.files("furniture_ai.resources").joinpath("furniture_catalog.json")
    return resource.read_text(encoding="utf-8")


@lru_cache(maxsize=8)
def load_catalog(path: str | Path | None = None) -> list[Product]:
    """Load an explicit catalog or the packaged default.

    Explicit/configured custom paths fail closed when missing. Only the
    untouched default setting may fall back to the wheel-packaged catalog.
    """
    payload = json.loads(_catalog_text(path))
    if not isinstance(payload, list):
        raise ValueError("Furniture catalog must contain a JSON array")
    return [Product.model_validate(item) for item in payload]


def room_polygon(points: list[Point]) -> Polygon:
    """Build the canonical strict room polygon used by all layout paths."""
    return room_polygon_from_coordinates((point.x, point.y) for point in points)


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


def _candidate_centers(
    room: Polygon,
    wall_preferred: bool,
    *,
    policy: PlacementPolicyName = "balanced",
) -> list[tuple[float, float]]:
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
    centroid = room.centroid

    if policy == "fit_first":
        candidates.sort(key=lambda point: (point.distance(centroid), point.y, point.x))
    elif wall_preferred:
        candidates.sort(key=lambda point: (point.distance(room.boundary), point.y, point.x))
    else:
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


def _ordered_products(
    products: list[Product],
    policy: PlacementPolicyName,
) -> list[Product]:
    ordered = list(products)
    if policy == "wall_first":
        ordered.sort(
            key=lambda product: (
                product.category not in WALL_CATEGORIES,
                product.category,
                product.id,
            )
        )
    elif policy == "fit_first":
        ordered.sort(
            key=lambda product: (
                product.width_cm * product.depth_cm,
                product.category,
                product.id,
            )
        )
    else:
        ordered.sort(key=lambda product: (product.category, product.id))
    return ordered


def furnish_floor_plan(
    floor_plan: FloorPlanAnalysis,
    *,
    room_type_overrides: dict[str, str] | None = None,
    catalog: list[Product] | None = None,
    placement_policy: PlacementPolicyName = "balanced",
) -> DesignResult:
    """Furnish a floor plan without mutating the caller's object.

    ``placement_policy`` changes deterministic search order only. It is not an
    aesthetic score or a claim that one policy is universally better.
    """
    if placement_policy not in VALID_PLACEMENT_POLICIES:
        allowed = ", ".join(sorted(VALID_PLACEMENT_POLICIES))
        raise ValueError(
            f"Unknown placement_policy {placement_policy!r}; expected one of: {allowed}"
        )

    active_catalog = catalog if catalog is not None else load_catalog()
    override = room_type_overrides or {}
    plan = floor_plan.model_copy(deep=True)
    placed_total = 0
    warnings = list(plan.warnings)
    gates = [
        LineString([(opening.start.x, opening.start.y), (opening.end.x, opening.end.y)])
        for opening in plan.openings
    ]

    for room in plan.rooms:
        room.room_type = override.get(room.id, room.room_type)
        polygon = room_polygon(room.polygon)
        min_x, min_y, max_x, max_y = polygon.bounds
        short_side = max(min(max_x - min_x, max_y - min_y), 1.0)
        clearance = short_side * 0.025
        wall_margin = short_side * 0.008
        products = [
            product for product in active_catalog if room.room_type in product.room_types
        ]
        products = _ordered_products(products, placement_policy)
        placed_shapes: list[Polygon] = []
        placements: list[FurniturePlacement] = []

        for product in products:
            width, depth, source = _dimensions(polygon, product, plan.pixels_per_cm)
            if width <= 0 or depth <= 0:
                continue
            accepted: tuple[float, float, float, Polygon] | None = None
            for cx, cy in _candidate_centers(
                polygon,
                product.category in WALL_CATEGORIES,
                policy=placement_policy,
            ):
                for angle in (0.0, 90.0):
                    candidate = rectangle(cx, cy, width, depth, angle)
                    if _valid(
                        polygon,
                        candidate,
                        placed_shapes,
                        gates,
                        clearance,
                        wall_margin,
                    ):
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
                    confidence=None,
                    source_product_id=product.id,
                )
            )
        room.furniture = placements
        placed_total += len(placements)
        if not placements:
            if not products:
                warnings.append(
                    f"No catalog products match room type {room.room_type!r} for {room.id}"
                )
            else:
                warnings.append(f"No catalog items fit inside {room.id}")

    return DesignResult(floor_plan=plan, placed_items=placed_total, warnings=warnings)
