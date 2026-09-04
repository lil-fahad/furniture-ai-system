from __future__ import annotations

import math
from collections.abc import Iterable

from shapely.geometry import Polygon

MIN_POLYGON_AREA = 1e-6


def room_polygon_from_coordinates(
    coordinates: Iterable[tuple[float, float]],
) -> Polygon:
    """Build a strict, validated room polygon from coordinate pairs.

    The function deliberately does not repair invalid geometry. In particular,
    self-intersections must be rejected rather than silently changed with
    ``buffer(0)`` because repairing a user/model polygon can change room shape
    and area without an explicit decision by the caller.
    """
    points = list(coordinates)
    if len(points) < 3:
        raise ValueError("Room polygon requires at least three points")
    if not all(math.isfinite(x) and math.isfinite(y) for x, y in points):
        raise ValueError("Room polygon coordinates must be finite")

    polygon = Polygon(points)
    # Keep the existing degenerate classification for collapsed/collinear
    # geometry without using the repaired geometry as the accepted result.
    if polygon.is_empty or polygon.buffer(0).is_empty:
        raise ValueError("Room polygon is degenerate (collinear or near-zero area)")
    if not polygon.is_valid:
        raise ValueError("Room polygon must be a simple (non-self-intersecting) polygon")
    if polygon.area < MIN_POLYGON_AREA:
        raise ValueError("Room polygon is degenerate (collinear or near-zero area)")
    return polygon
