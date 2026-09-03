from __future__ import annotations

import math
from itertools import combinations

from shapely.geometry import LineString, Polygon

from furniture_ai.contracts import (
    ConstraintIssue,
    ConstraintSeverity,
    FloorPlanAnalysis,
    LayoutValidationReport,
    OpeningKind,
)
from furniture_ai.layout import rectangle, room_polygon

GEOMETRY_EPSILON = 1e-9


def _error(
    *,
    code: str,
    message: str,
    room_id: str,
    item_ids: list[str],
    opening_id: str | None = None,
) -> ConstraintIssue:
    return ConstraintIssue(
        code=code,
        severity=ConstraintSeverity.ERROR,
        message=message,
        room_id=room_id,
        item_ids=item_ids,
        opening_id=opening_id,
    )


def _door_lines(floor_plan: FloorPlanAnalysis) -> list[tuple[str, LineString]]:
    return [
        (
            opening.id,
            LineString(
                [
                    (opening.start.x, opening.start.y),
                    (opening.end.x, opening.end.y),
                ]
            ),
        )
        for opening in floor_plan.openings
        if opening.kind is OpeningKind.DOOR
    ]


def _finite_footprint_values(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def validate_layout_constraints(
    floor_plan: FloorPlanAnalysis,
    *,
    minimum_clearance: float = 0.0,
) -> LayoutValidationReport:
    """Validate deterministic spatial constraints without inventing design distances.

    ``minimum_clearance`` is optional and expressed in the same coordinate units
    as the supplied geometry. The default is zero, so no domain-specific spacing
    requirement is assumed implicitly.
    """
    if not math.isfinite(minimum_clearance) or minimum_clearance < 0:
        raise ValueError("minimum_clearance must be finite and non-negative")

    issues: list[ConstraintIssue] = []
    checked_items = 0
    doors = _door_lines(floor_plan)

    for room in floor_plan.rooms:
        boundary = room_polygon(room.polygon)
        room_doors = [entry for entry in doors if boundary.intersects(entry[1])]
        shapes: list[tuple[str, Polygon]] = []

        for item in room.furniture:
            checked_items += 1
            footprint_values = (
                item.center.x,
                item.center.y,
                item.width,
                item.depth,
                item.rotation_degrees,
            )
            valid_values = _finite_footprint_values(*footprint_values)
            if (
                not valid_values
                or item.width <= GEOMETRY_EPSILON
                or item.depth <= GEOMETRY_EPSILON
            ):
                issues.append(
                    _error(
                        code="invalid_footprint",
                        message=(
                            "Furniture footprint coordinates and dimensions must be finite, "
                            "with positive width and depth."
                        ),
                        room_id=room.id,
                        item_ids=[item.id],
                    )
                )
                continue

            shape = rectangle(
                item.center.x,
                item.center.y,
                item.width,
                item.depth,
                item.rotation_degrees,
            )
            shapes.append((item.id, shape))

            if not boundary.covers(shape):
                issues.append(
                    _error(
                        code="outside_room",
                        message="Furniture footprint extends outside the room boundary.",
                        room_id=room.id,
                        item_ids=[item.id],
                    )
                )

            door_obstacle = shape.buffer(minimum_clearance) if minimum_clearance else shape
            for opening_id, door in room_doors:
                if door_obstacle.intersects(door):
                    issues.append(
                        _error(
                            code="door_blocked",
                            message=(
                                "Furniture footprint blocks a door opening "
                                "or its requested clearance."
                            ),
                            room_id=room.id,
                            item_ids=[item.id],
                            opening_id=opening_id,
                        )
                    )

        for (left_id, left), (right_id, right) in combinations(shapes, 2):
            overlap_area = left.intersection(right).area
            if overlap_area > GEOMETRY_EPSILON:
                issues.append(
                    _error(
                        code="collision",
                        message="Furniture footprints overlap.",
                        room_id=room.id,
                        item_ids=[left_id, right_id],
                    )
                )
                continue
            clearance_too_small = (
                minimum_clearance > 0
                and left.distance(right) + GEOMETRY_EPSILON < minimum_clearance
            )
            if clearance_too_small:
                issues.append(
                    _error(
                        code="clearance_violation",
                        message="Furniture separation is below the requested minimum clearance.",
                        room_id=room.id,
                        item_ids=[left_id, right_id],
                    )
                )

    return LayoutValidationReport(
        valid=not any(issue.severity is ConstraintSeverity.ERROR for issue in issues),
        checked_rooms=len(floor_plan.rooms),
        checked_items=checked_items,
        issues=issues,
    )
