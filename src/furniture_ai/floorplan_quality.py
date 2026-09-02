from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import Polygon


@dataclass(frozen=True)
class GeometryAssessment:
    score: float
    warnings: tuple[str, ...] = ()


def assess_room_geometry(room: Polygon, *, image_width: int, image_height: int) -> GeometryAssessment:
    """Score geometry quality without claiming semantic room-type accuracy.

    The score is intentionally based only on observable geometry. It is useful
    for deciding when to abstain or warn, but it must not be presented as a
    model accuracy/confidence metric.
    """
    if room.is_empty or room.area <= 0 or image_width <= 0 or image_height <= 0:
        return GeometryAssessment(0.0, ("Room geometry is unusable",))

    image_area = float(image_width * image_height)
    area_ratio = room.area / image_area
    min_x, min_y, max_x, max_y = room.bounds
    touches_border = min_x <= 1 or min_y <= 1 or max_x >= image_width - 1 or max_y >= image_height - 1

    score = 1.0
    warnings: list[str] = []

    # Very small components are more likely to be text/annotations than rooms.
    if area_ratio < 0.01:
        score -= 0.35
        warnings.append("Room geometry is very small relative to the source plan")
    elif area_ratio < 0.02:
        score -= 0.15
        warnings.append("Room geometry is small relative to the source plan")

    if touches_border:
        score -= 0.45
        warnings.append("Room geometry touches the source-image border")

    # Compactness catches extremely thin/noisy regions while remaining agnostic
    # to the actual architectural style.
    perimeter = room.length
    compactness = (4.0 * 3.141592653589793 * room.area / (perimeter * perimeter)) if perimeter else 0.0
    if compactness < 0.15:
        score -= 0.25
        warnings.append("Room geometry has unusually low compactness")
    elif compactness < 0.25:
        score -= 0.10

    score = max(0.0, min(1.0, score))
    return GeometryAssessment(score=round(score, 3), warnings=tuple(warnings))
