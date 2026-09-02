from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from furniture_ai.contracts import FloorPlanAnalysis, Point, Room, Unit
from furniture_ai.floorplan_quality import assess_room_geometry


class FloorPlanAnalyzer:
    def __init__(self, *, minimum_room_ratio: float = 0.012, maximum_rooms: int = 30) -> None:
        if not 0 < minimum_room_ratio < 0.5:
            raise ValueError("minimum_room_ratio must be between zero and 0.5")
        if maximum_rooms < 1:
            raise ValueError("maximum_rooms must be positive")
        self.minimum_room_ratio = minimum_room_ratio
        self.maximum_rooms = maximum_rooms

    def analyze(
        self,
        image: Image.Image,
        *,
        pixels_per_cm: float | None = None,
    ) -> FloorPlanAnalysis:
        rgb = np.asarray(image.convert("RGB"))
        room_polygons = self._extract_room_polygons(rgb)
        warnings: list[str] = []
        if not room_polygons:
            margin_x = max(image.width * 0.03, 1)
            margin_y = max(image.height * 0.03, 1)
            room_polygons = [
                Polygon(
                    [
                        (margin_x, margin_y),
                        (image.width - margin_x, margin_y),
                        (image.width - margin_x, image.height - margin_y),
                        (margin_x, image.height - margin_y),
                    ]
                )
            ]
            warnings.append("No enclosed rooms were detected; a whole-plan fallback room was used")

        room_types = infer_room_types(room_polygons)
        rooms: list[Room] = []
        for index, polygon in enumerate(room_polygons):
            room_id = f"room-{index + 1}"
            assessment = assess_room_geometry(
                polygon,
                image_width=image.width,
                image_height=image.height,
            )
            warnings.extend(f"{room_id}: {warning}" for warning in assessment.warnings)
            room_points = [
                Point(x=float(x), y=float(y))
                for x, y in polygon.exterior.coords[:-1]
            ]
            rooms.append(
                Room(
                    id=room_id,
                    room_type=room_types[index],
                    polygon=room_points,
                    area=float(polygon.area),
                    # This remains semantic-label confidence, not geometry quality.
                    confidence=0.55 if len(room_polygons) > 1 else 0.35,
                )
            )

        warnings.append(
            "Door and window extraction requires a trained segmenter and was not inferred"
        )
        return FloorPlanAnalysis(
            source_width=image.width,
            source_height=image.height,
            unit=Unit.PIXEL,
            pixels_per_cm=pixels_per_cm,
            rooms=rooms,
            openings=[],
            warnings=warnings,
            analysis_method="opencv-connected-components",
        )

    def _extract_room_polygons(self, rgb: np.ndarray) -> list[Polygon]:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, dark_lines = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        walls = cv2.morphologyEx(dark_lines, cv2.MORPH_CLOSE, kernel, iterations=2)
        walls = cv2.dilate(walls, kernel, iterations=1)
        free_space = cv2.bitwise_not(walls)

        flood = free_space.copy()
        seed = _find_exterior_seed(free_space)
        if seed is not None:
            mask = np.zeros((free_space.shape[0] + 2, free_space.shape[1] + 2), np.uint8)
            cv2.floodFill(flood, mask, seed, 128)
            interior = np.where(flood == 255, 255, 0).astype(np.uint8)
        else:
            # No reachable exterior free space: nothing can be a verified room interior.
            interior = np.zeros_like(free_space)
        interior = cv2.morphologyEx(interior, cv2.MORPH_OPEN, kernel, iterations=1)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(interior, connectivity=8)
        image_area = interior.shape[0] * interior.shape[1]
        minimum_area = image_area * self.minimum_room_ratio
        polygons: list[Polygon] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area or area > image_area * 0.92:
                continue
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            right = left + int(stats[label, cv2.CC_STAT_WIDTH])
            bottom = top + int(stats[label, cv2.CC_STAT_HEIGHT])
            if (
                left <= 0
                or top <= 0
                or right >= interior.shape[1]
                or bottom >= interior.shape[0]
            ):
                # Components touching the image border are exterior background,
                # not enclosed rooms.
                continue
            component = np.where(labels == label, 255, 0).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            epsilon = 0.008 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True).squeeze(1)
            if simplified.ndim != 2 or len(simplified) < 3:
                continue
            polygon = Polygon([(float(x), float(y)) for x, y in simplified]).buffer(0)
            if polygon.is_empty or not isinstance(polygon, Polygon) or polygon.area < minimum_area:
                continue
            polygons.append(polygon)

        polygons.sort(key=lambda polygon: (-polygon.area, polygon.centroid.y, polygon.centroid.x))
        return polygons[: self.maximum_rooms]


def _find_exterior_seed(free_space: np.ndarray) -> tuple[int, int] | None:
    """Return an (x, y) seed pixel guaranteed to be in free space.

    The exterior background is reachable from the image border, so border free
    pixels are scanned first (deterministic order). If the whole border is wall
    (e.g. a plan cropped exactly at its outer walls), any free pixel is used as
    a last resort. Returns None when the image contains no free space at all.
    """
    height, width = free_space.shape
    top = np.flatnonzero(free_space[0, :] == 255)
    if top.size:
        return (int(top[0]), 0)
    bottom = np.flatnonzero(free_space[height - 1, :] == 255)
    if bottom.size:
        return (int(bottom[-1]), height - 1)
    left = np.flatnonzero(free_space[1 : height - 1, 0] == 255)
    if left.size:
        return (0, int(left[0]) + 1)
    right = np.flatnonzero(free_space[1 : height - 1, width - 1] == 255)
    if right.size:
        return (width - 1, int(right[0]) + 1)
    free = np.argwhere(free_space == 255)
    if free.size:
        y, x = free[0]
        return (int(x), int(y))
    return None


def infer_room_types(polygons: Iterable[Polygon]) -> list[str]:
    polygons_list = list(polygons)
    if not polygons_list:
        return []
    ranked = sorted(range(len(polygons_list)), key=lambda index: -polygons_list[index].area)
    result = ["room"] * len(polygons_list)
    templates = [
        "living_room",
        "bedroom",
        "bedroom",
        "kitchen",
        "bathroom",
        "dining_room",
        "office",
    ]
    for rank, original_index in enumerate(ranked):
        if rank < len(templates):
            result[original_index] = templates[rank]
        # Beyond the template list keep the generic "room" label instead of
        # mislabeling every remaining room as "office".
    return result
