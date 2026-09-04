from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from furniture_ai.contracts import Point, Room
from furniture_ai.geometry import MIN_POLYGON_AREA, room_polygon_from_coordinates
from furniture_ai.layout import room_polygon


def test_canonical_room_polygon_accepts_valid_square() -> None:
    polygon = room_polygon_from_coordinates([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])

    assert polygon.area == pytest.approx(100.0)
    assert polygon.is_valid


def test_canonical_room_polygon_rejects_self_intersection_without_repair() -> None:
    coordinates = [(0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)]

    with pytest.raises(ValueError, match="simple \(non-self-intersecting\)"):
        room_polygon_from_coordinates(coordinates)


def test_canonical_room_polygon_rejects_collinear_geometry() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        room_polygon_from_coordinates([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])


def test_canonical_room_polygon_rejects_near_zero_area() -> None:
    side = math.sqrt(MIN_POLYGON_AREA)
    coordinates = [(0.0, 0.0), (side, 0.0), (0.0, side)]

    assert side * side / 2 < MIN_POLYGON_AREA
    with pytest.raises(ValueError, match="degenerate"):
        room_polygon_from_coordinates(coordinates)


def test_canonical_room_polygon_rejects_non_finite_coordinates() -> None:
    with pytest.raises(ValueError, match="finite"):
        room_polygon_from_coordinates([(0.0, 0.0), (1.0, 0.0), (math.inf, 1.0)])


def test_contract_and_layout_share_self_intersection_policy() -> None:
    coordinates = [(0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)]
    points = [Point(x=x, y=y) for x, y in coordinates]

    with pytest.raises(ValidationError, match="simple \(non-self-intersecting\)"):
        Room(id="room-1", room_type="living_room", polygon=points, area=100.0)

    with pytest.raises(ValueError, match="simple \(non-self-intersecting\)"):
        room_polygon(points)
