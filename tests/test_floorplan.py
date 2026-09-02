from __future__ import annotations

import pytest
from PIL import Image, ImageDraw
from shapely.geometry import box

import furniture_ai.floorplan as floorplan_module
from furniture_ai.floorplan import FloorPlanAnalyzer, infer_room_types
from furniture_ai.floorplan_quality import GeometryAssessment, assess_room_geometry


def synthetic_plan() -> Image.Image:
    image = Image.new("RGB", (600, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 570, 370), outline="black", width=12)
    draw.line((300, 30, 300, 370), fill="black", width=12)
    return image


def test_analyzer_is_deterministic_and_returns_rooms() -> None:
    analyzer = FloorPlanAnalyzer(minimum_room_ratio=0.02)
    first = analyzer.analyze(synthetic_plan())
    second = analyzer.analyze(synthetic_plan())
    assert len(first.rooms) >= 1
    assert first.model_dump() == second.model_dump()
    assert first.source_width == 600


def test_dark_corner_mark_does_not_turn_exterior_into_a_room() -> None:
    image = synthetic_plan()
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 40, 25), fill="black")
    result = FloorPlanAnalyzer().analyze(image)
    assert len(result.rooms) == 2
    image_area = image.width * image.height
    for room in result.rooms:
        xs = [point.x for point in room.polygon]
        ys = [point.y for point in room.polygon]
        assert room.area < image_area * 0.5
        assert not (min(xs) <= 40 and min(ys) <= 25)
    assert {room.room_type for room in result.rooms} == {"living_room", "bedroom"}


def test_border_touching_free_space_is_not_a_room() -> None:
    image = Image.new("RGB", (400, 400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((100, 100, 300, 300), outline="black", width=10)
    result = FloorPlanAnalyzer(minimum_room_ratio=0.02).analyze(image)
    assert len(result.rooms) == 1
    xs = [point.x for point in result.rooms[0].polygon]
    ys = [point.y for point in result.rooms[0].polygon]
    assert min(xs) > 0 and min(ys) > 0
    assert max(xs) < 400 and max(ys) < 400


def test_infer_room_types_falls_back_to_generic_room() -> None:
    labels = infer_room_types([box(index * 10, 0, index * 10 + 5, 5) for index in range(10)])
    assert labels.count("office") == 1
    assert labels[-3:] == ["room", "room", "room"]


def test_geometry_assessment_is_not_semantic_confidence() -> None:
    assessment = assess_room_geometry(
        box(100, 100, 500, 300),
        image_width=600,
        image_height=400,
    )
    assert assessment.score == pytest.approx(1.0)
    assert assessment.warnings == ()


def test_geometry_assessment_warns_on_border_touching_region() -> None:
    assessment = assess_room_geometry(
        box(0, 50, 300, 350),
        image_width=600,
        image_height=400,
    )
    assert assessment.score < 1
    assert any("touches the source-image border" in warning for warning in assessment.warnings)


def test_geometry_assessment_warns_on_tiny_region() -> None:
    assessment = assess_room_geometry(
        box(10, 10, 20, 20),
        image_width=600,
        image_height=400,
    )
    assert assessment.score < 1
    assert any("very small" in warning for warning in assessment.warnings)


def test_analyzer_surfaces_geometry_quality_warnings(monkeypatch: pytest.MonkeyPatch) -> None:
    def forced_assessment(*args, **kwargs) -> GeometryAssessment:
        return GeometryAssessment(0.4, ("synthetic geometry concern",))

    monkeypatch.setattr(floorplan_module, "assess_room_geometry", forced_assessment)
    result = FloorPlanAnalyzer(minimum_room_ratio=0.02).analyze(synthetic_plan())
    assert any(warning.endswith("synthetic geometry concern") for warning in result.warnings)
    # Geometry quality must not overwrite semantic classification confidence.
    assert all(room.confidence == pytest.approx(0.55) for room in result.rooms)
