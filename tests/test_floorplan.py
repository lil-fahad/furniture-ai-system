from __future__ import annotations

from PIL import Image, ImageDraw

from furniture_ai.floorplan import FloorPlanAnalyzer


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
