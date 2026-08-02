from __future__ import annotations

from pathlib import Path

from furniture_ai.models import ModelRegistry


def test_model_registry_reports_missing_models_without_failure() -> None:
    statuses = ModelRegistry(Path("models/manifest.json")).statuses()
    assert {status.id for status in statuses} == {
        "room-classifier-efficientnet-b0",
        "floorplan-segmenter-unet",
    }
    assert all(status.present is False for status in statuses)
