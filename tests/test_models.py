from __future__ import annotations

import hashlib
import json
from pathlib import Path

from furniture_ai.models import ModelRegistry


def test_model_registry_reports_missing_models_without_failure() -> None:
    statuses = ModelRegistry(Path("models/manifest.json")).statuses()
    assert len(statuses) == 9
    assert {status.id for status in statuses}.issuperset(
        {
            "room-classifier-efficientnet-b0",
            "floorplan-segmenter-unet",
            "detr_resnet50",
            "sam2_1_hiera_tiny",
            "depth_anything_v2_small",
        }
    )
    present = {status.id: status.present for status in statuses}
    assert present["supplier-ranker-ridge"] is True
    assert all(
        not is_present
        for model_id, is_present in present.items()
        if model_id != "supplier-ranker-ridge"
    )


def test_model_registry_verifies_size_and_hash(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    weight = models / "weight.bin"
    weight.write_bytes(b"verified")
    manifest = models / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "models": [
                    {
                        "id": "verified-model",
                        "name": "Verified Model",
                        "task": "test",
                        "path": "weight.bin",
                        "size_bytes": weight.stat().st_size,
                        "sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    status = ModelRegistry(manifest).statuses()[0]
    assert status.present is True
    assert status.verified is True
