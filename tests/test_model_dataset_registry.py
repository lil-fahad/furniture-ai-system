from __future__ import annotations

import json
from pathlib import Path

import pytest

from furniture_ai.model_dataset_registry import ModelDatasetRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "data" / "model_dataset_registry.json"


EXPECTED_TRAINED_MODELS = {
    "efficientnet_b0-efficientnet_b0_best",
    "efficientnet_b0_multiview-efficientnet_b0_best",
    "efficientnet_b0_multiview-efficientnet_b0_last",
    "supplier-ranker-ridge-v1",
}


def test_project_registry_contains_only_verified_trained_artifacts() -> None:
    registry = ModelDatasetRegistry.load(REGISTRY_PATH)

    assert {model.id for model in registry.models} == EXPECTED_TRAINED_MODELS
    assert "room-classifier-efficientnet-b0" not in {model.id for model in registry.models}
    assert "floorplan-segmenter-unet" not in {model.id for model in registry.models}


def test_every_trained_model_has_resolvable_dataset_lineage() -> None:
    registry = ModelDatasetRegistry.load(REGISTRY_PATH)
    dataset_ids = {dataset.id for dataset in registry.datasets}

    assert registry.models
    for model in registry.models:
        assert model.dataset_ids
        assert set(model.dataset_ids) <= dataset_ids
        assert not Path(model.artifact_path).is_absolute()
        assert Path(model.artifact_path).parts[0] == "models"


def test_registry_queries_models_and_datasets_bidirectionally() -> None:
    registry = ModelDatasetRegistry.load(REGISTRY_PATH)

    starter = registry.trained_models_for_dataset("abo-starter-recovered")
    assert [model.id for model in starter] == ["efficientnet_b0-efficientnet_b0_best"]

    datasets = registry.datasets_for_model("supplier-ranker-ridge-v1")
    assert [dataset.id for dataset in datasets] == ["supplier-master-v1"]


def test_registry_rejects_unknown_dataset_reference(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": [{"id": "known", "name": "Known", "kind": "test"}],
                "models": [
                    {
                        "id": "trained",
                        "task": "test",
                        "architecture": "test",
                        "artifact_path": "models/trained.bin",
                        "sha256": "a" * 64,
                        "dataset_ids": ["missing"],
                        "status": "trained",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown dataset"):
        ModelDatasetRegistry.load(path)


def test_registry_rejects_model_artifacts_inside_dataset_tree(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": [{"id": "known", "name": "Known", "kind": "test"}],
                "models": [
                    {
                        "id": "trained",
                        "task": "test",
                        "architecture": "test",
                        "artifact_path": "data/known/checkpoint.pt",
                        "sha256": "b" * 64,
                        "dataset_ids": ["known"],
                        "status": "trained",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must live outside data/"):
        ModelDatasetRegistry.load(path)


def test_runtime_manifest_exposes_dataset_ids_for_trained_classifiers() -> None:
    manifest = json.loads((PROJECT_ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    entries = {entry["id"]: entry for entry in manifest["models"]}

    assert entries["efficientnet_b0-efficientnet_b0_best"]["dataset_ids"] == [
        "abo-starter-recovered"
    ]
    assert entries["efficientnet_b0_multiview-efficientnet_b0_best"]["dataset_ids"] == [
        "abo-multiview-recovered"
    ]
    assert entries["efficientnet_b0_multiview-efficientnet_b0_last"]["dataset_ids"] == [
        "abo-multiview-recovered"
    ]
