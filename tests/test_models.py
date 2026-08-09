from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from furniture_ai.models import ModelRegistry


def test_model_registry_reports_missing_models_without_failure() -> None:
    statuses = ModelRegistry(Path("models/manifest.json")).statuses()
    assert len(statuses) == 8
    assert {status.id for status in statuses}.issuperset(
        {
            "room-classifier-efficientnet-b0",
            "floorplan-segmenter-unet",
            "detr_resnet50",
            "sam2_1_hiera_tiny",
            "depth_anything_v2_small",
        }
    )
    assert all(status.present is False for status in statuses)


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


def test_model_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "models": [
                    {"id": "dup", "task": "a", "path": "a.bin"},
                    {"id": "dup", "task": "b", "path": "b.bin"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate"):
        ModelRegistry(manifest)


torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from furniture_ai.models import _infer_num_classes, safe_load_room_classifier  # noqa: E402


def _write_classifier_checkpoint(
    path: Path, num_classes: int, *, include_classes: bool
) -> None:
    model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=num_classes)
    checkpoint = {
        "architecture": "tf_efficientnet_b0",
        "model_state": model.state_dict(),
    }
    if include_classes:
        checkpoint["classes"] = [f"class-{index}" for index in range(num_classes)]
    torch.save(checkpoint, path)


def test_safe_load_round_trip_with_classes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "classifier.pth"
    _write_classifier_checkpoint(checkpoint, 12, include_classes=True)
    model = safe_load_room_classifier(checkpoint)
    assert len(model.class_labels) == 12


def test_safe_load_infers_num_classes_from_state_dict(tmp_path: Path) -> None:
    # Regression: professional checkpoints are 12-class; the old magic default
    # of 8 built a mismatched head and failed the strict load.
    checkpoint = tmp_path / "classifier.pth"
    _write_classifier_checkpoint(checkpoint, 12, include_classes=False)
    model = safe_load_room_classifier(checkpoint)
    assert model.classifier.out_features == 12
    assert model.class_labels == tuple()


def test_safe_load_explicit_num_classes_mismatch_fails_strict(tmp_path: Path) -> None:
    checkpoint = tmp_path / "classifier.pth"
    _write_classifier_checkpoint(checkpoint, 12, include_classes=False)
    with pytest.raises(RuntimeError, match="size mismatch|Error"):
        safe_load_room_classifier(checkpoint, num_classes=8)


def test_safe_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        safe_load_room_classifier(tmp_path / "absent.pth")


def test_safe_load_non_dict_checkpoint_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.pth"
    torch.save([1, 2, 3], path)
    with pytest.raises(TypeError):
        safe_load_room_classifier(path)


def test_infer_num_classes_from_classifier_keys() -> None:
    state_dict = {
        "conv.weight": torch.zeros(16, 3, 3, 3),
        "classifier.weight": torch.zeros(12, 1280),
        "classifier.bias": torch.zeros(12),
    }
    assert _infer_num_classes(state_dict) == 12
    assert _infer_num_classes({"conv.weight": torch.zeros(16, 3, 3, 3)}) is None


def test_safe_load_torchvision_layout_detected_from_keys(tmp_path: Path) -> None:
    torchvision_models = pytest.importorskip("torchvision.models")
    model = torchvision_models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 12)
    checkpoint = tmp_path / "torchvision.pth"
    # Architecture string intentionally mismatches: key layout must win.
    torch.save(
        {"architecture": "tf_efficientnet_b0", "model_state": model.state_dict()},
        checkpoint,
    )
    loaded = safe_load_room_classifier(checkpoint)
    assert loaded.classifier[1].out_features == 12


def test_cached_loader_returns_same_instance(tmp_path: Path) -> None:
    from furniture_ai.models import clear_model_cache, load_room_classifier_cached

    clear_model_cache()
    checkpoint = tmp_path / "classifier.pth"
    _write_classifier_checkpoint(checkpoint, 12, include_classes=True)
    first = load_room_classifier_cached(checkpoint)
    second = load_room_classifier_cached(checkpoint)
    assert first is second
    assert len(first.class_labels) == 12
    clear_model_cache()


def test_cached_loader_reloads_after_checkpoint_replacement(tmp_path: Path) -> None:
    from furniture_ai.models import clear_model_cache, load_room_classifier_cached

    clear_model_cache()
    checkpoint = tmp_path / "classifier.pth"
    _write_classifier_checkpoint(checkpoint, 12, include_classes=True)
    first = load_room_classifier_cached(checkpoint)
    # Replacing the file (different size/mtime) must invalidate the cache.
    _write_classifier_checkpoint(checkpoint, 5, include_classes=True)
    second = load_room_classifier_cached(checkpoint)
    assert second is not first
    assert len(second.class_labels) == 5
    clear_model_cache()


def test_cached_loader_is_thread_safe(tmp_path: Path) -> None:
    import threading

    from furniture_ai.models import clear_model_cache, load_room_classifier_cached

    clear_model_cache()
    checkpoint = tmp_path / "classifier.pth"
    _write_classifier_checkpoint(checkpoint, 12, include_classes=False)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(load_room_classifier_cached(checkpoint)))
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 8
    assert all(model is results[0] for model in results)
    clear_model_cache()
