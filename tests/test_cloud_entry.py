"""Offline tests for training.common_io and training.cloud_entry (SPEC §3 WP-C item 3).

No network access: GCS_BUCKET is explicitly cleared and the room-classifier
smoke run uses --no-pretrained.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training import cloud_entry, common_io  # noqa: E402


def valid_metrics() -> dict:
    return {
        "task": "room",
        "run_id": "20260810T120000Z-room",
        "status": "succeeded",
        "epochs": 1,
        "final_val_metric_name": "val_accuracy",
        "final_val_metric_value": 0.5,
        "checkpoint_files": ["room_classifier.pt"],
        "timestamp_utc": "2026-08-10T12:00:00+00:00",
        "extra": {},
    }


@pytest.fixture(autouse=True)
def no_gcs_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    monkeypatch.delenv("RUN_ID", raising=False)


# --- write_metrics schema validation (SPEC §2.3) -----------------------------


def test_write_metrics_roundtrip(tmp_path: Path) -> None:
    metrics = valid_metrics()
    destination = common_io.write_metrics(metrics, tmp_path)
    assert destination == tmp_path / "metrics.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == metrics


@pytest.mark.parametrize("missing_key", sorted(common_io._METRICS_KEYS))
def test_write_metrics_missing_key_raises(tmp_path: Path, missing_key: str) -> None:
    metrics = valid_metrics()
    del metrics[missing_key]
    with pytest.raises(ValueError, match=missing_key):
        common_io.write_metrics(metrics, tmp_path)
    assert not (tmp_path / "metrics.json").exists()


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("task", "unknown-task"),
        ("run_id", ""),
        ("status", "partial"),
        ("epochs", "one"),
        ("epochs", -1),
        ("final_val_metric_name", 7),
        ("final_val_metric_value", "high"),
        ("checkpoint_files", "room_classifier.pt"),
        ("checkpoint_files", ["ok.pt", 3]),
        ("timestamp_utc", ""),
        ("extra", []),
    ],
)
def test_write_metrics_invalid_values_raise(tmp_path: Path, key: str, bad_value: object) -> None:
    metrics = valid_metrics()
    metrics[key] = bad_value
    with pytest.raises(ValueError):
        common_io.write_metrics(metrics, tmp_path)


# --- upload_run_artifacts / resolve_dataset offline behavior -----------------


def test_upload_run_artifacts_noop_without_bucket(tmp_path: Path) -> None:
    assert common_io.upload_run_artifacts(tmp_path, None, "run-1") == []


def test_resolve_dataset_unknown_task() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        common_io.resolve_dataset("not-a-task")


def test_resolve_dataset_local_staging_missing() -> None:
    # data/staging is absent in a fresh checkout and no GCS_BUCKET is set.
    with pytest.raises(FileNotFoundError, match="data/staging"):
        common_io.resolve_dataset("room")


def test_resolve_dataset_uses_local_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    staging = tmp_path / "data" / "staging" / "rooms"
    staging.mkdir(parents=True)
    monkeypatch.setattr(common_io, "_REPO_ROOT", tmp_path)
    assert common_io.resolve_dataset("room") == staging


def test_default_run_id_format() -> None:
    run_id = cloud_entry.default_run_id("room")
    assert run_id.endswith("-room")
    assert len(run_id.split("-")[0]) == 16  # YYYYMMDDTHHMMSSZ


def test_trainer_failure_writes_failed_metrics(tmp_path: Path) -> None:
    # An empty ImageFolder makes the room trainer exit non-zero; metrics.json
    # must still be written with status=failed and the run must exit non-zero.
    empty = tmp_path / "rooms"
    (empty / "placeholder").mkdir(parents=True)
    run_dir = tmp_path / "run"
    exit_code = cloud_entry.main(
        [
            "--task",
            "room",
            "--epochs",
            "1",
            "--run-id",
            "test-fail",
            "--run-dir",
            str(run_dir),
            "--data-dir",
            str(empty),
        ]
    )
    assert exit_code != 0
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "failed"
    assert metrics["checkpoint_files"] == []


# --- offline smoke: cloud_entry --task room (SPEC §3 WP-C item 3) ------------


def test_cloud_entry_room_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    pytest.importorskip("timm")
    import numpy as np
    from PIL import Image

    # Tiny ImageFolder: 2 classes x 4 small images.
    rng = np.random.default_rng(0)
    data_root = tmp_path / "rooms"
    for name, color in {"living_room": (200, 60, 60), "bedroom": (60, 200, 60)}.items():
        directory = data_root / name
        directory.mkdir(parents=True)
        for index in range(4):
            noise = rng.normal(0, 10, (32, 32, 3)).astype(np.int16)
            array = np.clip(np.array(color, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(array).save(directory / f"{index:03d}.png")

    run_dir = tmp_path / "run"
    exit_code = cloud_entry.main(
        [
            "--task",
            "room",
            "--epochs",
            "1",
            "--run-id",
            "test-room-smoke",
            "--run-dir",
            str(run_dir),
            "--data-dir",
            str(data_root),
            # forwarded to train_room_classifier.py (offline, tiny dataset):
            "--no-pretrained",
            "--min-images",
            "4",
            "--img-size",
            "64",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
        ]
    )
    assert exit_code == 0

    metrics_path = run_dir / "metrics.json"
    assert metrics_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    common_io.validate_metrics(metrics)  # SPEC §2.3 schema
    assert metrics["task"] == "room"
    assert metrics["run_id"] == "test-room-smoke"
    assert metrics["status"] == "succeeded"
    assert metrics["epochs"] == 1
    assert metrics["final_val_metric_name"] == "val_accuracy"
    assert 0.0 <= metrics["final_val_metric_value"] <= 1.0
    assert metrics["checkpoint_files"] == ["room_classifier.pt"]
    assert (run_dir / "checkpoints" / "room_classifier.pt").is_file()
    assert (run_dir / "logs.txt").is_file()
