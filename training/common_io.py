"""Shared I/O helpers for cloud and local training runs (SPEC §2.2/§2.3/§2.4).

All Google Cloud Storage access is lazy: ``google-cloud-storage`` (or WP-B's
``training.data_ingest.gcs`` wrapper) is imported inside functions only, so this
module imports cleanly in environments without any GCP dependencies.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TASKS = ("room", "segmenter", "ranker")

# SPEC §2.2/§2.4 dataset directory names per task. The GCS layout uses
# datasets/{rooms,plans,catalog} while the CLI tasks are {room,segmenter,ranker}.
TASK_DATASETS = {"room": "rooms", "segmenter": "plans", "ranker": "catalog"}

_METRICS_KEYS = {
    "task",
    "run_id",
    "status",
    "epochs",
    "final_val_metric_name",
    "final_val_metric_value",
    "checkpoint_files",
    "timestamp_utc",
    "extra",
}

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return _REPO_ROOT


def gcs_available() -> bool:
    """Return True when a google-cloud-storage client can be imported."""
    try:
        from google.cloud import storage  # noqa: F401
    except ImportError:
        return False
    return True


def _gcs_download_dir(bucket: str, prefix: str, local_dir: Path) -> int:
    """Download ``gs://<bucket>/<prefix>/`` into ``local_dir``; returns file count."""
    try:
        from training.data_ingest import gcs  # WP-B wrapper, lazy per SPEC §4

        return gcs.download_dir(bucket, prefix, local_dir)
    except ImportError:
        pass
    from google.cloud import storage

    client = storage.Client()
    blobs = client.list_blobs(bucket, prefix=prefix.rstrip("/") + "/")
    count = 0
    for blob in blobs:
        relative = blob.name[len(prefix.rstrip("/") + "/") :]
        if not relative or relative.endswith("/"):
            continue
        target = local_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        count += 1
    return count


def _gcs_upload_dir(local_dir: Path, bucket: str, prefix: str) -> list[str]:
    """Upload every file under ``local_dir`` to ``gs://<bucket>/<prefix>/...``."""
    try:
        from training.data_ingest import gcs  # WP-B wrapper, lazy per SPEC §4

        gcs.upload_dir(local_dir, bucket, prefix)
        return sorted(
            f"gs://{bucket}/{prefix}/{path.relative_to(local_dir).as_posix()}"
            for path in local_dir.rglob("*")
            if path.is_file()
        )
    except ImportError:
        pass
    from google.cloud import storage

    client = storage.Client()
    bucket_obj = client.bucket(bucket)
    uris: list[str] = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file():
            continue
        blob_name = f"{prefix}/{path.relative_to(local_dir).as_posix()}"
        bucket_obj.blob(blob_name).upload_from_filename(str(path))
        uris.append(f"gs://{bucket}/{blob_name}")
    return uris


def resolve_dataset(task: str, staging_root: Path = Path("/tmp/fai_data")) -> Path:
    """Resolve the local dataset directory for ``task`` (SPEC §3 WP-C item 1).

    When ``GCS_BUCKET`` is set and a google-cloud-storage client is importable,
    downloads ``gs://$GCS_BUCKET/datasets/<dataset>/`` into
    ``<staging_root>/<dataset>`` and returns that path. Otherwise falls back to
    the repo-local ``data/staging/<dataset>`` directory and raises a clear
    error when it is absent.
    """
    if task not in TASK_DATASETS:
        raise ValueError(f"Unknown task {task!r}; expected one of {sorted(TASK_DATASETS)}")
    dataset = TASK_DATASETS[task]
    bucket = os.getenv("GCS_BUCKET")
    if bucket and gcs_available():
        destination = staging_root / dataset
        destination.mkdir(parents=True, exist_ok=True)
        count = _gcs_download_dir(bucket, f"datasets/{dataset}", destination)
        if count == 0:
            raise FileNotFoundError(
                f"No objects found under gs://{bucket}/datasets/{dataset}/; "
                "stage datasets first (python -m training.data_ingest.stage_all)"
            )
        return destination
    staging = _repo_root() / "data" / "staging"
    candidates = [staging / dataset]
    if dataset != task:
        candidates.append(staging / task)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    reason = (
        f"GCS_BUCKET={bucket!r} but google-cloud-storage is not importable"
        if bucket
        else "GCS_BUCKET is not set"
    )
    raise FileNotFoundError(
        f"Local dataset for task {task!r} not found at {candidates[0]} and {reason}. "
        "Run `python -m training.data_ingest.stage_all` to build data/staging, "
        "or pass --data-dir explicitly."
    )


def upload_run_artifacts(run_dir: Path, bucket: str | None, run_id: str) -> list[str]:
    """Upload ``run_dir`` contents to ``gs://<bucket>/runs/<run_id>/``.

    Returns the uploaded ``gs://`` URIs. No-op returning ``[]`` when ``bucket``
    is None or the storage SDK is unavailable (SPEC §3 WP-C item 1).
    """
    if bucket is None or not gcs_available():
        return []
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    return _gcs_upload_dir(run_dir, bucket, f"runs/{run_id}")


def validate_metrics(metrics: dict) -> None:
    """Validate ``metrics`` against the SPEC §2.3 schema; raise ValueError."""
    if not isinstance(metrics, dict):
        raise ValueError(f"metrics must be a dict, got {type(metrics).__name__}")
    missing = sorted(_METRICS_KEYS - metrics.keys())
    if missing:
        raise ValueError(f"metrics.json is missing required keys: {', '.join(missing)}")
    if metrics["task"] not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}, got {metrics['task']!r}")
    if not isinstance(metrics["run_id"], str) or not metrics["run_id"]:
        raise ValueError("run_id must be a non-empty string")
    if metrics["status"] not in ("succeeded", "failed"):
        raise ValueError(f"status must be 'succeeded' or 'failed', got {metrics['status']!r}")
    epochs = metrics["epochs"]
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs < 0:
        raise ValueError(f"epochs must be a non-negative integer, got {epochs!r}")
    if not isinstance(metrics["final_val_metric_name"], str):
        raise ValueError("final_val_metric_name must be a string")
    value = metrics["final_val_metric_value"]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"final_val_metric_value must be a number, got {value!r}")
    checkpoint_files = metrics["checkpoint_files"]
    if not isinstance(checkpoint_files, list) or not all(
        isinstance(item, str) for item in checkpoint_files
    ):
        raise ValueError("checkpoint_files must be a list of file-name strings")
    if not isinstance(metrics["timestamp_utc"], str) or not metrics["timestamp_utc"]:
        raise ValueError("timestamp_utc must be a non-empty ISO-8601 string")
    if not isinstance(metrics["extra"], dict):
        raise ValueError("extra must be a dict")


def write_metrics(metrics: dict, run_dir: Path) -> Path:
    """Validate ``metrics`` (SPEC §2.3) and write ``<run_dir>/metrics.json``."""
    validate_metrics(metrics)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    destination = run_dir / "metrics.json"
    destination.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return destination
