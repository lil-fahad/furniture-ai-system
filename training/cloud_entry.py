"""Cloud/local training entrypoint (SPEC §1, §3 WP-C item 2).

Usage:
    python -m training.cloud_entry --task {room,segmenter,ranker} [--epochs N]
        [--run-id ID] [--run-dir PATH] [--data-dir PATH] [extra trainer args...]

Dispatches to the existing trainer scripts via subprocess so their behavior is
completely untouched, captures logs into the run dir, writes a SPEC §2.3
``metrics.json``, and uploads run artifacts when ``GCS_BUCKET`` is set. Fully
local mode (no GCP environment) is the default and is used by the tests.

Any arguments not recognized by this parser are forwarded verbatim to the
underlying trainer, e.g.::

    python -m training.cloud_entry --task room --epochs 1 --data-dir ./rooms \
        --no-pretrained --min-images 4 --img-size 64
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    from training import common_io
except ImportError:  # running as a loose script (python training/cloud_entry.py)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from training import common_io

TRAINING_DIR = Path(__file__).resolve().parent

# Default epochs per task, mirroring cloud/config.yaml (SPEC §3 WP-A item 3).
DEFAULT_EPOCHS = {"room": 15, "segmenter": 25, "ranker": 1}

CHECKPOINT_NAMES = {
    "room": "room_classifier.pt",
    "segmenter": "floorplan_segmenter.pt",
    "ranker": "supplier_ranker.json",
}


def default_run_id(task: str) -> str:
    """UTC timestamp + task, e.g. ``20260810T120000Z-room`` (SPEC §2.1)."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{task}"


def _room_imagefolder_root(data_dir: Path) -> Path:
    """Return the ImageFolder root for the room classifier.

    The GCS layout (SPEC §2.2) nests class folders under ``rooms/images/``,
    while plain local staging may hold class folders directly.
    """
    nested = data_dir / "images"
    if nested.is_dir() and any(child.is_dir() for child in nested.iterdir()):
        return nested
    return data_dir


def _ensure_ranker_csv(data_dir: Path, work_dir: Path) -> Path:
    """Locate the suppliers catalog CSV for the ranker.

    ``load_supplier_rows`` reads ``*.gz.b64`` natively and treats every other
    path as plain CSV text, so a real gzip payload is transparently decompressed
    here to keep the trainer's behavior untouched.
    """
    if data_dir.is_file():
        return data_dir
    encoded = data_dir / "suppliers_master.csv.gz.b64"
    if encoded.is_file():
        return encoded
    plain = data_dir / "suppliers_master.csv.gz"
    if plain.is_file():
        if plain.read_bytes()[:2] == b"\x1f\x8b":  # real gzip -> plain CSV text
            work_dir.mkdir(parents=True, exist_ok=True)
            target = work_dir / "suppliers_master.csv"
            target.write_bytes(gzip.decompress(plain.read_bytes()))
            return target
        return plain
    raise FileNotFoundError(
        f"No suppliers_master.csv.gz (or .b64) found under {data_dir}; "
        "stage the catalog dataset first or pass --data-dir pointing at it."
    )


def build_trainer_command(
    task: str, data_dir: Path, run_dir: Path, epochs: int, extra_args: list[str]
) -> list[str]:
    """Build the subprocess command for the existing trainer script."""
    checkpoints = run_dir / "checkpoints"
    if task == "room":
        return [
            sys.executable,
            str(TRAINING_DIR / "train_room_classifier.py"),
            str(_room_imagefolder_root(data_dir)),
            "--output",
            str(checkpoints / CHECKPOINT_NAMES["room"]),
            "--epochs",
            str(epochs),
            *extra_args,
        ]
    if task == "segmenter":
        return [
            sys.executable,
            str(TRAINING_DIR / "train_floorplan_segmenter.py"),
            str(data_dir),
            "--output",
            str(checkpoints / CHECKPOINT_NAMES["segmenter"]),
            "--epochs",
            str(epochs),
            *extra_args,
        ]
    if task == "ranker":
        csv_path = _ensure_ranker_csv(data_dir, run_dir / "catalog")
        # The ranker is a single scikit-learn fit; --epochs is recorded in
        # metrics.json but not forwarded (the trainer has no such argument).
        return [
            sys.executable,
            str(TRAINING_DIR / "train_supplier_ranker.py"),
            "--data",
            str(csv_path),
            "--model",
            str(checkpoints / CHECKPOINT_NAMES["ranker"]),
            "--metrics",
            str(run_dir / "ranker_metrics.json"),
            "--predictions",
            str(run_dir / "ranker_predictions.csv"),
            "--report",
            str(run_dir / "ranker_report.md"),
            *extra_args,
        ]
    raise ValueError(f"Unknown task {task!r}")  # pragma: no cover - argparse guards


def _final_metric(task: str, run_dir: Path, log_text: str) -> tuple[str, float]:
    """Extract (metric name, value) from trainer output without importing torch."""
    if task == "room":
        matches = re.findall(r"validation_accuracy=([0-9.]+)", log_text)
        if matches:
            return "val_accuracy", float(matches[-1])
        return "val_accuracy", 0.0
    if task == "segmenter":
        matches = re.findall(r"loss=([0-9.]+)", log_text)
        if matches:
            return "train_loss", float(matches[-1])
        return "train_loss", 0.0
    ranker_metrics = run_dir / "ranker_metrics.json"
    if ranker_metrics.is_file():
        payload = json.loads(ranker_metrics.read_text(encoding="utf-8"))
        return (
            "loo_rank_correlation",
            float(payload["leave_one_out"]["rank_correlation"]),
        )
    return "loo_rank_correlation", 0.0


def _checkpoint_files(run_dir: Path) -> list[str]:
    checkpoints = run_dir / "checkpoints"
    if not checkpoints.is_dir():
        return []
    return sorted(path.name for path in checkpoints.iterdir() if path.is_file())


def run_task(
    task: str,
    epochs: int,
    run_id: str,
    run_dir: Path,
    data_dir: Path | None,
    extra_args: list[str],
) -> int:
    """Run one training task end-to-end; return the process exit code."""
    run_dir.mkdir(parents=True, exist_ok=True)
    extra: dict = {}
    status = "failed"
    metric_name, metric_value = "val_accuracy", 0.0
    try:
        resolved = common_io.resolve_dataset(task) if data_dir is None else Path(data_dir)
        command = build_trainer_command(task, resolved, run_dir, epochs, extra_args)
        print(f"==> run {run_id}: {' '.join(command)}", flush=True)
        log_path = run_dir / "logs.txt"
        with log_path.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if process.returncode == 0:
            status = "succeeded"
            metric_name, metric_value = _final_metric(task, run_dir, log_text)
        else:
            extra["trainer_exit_code"] = process.returncode
            extra["log_tail"] = log_text.strip().splitlines()[-20:]
            print(f"ERROR: trainer exited with code {process.returncode}", flush=True)
    except Exception as exc:  # failure must still produce metrics.json
        extra["error"] = f"{exc.__class__.__name__}: {exc}"
        print(f"ERROR: {extra['error']}", flush=True)

    metrics = {
        "task": task,
        "run_id": run_id,
        "status": status,
        "epochs": epochs,
        "final_val_metric_name": metric_name,
        "final_val_metric_value": metric_value,
        "checkpoint_files": _checkpoint_files(run_dir),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "extra": extra,
    }
    metrics_path = common_io.write_metrics(metrics, run_dir)
    print(f"==> wrote {metrics_path} (status={status})", flush=True)

    bucket = os.getenv("GCS_BUCKET")
    uploaded = common_io.upload_run_artifacts(run_dir, bucket, run_id)
    if uploaded:
        print(f"==> uploaded {len(uploaded)} artifacts to gs://{bucket}/runs/{run_id}/", flush=True)
    return 0 if status == "succeeded" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="training.cloud_entry",
        description=(
            "Cloud/local training entrypoint. Unknown arguments are forwarded "
            "verbatim to the underlying trainer (e.g. --no-pretrained, "
            "--min-images, --img-size for --task room)."
        ),
    )
    parser.add_argument("--task", required=True, choices=["room", "segmenter", "ranker"])
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (default per task: room=15, segmenter=25, ranker=1)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run id (default: $RUN_ID, else UTC timestamp + task)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run output directory (default: runs/<run-id>)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Local dataset directory (default: resolve from GCS_BUCKET or data/staging)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args, extra_args = build_parser().parse_known_args(argv)
    epochs = args.epochs if args.epochs is not None else DEFAULT_EPOCHS[args.task]
    run_id = args.run_id or os.getenv("RUN_ID") or default_run_id(args.task)
    run_dir = args.run_dir or Path("runs") / run_id
    return run_task(args.task, epochs, run_id, run_dir, args.data_dir, extra_args)


if __name__ == "__main__":
    raise SystemExit(main())
