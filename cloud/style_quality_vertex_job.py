#!/usr/bin/env python3
"""Managed Open Images style job with dataset-quality and evaluation gates.

This wrapper deliberately reuses the licensed Open Images V7 collection and
SigLIP pseudo-labeling implementation in ``openimages_vertex_job.py``. It then
prepares leakage-resistant train/validation/test splits, trains the dedicated
style classifier, evaluates the untouched test split, and publishes all
identity/quality artifacts beside the checkpoint in GCS.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import openimages_vertex_job as base

QUALITY_PIPELINE_VERSION = 1
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10


def quality_plan(args: Any) -> dict[str, Any]:
    plan = base.job_plan(args)
    plan.update(
        {
            "quality_pipeline_version": QUALITY_PIPELINE_VERSION,
            "exact_duplicate_filter": True,
            "perceptual_duplicate_filter": True,
            "review_required_excluded_from_splits": True,
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": TEST_FRACTION,
            "selection_metric": "validation_macro_f1",
            "evaluation_kind": "pseudo_label_holdout",
            "production_test_human_review_required": True,
        }
    )
    return plan


def prepare_quality_dataset(
    args: Any,
    data_root: Path,
    manifest_path: Path,
    prepared_root: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "scripts/prepare_style_dataset.py",
        str(data_root),
        "--output",
        str(prepared_root),
        "--source-manifest",
        str(manifest_path),
        "--seed",
        str(args.seed),
        "--validation-fraction",
        str(VALIDATION_FRACTION),
        "--test-fraction",
        str(TEST_FRACTION),
        "--mode",
        "hardlink",
        "--allow-label-conflicts",
    ]
    print("quality command=" + " ".join(command), flush=True)
    subprocess.run(command, check=True)  # noqa: S603 - fixed executable and arguments.
    summary_path = prepared_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict):
        raise ValueError("quality summary is not a JSON object")
    accepted = int(summary.get("accepted_images", 0))
    if accepted < 100:
        raise RuntimeError(
            f"Only {accepted} usable images remain after dataset quality filtering; "
            "at least 100 are required for managed training"
        )
    return summary


def train_quality_classifier(args: Any, prepared_root: Path, output: Path) -> Path:
    command = [
        sys.executable,
        "training/train_style_classifier.py",
        str(prepared_root),
        "--output",
        str(output),
        "--device",
        "cuda",
        "--precision",
        "auto",
        "--class-balance",
        "sampler",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.train_batch_size),
        "--num-workers",
        str(args.train_workers),
        "--early-stopping-patience",
        "5",
    ]
    print("training command=" + " ".join(command), flush=True)
    subprocess.run(command, check=True)  # noqa: S603 - fixed executable and arguments.
    metrics_path = output.with_suffix(output.suffix + ".metrics.json")
    if not output.is_file() or not metrics_path.is_file():
        raise RuntimeError("style training completed without checkpoint or metrics artifact")
    return metrics_path


def upload_quality_artifacts(
    store: base.GCSStore,
    prepared_root: Path,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    mappings = {
        "quality_manifest_uri": ("quality/manifest.jsonl", prepared_root / "manifest.jsonl"),
        "quality_summary_uri": ("quality/summary.json", prepared_root / "summary.json"),
        "quality_conflicts_uri": ("quality/conflicts.json", prepared_root / "conflicts.json"),
        "quality_rejected_uri": ("quality/rejected.jsonl", prepared_root / "rejected.jsonl"),
    }
    for key, (remote, local) in mappings.items():
        if local.is_file():
            content_type = (
                "application/x-ndjson" if local.suffix == ".jsonl" else "application/json"
            )
            artifacts[key] = store.upload_file(remote, local, content_type)
    return artifacts


def run(args: Any) -> int:
    plan = quality_plan(args)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Vertex job has no NVIDIA GPU visible to PyTorch")
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)

    work_dir = args.work_dir.resolve()
    metadata_dir = work_dir / "metadata"
    data_root = work_dir / "data" / "styles"
    prepared_root = work_dir / "data" / "styles_prepared"
    model_path = work_dir / "models" / "style_classifier.pth"
    manifest_path = work_dir / "manifest.jsonl"
    store = base.GCSStore(args.bucket, f"runs/{args.run_id}")
    status = {**plan, "status": "running", "started_at": base.utc_now()}
    store.upload_bytes("status.json", json.dumps(status, indent=2).encode(), "application/json")

    try:
        class_path = base.download_file(
            base.CLASS_DESCRIPTIONS_URL,
            metadata_dir / "class-descriptions.csv",
            max_bytes=20 * 1024 * 1024,
            timeout=args.timeout,
        )
        boxes_path = base.download_file(
            base.BOX_ANNOTATIONS_URL,
            metadata_dir / "train-boxes.csv",
            max_bytes=8 * 1024 * 1024 * 1024,
            timeout=args.timeout,
        )
        metadata_path = base.download_file(
            base.TRAIN_METADATA_URL,
            metadata_dir / "train-images.csv",
            max_bytes=3 * 1024 * 1024 * 1024,
            timeout=args.timeout,
        )
        class_ids = base.load_furniture_class_ids(class_path)
        candidate_ids = base.candidate_ids_from_rows(
            base.csv_rows(boxes_path),
            class_ids,
            min_distinct_classes=args.min_distinct_classes,
            min_boxes=args.min_boxes,
        )
        pool_size = math.ceil(args.max_images * args.candidate_multiplier)
        selected = base.reservoir_select_metadata(
            base.csv_rows(metadata_path),
            candidate_ids,
            limit=pool_size,
            seed=args.seed,
        )
        print(
            f"candidate_ids={len(candidate_ids)} licensed_candidate_pool={len(selected)}",
            flush=True,
        )
        if len(selected) < args.max_images:
            raise RuntimeError(
                f"Only {len(selected)} licensed candidates are available, below "
                f"--max-images={args.max_images}"
            )

        raw_manifest = base.build_dataset(args, store, selected, data_root, manifest_path)
        quality_summary = prepare_quality_dataset(
            args, data_root, manifest_path, prepared_root
        )
        metrics_path = train_quality_classifier(args, prepared_root, model_path)
        automated_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        checkpoint_uri = store.upload_file(
            "models/style_classifier.pth", model_path, "application/octet-stream"
        )
        metrics_uri = store.upload_file(
            "models/style_classifier.pth.metrics.json", metrics_path, "application/json"
        )
        raw_manifest_uri = store.upload_file(
            "manifest.jsonl", manifest_path, "application/x-ndjson"
        )
        quality_artifacts = upload_quality_artifacts(store, prepared_root)
        status.update(
            {
                "status": "completed",
                "completed_at": base.utc_now(),
                "accepted_raw_images": len(raw_manifest),
                "raw_style_counts": dict(
                    sorted(Counter(str(item["style"]) for item in raw_manifest).items())
                ),
                "review_required_raw_images": sum(
                    bool(item.get("review_required")) for item in raw_manifest
                ),
                "quality": quality_summary,
                "automated_evaluation": automated_metrics,
                "evaluation_disclaimer": (
                    "Validation/test labels are SigLIP pseudo-labels. These metrics are for "
                    "pipeline regression and model selection, not production accuracy claims. "
                    "A separate human-reviewed test set is required for release claims."
                ),
                "checkpoint_uri": checkpoint_uri,
                "metrics_uri": metrics_uri,
                "manifest_uri": raw_manifest_uri,
                **quality_artifacts,
                "metadata_sha256": {
                    "class_descriptions": base.sha256_file(class_path),
                    "box_annotations": base.sha256_file(boxes_path),
                    "train_metadata": base.sha256_file(metadata_path),
                },
            }
        )
        store.upload_bytes(
            "status.json",
            json.dumps(status, indent=2, sort_keys=True).encode(),
            "application/json",
        )
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "failed_at": base.utc_now(),
                "error_type": exc.__class__.__name__,
                "error": base.clean_text(str(exc), 2_000),
            }
        )
        try:
            status.update(upload_quality_artifacts(store, prepared_root))
            store.upload_bytes(
                "status.json",
                json.dumps(status, indent=2, sort_keys=True).encode(),
                "application/json",
            )
            if manifest_path.exists():
                store.upload_file("manifest.jsonl", manifest_path, "application/x-ndjson")
        except Exception as status_error:  # pragma: no cover - last-resort reporting only.
            print(f"could not upload failure status: {status_error}", file=sys.stderr)
        raise


def main() -> None:
    try:
        raise SystemExit(run(base.parse_args()))
    except KeyboardInterrupt:
        print("Stopped by operator.", file=sys.stderr)
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
