from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from furniture_ai.evaluation.detection import (
    DetectionRecord,
    GroundTruthRecord,
    NormalizedBox,
    evaluate_detections,
)


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError("model SHA-256 must contain exactly 64 hexadecimal characters")
    return normalized


def _box(payload: dict[str, object]) -> NormalizedBox:
    return NormalizedBox(
        x_min=float(payload["x_min"]),
        y_min=float(payload["y_min"]),
        x_max=float(payload["x_max"]),
        y_max=float(payload["y_max"]),
    )


def load_ground_truth(path: Path) -> list[GroundTruthRecord]:
    records: list[GroundTruthRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(
                    GroundTruthRecord(
                        image_id=str(payload["image_id"]),
                        label=str(payload["label"]),
                        box=_box(payload["box"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid ground-truth JSONL at line {line_number}") from exc
    return records


def load_predictions(path: Path) -> list[DetectionRecord]:
    records: list[DetectionRecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                records.append(
                    DetectionRecord(
                        image_id=str(payload["image_id"]),
                        label=str(payload["label"]),
                        score=float(payload["score"]),
                        box=_box(payload["box"]),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid prediction JSONL at line {line_number}") from exc
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate pinned model detections against a pinned FurnitureAI benchmark."
    )
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    if not args.ground_truth.is_file():
        raise FileNotFoundError(args.ground_truth)
    if not args.predictions.is_file():
        raise FileNotFoundError(args.predictions)
    model_sha256 = _validate_sha256(args.model_sha256)

    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)
    report = evaluate_detections(
        ground_truth,
        predictions,
        iou_threshold=args.iou_threshold,
    )
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metric": "furnitureai-detection-ap-v1",
        "model": {
            "id": args.model_id,
            "revision": args.model_revision,
            "sha256": model_sha256,
        },
        "inputs": {
            "ground_truth": str(args.ground_truth),
            "ground_truth_sha256": _sha256(args.ground_truth),
            "predictions": str(args.predictions),
            "predictions_sha256": _sha256(args.predictions),
        },
        "report": asdict(report),
        "limitations": [
            "This metric is FurnitureAI detection AP v1, not the full Open Images challenge metric.",
            "Dataset-specific group-of/depiction policy is fixed when the benchmark manifest is built.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
