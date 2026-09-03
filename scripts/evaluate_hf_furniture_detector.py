from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Hugging Face furniture detector on an explicit local benchmark."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("benchmark must be a JSON array")
    for row in payload:
        if not isinstance(row, dict) or "image" not in row:
            raise ValueError("every benchmark row must contain an image path")
    return payload


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    if not args.benchmark.is_file():
        raise FileNotFoundError(args.benchmark)

    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForObjectDetection.from_pretrained(args.model_dir, local_files_only=True)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        model = model.to("cuda")
    model.eval()

    rows = load_manifest(args.benchmark)
    latencies: list[float] = []
    predictions: list[dict[str, object]] = []

    for row in rows:
        image_path = Path(str(row["image"]))
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        if args.device == "cuda":
            inputs = {key: value.to("cuda") for key, value in inputs.items()}

        start = time.perf_counter()
        outputs = model(**inputs)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)
        result = processor.post_process_object_detection(
            outputs,
            threshold=args.threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        predictions.append(
            {
                "image": str(image_path),
                "labels": [model.config.id2label[int(label)] for label in result["labels"]],
                "scores": [float(score) for score in result["scores"]],
                "boxes": [[float(value) for value in box] for box in result["boxes"]],
            }
        )

    artifacts = [
        {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(args.model_dir.glob("*.safetensors"))
    ]
    summary = {
        "model_dir": str(args.model_dir),
        "benchmark": str(args.benchmark),
        "images": len(rows),
        "threshold": args.threshold,
        "device": args.device,
        "latency_ms_p50": statistics.median(latencies) if latencies else None,
        "latency_ms_p95": (
            statistics.quantiles(latencies, n=20, method="inclusive")[18]
            if len(latencies) >= 2
            else (latencies[0] if latencies else None)
        ),
        "label_mapping": {str(key): value for key, value in model.config.id2label.items()},
        "artifacts": artifacts,
        "predictions": predictions,
        "note": (
            "Detection-quality metrics require ground-truth boxes/labels and are intentionally "
            "not fabricated by this smoke evaluator."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
