from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from furniture_ai.nvidia_acceleration import (
    inference_context,
    prepare_model,
    resolve_nvidia_runtime,
)

SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp")
MODEL_TO_CANONICAL = {"Chair": "Chair", "Sofa": "Sofa", "Table": "Table"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("model SHA-256 must contain exactly 64 hexadecimal characters")
    return normalized


def normalized_box(box: list[float], width: int, height: int) -> dict[str, float]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x_min, y_min, x_max, y_max = box
    normalized = {
        "x_min": min(max(x_min / width, 0.0), 1.0),
        "y_min": min(max(y_min / height, 0.0), 1.0),
        "x_max": min(max(x_max / width, 0.0), 1.0),
        "y_max": min(max(y_max / height, 0.0), 1.0),
    }
    if normalized["x_max"] < normalized["x_min"] or normalized["y_max"] < normalized["y_min"]:
        raise ValueError("detector returned an inverted bounding box")
    return normalized


def benchmark_image_ids(path: Path) -> list[str]:
    image_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                image_ids.add(str(json.loads(line)["image_id"]))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid benchmark JSONL at line {line_number}") from exc
    if not image_ids:
        raise ValueError("benchmark contains no image IDs")
    return sorted(image_ids)


def find_image(images_dir: Path, image_id: str) -> Path:
    matches = [images_dir / f"{image_id}{suffix}" for suffix in SUPPORTED_SUFFIXES]
    existing = [path for path in matches if path.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(
            f"expected exactly one local image for {image_id!r}; found {len(existing)}"
        )
    return existing[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FurnitureAI detection JSONL from a pinned HF model and real images."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--precision",
        choices=("auto", "fp32", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--torch-compile", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if not args.model_dir.is_dir():
        raise FileNotFoundError(args.model_dir)
    if not args.ground_truth.is_file():
        raise FileNotFoundError(args.ground_truth)
    if not args.images_dir.is_dir():
        raise FileNotFoundError(args.images_dir)
    model_sha256 = validate_sha256(args.model_sha256)

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    runtime = resolve_nvidia_runtime(
        args.device,
        precision=args.precision,
        enable_torch_compile=args.torch_compile,
    )
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForObjectDetection.from_pretrained(args.model_dir, local_files_only=True)
    labels = dict(model.config.id2label)
    model = prepare_model(model, runtime)

    image_ids = benchmark_image_ids(args.ground_truth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ignored_labels: Counter[str] = Counter()
    written = 0
    with args.output.open("w", encoding="utf-8") as output:
        for image_id in image_ids:
            image_path = find_image(args.images_dir, image_id)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            inputs = {
                key: value.to(runtime.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            with inference_context(runtime):
                outputs = model(**inputs)
            result = processor.post_process_object_detection(
                outputs,
                threshold=args.threshold,
                target_sizes=[(image.height, image.width)],
            )[0]
            for score, label, box in zip(
                result["scores"], result["labels"], result["boxes"], strict=True
            ):
                raw_label = str(labels.get(int(label), int(label)))
                canonical = MODEL_TO_CANONICAL.get(raw_label)
                if canonical is None:
                    ignored_labels[raw_label] += 1
                    continue
                payload = {
                    "image_id": image_id,
                    "label": canonical,
                    "score": float(score.detach().cpu().item()),
                    "box": normalized_box(
                        [float(value) for value in box.detach().cpu().tolist()],
                        image.width,
                        image.height,
                    ),
                }
                output.write(json.dumps(payload, sort_keys=True) + "\n")
                written += 1
            if runtime.device.startswith("cuda"):
                torch.cuda.synchronize()

    metadata_path = args.metadata or args.output.with_suffix(
        args.output.suffix + ".metadata.json"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "model": {
                    "id": args.model_id,
                    "revision": args.model_revision,
                    "sha256": model_sha256,
                },
                "ground_truth_sha256": sha256_file(args.ground_truth),
                "predictions_sha256": sha256_file(args.output),
                "images": len(image_ids),
                "predictions": written,
                "threshold": args.threshold,
                "runtime": runtime.as_public_dict(),
                "accepted_label_mapping": MODEL_TO_CANONICAL,
                "ignored_model_labels": dict(sorted(ignored_labels.items())),
                "fail_closed_on_missing_images": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
