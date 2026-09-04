from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from PIL import Image

from furniture_ai.professional_vision import ProfessionalVisionService

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark FurnitureAI professional vision on the target NVIDIA host."
    )
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--include-depth", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    image_paths = sorted(
        path
        for path in args.images.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not image_paths:
        raise ValueError("no supported benchmark images found")

    service = ProfessionalVisionService(
        args.models_root,
        device=args.device,
        precision=args.precision,
        enable_torch_compile=args.torch_compile,
    )

    warmup_image = Image.open(image_paths[0]).convert("RGB")
    for _ in range(args.warmup):
        service.analyze(warmup_image, include_depth=args.include_depth)

    latencies: list[float] = []
    object_counts: list[int] = []
    for image_path in image_paths:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        start = time.perf_counter()
        result = service.analyze(image, include_depth=args.include_depth)
        latencies.append((time.perf_counter() - start) * 1000)
        object_counts.append(len(result.objects))

    payload = {
        "runtime": service.runtime_info,
        "images": len(image_paths),
        "include_depth": args.include_depth,
        "warmup_iterations": args.warmup,
        "latency_ms": {
            "p50": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "min": min(latencies),
            "max": max(latencies),
            "mean": statistics.fmean(latencies),
        },
        "detections": {
            "total": sum(object_counts),
            "mean_per_image": statistics.fmean(object_counts),
        },
        "note": "Performance evidence only; this file does not measure detection accuracy.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
