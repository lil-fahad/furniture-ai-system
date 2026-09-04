from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from train_style_classifier import (
    build_model,
    build_transforms,
    evaluate,
    load_quality_metadata,
    resolve_device,
    resolve_precision,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="prepared dataset root")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-dataset-mismatch", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {args.checkpoint}")
    device = resolve_device(args.device)
    precision_name, autocast_dtype = resolve_precision(args.precision, device)
    quality = load_quality_metadata(args.data, allow_unversioned=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint payload is not a dictionary")
    classes = checkpoint.get("classes")
    architecture = checkpoint.get("architecture")
    image_size = checkpoint.get("image_size", 224)
    if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
        raise ValueError("checkpoint has no valid class list")
    if not isinstance(architecture, str):
        raise ValueError("checkpoint has no valid architecture")
    if not isinstance(image_size, int) or image_size < 32:
        raise ValueError("checkpoint has no valid image_size")

    expected_fingerprint = checkpoint.get("dataset_fingerprint")
    actual_fingerprint = quality["dataset_fingerprint"]
    if (
        expected_fingerprint is not None
        and expected_fingerprint != actual_fingerprint
        and not args.allow_dataset_mismatch
    ):
        raise ValueError(
            "dataset fingerprint does not match checkpoint; pass --allow-dataset-mismatch "
            "only for an intentional external evaluation"
        )

    _, eval_transform = build_transforms(image_size)
    dataset = datasets.ImageFolder(args.data / args.split, transform=eval_transform)
    if dataset.classes != classes:
        raise ValueError("dataset class order does not match checkpoint classes")
    loader_options: dict[str, object] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = 2
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, **loader_options
    )
    model = build_model(architecture, len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    metrics = evaluate(model, loader, device, autocast_dtype, classes)
    report = {
        "version": 1,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture": architecture,
        "precision": precision_name,
        "dataset_fingerprint": actual_fingerprint,
        "checkpoint_dataset_fingerprint": expected_fingerprint,
        "dataset_match": expected_fingerprint == actual_fingerprint,
        "metrics": metrics,
    }
    output = args.output or args.checkpoint.with_suffix(
        args.checkpoint.suffix + f".{args.split}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
