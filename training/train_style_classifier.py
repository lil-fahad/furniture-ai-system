from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no NVIDIA GPU is visible to PyTorch")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> tuple[str, torch.dtype | None]:
    if device.type != "cuda":
        if requested not in {"auto", "fp32"}:
            raise ValueError("fp16/bf16 precision requires an NVIDIA CUDA device")
        return "fp32", None
    if requested == "auto":
        requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "fp32":
        return requested, None
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("bf16 was requested but this CUDA device does not support it")
        return requested, torch.bfloat16
    return requested, torch.float16


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except (OSError, RuntimeError) as exc:
        if not pretrained:
            raise
        print(
            "WARNING: pretrained weights unavailable; using random initialization "
            f"({exc.__class__.__name__}: {exc})",
            flush=True,
        )
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    training = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.RandomGrayscale(p=0.03),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return training, evaluation


def load_quality_metadata(root: Path, allow_unversioned: bool) -> dict[str, object]:
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.jsonl"
    if not summary_path.is_file() or not manifest_path.is_file():
        if allow_unversioned:
            return {
                "dataset_fingerprint": None,
                "manifest_sha256": None,
                "summary": {},
            }
        raise ValueError(
            "prepared dataset must contain summary.json and manifest.jsonl; "
            "run scripts/prepare_style_dataset.py first"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not isinstance(summary.get("dataset_fingerprint"), str):
        raise ValueError("summary.json does not contain a valid dataset_fingerprint")
    validate_manifest_no_split_leakage(manifest_path)
    return {
        "dataset_fingerprint": summary["dataset_fingerprint"],
        "manifest_sha256": sha256_file(manifest_path),
        "summary": summary,
    }


def validate_manifest_no_split_leakage(path: Path) -> None:
    seen: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path} line {line_number}") from exc
        if not isinstance(row, dict) or row.get("status") != "accepted":
            continue
        digest = row.get("sha256")
        split = row.get("split")
        if not isinstance(digest, str) or split not in {"train", "validation", "test"}:
            raise ValueError(f"invalid accepted manifest record on line {line_number}")
        previous = seen.get(digest)
        if previous is not None and previous != split:
            raise ValueError(
                f"dataset leakage: sha256 {digest} appears in both {previous} and {split}"
            )
        seen[digest] = split


def imagefolder_splits(
    root: Path, train_transform: transforms.Compose, eval_transform: transforms.Compose
) -> tuple[datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder]:
    paths = {name: root / name for name in ("train", "validation", "test")}
    missing = [name for name, path in paths.items() if not path.is_dir()]
    if missing:
        raise ValueError("prepared dataset is missing split directories: " + ", ".join(missing))
    train = datasets.ImageFolder(paths["train"], transform=train_transform)
    validation = datasets.ImageFolder(paths["validation"], transform=eval_transform)
    test = datasets.ImageFolder(paths["test"], transform=eval_transform)
    if train.classes != validation.classes or train.classes != test.classes:
        raise ValueError("train, validation, and test class directories must match exactly")
    if not train.samples or not validation.samples or not test.samples:
        raise ValueError("train, validation, and test must all contain images")
    return train, validation, test


def balanced_sample_weights(targets: list[int]) -> list[float]:
    counts = Counter(int(target) for target in targets)
    if not counts:
        raise ValueError("cannot balance an empty training split")
    return [1.0 / counts[int(target)] for target in targets]


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        raise ValueError("probabilities and labels have incompatible shapes")
    if len(labels) == 0:
        return 0.0
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = predictions == labels
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)
        count = int(mask.sum())
        if not count:
            continue
        accuracy = float(correctness[mask].mean())
        confidence = float(confidences[mask].mean())
        ece += (count / len(labels)) * abs(accuracy - confidence)
    return float(ece)


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, classes: list[str]
) -> dict[str, object]:
    probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
    true = labels.cpu().numpy()
    predicted = probabilities.argmax(axis=1)
    labels_range = list(range(len(classes)))
    matrix = confusion_matrix(true, predicted, labels=labels_range)
    per_class: dict[str, dict[str, float | int]] = {}
    for index, class_name in enumerate(classes):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(score),
            "support": support,
        }
    clipped = np.clip(probabilities[np.arange(len(true)), true], 1e-12, 1.0)
    nll = float(-np.log(clipped).mean()) if len(true) else 0.0
    return {
        "accuracy": float((predicted == true).mean()) if len(true) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=labels_range, average="macro")),
        "negative_log_likelihood": nll,
        "expected_calibration_error_15bin": expected_calibration_error(probabilities, true),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "classes": classes,
        "images": int(len(true)),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    classes: list[str],
) -> dict[str, object]:
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                logits = model(images)
            logits_parts.append(logits.detach().float().cpu())
            label_parts.append(labels.detach().cpu())
    if not logits_parts:
        raise ValueError("evaluation split is empty")
    return classification_metrics(torch.cat(logits_parts), torch.cat(label_parts), classes)


def save_checkpoint(payload: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="prepared dataset root with train/validation/test")
    parser.add_argument(
        "--output", type=Path, default=Path("models/style_classifier/efficientnet_b0.pth")
    )
    parser.add_argument("--model-name", default="tf_efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--class-balance", choices=("none", "sampler"), default="sampler")
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True)
    parser.add_argument("--no-tf32", dest="tf32", action="store_false", default=True)
    parser.add_argument("--allow-unversioned-data", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be between 0 and 1")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    precision_name, autocast_dtype = resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.set_float32_matmul_precision("high")

    quality = load_quality_metadata(args.data, args.allow_unversioned_data)
    train_transform, eval_transform = build_transforms(args.img_size)
    train_set, validation_set, test_set = imagefolder_splits(
        args.data, train_transform, eval_transform
    )
    classes = train_set.classes
    loader_options: dict[str, object] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = 2

    sampler = None
    if args.class_balance == "sampler":
        weights = balanced_sample_weights(train_set.targets)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=False, **loader_options
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **loader_options)

    model = build_model(args.model_name, len(classes), args.pretrained).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_dtype == torch.float16)

    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device} precision={precision_name} classes={len(classes)} "
        f"train={len(train_set)} validation={len(validation_set)} test={len(test_set)}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_loss = 0.0
        for step, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"train {epoch}/{args.epochs}"), start=1
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                loss = criterion(model(images), labels) / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            training_loss += float(loss.detach()) * args.gradient_accumulation_steps
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        validation_metrics = evaluate(
            model, validation_loader, device, autocast_dtype, classes
        )
        macro_f1 = float(validation_metrics["macro_f1"])
        print(
            f"epoch={epoch} train_loss={training_loss / max(len(train_loader), 1):.4f} "
            f"val_accuracy={float(validation_metrics['accuracy']):.4f} "
            f"val_macro_f1={macro_f1:.4f}",
            flush=True,
        )
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "architecture": args.model_name,
                    "image_size": args.img_size,
                    "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                    "seed": args.seed,
                    "precision": precision_name,
                    "class_balance": args.class_balance,
                    "train_images": len(train_set),
                    "validation_images": len(validation_set),
                    "test_images": len(test_set),
                    "training_class_counts": dict(sorted(Counter(train_set.targets).items())),
                    "dataset_fingerprint": quality["dataset_fingerprint"],
                    "dataset_manifest_sha256": quality["manifest_sha256"],
                    "selection_metric": "validation_macro_f1",
                    "validation_metrics": validation_metrics,
                    "epoch": epoch,
                },
                args.output,
            )
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(f"early_stopping epoch={epoch}", flush=True)
                break

    if best_epoch == 0:
        raise RuntimeError("training did not produce a checkpoint")
    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, autocast_dtype, classes)
    checkpoint["test_metrics"] = test_metrics
    checkpoint["best_epoch"] = best_epoch
    save_checkpoint(checkpoint, args.output)

    report = {
        "version": 1,
        "checkpoint": str(args.output),
        "checkpoint_sha256": sha256_file(args.output),
        "architecture": args.model_name,
        "classes": classes,
        "image_size": args.img_size,
        "dataset_fingerprint": quality["dataset_fingerprint"],
        "dataset_manifest_sha256": quality["manifest_sha256"],
        "selection_metric": "validation_macro_f1",
        "best_epoch": best_epoch,
        "validation": checkpoint["validation_metrics"],
        "test": test_metrics,
    }
    metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    metrics_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
