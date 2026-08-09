from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no NVIDIA GPU is visible to PyTorch")
    return torch.device(requested)


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except (OSError, RuntimeError) as exc:
        if not pretrained:
            raise
        print(
            "WARNING: could not download pretrained weights "
            f"({exc.__class__.__name__}: {exc}). Falling back to random initialization. "
            "Pass --no-pretrained to skip the download attempt.",
            flush=True,
        )
        return timm.create_model(model_name, pretrained=False, num_classes=num_classes)


def save_checkpoint(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def stratified_split_indices(
    targets: list[int], validation_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)

    too_small = sorted(label for label, indices in by_class.items() if len(indices) < 2)
    if too_small:
        raise ValueError(
            "Every class needs at least two images for a train/validation split; "
            f"class ids with fewer than two: {too_small}"
        )

    generator = torch.Generator().manual_seed(seed)
    training: list[int] = []
    validation: list[int] = []
    for label in sorted(by_class):
        indices = by_class[label]
        permutation = torch.randperm(len(indices), generator=generator).tolist()
        shuffled = [indices[position] for position in permutation]
        validation_count = max(1, round(len(shuffled) * validation_fraction))
        validation_count = min(validation_count, len(shuffled) - 1)
        validation.extend(shuffled[:validation_count])
        training.extend(shuffled[validation_count:])
    return training, validation


def balanced_limit_indices(targets: list[int], limit: int, seed: int) -> list[int]:
    if limit >= len(targets):
        return list(range(len(targets)))
    if limit < len(set(targets)) * 2:
        raise ValueError("--limit must leave at least two images per class")

    by_class: dict[int, list[int]] = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[int(target)].append(index)
    rng = random.Random(seed)
    for indices in by_class.values():
        rng.shuffle(indices)

    selected: list[int] = []
    labels = sorted(by_class)
    while len(selected) < limit:
        made_progress = False
        for label in labels:
            if by_class[label] and len(selected) < limit:
                selected.append(by_class[label].pop())
                made_progress = True
        if not made_progress:
            break
    return selected


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    training = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    validation = transforms.Compose(
        [
            transforms.Resize(round(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return training, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="ImageFolder root: one directory per class")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/room_classifier/efficientnet_b0.pth"),
    )
    parser.add_argument("--model-name", default="tf_efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Square crop size applied to training and validation images",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Balanced cap on total images used (for smoke runs)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=20,
        help="Minimum dataset size; lower it for tiny synthetic smoke datasets",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
        help="Stop after this many non-improving epochs; 0 disables early stopping",
    )

    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        default=None,
        help="Download ImageNet weights (default; FURNITURE_PRETRAINED=1)",
    )
    pretrained_group.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Skip the download and train from random initialization (offline-friendly)",
    )
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument(
        "--no-tf32",
        dest="tf32",
        action="store_false",
        default=True,
        help="Disable TF32 acceleration on supported NVIDIA GPUs",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be greater than 0 and less than 0.5")
    return args


def main() -> None:
    args = parse_args()
    pretrained = args.pretrained
    if pretrained is None:
        pretrained = env_flag("FURNITURE_PRETRAINED", True)
    seed_everything(args.seed)

    device = resolve_device(args.device)
    amp_enabled = device.type == "cuda" if args.amp is None else args.amp
    if amp_enabled and device.type != "cuda":
        raise ValueError("--amp currently requires an NVIDIA CUDA device")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.set_float32_matmul_precision("high")

    train_transform, validation_transform = build_transforms(args.img_size)
    index_dataset = datasets.ImageFolder(args.data)
    if len(index_dataset) < args.min_images:
        raise ValueError(
            f"At least {args.min_images} labeled images are required "
            f"(found {len(index_dataset)}); pass --min-images to lower the threshold for smoke runs"
        )
    selected = list(range(len(index_dataset)))
    if args.limit is not None:
        selected = balanced_limit_indices(index_dataset.targets, args.limit, args.seed)
    selected_targets = [index_dataset.targets[index] for index in selected]
    train_relative, validation_relative = stratified_split_indices(
        selected_targets, args.validation_fraction, args.seed
    )
    train_indices = [selected[index] for index in train_relative]
    validation_indices = [selected[index] for index in validation_relative]

    train_dataset = datasets.ImageFolder(args.data, transform=train_transform)
    validation_dataset = datasets.ImageFolder(args.data, transform=validation_transform)
    classes = index_dataset.classes
    train_set = Subset(train_dataset, train_indices)
    validation_set = Subset(validation_dataset, validation_indices)

    loader_options: dict[str, object] = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )

    model = build_model(args.model_name, num_classes=len(classes), pretrained=pretrained).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    best_accuracy = 0.0
    epochs_without_improvement = 0
    checkpoint_written = False
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"device={device} amp={amp_enabled} tf32={args.tf32 and device.type == 'cuda'} "
        f"classes={len(classes)} train={len(train_set)} validation={len(validation_set)}",
        flush=True,
    )
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        training_loss = 0.0
        for step, (images, labels) in enumerate(
            tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}"), start=1
        ):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                loss = criterion(model(images), labels) / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            training_loss += float(loss.detach()) * args.gradient_accumulation_steps
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        model.eval()
        validation_loss = 0.0
        correct = total = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if device.type == "cuda":
                    images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=amp_enabled
                ):
                    logits = model(images)
                    validation_loss += float(criterion(logits, labels))
                correct += int((logits.argmax(dim=1) == labels).sum())
                total += len(labels)
        scheduler.step()
        accuracy = correct / max(total, 1)
        print(
            f"epoch={epoch + 1} train_loss={training_loss / max(len(train_loader), 1):.4f} "
            f"validation_loss={validation_loss / max(len(validation_loader), 1):.4f} "
            f"validation_accuracy={accuracy:.4f}",
            flush=True,
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            epochs_without_improvement = 0
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "architecture": args.model_name,
                    "validation_accuracy": accuracy,
                    "epoch": epoch + 1,
                    "seed": args.seed,
                    "image_size": args.img_size,
                    "train_images": len(train_set),
                    "validation_images": len(validation_set),
                    "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                },
                args.output,
            )
            checkpoint_written = True
        else:
            epochs_without_improvement += 1
            if (
                args.early_stopping_patience > 0
                and epochs_without_improvement >= args.early_stopping_patience
            ):
                print(f"early_stopping epoch={epoch + 1}", flush=True)
                break
    if not checkpoint_written:
        save_checkpoint(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "architecture": args.model_name,
                "validation_accuracy": best_accuracy,
                "epoch": 0,
                "seed": args.seed,
                "image_size": args.img_size,
                "train_images": len(train_set),
                "validation_images": len(validation_set),
                "normalization": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
            },
            args.output,
        )


if __name__ == "__main__":
    main()
