from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm import tqdm

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no NVIDIA GPU is visible to PyTorch")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(requested)


def _split_dirs(root: Path, split: str | None) -> tuple[Path, Path]:
    if split:
        split_images = root / "images" / split
        split_masks = root / "masks" / split
        if split_images.is_dir() and split_masks.is_dir():
            return split_images, split_masks
    return root / "images", root / "masks"


def has_named_split(root: Path, split: str) -> bool:
    return (root / "images" / split).is_dir() and (root / "masks" / split).is_dir()


def deterministic_split_indices(
    count: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    if count < 2:
        raise ValueError("At least two training pairs are required for a validation split")
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be greater than 0 and less than 0.5")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(count, generator=generator).tolist()
    validation_count = max(1, round(count * validation_fraction))
    validation_count = min(validation_count, count - 1)
    return indices[validation_count:], indices[:validation_count]


class FloorPlanDataset(Dataset):
    def __init__(
        self,
        root: Path,
        size: int = 512,
        classes: int = 5,
        mask_remap: str = "none",
        limit: int | None = None,
        *,
        split: str | None = None,
        remap_table: torch.Tensor | None = None,
        augment: bool = False,
    ) -> None:
        self.images_dir, self.masks = _split_dirs(root, split)
        self.images = sorted(
            path
            for path in self.images_dir.glob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.size = size
        self.classes = classes
        self.augment = augment
        if not self.images:
            raise ValueError(f"No training images found under {self.images_dir}")
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive")
            self.images = self.images[:limit]
        for image_path in self.images:
            mask_path = self.masks / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise ValueError(f"Missing mask for image {image_path}: expected {mask_path}")

        self.remap_values: list[int] | None = None
        self.remap_table: torch.Tensor | None = None
        if remap_table is not None:
            self.remap_table = remap_table.clone()
        elif mask_remap == "auto":
            self.remap_table, self.remap_values = self._build_remap_table()
        elif mask_remap != "none":
            raise ValueError(f"Unknown --mask-remap mode: {mask_remap!r} (use 'none' or 'auto')")

    def _mask_path(self, image_path: Path) -> Path:
        return self.masks / f"{image_path.stem}.png"

    def _build_remap_table(self) -> tuple[torch.Tensor, list[int]]:
        unique_values: set[int] = set()
        for image_path in self.images:
            mask = read_image(str(self._mask_path(image_path)))[:1].long()
            unique_values.update(int(value) for value in mask.unique().tolist())
        sorted_values = sorted(unique_values)
        if not sorted_values:
            raise ValueError("No mask values were discovered")
        if len(sorted_values) > self.classes:
            raise ValueError(
                f"Masks contain {len(sorted_values)} distinct values but --classes={self.classes}; "
                "raise --classes or fix the masks"
            )
        table = torch.full((max(sorted_values) + 1,), -1, dtype=torch.long)
        for new_id, old_value in enumerate(sorted_values):
            table[old_value] = new_id
        print(f"mask remap (auto): {sorted_values} -> {list(range(len(sorted_values)))}")
        return table, sorted_values

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        mask_path = self._mask_path(image_path)
        image = read_image(str(image_path)).float() / 255
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] >= 3:
            image = image[:3]
        else:
            raise ValueError(f"Unsupported channel count in {image_path}: {image.shape[0]}")
        image = resize(
            image,
            [self.size, self.size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = read_image(str(mask_path))[:1].long()
        if self.remap_table is not None:
            mask_max = int(mask.max())
            if mask_max >= len(self.remap_table):
                raise ValueError(
                    f"Mask {mask_path} contains value {mask_max} which was not seen "
                    "when the remap table was built"
                )
            mask = self.remap_table[mask]
            if int(mask.min()) < 0:
                raise ValueError(f"Mask {mask_path} contains an unmapped class value")
        mask = resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
        mask = mask.squeeze(0)
        mask_min, mask_max = int(mask.min()), int(mask.max())
        if mask_min < 0 or mask_max >= self.classes:
            raise ValueError(
                f"Mask {mask_path} contains class ids in [{mask_min}, {mask_max}] but the model "
                f"expects [0, {self.classes - 1}] (--classes={self.classes}). Pass "
                "--mask-remap auto to remap mask values to contiguous class ids."
            )
        if self.augment:
            if bool(torch.rand(()) < 0.5):
                image = torch.flip(image, dims=(2,))
                mask = torch.flip(mask, dims=(1,))
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                image = torch.rot90(image, rotations, dims=(1, 2))
                mask = torch.rot90(mask, rotations, dims=(0, 1))
        return image, mask


class Block(nn.Module):
    def __init__(self, incoming: int, outgoing: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, padding=1),
            nn.BatchNorm2d(outgoing),
            nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1),
            nn.BatchNorm2d(outgoing),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class SmallUNet(nn.Module):
    def __init__(self, classes: int = 5) -> None:
        super().__init__()
        self.enc1, self.enc2, self.bottleneck = Block(3, 32), Block(32, 64), Block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), Block(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), Block(64, 32)
        self.head = nn.Conv2d(32, classes, 1)

    def forward(self, inputs):
        first = self.enc1(inputs)
        second = self.enc2(self.pool(first))
        encoded = self.bottleneck(self.pool(second))
        decoded = self.dec2(torch.cat([self.up2(encoded), second], dim=1))
        decoded = self.dec1(torch.cat([self.up1(decoded), first], dim=1))
        return self.head(decoded)


def update_confusion_matrix(
    confusion: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    classes: int,
) -> None:
    predictions = logits.argmax(dim=1)
    valid = (targets >= 0) & (targets < classes)
    encoded = targets[valid] * classes + predictions[valid]
    counts = torch.bincount(encoded, minlength=classes * classes)
    confusion += counts.reshape(classes, classes).to(confusion.device)


def segmentation_metrics(confusion: torch.Tensor) -> dict[str, object]:
    matrix = confusion.double()
    true_positive = matrix.diag()
    target_total = matrix.sum(dim=1)
    predicted_total = matrix.sum(dim=0)
    union = target_total + predicted_total - true_positive
    denom_dice = target_total + predicted_total
    iou = torch.where(union > 0, true_positive / union, torch.nan)
    dice = torch.where(denom_dice > 0, 2 * true_positive / denom_dice, torch.nan)
    present_iou = iou[~torch.isnan(iou)]
    present_dice = dice[~torch.isnan(dice)]
    total = matrix.sum()
    return {
        "pixel_accuracy": float(true_positive.sum() / total) if total > 0 else 0.0,
        "mean_iou": float(present_iou.mean()) if len(present_iou) else 0.0,
        "mean_dice": float(present_dice.mean()) if len(present_dice) else 0.0,
        "per_class_iou": [None if math.isnan(float(value)) else float(value) for value in iou],
        "per_class_dice": [
            None if math.isnan(float(value)) else float(value) for value in dice
        ],
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    classes: int,
    amp_enabled: bool,
) -> dict[str, object]:
    model.eval()
    loss_total = 0.0
    confusion = torch.zeros((classes, classes), dtype=torch.int64, device=device)
    with torch.inference_mode():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = model(images)
                loss = criterion(logits, masks)
            loss_total += float(loss)
            update_confusion_matrix(confusion, logits, masks, classes)
    metrics = segmentation_metrics(confusion.cpu())
    metrics["loss"] = loss_total / max(len(loader), 1)
    return metrics


def save_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def save_checkpoint_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def export_torchscript(model: nn.Module, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu().eval())
    temporary = output.with_suffix(output.suffix + ".tmp")
    scripted.save(str(temporary))
    os.replace(temporary, output)


def build_loaders(args: argparse.Namespace):
    train_split = "train" if has_named_split(args.data, "train") else None
    train_dataset = FloorPlanDataset(
        args.data,
        size=args.size,
        classes=args.classes,
        mask_remap=args.mask_remap,
        limit=args.limit,
        split=train_split,
        augment=not args.no_augmentation,
    )
    remap_table = train_dataset.remap_table

    validation_dataset: Dataset
    if has_named_split(args.data, "validation"):
        validation_dataset = FloorPlanDataset(
            args.data,
            size=args.size,
            classes=args.classes,
            split="validation",
            remap_table=remap_table,
            augment=False,
        )
        training_dataset: Dataset = train_dataset
    elif has_named_split(args.data, "val"):
        validation_dataset = FloorPlanDataset(
            args.data,
            size=args.size,
            classes=args.classes,
            split="val",
            remap_table=remap_table,
            augment=False,
        )
        training_dataset = train_dataset
    else:
        train_indices, validation_indices = deterministic_split_indices(
            len(train_dataset), args.validation_fraction, args.seed
        )
        validation_base = FloorPlanDataset(
            args.data,
            size=args.size,
            classes=args.classes,
            mask_remap=args.mask_remap,
            limit=args.limit,
            split=train_split,
            remap_table=remap_table,
            augment=False,
        )
        training_dataset = Subset(train_dataset, train_indices)
        validation_dataset = Subset(validation_base, validation_indices)

    test_dataset: Dataset | None = None
    if has_named_split(args.data, "test"):
        test_dataset = FloorPlanDataset(
            args.data,
            size=args.size,
            classes=args.classes,
            split="test",
            remap_table=remap_table,
            augment=False,
        )

    loader_options: dict[str, object] = {
        "num_workers": args.num_workers,
        "pin_memory": args.device_resolved.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_options["prefetch_factor"] = 2
    train_loader = DataLoader(
        training_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options,
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **loader_options)
        if test_dataset is not None
        else None
    )
    return train_dataset, train_loader, validation_loader, test_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data",
        type=Path,
        help=(
            "Dataset root. Supports images/ + masks/ or named "
            "images/{train,validation,test} and masks/{train,validation,test} splits."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("models/floorplan_segmenter/unet.pt"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classes", type=int, default=5, help="Number of segmentation classes")
    parser.add_argument("--size", type=int, default=512, help="Square resize for images and masks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of training pairs")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--mask-remap",
        choices=["none", "auto"],
        default="none",
        help=(
            "'auto' remaps the distinct training-mask values to contiguous class ids 0..N-1 "
            "and applies the same mapping to validation/test masks."
        ),
    )
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.classes < 2:
        parser.error("--classes must be at least 2")
    if args.size < 32:
        parser.error("--size must be at least 32")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be greater than 0 and less than 0.5")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience cannot be negative")
    return args


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.device_resolved = resolve_device(args.device)
    if args.amp and args.device_resolved.type != "cuda":
        raise ValueError("--amp currently requires an NVIDIA CUDA device")
    amp_enabled = args.amp and args.device_resolved.type == "cuda"
    if args.device_resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    checkpoint_path = args.checkpoint or args.output.with_suffix(".checkpoint.pth")
    metrics_path = args.metrics or args.output.with_suffix(".metrics.json")
    train_dataset, train_loader, validation_loader, test_loader = build_loaders(args)

    model = SmallUNet(classes=args.classes).to(args.device_resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    best_miou = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, object]] = []

    if args.resume is not None:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if int(payload.get("classes", args.classes)) != args.classes:
            raise ValueError("Resume checkpoint class count does not match --classes")
        if int(payload.get("size", args.size)) != args.size:
            raise ValueError("Resume checkpoint image size does not match --size")
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload.get("epoch", 0))
        best_miou = float(payload.get("best_miou", -1.0))
        stored_best = payload.get("best_model_state_dict")
        if isinstance(stored_best, dict):
            best_state = stored_best
        print(f"resumed_from={args.resume} start_epoch={start_epoch} best_miou={best_miou:.5f}")

    epochs_without_improvement = 0
    print(
        f"device={args.device_resolved} amp={amp_enabled} train={len(train_loader.dataset)} "
        f"validation={len(validation_loader.dataset)} "
        f"test={len(test_loader.dataset) if test_loader is not None else 0} classes={args.classes}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for images, masks in tqdm(train_loader, desc=f"train {epoch + 1}/{args.epochs}"):
            images = images.to(args.device_resolved, non_blocking=True)
            masks = masks.to(args.device_resolved, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=args.device_resolved.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach())
        scheduler.step()

        validation = evaluate(
            model,
            validation_loader,
            criterion,
            args.device_resolved,
            args.classes,
            amp_enabled,
        )
        epoch_record: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": running / max(len(train_loader), 1),
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_record)
        current_miou = float(validation["mean_iou"])
        print(
            f"epoch={epoch + 1} train_loss={epoch_record['train_loss']:.5f} "
            f"val_loss={float(validation['loss']):.5f} "
            f"val_miou={current_miou:.5f} val_dice={float(validation['mean_dice']):.5f}",
            flush=True,
        )

        if current_miou > best_miou:
            best_miou = current_miou
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        save_checkpoint_atomic(
            {
                "model_state_dict": model.state_dict(),
                "best_model_state_dict": best_state,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_miou": best_miou,
                "classes": args.classes,
                "size": args.size,
                "seed": args.seed,
                "remap_values": train_dataset.remap_values,
            },
            checkpoint_path,
        )
        save_json_atomic(
            {
                "schema_version": "1.0",
                "architecture": "small-unet",
                "classes": args.classes,
                "image_size": args.size,
                "seed": args.seed,
                "device": str(args.device_resolved),
                "train_pairs": len(train_loader.dataset),
                "validation_pairs": len(validation_loader.dataset),
                "test_pairs": len(test_loader.dataset) if test_loader is not None else 0,
                "mask_remap_values": train_dataset.remap_values,
                "best_validation_miou": best_miou,
                "history": history,
            },
            metrics_path,
        )
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(f"early_stopping epoch={epoch + 1}", flush=True)
            break

    if best_state is None:
        best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    model = model.to(args.device_resolved)
    final_metrics: dict[str, object] = {
        "best_validation": evaluate(
            model,
            validation_loader,
            criterion,
            args.device_resolved,
            args.classes,
            amp_enabled,
        )
    }
    if test_loader is not None:
        final_metrics["test"] = evaluate(
            model,
            test_loader,
            criterion,
            args.device_resolved,
            args.classes,
            amp_enabled,
        )
    save_json_atomic(
        {
            "schema_version": "1.0",
            "architecture": "small-unet",
            "classes": args.classes,
            "image_size": args.size,
            "seed": args.seed,
            "device": str(args.device_resolved),
            "train_pairs": len(train_loader.dataset),
            "validation_pairs": len(validation_loader.dataset),
            "test_pairs": len(test_loader.dataset) if test_loader is not None else 0,
            "mask_remap_values": train_dataset.remap_values,
            "best_validation_miou": best_miou,
            "history": history,
            "final": final_metrics,
        },
        metrics_path,
    )
    export_torchscript(model, args.output)
    print(f"exported={args.output} metrics={metrics_path} checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
