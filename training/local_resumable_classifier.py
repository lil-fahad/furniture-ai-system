from __future__ import annotations

import argparse
import json
import os
import random
import signal
from collections import Counter
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets

from training import train_room_classifier as room
from training import train_style_classifier as style

PAUSED_EXIT_CODE = 75
_STOP_REQUESTED = False


def _request_stop(_signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def install_signal_handlers() -> None:
    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        value = getattr(signal, name, None)
        if value is not None:
            signal.signal(value, _request_stop)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def build_model(model_name: str, classes: int, pretrained: bool) -> nn.Module:
    try:
        return timm.create_model(model_name, pretrained=pretrained, num_classes=classes)
    except (OSError, RuntimeError) as exc:
        if not pretrained:
            raise
        print(
            "WARNING: pretrained weights unavailable; using random initialization "
            f"({exc.__class__.__name__}: {exc})",
            flush=True,
        )
        return timm.create_model(model_name, pretrained=False, num_classes=classes)


def loader_options(num_workers: int, device: torch.device) -> dict[str, object]:
    options: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": False,
    }
    if num_workers > 0:
        options["prefetch_factor"] = 2
    return options


def make_train_loader(
    dataset: torch.utils.data.Dataset,
    targets: list[int],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    class_balance: str,
    seed: int,
    epoch: int,
) -> DataLoader:
    epoch_seed = seed + epoch * 100_003
    generator = torch.Generator().manual_seed(epoch_seed)
    sampler = None
    if class_balance == "sampler":
        weights = style.balanced_sample_weights(targets)
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        **loader_options(num_workers, device),
    )


def prepare_style_data(
    args: argparse.Namespace,
) -> tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset | None,
    list[int],
    list[str],
    dict[str, object],
]:
    quality = style.load_quality_metadata(args.data, args.allow_unversioned_data)
    train_transform, eval_transform = style.build_transforms(args.img_size)
    train_set, validation_set, test_set = style.imagefolder_splits(
        args.data, train_transform, eval_transform
    )
    return (
        train_set,
        validation_set,
        test_set,
        list(train_set.targets),
        list(train_set.classes),
        quality,
    )


def prepare_room_data(
    args: argparse.Namespace,
) -> tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    None,
    list[int],
    list[str],
    dict[str, object],
]:
    train_transform, validation_transform = room.build_transforms(args.img_size)
    index_dataset = datasets.ImageFolder(args.data)
    if len(index_dataset) < args.min_images:
        raise ValueError(
            f"At least {args.min_images} labeled images are required "
            f"(found {len(index_dataset)})"
        )
    selected = list(range(len(index_dataset)))
    if args.limit is not None:
        selected = room.balanced_limit_indices(index_dataset.targets, args.limit, args.seed)
    selected_targets = [index_dataset.targets[index] for index in selected]
    train_relative, validation_relative = room.stratified_split_indices(
        selected_targets, args.validation_fraction, args.seed
    )
    train_indices = [selected[index] for index in train_relative]
    validation_indices = [selected[index] for index in validation_relative]
    train_dataset = datasets.ImageFolder(args.data, transform=train_transform)
    validation_dataset = datasets.ImageFolder(args.data, transform=validation_transform)
    train_set = Subset(train_dataset, train_indices)
    validation_set = Subset(validation_dataset, validation_indices)
    targets = [index_dataset.targets[index] for index in train_indices]
    return (
        train_set,
        validation_set,
        None,
        targets,
        list(index_dataset.classes),
        {"dataset_fingerprint": None, "manifest_sha256": None},
    )


def evaluate_room(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    criterion: nn.Module,
) -> dict[str, object]:
    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ):
                logits = model(images)
                loss_total += float(criterion(logits, labels))
            correct += int((logits.argmax(dim=1) == labels).sum())
            total += len(labels)
    return {
        "accuracy": correct / max(total, 1),
        "loss": loss_total / max(len(loader), 1),
        "images": total,
    }


def resume_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    mode: str,
    architecture: str,
    classes: list[str],
    image_size: int,
    dataset_fingerprint: object,
    epoch: int,
    batch_in_epoch: int,
    best_metric: float,
    best_epoch: int,
    epochs_without_improvement: int,
    loss_sum: float,
    loss_batches: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": mode,
        "architecture": architecture,
        "classes": classes,
        "image_size": image_size,
        "dataset_fingerprint": dataset_fingerprint,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "loss_sum": loss_sum,
        "loss_batches": loss_batches,
    }


def validate_resume(
    payload: dict[str, object],
    args: argparse.Namespace,
    classes: list[str],
    quality: dict[str, object],
) -> None:
    if payload.get("mode") != args.mode:
        raise ValueError("resume checkpoint mode does not match --mode")
    if payload.get("architecture") != args.model_name:
        raise ValueError("resume checkpoint architecture does not match --model-name")
    if payload.get("classes") != classes:
        raise ValueError("resume checkpoint classes do not match the current dataset")
    if int(payload.get("image_size", args.img_size)) != args.img_size:
        raise ValueError("resume checkpoint image size does not match --img-size")
    expected = quality.get("dataset_fingerprint")
    actual = payload.get("dataset_fingerprint")
    if expected is not None and actual != expected:
        raise ValueError("resume checkpoint dataset fingerprint does not match current data")


def save_best_checkpoint(
    args: argparse.Namespace,
    model: nn.Module,
    classes: list[str],
    quality: dict[str, object],
    precision_name: str,
    metric_name: str,
    validation: dict[str, object],
    epoch: int,
    train_images: int,
    validation_images: int,
    test_images: int,
    targets: list[int],
) -> None:
    payload: dict[str, object] = {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "architecture": args.model_name,
        "image_size": args.img_size,
        "normalization": {"mean": style.IMAGENET_MEAN, "std": style.IMAGENET_STD},
        "seed": args.seed,
        "precision": precision_name,
        "class_balance": args.class_balance,
        "train_images": train_images,
        "validation_images": validation_images,
        "test_images": test_images,
        "training_class_counts": dict(sorted(Counter(targets).items())),
        "dataset_fingerprint": quality.get("dataset_fingerprint"),
        "dataset_manifest_sha256": quality.get("manifest_sha256"),
        "selection_metric": metric_name,
        "validation_metrics": validation,
        "epoch": epoch,
    }
    atomic_torch_save(payload, args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--mode", choices=("style", "room"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-output", type=Path, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
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
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-images", type=int, default=20)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")
    if args.checkpoint_every_steps < 1:
        parser.error("--checkpoint-every-steps must be at least 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        parser.error("--label-smoothing must be between 0 and 1")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be greater than 0 and less than 0.5")
    return args


def main() -> int:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    args = parse_args()
    install_signal_handlers()
    seed_everything(args.seed)
    device = style.resolve_device(args.device)
    precision_name, autocast_dtype = style.resolve_precision(args.precision, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32
        torch.set_float32_matmul_precision("high")

    if args.mode == "style":
        train_set, validation_set, test_set, targets, classes, quality = prepare_style_data(args)
    else:
        train_set, validation_set, test_set, targets, classes, quality = prepare_room_data(args)

    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_options(args.num_workers, device),
    )
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            **loader_options(args.num_workers, device),
        )

    model = build_model(args.model_name, len(classes), args.pretrained).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=autocast_dtype == torch.float16)

    resume_path = args.resume_output or args.output.with_suffix(args.output.suffix + ".resume.pth")
    start_epoch = 1
    resume_batch = 0
    best_metric = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    resumed_loss_sum = 0.0
    resumed_loss_batches = 0

    if args.resume is not None and args.resume.is_file():
        payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("resume checkpoint must contain a mapping")
        validate_resume(payload, args, classes, quality)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        scaler.load_state_dict(payload.get("scaler_state_dict", {}))
        start_epoch = int(payload.get("epoch", 1))
        resume_batch = int(payload.get("batch_in_epoch", 0))
        best_metric = float(payload.get("best_metric", -1.0))
        best_epoch = int(payload.get("best_epoch", 0))
        epochs_without_improvement = int(payload.get("epochs_without_improvement", 0))
        resumed_loss_sum = float(payload.get("loss_sum", 0.0))
        resumed_loss_batches = int(payload.get("loss_batches", 0))
        print(
            f"resumed_from={args.resume} epoch={start_epoch} batch={resume_batch} "
            f"best_metric={best_metric:.5f}",
            flush=True,
        )

    metric_name = "validation_macro_f1" if args.mode == "style" else "validation_accuracy"
    print(
        f"mode={args.mode} device={device} precision={precision_name} classes={len(classes)} "
        f"train={len(train_set)} validation={len(validation_set)} "
        f"test={len(test_set) if test_set is not None else 0}",
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loader = make_train_loader(
            train_set,
            targets,
            args.batch_size,
            args.num_workers,
            device,
            args.class_balance,
            args.seed,
            epoch,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = resumed_loss_sum if epoch == start_epoch else 0.0
        loss_batches = resumed_loss_batches if epoch == start_epoch else 0
        skip_batches = resume_batch if epoch == start_epoch else 0
        last_completed_batch = skip_batches

        for step, (images, labels) in enumerate(train_loader, start=1):
            if step <= skip_batches:
                continue
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
            loss_sum += float(loss.detach()) * args.gradient_accumulation_steps
            loss_batches += 1
            should_step = (
                step % args.gradient_accumulation_steps == 0 or step == len(train_loader)
            )
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                last_completed_batch = step
                should_checkpoint = step % args.checkpoint_every_steps == 0
                if should_checkpoint or _STOP_REQUESTED:
                    atomic_torch_save(
                        resume_payload(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            scaler=scaler,
                            mode=args.mode,
                            architecture=args.model_name,
                            classes=classes,
                            image_size=args.img_size,
                            dataset_fingerprint=quality.get("dataset_fingerprint"),
                            epoch=epoch,
                            batch_in_epoch=step,
                            best_metric=best_metric,
                            best_epoch=best_epoch,
                            epochs_without_improvement=epochs_without_improvement,
                            loss_sum=loss_sum,
                            loss_batches=loss_batches,
                        ),
                        resume_path,
                    )
                if _STOP_REQUESTED:
                    print(f"paused checkpoint={resume_path} epoch={epoch} batch={step}", flush=True)
                    return PAUSED_EXIT_CODE

        if last_completed_batch < len(train_loader):
            raise RuntimeError("training epoch ended before all batches completed")

        if args.mode == "style":
            validation = style.evaluate(
                model, validation_loader, device, autocast_dtype, classes
            )
            current_metric = float(validation["macro_f1"])
        else:
            validation = evaluate_room(
                model, validation_loader, device, autocast_dtype, criterion
            )
            current_metric = float(validation["accuracy"])
        scheduler.step()
        print(
            f"epoch={epoch} train_loss={loss_sum / max(loss_batches, 1):.5f} "
            f"selection_metric={current_metric:.5f}",
            flush=True,
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            save_best_checkpoint(
                args,
                model,
                classes,
                quality,
                precision_name,
                metric_name,
                validation,
                epoch,
                len(train_set),
                len(validation_set),
                len(test_set) if test_set is not None else 0,
                targets,
            )
        else:
            epochs_without_improvement += 1

        atomic_torch_save(
            resume_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                mode=args.mode,
                architecture=args.model_name,
                classes=classes,
                image_size=args.img_size,
                dataset_fingerprint=quality.get("dataset_fingerprint"),
                epoch=epoch + 1,
                batch_in_epoch=0,
                best_metric=best_metric,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                loss_sum=0.0,
                loss_batches=0,
            ),
            resume_path,
        )
        resume_batch = 0
        resumed_loss_sum = 0.0
        resumed_loss_batches = 0
        if _STOP_REQUESTED:
            print(f"paused checkpoint={resume_path} next_epoch={epoch + 1}", flush=True)
            return PAUSED_EXIT_CODE
        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(f"early_stopping epoch={epoch}", flush=True)
            break

    if best_epoch == 0 or not args.output.is_file():
        raise RuntimeError("training did not produce a best checkpoint")
    checkpoint = torch.load(args.output, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if args.mode == "style" and test_loader is not None:
        test_metrics = style.evaluate(model, test_loader, device, autocast_dtype, classes)
        checkpoint["test_metrics"] = test_metrics
        checkpoint["best_epoch"] = best_epoch
        atomic_torch_save(checkpoint, args.output)
        report = {
            "version": 2,
            "checkpoint": str(args.output),
            "checkpoint_sha256": style.sha256_file(args.output),
            "architecture": args.model_name,
            "classes": classes,
            "image_size": args.img_size,
            "dataset_fingerprint": quality.get("dataset_fingerprint"),
            "dataset_manifest_sha256": quality.get("manifest_sha256"),
            "selection_metric": metric_name,
            "best_epoch": best_epoch,
            "validation": checkpoint["validation_metrics"],
            "test": test_metrics,
        }
        atomic_json_save(report, args.output.with_suffix(args.output.suffix + ".metrics.json"))

    resume_path.unlink(missing_ok=True)
    print(f"completed output={args.output} best_epoch={best_epoch} best_metric={best_metric:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
