from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    try:
        return timm.create_model(
            "tf_efficientnet_b0",
            pretrained=pretrained,
            num_classes=num_classes,
        )
    except (OSError, RuntimeError) as exc:
        if not pretrained:
            raise
        print(
            "WARNING: could not download pretrained weights "
            f"({exc.__class__.__name__}: {exc}). Falling back to random initialization. "
            "Pass --no-pretrained to skip the download attempt.",
            flush=True,
        )
        return timm.create_model(
            "tf_efficientnet_b0",
            pretrained=False,
            num_classes=num_classes,
        )


def save_checkpoint(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="ImageFolder root: one directory per room class")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/room_classifier/efficientnet_b0.pth"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Square resize applied to training images",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of images used (for smoke runs)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=20,
        help="Minimum dataset size; lower it for tiny synthetic smoke datasets",
    )
    parser.add_argument("--num-workers", type=int, default=2)
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
    args = parser.parse_args()
    pretrained = args.pretrained
    if pretrained is None:
        pretrained = env_flag("FURNITURE_PRETRAINED", True)
    seed_everything(args.seed)

    transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(args.data, transform=transform)
    if len(dataset) < args.min_images:
        raise ValueError(
            f"At least {args.min_images} labeled images are required "
            f"(found {len(dataset)}); pass --min-images to lower the threshold for smoke runs"
        )
    if args.limit is not None:
        keep = min(args.limit, len(dataset))
        dataset, _ = random_split(
            dataset,
            [keep, len(dataset) - keep],
            generator=torch.Generator().manual_seed(args.seed),
        )
        classes = dataset.dataset.classes
    else:
        classes = dataset.classes
    validation_size = max(1, int(len(dataset) * 0.2))
    train_set, validation_set = random_split(
        dataset,
        [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    validation_loader = DataLoader(
        validation_set, batch_size=args.batch_size, num_workers=args.num_workers
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(classes), pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"train {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                predictions = model(images.to(device)).argmax(dim=1).cpu()
                correct += int((predictions == labels).sum())
                total += len(labels)
        accuracy = correct / max(total, 1)
        print(f"epoch={epoch + 1} validation_accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_checkpoint(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "architecture": "tf_efficientnet_b0",
                    "validation_accuracy": accuracy,
                    "seed": args.seed,
                },
                args.output,
            )
    if best_accuracy == 0.0:
        # Accuracy never beat 0.0 (possible on a tiny smoke set); still write a checkpoint.
        save_checkpoint(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "architecture": "tf_efficientnet_b0",
                "validation_accuracy": best_accuracy,
                "seed": args.seed,
            },
            args.output,
        )


if __name__ == "__main__":
    main()
