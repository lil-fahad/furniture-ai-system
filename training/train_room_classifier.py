from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

_NORMALIZE = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def training_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            _NORMALIZE,
        ]
    )


def validation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            _NORMALIZE,
        ]
    )


def split_indices(size: int, validation_size: int, seed: int) -> tuple[list[int], list[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(size, generator=generator).tolist()
    validation = indices[:validation_size]
    training = indices[validation_size:]
    return training, validation


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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)

    metadata = datasets.ImageFolder(args.data)
    if len(metadata) < 20:
        raise ValueError("At least 20 labeled images are required")
    validation_size = max(1, int(len(metadata) * 0.2))
    train_indices, validation_indices = split_indices(
        len(metadata), validation_size, args.seed
    )

    training_dataset = datasets.ImageFolder(args.data, transform=training_transform())
    validation_dataset = datasets.ImageFolder(args.data, transform=validation_transform())
    if training_dataset.class_to_idx != validation_dataset.class_to_idx:
        raise RuntimeError("Training and validation class mappings do not match")

    train_set = Subset(training_dataset, train_indices)
    validation_set = Subset(validation_dataset, validation_indices)
    loader_generator = torch.Generator().manual_seed(args.seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, args.workers),
        pin_memory=pin_memory,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.workers),
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(
        "tf_efficientnet_b0",
        pretrained=True,
        num_classes=len(metadata.classes),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_accuracy = -1.0
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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": metadata.classes,
                    "architecture": "tf_efficientnet_b0",
                    "validation_accuracy": accuracy,
                    "seed": args.seed,
                    "training_samples": len(train_set),
                    "validation_samples": len(validation_set),
                },
                args.output,
            )


if __name__ == "__main__":
    main()
