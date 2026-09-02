from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no NVIDIA GPU is visible to PyTorch")
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device {requested!r}; expected auto, cpu, or cuda")


class FloorPlanDataset(Dataset):
    def __init__(
        self,
        root: Path,
        size: int = 512,
        classes: int = 5,
        mask_remap: str = "none",
        limit: int | None = None,
    ) -> None:
        self.images = sorted(
            path
            for path in (root / "images").glob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        self.masks = root / "masks"
        self.size = size
        self.classes = classes
        if not self.images:
            raise ValueError(f"No training images found under {root / 'images'}")
        if limit is not None:
            self.images = self.images[:limit]
        for image_path in self.images:
            mask_path = self.masks / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise ValueError(f"Missing mask for image {image_path}: expected {mask_path}")
        self.remap_table: torch.Tensor | None = None
        if mask_remap == "auto":
            self.remap_table = self._build_remap_table()
        elif mask_remap != "none":
            raise ValueError(f"Unknown --mask-remap mode: {mask_remap!r} (use 'none' or 'auto')")

    def _mask_path(self, image_path: Path) -> Path:
        return self.masks / f"{image_path.stem}.png"

    def _build_remap_table(self) -> torch.Tensor:
        unique_values: set[int] = set()
        for image_path in self.images:
            mask = read_image(str(self._mask_path(image_path)))[:1].long()
            unique_values.update(mask.unique().tolist())
        sorted_values = sorted(unique_values)
        if len(sorted_values) > self.classes:
            raise ValueError(
                f"Masks contain {len(sorted_values)} distinct values but --classes={self.classes}; "
                "raise --classes or fix the masks"
            )
        table = torch.zeros(max(sorted_values) + 1, dtype=torch.long)
        for new_id, old_value in enumerate(sorted_values):
            table[old_value] = new_id
        print(f"mask remap (auto): {sorted_values} -> {list(range(len(sorted_values)))}")
        return table

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        mask_path = self._mask_path(image_path)
        image = resize(read_image(str(image_path)).float() / 255, [self.size, self.size])
        mask = read_image(str(mask_path))[:1].long()
        if self.remap_table is not None:
            if int(mask.max()) >= len(self.remap_table):
                raise ValueError(
                    f"Mask {mask_path} contains value {int(mask.max())} which was not seen "
                    "when the remap table was built"
                )
            mask = self.remap_table[mask]
        mask = resize(mask, [self.size, self.size], interpolation=InterpolationMode.NEAREST)
        mask = mask.squeeze(0)
        mask_min, mask_max = int(mask.min()), int(mask.max())
        if mask_min < 0 or mask_max >= self.classes:
            raise ValueError(
                f"Mask {mask_path} contains class ids in [{mask_min}, {mask_max}] but the model "
                f"expects [0, {self.classes - 1}] (--classes={self.classes}). Pass "
                "--mask-remap auto to remap mask values to contiguous class ids."
            )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Directory containing images/ and masks/")
    parser.add_argument("--output", type=Path, default=Path("models/floorplan_segmenter/unet.pt"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--classes", type=int, default=5, help="Number of segmentation classes")
    parser.add_argument("--size", type=int, default=512, help="Square resize for images and masks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of training pairs")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device. Explicit cuda fails closed when no NVIDIA GPU is visible.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA automatic mixed precision. Ignored on CPU.",
    )
    parser.add_argument(
        "--mask-remap",
        choices=["none", "auto"],
        default="none",
        help=(
            "'auto' remaps the distinct mask values to contiguous class ids 0..N-1 "
            "(e.g. 0/255 masks become 0/1). 'none' requires masks to already use "
            "class ids 0..classes-1."
        ),
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    dataset = FloorPlanDataset(
        args.data,
        size=args.size,
        classes=args.classes,
        mask_remap=args.mask_remap,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if args.amp and not use_amp:
        print("AMP requested but CUDA is not active; continuing with full precision")
    if device.type == "cuda":
        print(f"training device=cuda gpu={torch.cuda.get_device_name(0)} amp={use_amp}")
    else:
        print("training device=cpu amp=False")

    model = SmallUNet(classes=args.classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, masks in tqdm(loader, desc=f"epoch {epoch + 1}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(images), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item())
        print(f"epoch={epoch + 1} loss={running / max(len(loader), 1):.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu().eval())
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    scripted.save(str(temporary))
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
