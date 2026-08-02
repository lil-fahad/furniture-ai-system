from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms.functional import resize
from tqdm import tqdm


class FloorPlanDataset(Dataset):
    def __init__(self, root: Path, size: int = 512) -> None:
        self.images = sorted((root / "images").glob("*"))
        self.masks = root / "masks"
        self.size = size
        if not self.images:
            raise ValueError("No training images found")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        mask_path = self.masks / f"{image_path.stem}.png"
        image = resize(read_image(str(image_path)).float() / 255, [self.size, self.size])
        mask = resize(read_image(str(mask_path))[:1], [self.size, self.size]).squeeze(0).long()
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
    args = parser.parse_args()

    dataset = FloorPlanDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for images, masks in tqdm(loader, desc=f"epoch {epoch + 1}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
        print(f"epoch={epoch + 1} loss={running / max(len(loader), 1):.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model.cpu().eval())
    scripted.save(str(args.output))


if __name__ == "__main__":
    main()
