from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"


def load_training_module(name: str):
    spec = importlib.util.spec_from_file_location(name, TRAINING_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


segmenter = load_training_module("train_floorplan_segmenter")


def make_pair(root: Path, name: str, mask_values: tuple[int, ...], size: int = 64) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (size, size), "white")
    mask = Image.new("L", (size, size), int(mask_values[0]))
    draw_image, draw_mask = ImageDraw.Draw(image), ImageDraw.Draw(mask)
    for offset, value in enumerate(mask_values[1:], start=1):
        x0 = offset * (size // (len(mask_values) + 1))
        draw_image.rectangle((x0, 8, x0 + 6, size - 8), fill="black")
        draw_mask.rectangle((x0, 8, x0 + 6, size - 8), fill=int(value))
    image.save(root / "images" / f"{name}.png")
    mask.save(root / "masks" / f"{name}.png")


def test_mask_resize_uses_nearest_and_preserves_class_ids(tmp_path: Path) -> None:
    make_pair(tmp_path, "a", (1, 3))
    dataset = segmenter.FloorPlanDataset(tmp_path, size=128, classes=4)
    _, mask = dataset[0]
    assert set(mask.unique().tolist()) <= {1, 3}


def test_mask_remap_auto_maps_values_to_contiguous_ids(tmp_path: Path) -> None:
    make_pair(tmp_path, "a", (0, 255))
    dataset = segmenter.FloorPlanDataset(tmp_path, size=64, classes=2, mask_remap="auto")
    _, mask = dataset[0]
    assert set(mask.unique().tolist()) <= {0, 1}


def test_out_of_range_mask_raises_clear_error(tmp_path: Path) -> None:
    make_pair(tmp_path, "a", (0, 255))
    dataset = segmenter.FloorPlanDataset(tmp_path, size=64, classes=5)
    with pytest.raises(ValueError, match="--mask-remap auto"):
        dataset[0]


def test_missing_mask_names_the_expected_file(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(tmp_path / "images" / "lonely.png")
    with pytest.raises(ValueError, match="Missing mask.*lonely"):
        segmenter.FloorPlanDataset(tmp_path)


def test_non_image_files_are_ignored(tmp_path: Path) -> None:
    make_pair(tmp_path, "a", (0, 1))
    (tmp_path / "images" / "notes.txt").write_text("stray file")
    dataset = segmenter.FloorPlanDataset(tmp_path, size=64, classes=2)
    assert len(dataset) == 1


def test_segmenter_trains_one_epoch_and_writes_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(4):
        make_pair(tmp_path / "data", f"{index:03d}", (0, 128, 255))
    output = tmp_path / "out" / "unet.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_floorplan_segmenter",
            str(tmp_path / "data"),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--classes",
            "3",
            "--size",
            "64",
            "--num-workers",
            "0",
            "--mask-remap",
            "auto",
        ],
    )
    segmenter.main()
    assert output.is_file()
    assert torch.jit.load(str(output)) is not None


classifier = load_training_module("train_room_classifier")


def make_room_dataset(root: Path, per_class: int = 4, size: int = 64) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    palette = {"living_room": (200, 60, 60), "bedroom": (60, 200, 60)}
    for name, color in palette.items():
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            noise = rng.normal(0, 10, (size, size, 3)).astype(np.int16)
            array = np.clip(np.array(color, dtype=np.int16) + noise, 0, 255).astype(np.uint8)
            Image.fromarray(array).save(directory / f"{index:03d}.png")


def test_classifier_trains_one_epoch_offline_and_writes_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_room_dataset(tmp_path / "rooms")
    output = tmp_path / "out" / "room_classifier.pth"
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_room_classifier",
            str(tmp_path / "rooms"),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--img-size",
            "64",
            "--min-images",
            "4",
            "--num-workers",
            "0",
            "--no-pretrained",
        ],
    )
    classifier.main()
    assert output.is_file()
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    assert checkpoint["architecture"] == "tf_efficientnet_b0"
    assert sorted(checkpoint["classes"]) == ["bedroom", "living_room"]


def test_classifier_min_images_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_room_dataset(tmp_path / "rooms", per_class=2)
    monkeypatch.setattr(
        "sys.argv", ["train_room_classifier", str(tmp_path / "rooms"), "--no-pretrained"]
    )
    with pytest.raises(ValueError, match="--min-images"):
        classifier.main()
