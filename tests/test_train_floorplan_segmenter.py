from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / "training" / "train_floorplan_segmenter.py"
SPEC = importlib.util.spec_from_file_location("train_floorplan_segmenter", MODULE_PATH)
assert SPEC and SPEC.loader
segmenter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(segmenter)


def _write_pair(root: Path, split: str, name: str, mask_value: int) -> None:
    images = root / "images" / split
    masks = root / "masks" / split
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), "white").save(images / f"{name}.png")
    mask = Image.new("L", (16, 16), 0)
    for x in range(8, 16):
        for y in range(16):
            mask.putpixel((x, y), mask_value)
    mask.save(masks / f"{name}.png")


def test_deterministic_split_is_stable_disjoint_and_complete() -> None:
    first_train, first_validation = segmenter.deterministic_split_indices(20, 0.2, 42)
    second_train, second_validation = segmenter.deterministic_split_indices(20, 0.2, 42)
    assert first_train == second_train
    assert first_validation == second_validation
    assert set(first_train).isdisjoint(first_validation)
    assert sorted(first_train + first_validation) == list(range(20))
    assert len(first_validation) == 4


def test_segmentation_metrics_known_confusion_matrix() -> None:
    confusion = torch.tensor([[5, 1], [2, 2]], dtype=torch.int64)
    metrics = segmenter.segmentation_metrics(confusion)
    assert metrics["pixel_accuracy"] == 0.7
    assert abs(metrics["mean_iou"] - 0.5125) < 1e-9
    assert metrics["per_class_iou"] == [0.625, 0.4]
    expected_dice = [(10 / 13), (4 / 7)]
    for actual, expected in zip(metrics["per_class_dice"], expected_dice, strict=True):
        assert actual is not None
        assert abs(actual - expected) < 1e-9


def test_named_split_reuses_training_mask_remap(tmp_path: Path) -> None:
    _write_pair(tmp_path, "train", "a", 255)
    _write_pair(tmp_path, "train", "b", 255)
    _write_pair(tmp_path, "validation", "c", 255)

    training = segmenter.FloorPlanDataset(
        tmp_path,
        size=16,
        classes=2,
        mask_remap="auto",
        split="train",
    )
    assert training.remap_values == [0, 255]
    assert training.remap_table is not None

    validation = segmenter.FloorPlanDataset(
        tmp_path,
        size=16,
        classes=2,
        split="validation",
        remap_table=training.remap_table,
    )
    _, mask = validation[0]
    assert set(mask.unique().tolist()) == {0, 1}


def test_update_confusion_matrix_matches_predictions() -> None:
    logits = torch.tensor(
        [
            [
                [[4.0, 0.0], [0.0, 3.0]],
                [[0.0, 4.0], [3.0, 0.0]],
            ]
        ]
    )
    targets = torch.tensor([[[0, 1], [1, 0]]])
    confusion = torch.zeros((2, 2), dtype=torch.int64)
    segmenter.update_confusion_matrix(confusion, logits, targets, 2)
    assert confusion.tolist() == [[2, 0], [0, 2]]
