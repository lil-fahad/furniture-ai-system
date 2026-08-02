from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    ast.parse(text)
    return text


def test_segmentation_masks_use_nearest_neighbor_resize() -> None:
    source = _source("training/train_floorplan_segmenter.py")
    assert "InterpolationMode.NEAREST" in source
    assert "Missing segmentation mask" in source


def test_classifier_validation_has_no_random_augmentation() -> None:
    source = _source("training/train_room_classifier.py")
    validation_body = source.split("def validation_transform", 1)[1].split(
        "def split_indices", 1
    )[0]
    assert "RandomHorizontalFlip" not in validation_body
    assert "ColorJitter" not in validation_body
    assert "Subset(training_dataset" in source
    assert "Subset(validation_dataset" in source
