from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_style_dataset", ROOT / "scripts" / "prepare_style_dataset.py")


def make_image(path: Path, color: tuple[int, int, int], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    array = np.full((64, 64, 3), color, dtype=np.int16)
    array += rng.integers(-8, 9, size=array.shape, dtype=np.int16)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path)


def make_dataset(root: Path, per_class: int = 5) -> None:
    classes = {"minimalist": (220, 220, 220), "industrial": (80, 80, 80)}
    for class_offset, (label, color) in enumerate(classes.items()):
        for index in range(per_class):
            make_image(
                root / label / f"{index:02d}.png",
                color,
                seed=class_offset * 100 + index,
            )


def args_for(input_root: Path, output_root: Path):
    return type(
        "Args",
        (),
        {
            "input": input_root,
            "output": output_root,
            "source_manifest": None,
            "seed": 42,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
            "max_hamming_distance": 0,
            "aspect_tolerance": 0.02,
            "color_tolerance": 18.0,
            "mode": "copy",
            "dry_run": False,
            "include_review_required": False,
            "allow_label_conflicts": False,
        },
    )()


def test_prepare_removes_exact_duplicate_and_versions_dataset(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_dataset(source)
    duplicate = source / "minimalist" / "copy.png"
    duplicate.write_bytes((source / "minimalist" / "00.png").read_bytes())
    output = tmp_path / "prepared"
    summary = prepare.prepare(args_for(source, output))
    assert summary["duplicate_images_removed"] == 1
    assert summary["label_conflict_clusters"] == 0
    assert isinstance(summary["dataset_fingerprint"], str)
    assert len(summary["dataset_fingerprint"]) == 64
    assert (output / "summary.json").is_file()
    assert (output / "manifest.jsonl").is_file()
    assert (output / "train").is_dir()
    assert (output / "validation").is_dir()
    assert (output / "test").is_dir()


def test_prepare_quarantines_cross_label_duplicate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_dataset(source)
    conflicting = source / "industrial" / "conflict.png"
    conflicting.write_bytes((source / "minimalist" / "00.png").read_bytes())
    output = tmp_path / "prepared"
    summary = prepare.prepare(args_for(source, output))
    assert summary["label_conflict_clusters"] == 1
    assert summary["label_conflict_images"] == 2
    conflicts = json.loads((output / "conflicts.json").read_text(encoding="utf-8"))
    assert len(conflicts) == 1
    assert sorted(conflicts[0]["labels"]) == ["industrial", "minimalist"]


def test_dataset_fingerprint_is_stable_across_output_directories(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_dataset(source)
    first = prepare.prepare(args_for(source, tmp_path / "one"))
    second = prepare.prepare(args_for(source, tmp_path / "two"))
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]


def training_module():
    pytest.importorskip("timm")
    return load_module(
        "train_style_classifier_test", ROOT / "training" / "train_style_classifier.py"
    )


def test_training_manifest_validator_rejects_split_leakage(tmp_path: Path) -> None:
    training = training_module()
    manifest = tmp_path / "manifest.jsonl"
    digest = "a" * 64
    rows = [
        {"status": "accepted", "split": "train", "sha256": digest},
        {"status": "accepted", "split": "test", "sha256": digest},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dataset leakage"):
        training.validate_manifest_no_split_leakage(manifest)


def test_classifier_metrics_include_calibration_and_per_class_data() -> None:
    training = training_module()
    logits = torch.tensor([[4.0, 0.1], [0.3, 2.0], [1.0, 1.2], [3.0, 0.0]])
    labels = torch.tensor([0, 1, 1, 0])
    metrics = training.classification_metrics(logits, labels, ["minimalist", "industrial"])
    assert 0.0 <= metrics["expected_calibration_error_15bin"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert set(metrics["per_class"]) == {"minimalist", "industrial"}
    assert len(metrics["confusion_matrix"]) == 2
