from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.predict_openimages_furniture import (
    MODEL_TO_CANONICAL,
    benchmark_image_ids,
    find_image,
    normalized_box,
    validate_sha256,
)


def test_candidate_label_mapping_is_explicit_and_does_not_invent_classes() -> None:
    assert MODEL_TO_CANONICAL == {"Chair": "Chair", "Sofa": "Sofa", "Table": "Table"}
    assert "furniture" not in MODEL_TO_CANONICAL
    assert "Bed" not in MODEL_TO_CANONICAL


def test_normalized_box_converts_pixels_and_clamps() -> None:
    assert normalized_box([-10.0, 20.0, 220.0, 120.0], 200, 100) == {
        "x_min": 0.0,
        "y_min": 0.2,
        "x_max": 1.0,
        "y_max": 1.0,
    }


def test_normalized_box_rejects_inverted_detector_output() -> None:
    with pytest.raises(ValueError, match="inverted"):
        normalized_box([80.0, 20.0, 10.0, 60.0], 100, 100)


def test_model_sha256_validation_is_fail_closed() -> None:
    assert validate_sha256("A" * 64) == "a" * 64
    with pytest.raises(ValueError, match="64 hexadecimal"):
        validate_sha256("not-a-sha")


def test_benchmark_image_ids_are_unique_and_sorted(tmp_path: Path) -> None:
    path = tmp_path / "ground_truth.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"image_id": "b"}),
                json.dumps({"image_id": "a"}),
                json.dumps({"image_id": "b"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert benchmark_image_ids(path) == ["a", "b"]


def test_find_image_fails_closed_for_missing_or_ambiguous_files(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="found 0"):
        find_image(tmp_path, "img")

    (tmp_path / "img.jpg").write_bytes(b"jpg")
    (tmp_path / "img.png").write_bytes(b"png")
    with pytest.raises(FileNotFoundError, match="found 2"):
        find_image(tmp_path, "img")
