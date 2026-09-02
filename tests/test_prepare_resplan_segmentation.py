from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon, box

MODULE_PATH = Path(__file__).resolve().parents[1] / "training" / "prepare_resplan_segmentation.py"
SPEC = importlib.util.spec_from_file_location("prepare_resplan_segmentation", MODULE_PATH)
assert SPEC and SPEC.loader
preparer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preparer)


def _record(
    plan_id: str,
    source_index: int,
    split: str,
    room: Polygon,
) -> dict[str, object]:
    min_x, min_y, max_x, max_y = room.bounds
    wall = LineString(
        [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
            (min_x, min_y),
        ]
    )
    door = LineString([(min_x + 1, min_y), (min_x + 3, min_y)])
    window = LineString([(max_x, min_y + 1), (max_x, min_y + 3)])
    return {
        "schema": preparer.SAFE_SCHEMA,
        "plan_id": plan_id,
        "source_index": source_index,
        "split": split,
        "wall_depth": 0.2,
        "geometries": {
            "inner": room.wkt,
            "living": room.wkt,
            "wall": wall.wkt,
            "door": door.wkt,
            "window": window.wkt,
        },
    }


def test_render_pair_contains_all_segmentation_classes() -> None:
    record = _record("a", 1, "train", box(0, 0, 10, 8))
    image, mask = preparer.render_pair(
        preparer.parse_geometries(record),
        size=128,
        padding_ratio=0.05,
    )
    assert image.shape == (128, 128)
    assert mask.shape == (128, 128)
    values = set(int(value) for value in np.unique(mask))
    assert {0, 1, 2, 3, 4}.issubset(values)


def test_geometry_signature_is_rotation_and_flip_invariant() -> None:
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:45, 8:26] = 1
    mask[8:12, 8:26] = 2
    mask[20:25, 23:29] = 3
    mask[35:42, 7:11] = 4
    signature = preparer.canonical_geometry_signature(mask)
    for rotations in range(4):
        transformed = np.rot90(mask, rotations)
        assert preparer.canonical_geometry_signature(transformed) == signature
        assert preparer.canonical_geometry_signature(np.fliplr(transformed)) == signature


def test_process_records_filters_train_leakage_and_augmented(tmp_path: Path) -> None:
    train = _record("train", 1, "train", box(0, 0, 10, 10))
    leaked_validation = _record("leaked", 2, "validation", box(0, 0, 10, 10))
    validation = _record("validation", 3, "validation", box(0, 0, 20, 8))
    test = _record("test", 4, "test", Polygon([(0, 0), (14, 0), (14, 4), (7, 4), (7, 10), (0, 10)]))
    augmented = _record("aug", 5, "augmented", box(0, 0, 7, 5))

    preparer.prepare_output(tmp_path, force=False)
    samples, counts, skipped = preparer.process_records(
        [train, leaked_validation, validation, test, augmented],
        tmp_path,
        size=128,
        padding_ratio=0.05,
        include_augmented=False,
    )

    assert counts == {"train": 1, "validation": 1, "test": 1}
    assert skipped["train_leakage_validation"] == 1
    assert skipped["augmented_excluded"] == 1
    assert {sample["plan_id"] for sample in samples} == {"train", "validation", "test"}
    for sample in samples:
        assert (tmp_path / sample["image"]).is_file()
        assert (tmp_path / sample["mask"]).is_file()
        assert len(sample["geometry_signature"]) == 64
        assert len(sample["image_sha256"]) == 64
        assert len(sample["mask_sha256"]) == 64


def test_augmented_can_only_enter_training(tmp_path: Path) -> None:
    records = [
        _record("train", 1, "train", box(0, 0, 10, 10)),
        _record("validation", 2, "validation", box(0, 0, 20, 8)),
        _record("test", 3, "test", box(0, 0, 6, 14)),
        _record("aug", 4, "augmented", Polygon([(0, 0), (11, 0), (9, 6), (0, 8)])),
    ]
    preparer.prepare_output(tmp_path, force=False)
    samples, counts, _ = preparer.process_records(
        records,
        tmp_path,
        size=128,
        padding_ratio=0.05,
        include_augmented=True,
    )
    augmented = next(sample for sample in samples if sample["plan_id"] == "aug")
    assert augmented["source_split"] == "augmented"
    assert augmented["split"] == "train"
    assert counts["train"] == 2


def test_safe_jsonl_loader_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"schema": "wrong", "geometries": {}}) + "\n", encoding="utf-8")
    try:
        preparer.load_safe_records(path)
    except ValueError as exc:
        assert "Invalid safe-export record" in str(exc)
    else:
        raise AssertionError("Expected invalid schema to be rejected")


def test_signature_stays_same_after_shapely_rotation_render() -> None:
    original = _record(
        "original",
        1,
        "train",
        Polygon([(0, 0), (12, 0), (12, 4), (5, 4), (5, 10), (0, 10)]),
    )
    rotated_room = affinity.rotate(
        Polygon([(0, 0), (12, 0), (12, 4), (5, 4), (5, 10), (0, 10)]),
        90,
        origin="centroid",
    )
    rotated = _record("rotated", 2, "validation", rotated_room)
    _, original_mask = preparer.render_pair(
        preparer.parse_geometries(original), size=128, padding_ratio=0.05
    )
    _, rotated_mask = preparer.render_pair(
        preparer.parse_geometries(rotated), size=128, padding_ratio=0.05
    )
    assert preparer.canonical_geometry_signature(original_mask) == preparer.canonical_geometry_signature(
        rotated_mask
    )
