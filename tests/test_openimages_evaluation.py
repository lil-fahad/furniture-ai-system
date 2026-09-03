from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from furniture_ai.evaluation.openimages import (
    OpenImagesBenchmarkPolicy,
    build_openimages_furniture_ground_truth,
    write_benchmark_manifest,
)

FIELDS = [
    "ImageID",
    "LabelName",
    "Confidence",
    "XMin",
    "XMax",
    "YMin",
    "YMax",
    "IsGroupOf",
    "IsDepiction",
]


def write_annotations(path: Path) -> str:
    rows = [
        ["img-chair", "/m/01mzpv", "1", "0.1", "0.4", "0.2", "0.6", "0", "0"],
        ["img-table", "/m/04bcr3", "1", "0.2", "0.6", "0.3", "0.7", "0", "0"],
        ["img-group", "/m/01mzpv", "1", "0.1", "0.5", "0.1", "0.5", "1", "0"],
        ["img-depiction", "/m/03ssj5", "1", "0.1", "0.5", "0.1", "0.5", "0", "1"],
        ["img-other", "/m/not-furniture", "1", "0.1", "0.5", "0.1", "0.5", "0", "0"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_openimages_builder_filters_policy_and_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "validation-annotations-bbox.csv"
    digest = write_annotations(source)
    records, metadata = build_openimages_furniture_ground_truth(
        source,
        expected_sha256=digest,
        source_url="https://example.test/annotations.csv",
    )
    assert [(record.image_id, record.label) for record in records] == [
        ("img-chair", "Chair"),
        ("img-table", "Table"),
    ]
    assert metadata.source_sha256 == digest
    assert metadata.records == 2
    assert metadata.images == 2
    assert metadata.class_counts == {"Chair": 1, "Table": 1}
    assert metadata.skipped_group_of == 1
    assert metadata.skipped_depictions == 1
    assert metadata.skipped_non_target == 1


def test_openimages_builder_can_include_group_and_depictions(tmp_path: Path) -> None:
    source = tmp_path / "annotations.csv"
    write_annotations(source)
    records, metadata = build_openimages_furniture_ground_truth(
        source,
        policy=OpenImagesBenchmarkPolicy(include_group_of=True, include_depictions=True),
    )
    assert len(records) == 4
    assert metadata.skipped_group_of == 0
    assert metadata.skipped_depictions == 0


def test_openimages_builder_rejects_sha_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "annotations.csv"
    write_annotations(source)
    with pytest.raises(ValueError, match="SHA-256"):
        build_openimages_furniture_ground_truth(source, expected_sha256="0" * 64)


def test_manifest_write_is_deterministic_and_has_metadata(tmp_path: Path) -> None:
    source = tmp_path / "annotations.csv"
    write_annotations(source)
    records, metadata = build_openimages_furniture_ground_truth(source)
    manifest, meta = write_benchmark_manifest(records, metadata, tmp_path / "out")
    lines = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [line["image_id"] for line in lines] == ["img-chair", "img-table"]
    meta_payload = json.loads(meta.read_text(encoding="utf-8"))
    assert meta_payload["source_sha256"] == metadata.source_sha256
    assert meta_payload["policy"]["include_group_of"] is False
