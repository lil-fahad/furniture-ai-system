from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CLOUD_JOB = ROOT / "cloud" / "openimages_vertex_job.py"


def load_vertex_module():
    spec = importlib.util.spec_from_file_location("openimages_vertex_job", CLOUD_JOB)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vertex = load_vertex_module()


def test_bucket_and_run_id_validation() -> None:
    assert (
        vertex.normalize_bucket_name("gs://licensed-style-data-123/") == "licensed-style-data-123"
    )
    assert vertex.validate_run_id("style-run-001") == "style-run-001"
    with pytest.raises(ValueError, match="bucket"):
        vertex.normalize_bucket_name("gs://Bad Bucket")
    with pytest.raises(ValueError, match="run-id"):
        vertex.validate_run_id("../../unsafe")


def test_only_allowlisted_open_images_licenses_are_accepted() -> None:
    assert vertex.canonical_license("http://creativecommons.org/licenses/by/2.0/") == (
        "https://creativecommons.org/licenses/by/2.0/"
    )
    assert vertex.canonical_license("https://creativecommons.org/publicdomain/zero/1.0/") == (
        "https://creativecommons.org/publicdomain/zero/1.0/"
    )
    assert vertex.canonical_license("https://creativecommons.org/licenses/by-sa/2.0/") is None
    assert vertex.canonical_license("") is None


def test_candidate_filter_requires_two_distinct_physical_furniture_classes() -> None:
    class_ids = {"/chair": 0, "/table": 1}
    rows = [
        {"ImageID": "0000000000000001", "LabelName": "/chair", "IsDepiction": "0"},
        {"ImageID": "0000000000000001", "LabelName": "/table", "IsDepiction": "0"},
        {"ImageID": "0000000000000002", "LabelName": "/chair", "IsDepiction": "0"},
        {"ImageID": "0000000000000002", "LabelName": "/chair", "IsDepiction": "0"},
        {"ImageID": "0000000000000003", "LabelName": "/chair", "IsDepiction": "1"},
        {"ImageID": "0000000000000003", "LabelName": "/table", "IsDepiction": "0"},
    ]
    assert vertex.candidate_ids_from_rows(rows, class_ids, min_distinct_classes=2, min_boxes=2) == {
        "0000000000000001"
    }


def test_metadata_selection_keeps_attribution_and_is_deterministic() -> None:
    candidate_ids = {f"{index:016x}" for index in range(1, 6)}
    rows = [
        {
            "ImageID": image_id,
            "License": "https://creativecommons.org/licenses/by/2.0/",
            "OriginalLandingURL": f"https://example.test/{image_id}",
            "Author": "Example Author",
            "Title": "Example title",
        }
        for image_id in sorted(candidate_ids)
    ]
    first = vertex.reservoir_select_metadata(rows, candidate_ids, limit=3, seed=42)
    second = vertex.reservoir_select_metadata(rows, candidate_ids, limit=3, seed=42)
    assert first == second
    assert len(first) == 3
    assert all(record.author == "Example Author" for record in first)


def test_normalized_jpeg_is_rgb_and_bounded() -> None:
    source = io.BytesIO()
    Image.new("RGBA", (1600, 800), (10, 20, 30, 128)).save(source, format="PNG")
    payload = vertex.normalized_jpeg(source.getvalue())
    with Image.open(io.BytesIO(payload)) as result:
        assert result.format == "JPEG"
        assert result.mode == "RGB"
        assert max(result.size) <= vertex.OUTPUT_MAX_EDGE


def test_dry_run_plan_needs_no_cloud_or_model_dependencies(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = vertex.parse_args(
        [
            "--bucket",
            "licensed-style-data-123",
            "--run-id",
            "style-run-001",
            "--max-images",
            "100",
            "--work-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )
    assert vertex.run(args) == 0
    output = capsys.readouterr().out
    assert '"target_images": 100' in output
    assert vertex.SIGLIP_REVISION in output


def test_manifest_rejects_duplicate_or_unlicensed_records(tmp_path: Path) -> None:
    valid = {
        "image_id": "0000000000000001",
        "style": "minimalist",
        "license": "https://creativecommons.org/licenses/by/2.0/",
        "weak_label_revision": vertex.SIGLIP_REVISION,
        "sha256": "a" * 64,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(valid) + "\n" + json.dumps(valid) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="line 2"):
        vertex.load_manifest(manifest)

    invalid_license = {**valid, "license": "https://example.test/no-license"}
    manifest.write_text(json.dumps(invalid_license) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        vertex.load_manifest(manifest)


def test_cloud_launcher_dry_plan_does_not_need_gcloud() -> None:
    result = subprocess.run(
        [
            "bash",
            "cloud/launch_gcp_training.sh",
            "--project",
            "round-office-505007-q4",
            "--run-id",
            "style-test-001",
            "--max-images",
            "100000",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Dry plan only" in result.stdout
    assert "gpu=NVIDIA_L4" in result.stdout
