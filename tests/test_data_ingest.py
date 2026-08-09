"""Offline tests for training/data_ingest (WP-B).

Everything here runs WITHOUT network access and WITHOUT torch. Any test that
would touch the network is marked ``skipif(True)`` with an explicit comment.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # namespace-package import of training/

from training.data_ingest import catalog, floorplans, gcs, stage_all  # noqa: E402

VALID_MASK_VALUES = set(range(floorplans.NUM_CLASSES))


def _read_png_values(path: Path) -> set[int]:
    from PIL import Image

    with Image.open(path) as image:
        return set(image.getdata())


def test_synthetic_floorplans_paired_and_valid(tmp_path: Path) -> None:
    summary = floorplans.generate_synthetic(tmp_path, n=5, seed=7, size=64)
    assert summary == {"pairs": 5, "images": 5, "masks": 5}

    images = sorted((tmp_path / "images").glob("*.png"))
    masks = sorted((tmp_path / "masks").glob("*.png"))
    assert len(images) == len(masks) == 5
    assert [p.stem for p in images] == [p.stem for p in masks]  # paired by stem

    for mask_path in masks:
        values = _read_png_values(mask_path)
        assert values <= VALID_MASK_VALUES
        assert len(values) > 1  # not a degenerate single-class mask
        # Walls should always be present in a generated plan.
        assert floorplans.CLASS_WALL in values


def test_synthetic_floorplans_deterministic_with_seed(tmp_path: Path) -> None:
    floorplans.generate_synthetic(tmp_path / "a", n=3, seed=7, size=64)
    floorplans.generate_synthetic(tmp_path / "b", n=3, seed=7, size=64)
    floorplans.generate_synthetic(tmp_path / "c", n=3, seed=8, size=64)

    for name in ("images", "masks"):
        for file_a in sorted((tmp_path / "a" / name).glob("*.png")):
            file_b = tmp_path / "b" / name / file_a.name
            assert file_a.read_bytes() == file_b.read_bytes()
        first_a = sorted((tmp_path / "a" / name).glob("*.png"))[0].read_bytes()
        first_c = sorted((tmp_path / "c" / name).glob("*.png"))[0].read_bytes()
        assert first_a != first_c  # different seed -> different plan


def test_prepare_catalog_decodes_real_repo_file(tmp_path: Path) -> None:
    source = REPO_ROOT / "data" / "suppliers_master.csv.gz.b64"
    assert source.is_file(), "repo must ship data/suppliers_master.csv.gz.b64"

    gz_path = catalog.prepare_catalog(tmp_path, REPO_ROOT)
    assert gz_path == tmp_path / "suppliers_master.csv.gz"
    assert gz_path.read_bytes().startswith(b"\x1f\x8b")  # gzip magic

    csv_path = tmp_path / "suppliers_master.csv"
    assert csv_path.is_file()
    with gzip.open(gz_path, "rt", encoding="utf-8-sig") as handle:
        reader = __import__("csv").reader(handle)
        header = next(reader)
        first_row = next(reader)
    assert header[0] == "Supplier Name"
    assert len(header) > 10 and len(first_row) == len(header)
    assert csv_path.read_bytes() == gzip.decompress(gz_path.read_bytes())


def test_prepare_catalog_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        catalog.prepare_catalog(tmp_path / "out", tmp_path)


def test_manifest_schema_keys(tmp_path: Path) -> None:
    # Fake a staging tree without any network work.
    (tmp_path / "rooms" / "images" / "Chair").mkdir(parents=True)
    (tmp_path / "rooms" / "images" / "Chair" / "a.jpg").write_bytes(b"x")
    floorplans.generate_synthetic(tmp_path / "plans", n=2, seed=7, size=64)
    (tmp_path / "catalog").mkdir()
    (tmp_path / "catalog" / "suppliers_master.csv.gz").write_bytes(b"\x1f\x8bfake")

    manifest_path = stage_all.write_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text())

    assert set(manifest) == {"datasets", "created_utc"}
    assert set(manifest["datasets"]) == {"rooms", "plans", "catalog"}
    assert manifest["datasets"]["rooms"]["files"] == 1
    assert manifest["datasets"]["plans"]["images"] == 2
    assert manifest["datasets"]["plans"]["masks"] == 2
    assert manifest["datasets"]["catalog"]["files"] == 1
    assert isinstance(manifest["created_utc"], str) and manifest["created_utc"]


def test_gcs_available_is_bool_and_lazy() -> None:
    # Must not raise even when google-cloud-storage is absent.
    assert isinstance(gcs.gcs_available(), bool)


def test_openimages_furniture_classes_are_mids() -> None:
    from training.data_ingest import openimages_furniture

    expected = {"Chair", "Table", "Sofa", "Bed", "Cabinetry", "Desk", "Shelf"}
    assert expected <= set(openimages_furniture.FURNITURE_CLASSES)
    for name, mid in openimages_furniture.FURNITURE_CLASSES.items():
        assert mid.startswith("/m/"), f"{name} -> {mid} is not an Open Images MID"


def test_download_subset_skips_ids_without_urls(tmp_path: Path) -> None:
    from training.data_ingest import openimages_furniture

    # No selection.csv sidecar -> clear error, no network access attempted.
    with pytest.raises(FileNotFoundError, match="select_image_ids"):
        openimages_furniture.download_subset({"Chair": ["deadbeef"]}, tmp_path)

    # Sidecar without a usable URL -> id is skipped offline.
    (tmp_path / openimages_furniture.SELECTION_FILENAME).write_text(
        "class_name,image_id,url\nChair,deadbeef,\n", encoding="utf-8"
    )
    counts = openimages_furniture.download_subset({"Chair": ["deadbeef"]}, tmp_path)
    assert counts == {"Chair": 0}


@pytest.mark.skipif(True, reason="network test: never hit Open Images endpoints in CI")
def test_fetch_class_index_network(tmp_path: Path) -> None:  # pragma: no cover
    from training.data_ingest import openimages_furniture

    path = openimages_furniture.fetch_class_index(tmp_path)
    assert path.is_file()
