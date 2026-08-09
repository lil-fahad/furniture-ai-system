"""Stage all training datasets locally, then upload to GCS per SPEC section 2.2.

CLI::

    python -m training.data_ingest.stage_all --bucket B \
        [--rooms-max 300] [--plans-synthetic 200] [--skip-download]

Local staging layout (under ``data/staging/``)::

    rooms/images/<class>/<id>.jpg     # Open Images furniture (ImageFolder)
    plans/images/*.png                # synthetic floor plans
    plans/masks/*.png                 # class-index masks (values 0..4)
    catalog/suppliers_master.csv.gz   # decoded supplier catalog
    manifest.json                     # dataset summary (see build_manifest)

GCS layout (SPEC section 2.2): ``gs://B/datasets/{rooms,plans,catalog}/...``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from training.data_ingest import catalog, floorplans, gcs, openimages_furniture

DEFAULT_STAGING_DIR = Path("data") / "staging"
REPO_ROOT = Path(__file__).resolve().parents[2]


def count_files(root: Path) -> int:
    """Count regular files under ``root`` (0 when missing)."""
    root = Path(root)
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file())


def build_manifest(staging_dir: Path) -> dict:
    """Build the manifest dict for a staging tree (SPEC section 3 WP-B item 6)."""
    staging_dir = Path(staging_dir)
    plans = staging_dir / "plans"
    return {
        "datasets": {
            "rooms": {"files": count_files(staging_dir / "rooms")},
            "plans": {
                "files": count_files(plans),
                "images": count_files(plans / "images"),
                "masks": count_files(plans / "masks"),
            },
            "catalog": {"files": count_files(staging_dir / "catalog")},
        },
        "created_utc": datetime.now(UTC).isoformat(),
    }


def write_manifest(staging_dir: Path) -> Path:
    """Write ``manifest.json`` into the staging dir and return its path."""
    manifest_path = Path(staging_dir) / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(build_manifest(staging_dir), indent=2) + "\n")
    return manifest_path


def stage_datasets(
    staging_dir: Path,
    repo_root: Path,
    rooms_max: int = 300,
    plans_synthetic: int = 200,
    workers: int = 8,
    skip_download: bool = False,
) -> dict[str, int]:
    """Build the local staging tree; returns per-dataset file counts."""
    staging_dir = Path(staging_dir)
    rooms_dir = staging_dir / "rooms" / "images"
    rooms_meta_dir = staging_dir / "rooms" / "_meta"
    plans_dir = staging_dir / "plans"
    catalog_dir = staging_dir / "catalog"

    if skip_download:
        print("==> --skip-download: skipping Open Images rooms download")
    else:
        print(f"==> staging rooms dataset (Open Images, max {rooms_max}/class)")
        class_names = sorted(openimages_furniture.FURNITURE_CLASSES)
        openimages_furniture.fetch_class_index(rooms_meta_dir)
        mapping = openimages_furniture.select_image_ids(class_names, rooms_max, rooms_meta_dir)
        counts = openimages_furniture.download_subset(
            mapping, rooms_dir, workers=workers, selection_dir=rooms_meta_dir
        )
        print(f"    rooms downloaded: {sum(counts.values())}")

    print(f"==> generating {plans_synthetic} synthetic floor-plan pairs")
    floorplans.generate_synthetic(plans_dir, n=plans_synthetic)

    print("==> decoding supplier catalog")
    catalog.prepare_catalog(catalog_dir, repo_root)

    return {
        "rooms": count_files(staging_dir / "rooms"),
        "plans": count_files(plans_dir),
        "catalog": count_files(catalog_dir),
    }


def upload_staging(staging_dir: Path, bucket: str) -> dict[str, int]:
    """Upload the staging tree to ``gs://bucket/datasets/...``; returns counts."""
    staging_dir = Path(staging_dir)
    uploaded: dict[str, int] = {}
    for name in ("rooms", "plans", "catalog"):
        local = staging_dir / name
        if not local.is_dir():
            continue
        print(f"==> uploading {local} -> gs://{bucket}/datasets/{name}/")
        uploaded[name] = gcs.upload_dir(local, bucket, f"datasets/{name}")
    manifest_path = staging_dir / "manifest.json"
    if manifest_path.is_file():
        manifest_dir = staging_dir / "_manifest_upload"
        manifest_dir.mkdir(exist_ok=True)
        staged = manifest_dir / "manifest.json"
        staged.write_bytes(manifest_path.read_bytes())
        uploaded["manifest"] = gcs.upload_dir(manifest_dir, bucket, "datasets")
    return uploaded


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m training.data_ingest.stage_all",
        description="Stage all training datasets locally and upload to GCS.",
    )
    parser.add_argument("--bucket", required=True, help="GCS bucket name (no gs://)")
    parser.add_argument("--rooms-max", type=int, default=300, help="Max room images per class")
    parser.add_argument(
        "--plans-synthetic", type=int, default=200, help="Synthetic floor-plan pairs"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip network downloads (rooms); synthetic + catalog still staged",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel download workers")
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=DEFAULT_STAGING_DIR,
        help="Local staging directory (default: data/staging)",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root containing data/suppliers_master.csv.gz.b64",
    )
    args = parser.parse_args(argv)

    counts = stage_datasets(
        args.staging_dir,
        args.repo_root,
        rooms_max=args.rooms_max,
        plans_synthetic=args.plans_synthetic,
        workers=args.workers,
        skip_download=args.skip_download,
    )
    manifest_path = write_manifest(args.staging_dir)
    print(f"==> wrote manifest {manifest_path}: {counts}")

    if not gcs.gcs_available():
        print(
            "warning: google-cloud-storage not installed; staged locally only "
            f"(install it and re-run to upload to gs://{args.bucket}/datasets/)",
            file=sys.stderr,
        )
        return 0
    uploaded = upload_staging(args.staging_dir, args.bucket)
    print(f"==> upload complete: {uploaded}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
