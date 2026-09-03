from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from furniture_ai.evaluation.detection import GroundTruthRecord, NormalizedBox

# Keep the benchmark class mapping aligned with the existing Open Images
# furniture ingestion pipeline. Open Images calls the Sofa MID "Couch".
FURNITURE_MIDS: dict[str, str] = {
    "Chair": "/m/01mzpv",
    "Table": "/m/04bcr3",
    "Sofa": "/m/02crq1",
    "Bed": "/m/03ssj5",
    "Cabinetry": "/m/01s105",
    "Desk": "/m/01y9k5",
    "Shelf": "/m/0dt3t",
}
MID_TO_FURNITURE = {mid: name for name, mid in FURNITURE_MIDS.items()}

# Open Images V6 reuses the V5 validation/test box files on the official
# download surface. This URL is metadata only; callers provide/download the
# local CSV explicitly so benchmark creation has a verifiable local SHA-256.
OPENIMAGES_VALIDATION_BBOX_URL = (
    "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
)

_REQUIRED_COLUMNS = {
    "ImageID",
    "LabelName",
    "Confidence",
    "XMin",
    "XMax",
    "YMin",
    "YMax",
    "IsGroupOf",
    "IsDepiction",
}


@dataclass(frozen=True)
class OpenImagesBenchmarkPolicy:
    include_group_of: bool = False
    include_depictions: bool = False


@dataclass(frozen=True)
class OpenImagesBenchmarkMetadata:
    schema_version: str
    source_file: str
    source_url: str | None
    source_sha256: str
    generated_at: str
    policy: OpenImagesBenchmarkPolicy
    records: int
    images: int
    class_counts: dict[str, int]
    skipped_group_of: int
    skipped_depictions: int
    skipped_non_target: int
    skipped_invalid: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flag(row: dict[str, str], name: str) -> bool:
    return row.get(name, "0").strip() == "1"


def build_openimages_furniture_ground_truth(
    annotation_csv: Path,
    *,
    policy: OpenImagesBenchmarkPolicy | None = None,
    source_url: str | None = None,
    expected_sha256: str | None = None,
) -> tuple[list[GroundTruthRecord], OpenImagesBenchmarkMetadata]:
    """Filter Open Images box annotations into a reproducible furniture benchmark.

    The source CSV is never silently downloaded. Callers can pin an expected
    SHA-256, and the exact observed source hash is always recorded in metadata.
    """
    source = Path(annotation_csv)
    if not source.is_file():
        raise FileNotFoundError(source)
    active_policy = policy or OpenImagesBenchmarkPolicy()
    source_hash = sha256_file(source)
    if expected_sha256 and source_hash.lower() != expected_sha256.lower():
        raise ValueError("Open Images annotation SHA-256 does not match expected value")

    records: list[GroundTruthRecord] = []
    images: set[str] = set()
    class_counts: Counter[str] = Counter()
    skipped_group_of = 0
    skipped_depictions = 0
    skipped_non_target = 0
    skipped_invalid = 0

    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(_REQUIRED_COLUMNS - fields)
        if missing:
            raise ValueError(f"Open Images box CSV is missing required columns: {missing}")

        for row in reader:
            label = MID_TO_FURNITURE.get(row["LabelName"].strip())
            if label is None or row["Confidence"].strip() != "1":
                skipped_non_target += 1
                continue
            if _flag(row, "IsGroupOf") and not active_policy.include_group_of:
                skipped_group_of += 1
                continue
            if _flag(row, "IsDepiction") and not active_policy.include_depictions:
                skipped_depictions += 1
                continue
            try:
                box = NormalizedBox(
                    x_min=float(row["XMin"]),
                    y_min=float(row["YMin"]),
                    x_max=float(row["XMax"]),
                    y_max=float(row["YMax"]),
                )
                image_id = row["ImageID"].strip()
                record = GroundTruthRecord(image_id=image_id, label=label, box=box)
            except (TypeError, ValueError):
                skipped_invalid += 1
                continue
            records.append(record)
            images.add(record.image_id)
            class_counts[label] += 1

    metadata = OpenImagesBenchmarkMetadata(
        schema_version="1.0",
        source_file=source.name,
        source_url=source_url,
        source_sha256=source_hash,
        generated_at=datetime.now(timezone.utc).isoformat(),
        policy=active_policy,
        records=len(records),
        images=len(images),
        class_counts=dict(sorted(class_counts.items())),
        skipped_group_of=skipped_group_of,
        skipped_depictions=skipped_depictions,
        skipped_non_target=skipped_non_target,
        skipped_invalid=skipped_invalid,
    )
    return records, metadata


def write_benchmark_manifest(
    records: list[GroundTruthRecord],
    metadata: OpenImagesBenchmarkMetadata,
    output_dir: Path,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "ground_truth.jsonl"
    meta = destination / "metadata.json"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in sorted(
            records,
            key=lambda item: (
                item.image_id,
                item.label,
                item.box.x_min,
                item.box.y_min,
            ),
        ):
            payload = {
                "image_id": record.image_id,
                "label": record.label,
                "box": asdict(record.box),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    meta.write_text(
        json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, meta
