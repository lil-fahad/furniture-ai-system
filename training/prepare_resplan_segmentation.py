from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
from shapely import affinity, wkt
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

SAFE_SCHEMA = "furnitureai-resplan-wkt-v1"
ROOM_KEYS = ("living", "bedroom", "bathroom", "kitchen", "storage", "stair", "balcony")
CLASS_MAP = {
    0: "background",
    1: "room",
    2: "wall",
    3: "door",
    4: "window",
}
MAX_JSONL_LINE_BYTES = 5 * 1024 * 1024


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_geometry(value: object, *, field: str) -> BaseGeometry | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 4_000_000:
        raise ValueError(f"Geometry field {field!r} must be a bounded WKT string")
    geometry = wkt.loads(value)
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if geometry.is_empty:
        return None
    return geometry


def load_safe_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if len(raw_line) > MAX_JSONL_LINE_BYTES:
                raise ValueError(f"JSONL line {line_number} exceeds the safety limit")
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            if not isinstance(payload, dict) or payload.get("schema") != SAFE_SCHEMA:
                raise ValueError(f"Invalid safe-export record on line {line_number}")
            if payload.get("split") not in {
                "train",
                "validation",
                "test",
                "augmented",
                "unassigned",
            }:
                raise ValueError(f"Unknown split on line {line_number}: {payload.get('split')!r}")
            geometries = payload.get("geometries")
            if not isinstance(geometries, dict):
                raise ValueError(f"Missing geometries object on line {line_number}")
            records.append(payload)
    if not records:
        raise ValueError("Safe ResPlan export contains no records")
    return records


def parse_geometries(record: dict[str, Any]) -> dict[str, BaseGeometry]:
    geometries: dict[str, BaseGeometry] = {}
    for key, value in record["geometries"].items():
        if key not in {*ROOM_KEYS, "inner", "wall", "door", "front_door", "window"}:
            continue
        geometry = _safe_geometry(value, field=key)
        if geometry is not None:
            geometries[key] = geometry
    if not geometries:
        raise ValueError(f"Plan {record.get('plan_id')} contains no usable geometry")
    return geometries


def _combined_bounds(geometries: dict[str, BaseGeometry]) -> tuple[float, float, float, float]:
    preferred = geometries.get("inner")
    if preferred is not None and not preferred.is_empty:
        bounds = preferred.bounds
    else:
        bounds = unary_union(list(geometries.values())).bounds
    min_x, min_y, max_x, max_y = (float(value) for value in bounds)
    if not np.isfinite([min_x, min_y, max_x, max_y]).all():
        raise ValueError("Plan geometry contains non-finite bounds")
    if max_x <= min_x or max_y <= min_y:
        raise ValueError("Plan geometry has degenerate bounds")
    return min_x, min_y, max_x, max_y


def normalize_geometries(
    geometries: dict[str, BaseGeometry],
    *,
    size: int,
    padding_ratio: float,
) -> dict[str, BaseGeometry]:
    min_x, min_y, max_x, max_y = _combined_bounds(geometries)
    usable = size * (1 - 2 * padding_ratio)
    scale = usable / max(max_x - min_x, max_y - min_y)
    scaled_width = (max_x - min_x) * scale
    scaled_height = (max_y - min_y) * scale
    offset_x = (size - scaled_width) / 2
    offset_y = (size - scaled_height) / 2

    result: dict[str, BaseGeometry] = {}
    for key, geometry in geometries.items():
        moved = affinity.translate(geometry, xoff=-min_x, yoff=-min_y)
        moved = affinity.scale(moved, xfact=scale, yfact=-scale, origin=(0, 0))
        moved = affinity.translate(moved, xoff=offset_x, yoff=size - offset_y)
        result[key] = moved
    return result


def _iter_geometries(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
    if isinstance(geometry, (MultiPolygon, MultiLineString, GeometryCollection)):
        for item in geometry.geoms:
            yield from _iter_geometries(item)
    else:
        yield geometry


def _points(coords: Iterable[tuple[float, float]]) -> np.ndarray:
    return np.asarray([(round(x), round(y)) for x, y in coords], dtype=np.int32)


def draw_geometry(
    canvas: np.ndarray,
    geometry: BaseGeometry,
    value: int,
    *,
    line_thickness: int,
) -> None:
    for item in _iter_geometries(geometry):
        if isinstance(item, Polygon):
            exterior = _points(item.exterior.coords)
            cv2.fillPoly(canvas, [exterior], value)
            for interior in item.interiors:
                cv2.fillPoly(canvas, [_points(interior.coords)], 0)
        elif isinstance(item, LineString):
            cv2.polylines(
                canvas,
                [_points(item.coords)],
                isClosed=False,
                color=value,
                thickness=max(1, line_thickness),
                lineType=cv2.LINE_8,
            )
        elif isinstance(item, Point):
            cv2.circle(
                canvas,
                (round(item.x), round(item.y)),
                max(1, line_thickness),
                value,
                -1,
                lineType=cv2.LINE_8,
            )


def render_pair(
    geometries: dict[str, BaseGeometry],
    *,
    size: int,
    padding_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = normalize_geometries(geometries, size=size, padding_ratio=padding_ratio)
    mask = np.zeros((size, size), dtype=np.uint8)

    room_parts = [normalized[key] for key in ROOM_KEYS if key in normalized]
    if room_parts:
        draw_geometry(mask, unary_union(room_parts), 1, line_thickness=1)
    if "wall" in normalized:
        draw_geometry(mask, normalized["wall"], 2, line_thickness=max(2, size // 128))
    door_parts = [normalized[key] for key in ("door", "front_door") if key in normalized]
    if door_parts:
        draw_geometry(mask, unary_union(door_parts), 3, line_thickness=max(2, size // 128))
    if "window" in normalized:
        draw_geometry(mask, normalized["window"], 4, line_thickness=max(2, size // 128))

    # Render a neutral architectural drawing rather than semantic class colors.
    image = np.full((size, size), 255, dtype=np.uint8)
    room_region = mask == 1
    image[room_region] = 245
    image[mask == 2] = 25
    image[mask == 3] = 95
    image[mask == 4] = 150
    return image, mask


def canonical_geometry_signature(mask: np.ndarray, *, coarse_size: int = 32) -> str:
    if mask.ndim != 2:
        raise ValueError("Geometry signature expects a 2D class mask")
    coarse = cv2.resize(mask, (coarse_size, coarse_size), interpolation=cv2.INTER_NEAREST)
    variants: list[bytes] = []
    for rotations in range(4):
        rotated = np.rot90(coarse, rotations)
        variants.append(np.ascontiguousarray(rotated).tobytes())
        variants.append(np.ascontiguousarray(np.fliplr(rotated)).tobytes())
    return sha256_bytes(min(variants))


def sample_name(record: dict[str, Any]) -> str:
    source_index = int(record.get("source_index", 0))
    plan_id = str(record.get("plan_id", source_index))
    token = hashlib.sha256(plan_id.encode("utf-8")).hexdigest()[:10]
    return f"{source_index:06d}-{token}"


def prepare_output(output: Path, force: bool) -> None:
    if output.exists() and any(output.iterdir()):
        if not force:
            raise FileExistsError(f"Output directory is not empty: {output}; use --force")
        shutil.rmtree(output)
    for split in ("train", "validation", "test"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "masks" / split).mkdir(parents=True, exist_ok=True)


def save_png(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path, format="PNG", optimize=True)


def process_records(
    records: list[dict[str, Any]],
    output: Path,
    *,
    size: int,
    padding_ratio: float,
    include_augmented: bool,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    split_order = {"train": 0, "validation": 1, "test": 2, "augmented": 3, "unassigned": 4}
    records = sorted(
        records,
        key=lambda record: (
            split_order[str(record.get("split", "unassigned"))],
            int(record.get("source_index", 0)),
        ),
    )
    accepted_signatures: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    train_signatures: set[str] = set()
    samples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for record in records:
        source_split = str(record.get("split"))
        if source_split == "augmented":
            if not include_augmented:
                skipped["augmented_excluded"] += 1
                continue
            split = "train"
        elif source_split in {"train", "validation", "test"}:
            split = source_split
        else:
            skipped["unassigned"] += 1
            continue

        try:
            geometries = parse_geometries(record)
            image, mask = render_pair(
                geometries,
                size=size,
                padding_ratio=padding_ratio,
            )
        except (TypeError, ValueError):
            skipped["invalid_geometry"] += 1
            continue
        if not np.any(mask):
            skipped["empty_mask"] += 1
            continue
        signature = canonical_geometry_signature(mask)
        if signature in accepted_signatures[split]:
            skipped[f"duplicate_within_{split}"] += 1
            continue
        if split in {"validation", "test"} and signature in train_signatures:
            skipped[f"train_leakage_{split}"] += 1
            continue

        name = sample_name(record)
        image_path = output / "images" / split / f"{name}.png"
        mask_path = output / "masks" / split / f"{name}.png"
        save_png(image, image_path)
        save_png(mask, mask_path)
        image_sha = sha256_file(image_path)
        mask_sha = sha256_file(mask_path)
        sample = {
            "plan_id": str(record.get("plan_id")),
            "source_index": int(record.get("source_index", 0)),
            "source_split": source_split,
            "split": split,
            "geometry_signature": signature,
            "image": str(image_path.relative_to(output)),
            "mask": str(mask_path.relative_to(output)),
            "image_sha256": image_sha,
            "mask_sha256": mask_sha,
        }
        samples.append(sample)
        counts[split] += 1
        accepted_signatures[split].add(signature)
        if split == "train":
            train_signatures.add(signature)

    if counts["train"] == 0 or counts["validation"] == 0 or counts["test"] == 0:
        raise ValueError(
            "Prepared dataset must retain at least one train, validation, and test sample"
        )
    return samples, counts, skipped


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("safe_jsonl", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/resplan/segmentation"))
    parser.add_argument("--source-config", type=Path, default=Path("config/resplan_source.json"))
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--include-augmented", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.size < 64 or args.size > 4096:
        raise ValueError("--size must be between 64 and 4096")
    if not 0 <= args.padding_ratio < 0.25:
        raise ValueError("--padding-ratio must be in [0, 0.25)")
    if not args.safe_jsonl.is_file():
        raise FileNotFoundError(args.safe_jsonl)
    if not args.source_config.is_file():
        raise FileNotFoundError(args.source_config)

    source = json.loads(args.source_config.read_text(encoding="utf-8"))
    if source.get("safe_export_schema") != SAFE_SCHEMA:
        raise ValueError("ResPlan source config safe-export schema does not match this preparer")
    records = load_safe_records(args.safe_jsonl)
    prepare_output(args.output, args.force)
    samples, counts, skipped = process_records(
        records,
        args.output,
        size=args.size,
        padding_ratio=args.padding_ratio,
        include_augmented=args.include_augmented,
    )
    write_jsonl(samples, args.output / "samples.jsonl")
    manifest = {
        "schema_version": "1.0",
        "dataset": "FurnitureAI ResPlan-derived segmentation",
        "source_repository": source["source_repository"],
        "source_revision": source["revision"],
        "source_license": source["data_license"],
        "safe_export_sha256": sha256_file(args.safe_jsonl),
        "image_size": args.size,
        "padding_ratio": args.padding_ratio,
        "class_map": {str(key): value for key, value in CLASS_MAP.items()},
        "counts": dict(sorted(counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "include_augmented_as_training_only": args.include_augmented,
        "leakage_filter": "canonical 32x32 semantic geometry signature (dihedral-invariant)",
        "limitations": [
            "The leakage signature is a coarse heuristic and is not a full near-duplicate detector.",
            "Inputs are deterministic vector-derived raster drawings, not photographs or scanned plans.",
            "Use a separate real-raster benchmark before claiming production generalization.",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"prepared={args.output} train={counts['train']} validation={counts['validation']} "
        f"test={counts['test']} skipped={sum(skipped.values())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
