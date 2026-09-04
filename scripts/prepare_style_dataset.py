#!/usr/bin/env python3
"""Build leakage-resistant train/validation/test splits from a style ImageFolder.

The source directory is never modified. Images are decoded, hashed, checked for
exact and perceptual duplicates, and then split deterministically by class.
Conflicting duplicate clusters are quarantined instead of being silently used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_PIXELS = 40_000_000


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative_path: str
    label: str
    sha256: str
    perceptual_hash: int
    width: int
    height: int
    mean_rgb: tuple[float, float, float]

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


@dataclass
class _BKNode:
    value: int
    indices: list[int]
    children: dict[int, int]


class BKTree:
    """BK-tree over integer perceptual hashes using Hamming distance."""

    def __init__(self) -> None:
        self.nodes: list[_BKNode] = []

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, index: int) -> None:
        if not self.nodes:
            self.nodes.append(_BKNode(value=value, indices=[index], children={}))
            return
        node_index = 0
        while True:
            node = self.nodes[node_index]
            distance = self.distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child_index = node.children.get(distance)
            if child_index is None:
                node.children[distance] = len(self.nodes)
                self.nodes.append(_BKNode(value=value, indices=[index], children={}))
                return
            node_index = child_index

    def query(self, value: int, max_distance: int) -> Iterable[int]:
        if not self.nodes:
            return ()
        matches: list[int] = []
        stack = [0]
        while stack:
            node = self.nodes[stack.pop()]
            distance = self.distance(value, node.value)
            if distance <= max_distance:
                matches.extend(node.indices)
            lower = distance - max_distance
            upper = distance + max_distance
            for edge, child in node.children.items():
                if lower <= edge <= upper:
                    stack.append(child)
        return matches


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference_bits(image: Image.Image, horizontal: bool) -> int:
    size = (9, 8) if horizontal else (8, 9)
    pixels = image.resize(size, Image.Resampling.LANCZOS).tobytes()
    width, height = size
    result = 0
    bit = 0
    if horizontal:
        for y in range(height):
            offset = y * width
            for x in range(width - 1):
                if pixels[offset + x] > pixels[offset + x + 1]:
                    result |= 1 << bit
                bit += 1
    else:
        for y in range(height - 1):
            top = y * width
            bottom = (y + 1) * width
            for x in range(width):
                if pixels[top + x] > pixels[bottom + x]:
                    result |= 1 << bit
                bit += 1
    return result


def perceptual_hash(path: Path) -> tuple[int, int, int, tuple[float, float, float]]:
    try:
        with Image.open(path) as source:
            if source.width * source.height > MAX_PIXELS:
                raise ValueError(f"image exceeds {MAX_PIXELS:,} pixels")
            source.load()
            normalized = ImageOps.exif_transpose(source).convert("RGB")
            image = normalized.convert("L")
            horizontal = _difference_bits(image, horizontal=True)
            vertical = _difference_bits(image, horizontal=False)
            color_sample = normalized.resize((16, 16), Image.Resampling.LANCZOS)
            mean_rgb = tuple(float(value) for value in ImageStat.Stat(color_sample).mean)
            return (horizontal << 64) | vertical, image.width, image.height, mean_rgb
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid image: {exc}") from exc


def discover(root: Path) -> list[tuple[Path, str]]:
    if not root.is_dir():
        raise ValueError(f"input root does not exist: {root}")
    discovered: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                discovered.append((path, class_dir.name))
    if not discovered:
        raise ValueError("no supported images found beneath class directories")
    return discovered


def scan_images(root: Path) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    records: list[ImageRecord] = []
    rejected: list[dict[str, str]] = []
    for path, label in discover(root):
        relative = path.relative_to(root).as_posix()
        try:
            digest = sha256_file(path)
            image_hash, width, height, mean_rgb = perceptual_hash(path)
        except ValueError as exc:
            rejected.append({"path": relative, "label": label, "reason": str(exc)})
            continue
        records.append(
            ImageRecord(
                path=path,
                relative_path=relative,
                label=label,
                sha256=digest,
                perceptual_hash=image_hash,
                width=width,
                height=height,
                mean_rgb=mean_rgb,
            )
        )
    if not records:
        raise ValueError("all discovered images were rejected")
    return records, rejected


def group_duplicates(
    records: list[ImageRecord],
    max_hamming_distance: int,
    aspect_tolerance: float,
    color_tolerance: float,
) -> tuple[list[list[int]], int, int]:
    groups = DisjointSet(len(records))
    exact: dict[str, int] = {}
    exact_links = 0
    near_links = 0
    tree = BKTree()

    for index, record in enumerate(records):
        previous = exact.get(record.sha256)
        if previous is None:
            exact[record.sha256] = index
        else:
            groups.union(index, previous)
            exact_links += 1

        for candidate_index in tree.query(record.perceptual_hash, max_hamming_distance):
            candidate = records[candidate_index]
            if record.sha256 == candidate.sha256:
                continue
            ratio_delta = abs(record.aspect_ratio - candidate.aspect_ratio)
            ratio_scale = max(record.aspect_ratio, candidate.aspect_ratio, 1e-12)
            color_delta = max(
                abs(left - right)
                for left, right in zip(record.mean_rgb, candidate.mean_rgb, strict=True)
            )
            if ratio_delta / ratio_scale <= aspect_tolerance and color_delta <= color_tolerance:
                before = groups.find(index) != groups.find(candidate_index)
                groups.union(index, candidate_index)
                if before:
                    near_links += 1
        tree.add(record.perceptual_hash, index)

    clustered: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        clustered[groups.find(index)].append(index)
    ordered = sorted(
        (
            sorted(indices, key=lambda item: records[item].relative_path)
            for indices in clustered.values()
        ),
        key=lambda indices: records[indices[0]].relative_path,
    )
    return ordered, exact_links, near_links


def classify_clusters(
    records: list[ImageRecord], clusters: list[list[int]]
) -> tuple[list[int], dict[int, str], list[dict[str, object]]]:
    canonical_indices: list[int] = []
    duplicate_of: dict[int, str] = {}
    conflicts: list[dict[str, object]] = []
    for cluster in clusters:
        labels = sorted({records[index].label for index in cluster})
        paths = [records[index].relative_path for index in cluster]
        if len(labels) != 1:
            conflicts.append({"labels": labels, "paths": paths})
            continue
        canonical = cluster[0]
        canonical_indices.append(canonical)
        canonical_path = records[canonical].relative_path
        for duplicate in cluster[1:]:
            duplicate_of[duplicate] = canonical_path
    return canonical_indices, duplicate_of, conflicts


def split_indices(
    records: list[ImageRecord],
    canonical_indices: list[int],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[int, str]:
    by_class: dict[str, list[int]] = defaultdict(list)
    for index in canonical_indices:
        by_class[records[index].label].append(index)

    too_small = sorted(label for label, values in by_class.items() if len(values) < 3)
    if too_small:
        raise ValueError(
            "every class needs at least three unique images for train/validation/test; "
            f"too small: {too_small}"
        )

    assignments: dict[int, str] = {}
    rng = random.Random(seed)
    for label in sorted(by_class):
        values = sorted(by_class[label], key=lambda item: records[item].relative_path)
        rng.shuffle(values)
        count = len(values)
        test_count = max(1, round(count * test_fraction))
        validation_count = max(1, round(count * validation_fraction))
        while test_count + validation_count >= count:
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise ValueError(f"class {label!r} is too small for requested fractions")
        for index in values[:test_count]:
            assignments[index] = "test"
        for index in values[test_count : test_count + validation_count]:
            assignments[index] = "validation"
        for index in values[test_count + validation_count :]:
            assignments[index] = "train"
    return assignments


def materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)


def load_source_manifest(path: Path | None) -> dict[str, dict[str, object]]:
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"source manifest does not exist: {path}")
    records: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in source manifest line {line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"source manifest line {line_number} is not an object")
        digest = row.get("sha256")
        if isinstance(digest, str) and len(digest) == 64:
            records[digest.lower()] = row
    return records


def dataset_fingerprint(records: list[ImageRecord], assignments: dict[int, str]) -> str:
    digest = hashlib.sha256()
    for index in sorted(assignments, key=lambda item: records[item].relative_path):
        record = records[index]
        digest.update(f"{record.sha256}|{record.label}|{assignments[index]}\n".encode())
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def prepare(args: argparse.Namespace) -> dict[str, object]:
    source_manifest = load_source_manifest(args.source_manifest)
    records, rejected = scan_images(args.input)
    clusters, exact_links, near_links = group_duplicates(
        records, args.max_hamming_distance, args.aspect_tolerance, args.color_tolerance
    )
    canonical, duplicate_of, conflicts = classify_clusters(records, clusters)
    review_required_indices = {
        index
        for index in canonical
        if not args.include_review_required
        and bool(source_manifest.get(records[index].sha256.lower(), {}).get("review_required"))
    }
    split_candidates = [index for index in canonical if index not in review_required_indices]
    assignments = split_indices(
        records,
        split_candidates,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    conflict_paths = {path for conflict in conflicts for path in conflict["paths"]}
    rows: list[dict[str, object]] = []
    source_matches = 0
    split_counts: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)

    if args.output.exists() and not args.dry_run:
        shutil.rmtree(args.output)
    for index, record in enumerate(records):
        if record.relative_path in conflict_paths:
            status = "label_conflict"
            split = None
        elif index in duplicate_of:
            status = "duplicate"
            split = None
        elif index in review_required_indices:
            status = "review_required"
            split = None
        else:
            status = "accepted"
            split = assignments[index]
            split_counts[split] += 1
            class_counts[record.label][split] += 1
            if not args.dry_run:
                destination = args.output / split / record.label / record.path.name
                if destination.exists():
                    destination = destination.with_name(f"{record.sha256[:12]}_{record.path.name}")
                materialize(record.path, destination, args.mode)
        source = source_manifest.get(record.sha256.lower())
        if source is not None:
            source_matches += 1
        rows.append(
            {
                "path": record.relative_path,
                "label": record.label,
                "sha256": record.sha256,
                "perceptual_hash": f"{record.perceptual_hash:032x}",
                "width": record.width,
                "height": record.height,
                "mean_rgb": [round(value, 3) for value in record.mean_rgb],
                "status": status,
                "split": split,
                "duplicate_of": duplicate_of.get(index),
                "source": source,
            }
        )

    summary: dict[str, object] = {
        "version": 1,
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "max_hamming_distance": args.max_hamming_distance,
        "aspect_tolerance": args.aspect_tolerance,
        "color_tolerance": args.color_tolerance,
        "scanned": len(records) + len(rejected),
        "valid_images": len(records),
        "rejected_images": len(rejected),
        "exact_duplicate_links": exact_links,
        "near_duplicate_links": near_links,
        "duplicate_images_removed": len(duplicate_of),
        "label_conflict_clusters": len(conflicts),
        "label_conflict_images": len(conflict_paths),
        "review_required_images_excluded": len(review_required_indices),
        "accepted_images": len(assignments),
        "dataset_fingerprint": dataset_fingerprint(records, assignments),
        "source_manifest": str(args.source_manifest) if args.source_manifest else None,
        "source_records_matched": source_matches,
        "source_records_missing": len(records) - source_matches if source_manifest else None,
        "split_counts": dict(sorted(split_counts.items())),
        "class_split_counts": {
            label: dict(sorted(counts.items())) for label, counts in sorted(class_counts.items())
        },
    }
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.output / "manifest.jsonl", rows)
        write_jsonl(args.output / "rejected.jsonl", rejected)
        (args.output / "conflicts.json").write_text(
            json.dumps(conflicts, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="source ImageFolder root")
    parser.add_argument("--output", type=Path, default=Path("data/styles_prepared"))
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="optional upstream JSONL provenance manifest keyed by image sha256",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument(
        "--max-hamming-distance",
        type=int,
        default=4,
        help="maximum 128-bit difference-hash distance considered a near duplicate",
    )
    parser.add_argument(
        "--aspect-tolerance",
        type=float,
        default=0.02,
        help="relative aspect-ratio tolerance for perceptual duplicate matching",
    )
    parser.add_argument(
        "--color-tolerance",
        type=float,
        default=18.0,
        help="maximum per-channel mean RGB difference for near-duplicate matching",
    )
    parser.add_argument("--mode", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--include-review-required",
        action="store_true",
        help="include upstream records marked review_required in train/validation/test",
    )
    parser.add_argument(
        "--allow-label-conflicts",
        action="store_true",
        help="return success even when duplicate clusters contain conflicting labels",
    )
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    if not 0 < args.test_fraction < 0.5:
        parser.error("--test-fraction must be between 0 and 0.5")
    if args.validation_fraction + args.test_fraction >= 0.8:
        parser.error("validation + test fractions must leave at least 20% for training")
    if not 0 <= args.max_hamming_distance <= 32:
        parser.error("--max-hamming-distance must be between 0 and 32")
    if not 0 <= args.aspect_tolerance <= 0.25:
        parser.error("--aspect-tolerance must be between 0 and 0.25")
    if not 0 <= args.color_tolerance <= 255:
        parser.error("--color-tolerance must be between 0 and 255")
    return args


def main() -> None:
    args = parse_args()
    try:
        summary = prepare(args)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2) from None
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    if summary["label_conflict_clusters"] and not args.allow_label_conflicts:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
