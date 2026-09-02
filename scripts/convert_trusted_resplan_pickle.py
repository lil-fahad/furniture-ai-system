#!/usr/bin/env python3
"""Convert a *verified and explicitly trusted* ResPlan pickle to safe JSONL/WKT.

Pickle is executable by design. This script therefore refuses to unpickle unless
all of the following are true:

1. the caller supplies an expected SHA-256;
2. the local file matches that SHA-256 exactly; and
3. the caller passes ``--allow-trusted-pickle``.

The generated JSONL contains only scalar values and WKT strings. Downstream
training must consume that safe export rather than the pickle itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import pickletools
from collections import Counter
from pathlib import Path
from typing import Any

GEOMETRY_KEYS = (
    "inner",
    "living",
    "bedroom",
    "bathroom",
    "kitchen",
    "storage",
    "stair",
    "door",
    "front_door",
    "window",
    "wall",
    "balcony",
    "balacony",
)
SCHEMA = "furnitureai-resplan-wkt-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_pickle(path: Path) -> dict[str, object]:
    """Inspect pickle opcodes without executing the pickle."""
    counts: Counter[str] = Counter()
    protocols: set[int] = set()
    with path.open("rb") as stream:
        for opcode, argument, _ in pickletools.genops(stream):
            counts[opcode.name] += 1
            if opcode.name == "PROTO" and isinstance(argument, int):
                protocols.add(argument)
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "protocols": sorted(protocols),
        "opcode_counts": dict(sorted(counts.items())),
    }


def _split_lookup(split_payload: dict[str, list[Any]]) -> dict[str, str]:
    aliases = {"val": "validation", "validation": "validation"}
    result: dict[str, str] = {}
    for source_name, ids in split_payload.items():
        normalized = aliases.get(source_name, source_name)
        if normalized not in {"train", "validation", "test", "augmented"}:
            continue
        for plan_id in ids:
            result[str(plan_id)] = normalized
    return result


def _geometry_to_wkt(value: Any) -> str | None:
    if value is None:
        return None
    if getattr(value, "is_empty", False):
        return None
    wkt = getattr(value, "wkt", None)
    if not isinstance(wkt, str):
        raise TypeError(f"Expected Shapely-like geometry, got {type(value)!r}")
    return wkt


def safe_record(index: int, plan: dict[str, Any], split_lookup: dict[str, str]) -> dict[str, Any]:
    plan_id = plan.get("id", index)
    geometries: dict[str, str] = {}
    for key in GEOMETRY_KEYS:
        normalized = "balcony" if key == "balacony" else key
        if normalized in geometries:
            continue
        value = plan.get(key)
        serialized = _geometry_to_wkt(value)
        if serialized is not None:
            geometries[normalized] = serialized
    return {
        "schema": SCHEMA,
        "plan_id": str(plan_id),
        "source_index": index,
        "split": split_lookup.get(str(plan_id), "unassigned"),
        "wall_depth": float(plan.get("wall_depth") or 0.0),
        "geometries": geometries,
    }


def write_jsonl_atomic(records: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pickle", type=Path)
    parser.add_argument("split_json", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/resplan/resplan.safe.jsonl"))
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument(
        "--allow-trusted-pickle",
        action="store_true",
        help="Required acknowledgement after the SHA-256 has been verified.",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = args.expected_sha256.strip().lower()
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("--expected-sha256 must be exactly 64 hexadecimal characters")
    if not args.pickle.is_file():
        raise FileNotFoundError(args.pickle)
    if not args.split_json.is_file():
        raise FileNotFoundError(args.split_json)

    inspection = inspect_pickle(args.pickle)
    if inspection["sha256"] != expected:
        raise RuntimeError(
            "ResPlan pickle SHA-256 mismatch; refusing to inspect or deserialize untrusted bytes"
        )
    if args.inspect_only:
        print(json.dumps(inspection, indent=2, sort_keys=True))
        return 0
    if not args.allow_trusted_pickle:
        raise RuntimeError(
            "Refusing to unpickle. Re-run only after provenance review with "
            "--allow-trusted-pickle and the verified SHA-256."
        )

    split_payload = json.loads(args.split_json.read_text(encoding="utf-8"))
    if not isinstance(split_payload, dict):
        raise ValueError("split.json must contain an object")
    split_lookup = _split_lookup(split_payload)

    # SECURITY BOUNDARY: pickle.load is intentionally isolated in this converter.
    # The file has already matched a caller-supplied SHA-256 and the caller has
    # explicitly opted into trusting that exact artifact.
    with args.pickle.open("rb") as stream:
        plans = pickle.load(stream)  # noqa: S301 -- documented, hash-gated trust boundary
    if not isinstance(plans, (list, tuple)):
        raise TypeError("Expected ResPlan pickle to contain a list/tuple of plans")

    maximum = len(plans) if args.limit is None else min(len(plans), args.limit)
    if maximum < 1:
        raise ValueError("No plans selected")
    records: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    for index, plan in enumerate(plans[:maximum]):
        if not isinstance(plan, dict):
            raise TypeError(f"Plan {index} is not a dictionary")
        record = safe_record(index, plan, split_lookup)
        records.append(record)
        split_counts[record["split"]] += 1

    write_jsonl_atomic(records, args.output)
    metadata_path = args.metadata or args.output.with_suffix(".metadata.json")
    metadata = {
        "schema": SCHEMA,
        "source_pickle_sha256": expected,
        "source_pickle_bytes": args.pickle.stat().st_size,
        "records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "pickle_inspection": inspection,
        "training_input": False,
        "safe_export": str(args.output),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"exported={args.output} records={len(records)} metadata={metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
