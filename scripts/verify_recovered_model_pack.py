#!/usr/bin/env python3
"""Verify FurnitureAI v0.5 recovered release artifacts without extracting them."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

EXPECTED = {
    "FurnitureAI_Professional_Models_v0.5_FINAL.zip": {
        "sha256": "7f6970f1c7d417f8a3308609a040857a1762ed9aff6d7f5d854201392e1af6d9",
        "manifest_release": "FurnitureAI Professional Models v0.5 — recovered upgrade",
        "dataset_status": "partial_recovery",
    },
    "FurnitureAI_Intelligence_Source_v0.5_FINAL.zip": {
        "sha256": "2d19594a901d13b979c82670c36e0311adfef237ef572df456c9e2913b98fbcb",
        "manifest_release": "FurnitureAI Intelligence Source v0.5",
    },
    "furniture_ai-0.5.0-py3-none-any.whl": {
        "sha256": "ad5bcb41f567520e0a64f4aaf99f365171daf026ed0ebf08d7f0bc820bf95f23",
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> dict[str, object]:
    expected = EXPECTED.get(path.name)
    if expected is None:
        raise ValueError(f"unsupported artifact name: {path.name}")
    if not path.is_file():
        raise FileNotFoundError(path)

    actual_hash = sha256_file(path)
    if actual_hash != expected["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}")

    result: dict[str, object] = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": actual_hash,
        "status": "pass",
    }
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"ZIP CRC failure in {corrupt}")
        result["members"] = len(archive.infolist())
        if "release_manifest.json" in archive.namelist():
            manifest = json.loads(archive.read("release_manifest.json"))
            expected_release = expected.get("manifest_release")
            if expected_release and manifest.get("release") != expected_release:
                raise ValueError("release manifest name mismatch")
            expected_dataset = expected.get("dataset_status")
            if expected_dataset and manifest.get("dataset_status") != expected_dataset:
                raise ValueError("release manifest dataset status mismatch")
            result["manifest_release"] = manifest.get("release")
            result["dataset_status"] = manifest.get("dataset_status")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    results = [verify(path) for path in args.artifacts]
    print(json.dumps({"status": "pass", "artifacts": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
