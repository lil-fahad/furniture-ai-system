#!/usr/bin/env python3
"""Download verified real-image furniture datasets without committing raw bytes to Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "config" / "real_furniture_datasets.json"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "real_furniture"


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "FurnitureAI-Dataset-Importer/1.0"}
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        mode = "ab" if start and response.status == 206 else "wb"
        if mode == "wb":
            start = 0
        total_header = response.headers.get("Content-Length")
        total = start + int(total_header) if total_header else None
        written = start
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r{target.name}: {written / total:.1%}", end="", flush=True)
    print()
    partial.replace(target)


def extract_zip(archive: Path, destination: Path) -> None:
    marker = destination / ".complete"
    if marker.exists():
        print(f"Already extracted: {destination}")
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            resolved = (destination / member.filename).resolve()
            if destination.resolve() not in resolved.parents and resolved != destination.resolve():
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)
    marker.write_text(sha256(archive) + "\n", encoding="utf-8")


def clone_ikea(destination: Path) -> None:
    if (destination / ".git").exists():
        subprocess.run(["git", "-C", str(destination), "pull", "--ff-only"], check=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/valexande/IKEA-Dataset.git", str(destination)],
        check=True,
    )


def record_provenance(dataset: dict, destination: Path, archive: Path | None = None) -> None:
    record = {
        "id": dataset["id"],
        "name": dataset["name"],
        "source": dataset["url"],
        "license": dataset["license"],
        "real_camera_images": True,
    }
    if archive and archive.exists():
        record["archive_sha256"] = sha256(archive)
        record["archive_bytes"] = archive.stat().st_size
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "SOURCE.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def install(dataset: dict, output: Path, keep_archives: bool) -> None:
    dataset_id = dataset["id"]
    destination = output / dataset_id
    if not dataset.get("automatic"):
        print(f"MANUAL {dataset_id}: {dataset['url']}")
        return

    if dataset_id == "ikea-products":
        clone_ikea(destination)
        record_provenance(dataset, destination)
        return

    archive = output / "_archives" / f"{dataset_id}.zip"
    if not archive.exists():
        download(dataset["url"], archive)
    else:
        print(f"Already downloaded: {archive}")
    extract_zip(archive, destination)
    record_provenance(dataset, destination, archive)
    if not keep_archives:
        archive.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset id; repeat the option. Default: every automatic dataset.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    registry = load_registry()
    datasets = {item["id"]: item for item in registry["datasets"]}
    if args.list:
        for item in datasets.values():
            mode = "automatic" if item.get("automatic") else "manual/cloud"
            print(f"{item['id']:<28} {mode:<12} {item['name']}")
        return 0

    selected = args.dataset or [
        item["id"] for item in datasets.values() if item.get("automatic")
    ]
    unknown = sorted(set(selected) - datasets.keys())
    if unknown:
        parser.error("unknown dataset(s): " + ", ".join(unknown))

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for dataset_id in selected:
        print(f"\n== {dataset_id} ==")
        install(datasets[dataset_id], output, args.keep_archives)

    print(f"\nData directory: {output}")
    print("Manual/cloud datasets:")
    for item in datasets.values():
        if not item.get("automatic"):
            print(f"- {item['id']}: {item['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
