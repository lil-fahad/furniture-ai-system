from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"manifest not found: {args.manifest}")
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    seen: set[str] = set()
    failures: list[str] = []
    for model in payload["models"]:
        model_id = str(model["id"])
        if model_id in seen:
            failures.append(f"duplicate model id: {model_id}")
            continue
        seen.add(model_id)
        path = args.manifest.parent / model["path"]
        if path.is_file():
            digest = sha256_file(path)
            size = path.stat().st_size
            expected_hash = model.get("sha256")
            expected_size = model.get("size_bytes")
            valid = (not expected_hash or digest == expected_hash) and (
                expected_size is None or size == int(expected_size)
            )
            print(model_id, size, digest, "verified" if valid else "INVALID")
            if not valid:
                failures.append(model_id)
            if args.write and not expected_hash:
                model["sha256"] = digest
                model["size_bytes"] = size
        else:
            print(model_id, "missing (optional)" if not model.get("required") else "MISSING")
            if model.get("required"):
                failures.append(model_id)
    if args.write:
        args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("Model manifest validation failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
