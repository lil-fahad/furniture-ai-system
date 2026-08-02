from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify recovered FurnitureAI model assets")
    parser.add_argument("--root", type=Path, required=True, help="Extracted v0.5 full-package root")
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).with_name("models.lock.json"),
    )
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    failures: list[str] = []
    for group in ("pretrained_models", "trained_checkpoints"):
        for model in lock[group]:
            if group == "pretrained_models":
                # The detailed pretrained file lock remains authoritative in the full package.
                continue
            path = args.root / model["path"]
            if not path.is_file():
                failures.append(f"missing: {path}")
                continue
            if path.stat().st_size != model["size_bytes"]:
                failures.append(f"size mismatch: {path}")
            if sha256(path) != model["sha256"]:
                failures.append(f"sha256 mismatch: {path}")
    if failures:
        print("\n".join(failures))
        return 1
    print("Recovered trained checkpoints verified.")
    print("Run `furniture-ai verify-models --root <root>` for all pretrained files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
