from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
IGNORED_SUFFIXES = {".pyc"}


def main() -> None:
    if (ROOT / ".gitmodules").exists():
        raise SystemExit("Submodules are not allowed")
    nested_git = [
        path.relative_to(ROOT)
        for path in ROOT.rglob(".git")
        if path != ROOT / ".git" and not any(part in IGNORED_PARTS for part in path.parts)
    ]
    if nested_git:
        raise SystemExit(f"Nested Git repositories found: {nested_git}")

    digests: dict[str, list[Path]] = defaultdict(list)
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.suffix in IGNORED_SUFFIXES or path.stat().st_size == 0:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests[digest].append(path.relative_to(ROOT))
    duplicates = [paths for paths in digests.values() if len(paths) > 1]
    if duplicates:
        formatted = "\n".join(", ".join(map(str, paths)) for paths in duplicates)
        raise SystemExit("Exact duplicate files detected:\n" + formatted)
    print("Repository audit passed: no submodules, nested repositories, or exact duplicates")


if __name__ == "__main__":
    main()
