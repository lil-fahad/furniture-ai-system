from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".coverage",
    "htmlcov",
}
IGNORED_SUFFIXES = {".pyc"}

# These are intentional byte-for-byte mirrors required so wheels can load the
# same audited defaults without depending on repository-relative paths. They
# are checked for existence and equality before duplicate detection; divergence
# is an audit failure rather than silently creating a second source of truth.
APPROVED_RESOURCE_MIRRORS = (
    (
        Path("data/furniture_catalog.json"),
        Path("src/furniture_ai/resources/furniture_catalog.json"),
    ),
    (
        Path("models/manifest.json"),
        Path("src/furniture_ai/resources/model_manifest.json"),
    ),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _approved_groups(root: Path) -> set[frozenset[Path]]:
    groups: set[frozenset[Path]] = set()
    for relative_paths in APPROVED_RESOURCE_MIRRORS:
        absolute_paths = [root / path for path in relative_paths]
        missing = [path.relative_to(root) for path in absolute_paths if not path.is_file()]
        if missing:
            raise SystemExit(f"Required runtime resource mirror missing: {missing}")
        hashes = {_sha256(path) for path in absolute_paths}
        if len(hashes) != 1:
            joined = ", ".join(str(path) for path in relative_paths)
            raise SystemExit(f"Runtime resource mirrors diverged: {joined}")
        groups.add(frozenset(relative_paths))
    return groups


def audit_repository(root: Path) -> None:
    if (root / ".gitmodules").exists():
        raise SystemExit("Submodules are not allowed")
    nested_git = [
        path.relative_to(root)
        for path in root.rglob(".git")
        if path != root / ".git" and not any(part in IGNORED_PARTS for part in path.parts)
    ]
    if nested_git:
        raise SystemExit(f"Nested Git repositories found: {nested_git}")

    approved_groups = _approved_groups(root)
    digests: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        if path.suffix in IGNORED_SUFFIXES or path.stat().st_size == 0:
            continue
        digests[_sha256(path)].append(path.relative_to(root))

    duplicates = [
        paths
        for paths in digests.values()
        if len(paths) > 1 and frozenset(paths) not in approved_groups
    ]
    if duplicates:
        formatted = "\n".join(", ".join(map(str, paths)) for paths in duplicates)
        raise SystemExit("Exact duplicate files detected:\n" + formatted)


def main() -> None:
    audit_repository(ROOT)
    print(
        "Repository audit passed: no submodules, nested repositories, unexpected exact "
        "duplicates, or divergent runtime mirrors"
    )


if __name__ == "__main__":
    main()
