from __future__ import annotations

from pathlib import Path

import pytest

from scripts import repository_audit


def _write_approved_mirrors(root: Path) -> None:
    payloads = (b"catalog", b"manifest")
    for paths, payload in zip(repository_audit.APPROVED_RESOURCE_MIRRORS, payloads, strict=True):
        for relative in paths:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)


def test_repository_audit_accepts_only_declared_identical_mirrors(tmp_path: Path) -> None:
    _write_approved_mirrors(tmp_path)
    repository_audit.audit_repository(tmp_path)


def test_repository_audit_rejects_divergent_runtime_mirror(tmp_path: Path) -> None:
    _write_approved_mirrors(tmp_path)
    mirror = tmp_path / repository_audit.APPROVED_RESOURCE_MIRRORS[0][1]
    mirror.write_bytes(b"diverged")
    with pytest.raises(SystemExit, match="Runtime resource mirrors diverged"):
        repository_audit.audit_repository(tmp_path)


def test_repository_audit_still_rejects_unexpected_duplicates(tmp_path: Path) -> None:
    _write_approved_mirrors(tmp_path)
    (tmp_path / "one.txt").write_text("unexpected duplicate", encoding="utf-8")
    (tmp_path / "two.txt").write_text("unexpected duplicate", encoding="utf-8")
    with pytest.raises(SystemExit, match="Exact duplicate files detected"):
        repository_audit.audit_repository(tmp_path)
