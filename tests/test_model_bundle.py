from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from furniture_ai.model_bundle import (
    install_bundle,
    load_bundle_spec,
    validate_bundle_spec,
    verify_bundle_archive,
    verify_installed_bundle,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_spec(spec_path: Path, archive: Path, member: str, content: bytes) -> None:
    spec = {
        "schema_version": 1,
        "id": "test-bundle",
        "name": "Test Bundle",
        "archive_filename": archive.name,
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive),
        "source_status": "test",
        "integrity_note": "test bundle",
        "install_root": "models/professional/installed",
        "files": [
            {
                "archive_path": member,
                "target_path": "pretrained/test/model.bin",
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        ],
    }
    spec_path.write_text(json.dumps(spec), encoding="utf-8")


def test_committed_bundle_spec_is_valid() -> None:
    spec = load_bundle_spec()
    report = validate_bundle_spec()
    assert spec.id == "furnitureai-professional-models-v0.4.1-repaired"
    assert report["files"] == 22
    assert report["model_bytes"] == 535635194


def test_install_and_verify_small_bundle(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.zip"
    content = b"verified-model-bytes"
    member = "models/pretrained/test/model.bin"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, content)
    spec_path = tmp_path / "bundle.json"
    _write_spec(spec_path, archive, member, content)

    archive_report = verify_bundle_archive(archive, spec_path)
    destination = tmp_path / "installed"
    install_report = install_bundle(archive, destination, spec_path)
    installed_report = verify_installed_bundle(destination, spec_path)

    assert archive_report["required_files"] == 1
    assert install_report.files_installed == 1
    assert installed_report["status"] == "pass"
    assert (destination / "pretrained/test/model.bin").read_bytes() == content


def test_bundle_rejects_wrong_archive_hash(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.zip"
    content = b"model"
    member = "models/pretrained/test/model.bin"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(member, content)
    spec_path = tmp_path / "bundle.json"
    _write_spec(spec_path, archive, member, content)
    payload = json.loads(spec_path.read_text())
    payload["archive_sha256"] = "0" * 64
    spec_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="SHA-256"):
        verify_bundle_archive(archive, spec_path)


def test_bundle_rejects_zip_slip_member(tmp_path: Path) -> None:
    archive = tmp_path / "bundle.zip"
    content = b"model"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.bin", b"escape")
        bundle.writestr("models/pretrained/test/model.bin", content)
    spec_path = tmp_path / "bundle.json"
    _write_spec(spec_path, archive, "models/pretrained/test/model.bin", content)

    with pytest.raises(ValueError, match="Unsafe relative path"):
        verify_bundle_archive(archive, spec_path)
