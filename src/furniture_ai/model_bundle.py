from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

DEFAULT_SPEC_PATH = Path("models/professional/bundle.json")
DEFAULT_INSTALL_ROOT = Path("models/professional/installed")


@dataclass(frozen=True)
class BundleFile:
    archive_path: str
    target_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class BundleSpec:
    id: str
    name: str
    archive_filename: str
    archive_size_bytes: int
    archive_sha256: str
    install_root: str
    source_status: str
    integrity_note: str
    files: tuple[BundleFile, ...]


@dataclass(frozen=True)
class BundleInstallReport:
    bundle_id: str
    archive: str
    destination: str
    archive_sha256: str
    files_installed: int
    bytes_installed: int

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "" in path.parts:
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def load_bundle_spec(path: Path = DEFAULT_SPEC_PATH) -> BundleSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = tuple(
        BundleFile(
            archive_path=_safe_relative_path(str(item["archive_path"])),
            target_path=_safe_relative_path(str(item["target_path"])),
            size_bytes=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
        )
        for item in payload["files"]
    )
    archive_paths = [item.archive_path for item in files]
    target_paths = [item.target_path for item in files]
    if len(archive_paths) != len(set(archive_paths)):
        raise ValueError("Bundle spec contains duplicate archive paths")
    if len(target_paths) != len(set(target_paths)):
        raise ValueError("Bundle spec contains duplicate target paths")
    return BundleSpec(
        id=str(payload["id"]),
        name=str(payload["name"]),
        archive_filename=str(payload["archive_filename"]),
        archive_size_bytes=int(payload["archive_size_bytes"]),
        archive_sha256=str(payload["archive_sha256"]),
        install_root=_safe_relative_path(str(payload["install_root"])),
        source_status=str(payload["source_status"]),
        integrity_note=str(payload["integrity_note"]),
        files=files,
    )


def validate_bundle_spec(path: Path = DEFAULT_SPEC_PATH) -> dict[str, int | str]:
    spec = load_bundle_spec(path)
    total = sum(item.size_bytes for item in spec.files)
    return {
        "bundle_id": spec.id,
        "files": len(spec.files),
        "model_bytes": total,
        "archive_size_bytes": spec.archive_size_bytes,
    }


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(unix_mode)


def _validate_zip_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        name = _safe_relative_path(info.filename.rstrip("/")) if info.filename.rstrip("/") else ""
        if not name:
            continue
        if _is_symlink(info):
            raise ValueError(f"Bundle contains a symbolic link: {info.filename}")
        if name in infos:
            raise ValueError(f"Bundle contains a duplicate member: {name}")
        infos[name] = info
    return infos


def verify_bundle_archive(
    archive_path: Path, spec_path: Path = DEFAULT_SPEC_PATH
) -> dict[str, int | str]:
    spec = load_bundle_spec(spec_path)
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    actual_size = archive_path.stat().st_size
    if actual_size != spec.archive_size_bytes:
        raise ValueError(
            f"Bundle size mismatch: expected {spec.archive_size_bytes}, got {actual_size}"
        )
    actual_hash = sha256_file(archive_path)
    if actual_hash != spec.archive_sha256:
        raise ValueError("Bundle SHA-256 mismatch")

    with zipfile.ZipFile(archive_path) as archive:
        infos = _validate_zip_members(archive)
        missing = [item.archive_path for item in spec.files if item.archive_path not in infos]
        if missing:
            raise ValueError(f"Bundle is missing {len(missing)} required files")
        for item in spec.files:
            if infos[item.archive_path].file_size != item.size_bytes:
                raise ValueError(f"Member size mismatch: {item.archive_path}")
    return {
        "bundle_id": spec.id,
        "archive_sha256": actual_hash,
        "required_files": len(spec.files),
        "archive_members": len(infos),
    }


def install_bundle(
    archive_path: Path,
    destination: Path = DEFAULT_INSTALL_ROOT,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> BundleInstallReport:
    spec = load_bundle_spec(spec_path)
    verified = verify_bundle_archive(archive_path, spec_path)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="furniture-models-",
        dir=destination.parent,
    ) as temp_dir:
        staging = Path(temp_dir) / "professional"
        staging.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            infos = _validate_zip_members(archive)
            for item in spec.files:
                target = staging / item.target_path
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                written = 0
                with archive.open(infos[item.archive_path]) as source, target.open("wb") as output:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                if written != item.size_bytes:
                    raise ValueError(f"Extracted size mismatch: {item.archive_path}")
                if digest.hexdigest() != item.sha256:
                    raise ValueError(f"Extracted SHA-256 mismatch: {item.archive_path}")

        metadata = {
            "bundle_id": spec.id,
            "archive_filename": archive_path.name,
            "archive_sha256": verified["archive_sha256"],
            "files_installed": len(spec.files),
            "bytes_installed": sum(item.size_bytes for item in spec.files),
        }
        (staging / "installed.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)

    return BundleInstallReport(
        bundle_id=spec.id,
        archive=str(archive_path.resolve()),
        destination=str(destination),
        archive_sha256=str(verified["archive_sha256"]),
        files_installed=len(spec.files),
        bytes_installed=sum(item.size_bytes for item in spec.files),
    )


def verify_installed_bundle(
    destination: Path = DEFAULT_INSTALL_ROOT,
    spec_path: Path = DEFAULT_SPEC_PATH,
) -> dict[str, int | str]:
    spec = load_bundle_spec(spec_path)
    missing: list[str] = []
    invalid: list[str] = []
    for item in spec.files:
        path = destination / item.target_path
        if not path.is_file():
            missing.append(item.target_path)
            continue
        if path.stat().st_size != item.size_bytes or sha256_file(path) != item.sha256:
            invalid.append(item.target_path)
    if missing or invalid:
        raise ValueError(
            f"Installed bundle verification failed: {len(missing)} missing, {len(invalid)} invalid"
        )
    return {
        "bundle_id": spec.id,
        "files": len(spec.files),
        "bytes": sum(item.size_bytes for item in spec.files),
        "status": "pass",
    }
