from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from furniture_ai.image_io import ImageValidationError, load_validated_image
from furniture_ai.layout import load_catalog
from furniture_ai.models import ModelRegistry


def image_bytes(format_name: str, size: tuple[int, int] = (128, 128)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "white").save(buffer, format=format_name)
    return buffer.getvalue()


def test_packaged_defaults_work_outside_repository(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    load_catalog.cache_clear()
    assert load_catalog()[0].id == "sofa-3-seat"
    statuses = ModelRegistry().statuses()
    assert {status.id for status in statuses} == {
        "room-classifier-efficientnet-b0",
        "floorplan-segmenter-unet",
    }


def test_explicit_missing_runtime_files_fail(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        ModelRegistry(tmp_path / "missing.json")


def test_model_registry_verifies_checksum_in_chunks(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    weight = model_dir / "weight.bin"
    weight.write_bytes(b"verified model bytes")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"models":[{"id":"test","task":"unit","path":"models/weight.bin",'
        f'"sha256":"{digest}"}}]}}',
        encoding="utf-8",
    )
    status = ModelRegistry(manifest).statuses()[0]
    assert status.present is True
    assert status.verified is True


def test_image_media_type_must_match_content() -> None:
    settings = SimpleNamespace(max_upload_bytes=1_000_000, max_image_pixels=1_000_000)
    with pytest.raises(ImageValidationError, match="does not match"):
        load_validated_image(image_bytes("PNG"), "image/jpeg", settings)


def test_unsupported_image_format_is_rejected_without_declared_type() -> None:
    settings = SimpleNamespace(max_upload_bytes=1_000_000, max_image_pixels=1_000_000)
    with pytest.raises(ImageValidationError, match="Only PNG"):
        load_validated_image(image_bytes("BMP"), None, settings)


def test_pixel_limit_is_checked_before_conversion() -> None:
    settings = SimpleNamespace(max_upload_bytes=1_000_000, max_image_pixels=10_000)
    with pytest.raises(ImageValidationError, match="pixel limit"):
        load_validated_image(image_bytes("PNG"), "image/png", settings)
