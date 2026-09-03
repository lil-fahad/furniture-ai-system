from __future__ import annotations

from pathlib import Path

import pytest

from furniture_ai.config import Settings
from furniture_ai.layout import load_catalog
from furniture_ai.models import ModelRegistry


def test_packaged_catalog_fallback_is_default_only(tmp_path: Path, monkeypatch) -> None:
    from furniture_ai import config

    settings = Settings(environment="test")
    assert settings.catalog_path_overridden is False
    # Simulate a standalone install whose default repository-local path is absent.
    settings.catalog_path = tmp_path / "missing-default-catalog.json"
    assert settings.catalog_path_overridden is False
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    load_catalog.cache_clear()

    catalog = load_catalog()
    assert catalog
    assert catalog[0].id == "sofa-3-seat"


def test_configured_missing_catalog_fails_closed(tmp_path: Path, monkeypatch) -> None:
    from furniture_ai import config

    missing = tmp_path / "custom-catalog.json"
    settings = Settings(environment="test", catalog_path=missing)
    assert settings.catalog_path_overridden is True
    monkeypatch.setattr(config, "get_settings", lambda: settings)
    load_catalog.cache_clear()

    with pytest.raises(FileNotFoundError, match="Furniture catalog not found"):
        load_catalog()


def test_environment_catalog_override_is_preserved(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "env-catalog.json"
    monkeypatch.setenv("CATALOG_PATH", str(missing))
    settings = Settings(environment="test")
    assert settings.catalog_path == missing
    assert settings.catalog_path_overridden is True


def test_environment_manifest_override_is_preserved(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "env-manifest.json"
    monkeypatch.setenv("MODEL_MANIFEST_PATH", str(missing))
    settings = Settings(environment="test")
    assert settings.model_manifest_path == missing
    assert settings.model_manifest_path_overridden is True


def test_explicit_missing_catalog_fails_closed(tmp_path: Path) -> None:
    load_catalog.cache_clear()
    with pytest.raises(FileNotFoundError, match="Furniture catalog not found"):
        load_catalog(tmp_path / "missing.json")


def test_packaged_manifest_fallback_preserves_registry_contract(tmp_path: Path) -> None:
    registry = ModelRegistry(
        tmp_path / "models" / "manifest.json",
        allow_packaged_default=True,
    )
    statuses = registry.statuses()
    assert len(statuses) == 8
    assert {status.id for status in statuses} >= {
        "room-classifier-efficientnet-b0",
        "floorplan-segmenter-unet",
        "detr_resnet50",
        "depth_anything_v2_small",
    }
    assert all(status.present is False for status in statuses)


def test_missing_manifest_without_default_permission_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Model manifest not found"):
        ModelRegistry(tmp_path / "custom-manifest.json")
