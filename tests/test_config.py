from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from furniture_ai.config import PROJECT_ROOT, Settings


def test_relative_paths_resolve_against_project_root() -> None:
    settings = Settings(
        environment="test",
        database_path=Path("data/furniture_ai.sqlite3"),
        catalog_path=Path("data/furniture_catalog.json"),
        model_manifest_path=Path("models/manifest.json"),
    )
    assert settings.database_path == PROJECT_ROOT / "data/furniture_ai.sqlite3"
    assert settings.catalog_path == PROJECT_ROOT / "data/furniture_catalog.json"
    assert settings.model_manifest_path == PROJECT_ROOT / "models/manifest.json"
    assert settings.database_path.is_absolute()
    assert settings.catalog_path.is_absolute()
    assert settings.model_manifest_path.is_absolute()


def test_absolute_paths_are_preserved(tmp_path: Path) -> None:
    db = tmp_path / "custom.sqlite3"
    catalog = tmp_path / "catalog.json"
    manifest = tmp_path / "manifest.json"
    settings = Settings(
        environment="test",
        database_path=db,
        catalog_path=catalog,
        model_manifest_path=manifest,
    )
    assert settings.database_path == db
    assert settings.catalog_path == catalog
    assert settings.model_manifest_path == manifest


def test_default_manifest_exists_at_project_root() -> None:
    settings = Settings(environment="test")
    assert settings.model_manifest_path.is_file()


def test_default_catalog_exists_at_project_root() -> None:
    settings = Settings(environment="test")
    assert settings.catalog_path.is_file()


def test_relative_paths_fall_back_to_cwd_when_project_root_copy_missing(
    tmp_path: Path, monkeypatch
) -> None:
    # Wheel installs land in site-packages, so PROJECT_ROOT has no data/ or
    # models/; relative paths must then resolve against the working directory
    # (e.g. a repo mounted at the Docker WORKDIR).
    from furniture_ai import config

    missing_root = tmp_path / "site-packages"
    missing_root.mkdir()
    monkeypatch.setattr(config, "PROJECT_ROOT", missing_root)

    checkout = tmp_path / "checkout"
    (checkout / "data").mkdir(parents=True)
    (checkout / "models").mkdir()
    (checkout / "data" / "furniture_catalog.json").write_text("[]", encoding="utf-8")
    (checkout / "models" / "manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(checkout)

    settings = Settings(
        environment="test",
        catalog_path=Path("data/furniture_catalog.json"),
        model_manifest_path=Path("models/manifest.json"),
        database_path=Path("data/furniture_ai.sqlite3"),
    )
    assert settings.catalog_path == checkout / "data" / "furniture_catalog.json"
    assert settings.model_manifest_path == checkout / "models" / "manifest.json"
    assert settings.database_path == checkout / "data" / "furniture_ai.sqlite3"


def test_allowed_origins_accepts_comma_separated_string() -> None:
    settings = Settings(environment="test", allowed_origins="http://a.test, http://b.test ,")
    assert settings.allowed_origins == ["http://a.test", "http://b.test"]


def test_production_requires_a_long_service_key() -> None:
    with pytest.raises(ValidationError, match="SERVICE_API_KEY"):
        Settings(environment="production", service_api_key=SecretStr("too-short"))


def test_wildcard_cors_origin_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Wildcard CORS"):
        Settings(environment="test", allowed_origins=["*"])
