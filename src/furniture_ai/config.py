from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _anchor(path: Path) -> Path:
    """Anchor a relative path at PROJECT_ROOT, falling back to the cwd.

    Dev checkouts keep ``data/`` and ``models/`` next to the package source,
    so relative paths resolve at PROJECT_ROOT regardless of the launch
    directory. In a wheel install (e.g. the Docker image) the package lands
    in site-packages and those directories are absent; fall back to the
    current working directory so a mounted repo at WORKDIR still works.
    The parent-directory check keeps not-yet-created files (the SQLite
    database) anchored in dev checkouts before they exist.
    """
    anchored = PROJECT_ROOT / path
    if anchored.exists() or anchored.parent.exists():
        return anchored
    return Path.cwd() / path


class Settings(BaseSettings):
    environment: Literal["development", "test", "production"] = "development"
    service_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"
    database_path: Path = Path("data/furniture_ai.sqlite3")
    catalog_path: Path = Path("data/furniture_catalog.json")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)
    max_image_pixels: int = Field(default=25_000_000, ge=1_000_000, le=100_000_000)
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"]
    )
    model_manifest_path: Path = Path("models/manifest.json")
    professional_models_root: Path = Path("models/professional/installed/pretrained")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def validate_production_security(self) -> Settings:
        service_key = self.service_api_key.get_secret_value() if self.service_api_key else ""
        if self.environment == "production" and len(service_key) < 24:
            raise ValueError("SERVICE_API_KEY must contain at least 24 characters in production")
        if "*" in self.allowed_origins:
            raise ValueError("Wildcard CORS origins are not allowed")
        # Anchor relative paths at the project root so the app works when
        # launched from any working directory, not just the repo root. When
        # the anchored path is absent (wheel install in site-packages), fall
        # back to the current working directory.
        if not self.database_path.is_absolute():
            self.database_path = _anchor(self.database_path)
        if not self.catalog_path.is_absolute():
            self.catalog_path = _anchor(self.catalog_path)
        if not self.model_manifest_path.is_absolute():
            self.model_manifest_path = _anchor(self.model_manifest_path)
        if not self.professional_models_root.is_absolute():
            self.professional_models_root = _anchor(self.professional_models_root)
        return self

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())

    @property
    def service_auth_enabled(self) -> bool:
        return bool(self.service_api_key and self.service_api_key.get_secret_value().strip())

    @property
    def professional_vision_available(self) -> bool:
        required = (
            self.professional_models_root / "detr_resnet50" / "model.safetensors",
            self.professional_models_root / "depth_anything_v2_small" / "model.safetensors",
        )
        return all(path.is_file() for path in required)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
