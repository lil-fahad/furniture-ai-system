from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PrivateAttr, SecretStr, field_validator, model_validator
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
    openai_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    database_path: Path = Path("data/furniture_ai.sqlite3")
    catalog_path: Path = Path("data/furniture_catalog.json")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)
    max_image_pixels: int = Field(default=25_000_000, ge=1_000_000, le=100_000_000)
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"]
    )
    model_manifest_path: Path = Path("models/manifest.json")
    professional_models_root: Path = Path("models/professional/installed/pretrained")

    _catalog_path_overridden: bool = PrivateAttr(default=False)
    _model_manifest_path_overridden: bool = PrivateAttr(default=False)

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
        # Capture whether runtime-resource paths were explicitly provided before
        # anchoring assignments mutate Pydantic's model_fields_set bookkeeping.
        supplied_fields = set(self.model_fields_set)
        self._catalog_path_overridden = "catalog_path" in supplied_fields
        self._model_manifest_path_overridden = "model_manifest_path" in supplied_fields

        service_key = self.service_api_key.get_secret_value() if self.service_api_key else ""
        if self.environment == "production" and len(service_key) < 24:
            raise ValueError("SERVICE_API_KEY must contain at least 24 characters in production")
        if "*" in self.allowed_origins:
            raise ValueError("Wildcard CORS origins are not allowed")
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
    def catalog_path_overridden(self) -> bool:
        return self._catalog_path_overridden

    @property
    def model_manifest_path_overridden(self) -> bool:
        return self._model_manifest_path_overridden

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())

    @property
    def service_auth_enabled(self) -> bool:
        return bool(self.service_api_key and self.service_api_key.get_secret_value().strip())

    @property
    def professional_vision_available(self) -> bool:
        required_names = ("config.json", "model.safetensors", "preprocessor_config.json")
        model_dirs = (
            self.professional_models_root / "detr_resnet50",
            self.professional_models_root / "depth_anything_v2_small",
        )
        return all(
            (model_dir / name).is_file()
            for model_dir in model_dirs
            for name in required_names
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
