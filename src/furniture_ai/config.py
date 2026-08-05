from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    environment: Literal["development", "test", "production"] = "development"
    service_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    openai_model: str = "gpt-5-mini"
    database_path: Path = Path("data/furniture_ai.sqlite3")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024, le=100 * 1024 * 1024)
    max_image_pixels: int = Field(default=25_000_000, ge=1_000_000, le=100_000_000)
    allowed_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:8501", "http://127.0.0.1:8501"]
    )
    model_manifest_path: Path = Path("models/manifest.json")
    supplier_data_path: Path = Path("data/suppliers_master.csv.gz.b64")
    supplier_model_path: Path = Path("models/supplier_ranker/model.json")

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
        return self

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())

    @property
    def service_auth_enabled(self) -> bool:
        return bool(self.service_api_key and self.service_api_key.get_secret_value().strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
