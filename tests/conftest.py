from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def test_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.delenv("SERVICE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from furniture_ai.api import get_booking_store, get_supplier_model, get_supplier_rows
    from furniture_ai.config import get_settings

    get_settings.cache_clear()
    get_booking_store.cache_clear()
    get_supplier_model.cache_clear()
    get_supplier_rows.cache_clear()
    yield
    get_settings.cache_clear()
    get_booking_store.cache_clear()
    get_supplier_model.cache_clear()
    get_supplier_rows.cache_clear()
