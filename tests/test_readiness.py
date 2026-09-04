from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient

from furniture_ai.api import app


def test_ready_returns_503_when_sqlite_probe_fails(monkeypatch) -> None:
    from furniture_ai import api

    class HealthyRegistry:
        def statuses(self):
            return []

    class BrokenStore:
        def ping(self) -> None:
            raise sqlite3.OperationalError("database is unavailable")

    monkeypatch.setattr(api, "_model_registry", lambda settings: HealthyRegistry())
    monkeypatch.setattr(api, "get_booking_store", lambda: BrokenStore())

    response = TestClient(app).get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"] == "Application dependencies are not ready"
