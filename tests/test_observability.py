from __future__ import annotations

import re

from fastapi.testclient import TestClient

from furniture_ai.api import app
from furniture_ai.observability import normalize_request_id


def test_normalize_request_id_preserves_safe_value() -> None:
    assert normalize_request_id("req-123.alpha") == "req-123.alpha"


def test_normalize_request_id_replaces_unsafe_value() -> None:
    value = normalize_request_id("bad request id\n")
    assert re.fullmatch(r"[0-9a-f]{32}", value)


def test_api_emits_request_id_and_server_timing() -> None:
    response = TestClient(app).get("/health", headers={"X-Request-ID": "trace-123"})
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "trace-123"
    assert response.headers["Server-Timing"].startswith("app;dur=")


def test_api_generates_request_id_when_missing() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert re.fullmatch(r"[0-9a-f]{32}", response.headers["X-Request-ID"])
