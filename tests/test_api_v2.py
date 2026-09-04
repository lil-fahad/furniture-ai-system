from __future__ import annotations

from fastapi.testclient import TestClient

from furniture_ai.api_entry import app


def test_v2_capabilities_are_exposed() -> None:
    response = TestClient(app).get("/api/v2/capabilities")
    assert response.status_code == 200
    payload = response.json()
    assert payload["api_version"] == "2.0"
    assert payload["design_portfolio"] is True
    assert payload["decision_graph"] is True
    assert payload["ranking_is_confidence"] is False
