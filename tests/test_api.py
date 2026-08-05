from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from furniture_ai.api import app


def plan_bytes() -> bytes:
    image = Image.new("RGB", (500, 350), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 480, 330), outline="black", width=10)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_health_never_exposes_secret(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-secret-value")
    from furniture_ai.config import get_settings

    get_settings.cache_clear()
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    text = response.text
    assert "not-a-real-secret-value" not in text
    assert response.json()["openai_configured"] is True


def test_analyze_endpoint() -> None:
    response = TestClient(app).post(
        "/api/v1/analyze",
        files={"image": ("plan.png", plan_bytes(), "image/png")},
        data={"use_openai": "false", "preferences": ""},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["floor_plan"]["rooms"]
    assert payload["placed_items"] >= 0


def test_catalog_and_booking() -> None:
    client = TestClient(app)
    assert client.get("/api/v1/catalog").status_code == 200
    created = client.post(
        "/api/v1/bookings",
        json={
            "customer_name": "Test User",
            "contact": "test@example.com",
            "requested_at": "2026-08-10T10:00:00+03:00",
            "notes": "Living room consultation",
        },
    )
    assert created.status_code == 200
    assert client.get("/api/v1/bookings").json()[0]["id"] == created.json()["id"]


def test_supplier_recommendation_endpoint() -> None:
    response = TestClient(app).get(
        "/api/v1/suppliers/recommend",
        params={
            "requires_dropshipping": "true",
            "requires_3d_models": "true",
            "top_k": 5,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert len(payload) == 5
    assert payload[0]["final_score"] >= payload[-1]["final_score"]
