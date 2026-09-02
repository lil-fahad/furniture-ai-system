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


def room_bytes() -> bytes:
    image = Image.new("RGB", (320, 240), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
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
    assert "professional_vision_available" in response.json()


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


def test_scene_endpoint_without_loading_real_models(monkeypatch) -> None:
    from furniture_ai import api
    from furniture_ai.contracts import RelativeDepthSummary, SceneAnalysis

    class FakeVisionService:
        def __init__(self, models_root) -> None:
            self.models_root = models_root

        def analyze(self, image, *, detection_threshold, include_depth):
            assert detection_threshold == 0.6
            assert include_depth is True
            return SceneAnalysis(
                source_width=image.width,
                source_height=image.height,
                relative_depth=RelativeDepthSummary(p10=0.1, median=0.5, p90=0.9),
                model_ids=["fake-detr", "fake-depth"],
            )

    monkeypatch.setattr(api, "ProfessionalVisionService", FakeVisionService)
    response = TestClient(app).post(
        "/api/v1/scene",
        files={"image": ("room.jpg", room_bytes(), "image/jpeg")},
        data={"detection_threshold": "0.6", "include_depth": "true"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["source_width"] == 320
    assert payload["relative_depth"]["median"] == 0.5
    assert payload["model_ids"] == ["fake-detr", "fake-depth"]


def test_scene_endpoint_reports_missing_professional_bundle(monkeypatch) -> None:
    from furniture_ai import api
    from furniture_ai.professional_vision import ProfessionalVisionUnavailable

    class MissingVisionService:
        def __init__(self, models_root) -> None:
            raise ProfessionalVisionUnavailable("professional bundle missing")

    monkeypatch.setattr(api, "ProfessionalVisionService", MissingVisionService)
    response = TestClient(app).post(
        "/api/v1/scene",
        files={"image": ("room.jpg", room_bytes(), "image/jpeg")},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "professional bundle missing"


def test_analyze_and_catalog_work_from_neutral_cwd(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    from furniture_ai.layout import load_catalog

    catalog = load_catalog()
    assert catalog, "default catalog must load from a neutral working directory"
    response = TestClient(app).post(
        "/api/v1/analyze",
        files={"image": ("plan.png", plan_bytes(), "image/png")},
        data={"use_openai": "false", "preferences": ""},
    )
    assert response.status_code == 200, response.text
    assert response.json()["floor_plan"]["rooms"]


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


def test_catalog_room_type_filter() -> None:
    client = TestClient(app)
    all_products = client.get("/api/v1/catalog").json()
    filtered = client.get("/api/v1/catalog", params={"room_type": "bedroom"}).json()
    assert 0 < len(filtered) < len(all_products)
    assert all("bedroom" in product["room_types"] for product in filtered)


def test_ready_returns_503_when_dependencies_missing(monkeypatch) -> None:
    from furniture_ai import api

    class BrokenRegistry:
        def __init__(self, path) -> None:
            raise ValueError("manifest is corrupt")

    monkeypatch.setattr(api, "ModelRegistry", BrokenRegistry)
    response = TestClient(app).get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"] == "Application dependencies are not ready"


def test_models_endpoint_lists_registered_models() -> None:
    response = TestClient(app).get("/api/v1/models")
    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert all("present" in status and "id" in status for status in payload)
