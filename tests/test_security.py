from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from furniture_ai.api import app
from furniture_ai.contracts import FloorPlanAnalysis, LayoutRequest, Point, Room

VALID_POLYGON = [
    {"x": 0, "y": 0},
    {"x": 400, "y": 0},
    {"x": 400, "y": 300},
    {"x": 0, "y": 300},
]


def layout_payload(polygon: list[dict[str, float]]) -> dict:
    return {
        "floor_plan": {
            "source_width": 500,
            "source_height": 400,
            "rooms": [
                {
                    "id": "room-1",
                    "room_type": "living_room",
                    "polygon": polygon,
                    "area": 120000,
                }
            ],
        }
    }


def test_layout_accepts_valid_polygon() -> None:
    response = TestClient(app).post("/api/v1/layout", json=layout_payload(VALID_POLYGON))
    assert response.status_code == 200, response.text
    assert response.json()["floor_plan"]["rooms"][0]["id"] == "room-1"


def test_layout_rejects_collinear_polygon_with_422() -> None:
    collinear = [{"x": 0, "y": 0}, {"x": 100, "y": 100}, {"x": 200, "y": 200}]
    response = TestClient(app).post("/api/v1/layout", json=layout_payload(collinear))
    assert response.status_code == 422, response.text


def test_layout_rejects_self_intersecting_polygon_with_422() -> None:
    bowtie = [{"x": 0, "y": 0}, {"x": 100, "y": 100}, {"x": 100, "y": 0}, {"x": 0, "y": 100}]
    response = TestClient(app).post("/api/v1/layout", json=layout_payload(bowtie))
    assert response.status_code == 422, response.text


def test_layout_engine_value_error_becomes_422(monkeypatch: pytest.MonkeyPatch) -> None:
    from furniture_ai import api

    def boom(*args, **kwargs):
        raise ValueError("Room polygon is invalid")

    monkeypatch.setattr(api, "furnish_floor_plan", boom)
    response = TestClient(app).post("/api/v1/layout", json=layout_payload(VALID_POLYGON))
    assert response.status_code == 422
    assert "Room polygon is invalid" in response.json()["detail"]


def test_room_contract_rejects_degenerate_polygons() -> None:
    with pytest.raises(ValidationError, match="degenerate"):
        Room(
            id="room-1",
            room_type="room",
            polygon=[Point(x=0, y=0), Point(x=5, y=5), Point(x=9, y=9)],
            area=0,
        )
    with pytest.raises(ValidationError, match="simple"):
        Room(
            id="room-1",
            room_type="room",
            polygon=[Point(x=0, y=0), Point(x=2, y=2), Point(x=2, y=0), Point(x=0, y=2)],
            area=2,
        )
    with pytest.raises(ValidationError, match="finite"):
        Room(
            id="room-1",
            room_type="room",
            polygon=[Point(x=0, y=0), Point(x=float("nan"), y=1), Point(x=1, y=1)],
            area=1,
        )


def test_room_contract_accepts_valid_polygon() -> None:
    room = Room(
        id="room-1",
        room_type="room",
        polygon=[Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=8), Point(x=0, y=8)],
        area=80,
    )
    plan = FloorPlanAnalysis(source_width=20, source_height=20, rooms=[room])
    request = LayoutRequest(floor_plan=plan, room_types={"room-1": "bedroom"})
    assert request.floor_plan.rooms[0].area == 80


API_KEY = "test-service-key-0123456789abcdef"


@pytest.fixture
def auth_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("SERVICE_API_KEY", API_KEY)
    from furniture_ai.config import get_settings

    get_settings.cache_clear()
    return TestClient(app)


def test_protected_routes_require_api_key(auth_client: TestClient) -> None:
    for path in ("/api/v1/catalog", "/api/v1/models", "/api/v1/bookings"):
        response = auth_client.get(path)
        assert response.status_code == 401, f"{path} should reject anonymous access"
        response = auth_client.get(path, headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 401, f"{path} should reject a wrong key"


def test_protected_routes_accept_valid_api_key(auth_client: TestClient) -> None:
    headers = {"X-API-Key": API_KEY}
    assert auth_client.get("/api/v1/catalog", headers=headers).status_code == 200
    assert auth_client.get("/api/v1/models", headers=headers).status_code == 200


def test_health_and_ready_stay_anonymous(auth_client: TestClient) -> None:
    assert auth_client.get("/health").status_code == 200
    assert auth_client.get("/ready").status_code == 200


def test_oversize_upload_returns_413(monkeypatch: pytest.MonkeyPatch) -> None:
    from io import BytesIO

    from PIL import Image

    from furniture_ai.config import get_settings

    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    get_settings.cache_clear()
    buffer = BytesIO()
    Image.new("RGB", (400, 300), "white").save(buffer, format="PNG")
    payload = buffer.getvalue()
    assert len(payload) > 1024
    response = TestClient(app).post(
        "/api/v1/analyze",
        files={"image": ("plan.png", payload, "image/png")},
    )
    assert response.status_code == 413, response.text


def test_invalid_image_upload_returns_422() -> None:
    response = TestClient(app).post(
        "/api/v1/analyze",
        files={"image": ("plan.png", b"not an image at all" * 10, "image/png")},
    )
    assert response.status_code == 422, response.text
