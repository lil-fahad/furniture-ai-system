from __future__ import annotations

from fastapi.testclient import TestClient

from furniture_ai.api import app


def main() -> None:
    client = TestClient(app)
    health = client.get("/health")
    health.raise_for_status()
    assert health.json()["status"] == "ok"
    assert health.headers.get("X-Correlation-ID")

    ready = client.get("/ready")
    if ready.status_code not in (200, 503):
        raise RuntimeError(f"Unexpected readiness status: {ready.status_code}")
    if ready.status_code == 200:
        assert ready.json()["status"] == "ready"

    print("runtime smoke passed: /health and /ready responded with expected states")


if __name__ == "__main__":
    main()
