from pathlib import Path

from fastapi.testclient import TestClient

from furniture_system.main import app
from furniture_system.registry import SourceRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_registry_is_valid_and_reproducible() -> None:
    registry = SourceRegistry.load(ROOT / "sources.lock.json")
    assert registry.summary()["total_sources"] == 10
    assert registry.summary()["importable_sources"] == 9
    assert registry.summary()["blocked_sources"] == 1
    assert all(len(source.commit) == 40 for source in registry.sources)


def test_blocked_source_has_no_import_path() -> None:
    registry = SourceRegistry.load(ROOT / "sources.lock.json")
    blocked = [source for source in registry.sources if source.tier == "blocked"]
    assert len(blocked) == 1
    assert blocked[0].path is None


def test_capability_index_excludes_blocked_sources() -> None:
    registry = SourceRegistry.load(ROOT / "sources.lock.json")
    capability_sources = {
        source_id for source_ids in registry.capabilities().values() for source_id in source_ids
    }
    assert "blocked-legacy-monolith" not in capability_sources
    assert "floorplan-engine" in capability_sources


def test_api_health_and_filters() -> None:
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["registry"]["blocked_sources"] == 1

    core = client.get("/api/v1/components", params={"tier": "core"})
    assert core.status_code == 200
    assert {item["id"] for item in core.json()} == {
        "classification-suite",
        "floorplan-engine",
    }


def test_private_sources_are_marked_private() -> None:
    registry = SourceRegistry.load(ROOT / "sources.lock.json")
    private_sources = registry.list(visibility="private")
    assert len(private_sources) == 3
    assert all(source.path and source.path.startswith("private/") for source in private_sources)
