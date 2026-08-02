from pathlib import Path

from fastapi.testclient import TestClient

from furniture_system.contracts import PlanRequest
from furniture_system.main import app
from furniture_system.orchestrator import build_execution_plan
from furniture_system.registry import SourceRegistry

ROOT = Path(__file__).resolve().parents[1]


def test_planner_prefers_core_sources_and_groups_capabilities() -> None:
    registry = SourceRegistry.load(ROOT / "sources.lock.json")
    plan = build_execution_plan(
        registry,
        PlanRequest(capabilities=["segmentation", "constraint-layout", "fastapi"]),
    )
    assert plan.unresolved_capabilities == []
    assert plan.steps[0].source_id == "floorplan-engine"
    assert plan.steps[0].capabilities == ["segmentation", "constraint-layout", "fastapi"]


def test_planner_excludes_experimental_sources_by_default() -> None:
    registry = SourceRegistry.load(ROOT / "sources.lock.json")
    plan = build_execution_plan(
        registry,
        PlanRequest(capabilities=["pix2pix", "segment-anything"]),
    )
    assert plan.steps == []
    assert plan.unresolved_capabilities == ["pix2pix", "segment-anything"]


def test_planner_can_opt_in_to_experimental_sources() -> None:
    registry = SourceRegistry.load(ROOT / "sources.lock.json")
    plan = build_execution_plan(
        registry,
        PlanRequest(
            capabilities=["pix2pix", "segment-anything"],
            allow_experimental=True,
        ),
    )
    assert {step.source_id for step in plan.steps} == {
        "generative-lab",
        "vision-prototypes",
    }


def test_plan_and_schema_api() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/plans",
        json={"capabilities": ["segmentation", "constraint-layout"]},
    )
    assert response.status_code == 200
    assert response.json()["steps"][0]["source_id"] == "floorplan-engine"

    schema = client.get("/api/v1/schemas/floor-plan-analysis")
    assert schema.status_code == 200
    assert schema.json()["title"] == "FloorPlanAnalysis"
