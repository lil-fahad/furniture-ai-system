from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType


def _load_coordination_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "ai_coordination.py"
    spec = importlib.util.spec_from_file_location("furniture_ai_coordination", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load coordination helper from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_coordination = _load_coordination_module()
active_leases = _coordination.active_leases
has_coordination_override = _coordination.has_coordination_override
overlap_paths = _coordination.overlap_paths
parse_agent_record = _coordination.parse_agent_record


def _comment(body: str, created_at: str) -> dict[str, str]:
    return {"body": body, "created_at": created_at}


def test_overlap_paths_is_deterministic() -> None:
    assert overlap_paths({"b.py", "a.py"}, {"c.py", "a.py", "b.py"}) == ["a.py", "b.py"]


def test_coordination_override_is_explicit() -> None:
    body = "Coordination-Override: #65\nReason: reviewed together"
    assert has_coordination_override(body, 65) is True
    assert has_coordination_override(body, 63) is False


def test_agent_record_parser_requires_marker() -> None:
    assert parse_agent_record("not-a-lease\nagent: x", "AI-LEASE") is None
    record = parse_agent_record("AI-LEASE\nagent: GPT\nbranch: feat/x", "AI-LEASE")
    assert record == {"agent": "GPT", "branch": "feat/x"}


def test_active_leases_obey_release_and_expiry() -> None:
    comments = [
        _comment(
            "AI-LEASE\n"
            "agent: Agent A\n"
            "task: API\n"
            "branch: feat/a\n"
            "lease_until: 2026-09-04T03:00:00Z\n"
            "status: active",
            "2026-09-04T01:00:00Z",
        ),
        _comment(
            "AI-LEASE\n"
            "agent: Agent B\n"
            "task: UI\n"
            "branch: feat/b\n"
            "lease_until: 2026-09-04T01:30:00Z\n"
            "status: active",
            "2026-09-04T01:05:00Z",
        ),
        _comment(
            "AI-RELEASE\nagent: Agent A\nbranch: feat/a\nstatus: completed",
            "2026-09-04T01:10:00Z",
        ),
        _comment(
            "AI-LEASE\n"
            "agent: Agent C\n"
            "task: Storage\n"
            "branch: feat/c\n"
            "lease_until: 2026-09-04T04:00:00Z\n"
            "status: active",
            "2026-09-04T01:20:00Z",
        ),
    ]
    now = datetime(2026, 9, 4, 2, 0, tzinfo=UTC)
    leases = active_leases(comments, now=now)
    assert [lease["branch"] for lease in leases] == ["feat/c"]
