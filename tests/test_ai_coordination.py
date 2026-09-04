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
lease_for_branch = _coordination.lease_for_branch
overlap_paths = _coordination.overlap_paths
parse_agent_record = _coordination.parse_agent_record
pr_coordination_errors = _coordination.pr_coordination_errors


def _comment(body: str, created_at: str) -> dict[str, str]:
    return {"body": body, "created_at": created_at}


def _lease(branch: str = "feat/x", base_sha: str = "abc") -> dict[str, str]:
    return {
        "agent": "Agent X",
        "task": "Coordination test",
        "branch": branch,
        "base_sha": base_sha,
        "files": "src/example.py",
        "lease_until": "2026-09-04T05:00:00Z",
        "status": "active",
    }


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


def test_lease_for_branch_requires_exact_branch() -> None:
    leases = [_lease("feat/a"), _lease("feat/b")]
    assert lease_for_branch(leases, "feat/b") == leases[1]
    assert lease_for_branch(leases, "feat/missing") is None
    assert lease_for_branch(leases, None) is None


def test_pr_coordination_requires_active_branch_lease() -> None:
    errors = pr_coordination_errors(
        head_branch="feat/x",
        base_sha="abc",
        live_main_sha="abc",
        leases=[],
    )
    assert errors == ["PR branch `feat/x` has no active AI-LEASE on issue #66"]


def test_pr_coordination_rejects_stale_base() -> None:
    errors = pr_coordination_errors(
        head_branch="feat/x",
        base_sha="old",
        live_main_sha="new",
        leases=[_lease(base_sha="old")],
    )
    assert any("current main SHA" in error for error in errors)


def test_pr_coordination_requires_lease_renewal_after_base_update() -> None:
    errors = pr_coordination_errors(
        head_branch="feat/x",
        base_sha="new",
        live_main_sha="new",
        leases=[_lease(base_sha="old")],
    )
    assert errors == [
        "AI-LEASE base_sha does not match the PR base SHA; "
        "renew the lease after updating the branch"
    ]


def test_pr_coordination_accepts_current_declared_branch() -> None:
    errors = pr_coordination_errors(
        head_branch="feat/x",
        base_sha="abc",
        live_main_sha="abc",
        leases=[_lease()],
    )
    assert errors == []


def test_coordination_workflow_runs_from_trusted_main() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ai-coordination.yml").read_text(
        encoding="utf-8"
    )
    assert "pull_request_target:" in workflow
    assert "ref: main" in workflow
    assert "persist-credentials: false" in workflow
    assert "if: github.event_name == 'pull_request_target'" in workflow
