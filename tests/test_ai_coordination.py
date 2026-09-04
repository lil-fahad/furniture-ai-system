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
active_collaborations = _coordination.active_collaborations
active_leases = _coordination.active_leases
collaboration_authorizes = _coordination.collaboration_authorizes
has_coordination_override = _coordination.has_coordination_override
lease_for_branch = _coordination.lease_for_branch
overlap_paths = _coordination.overlap_paths
parse_agent_record = _coordination.parse_agent_record
pr_coordination_errors = _coordination.pr_coordination_errors


def _comment(body: str, created_at: str) -> dict[str, str]:
    return {"body": body, "created_at": created_at}


def _lease(branch: str = "feat/x", base_sha: str = "abc") -> dict[str, str]:
    return {"agent": "Agent X", "task": "Coordination test", "branch": branch, "base_sha": base_sha, "files": "src/example.py", "lease_until": "2026-09-04T10:00:00Z", "status": "active"}


def test_overlap_paths_is_deterministic() -> None:
    assert overlap_paths({"b.py", "a.py"}, {"c.py", "a.py", "b.py"}) == ["a.py", "b.py"]


def test_coordination_override_is_explicit() -> None:
    body = "Coordination-Override: #65\nReason: reviewed together"
    assert has_coordination_override(body, 65) is True
    assert has_coordination_override(body, 63) is False


def test_agent_record_parser_requires_marker() -> None:
    assert parse_agent_record("not-a-lease\nagent: x", "AI-LEASE") is None
    assert parse_agent_record("AI-LEASE\nagent: GPT\nbranch: feat/x", "AI-LEASE") == {"agent": "GPT", "branch": "feat/x"}


def test_active_leases_obey_release_and_expiry() -> None:
    comments = [
        _comment("AI-LEASE\nagent: Agent A\ntask: API\nbranch: feat/a\nlease_until: 2026-09-04T03:00:00Z\nstatus: active", "2026-09-04T01:00:00Z"),
        _comment("AI-LEASE\nagent: Agent B\ntask: UI\nbranch: feat/b\nlease_until: 2026-09-04T01:30:00Z\nstatus: active", "2026-09-04T01:05:00Z"),
        _comment("AI-RELEASE\nagent: Agent A\nbranch: feat/a\nstatus: completed", "2026-09-04T01:10:00Z"),
        _comment("AI-LEASE\nagent: Agent C\ntask: Storage\nbranch: feat/c\nlease_until: 2026-09-04T04:00:00Z\nstatus: active", "2026-09-04T01:20:00Z"),
    ]
    leases = active_leases(comments, now=datetime(2026, 9, 4, 2, 0, tzinfo=UTC))
    assert [lease["branch"] for lease in leases] == ["feat/c"]


def test_lease_for_branch_requires_exact_branch() -> None:
    leases = [_lease("feat/a"), _lease("feat/b")]
    assert lease_for_branch(leases, "feat/b") == leases[1]
    assert lease_for_branch(leases, "feat/missing") is None
    assert lease_for_branch(leases, None) is None


def test_pr_coordination_requires_active_branch_lease() -> None:
    assert pr_coordination_errors(head_branch="feat/x", base_sha="abc", live_main_sha="abc", leases=[]) == ["PR branch `feat/x` has no active AI-LEASE on issue #66"]


def test_pr_coordination_rejects_stale_base() -> None:
    errors = pr_coordination_errors(head_branch="feat/x", base_sha="old", live_main_sha="new", leases=[_lease(base_sha="old")])
    assert any("current main SHA" in error for error in errors)


def test_pr_coordination_requires_lease_renewal_after_base_update() -> None:
    errors = pr_coordination_errors(head_branch="feat/x", base_sha="new", live_main_sha="new", leases=[_lease(base_sha="old")])
    assert errors == ["AI-LEASE base_sha does not match the PR base SHA; renew the lease after updating the branch"]


def test_pr_coordination_accepts_current_declared_branch() -> None:
    assert pr_coordination_errors(head_branch="feat/x", base_sha="abc", live_main_sha="abc", leases=[_lease()]) == []


def test_collaboration_requires_complete_joint_evidence() -> None:
    comments = [_comment(
        "AI-COLLAB\n"
        "agents: Agent A | Agent B\n"
        "branches: feat/a | feat/b\n"
        "prs: #80 | #81\n"
        "files: src/shared.py | tests/test_shared.py\n"
        "integration_owner: Agent A / feat/a\n"
        "plan: reconcile both implementations on a current-main integration branch\n"
        "tests: unit + integration + regression + exact-head CI\n"
        "ack: ack-agent-a | ack-agent-b\n"
        "status: active",
        "2026-09-04T03:00:00Z",
    )]
    collaborations = active_collaborations(comments)
    assert collaboration_authorizes(collaborations, current_pr=80, other_pr=81, current_branch="feat/a", other_branch="feat/b", shared=["src/shared.py"]) is True
    assert collaboration_authorizes(collaborations, current_pr=80, other_pr=81, current_branch="feat/a", other_branch="feat/b", shared=["src/not-declared.py"]) is False


def test_override_text_alone_does_not_authorize_collaboration() -> None:
    assert collaboration_authorizes([], current_pr=80, other_pr=81, current_branch="feat/a", other_branch="feat/b", shared=["src/shared.py"]) is False


def test_coordination_workflow_runs_from_trusted_main() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ai-coordination.yml").read_text(encoding="utf-8")
    assert "pull_request_target:" in workflow
    assert "ref: main" in workflow
    assert "persist-credentials: false" in workflow
    assert "if: github.event_name == 'pull_request_target'" in workflow


def test_agents_requires_full_project_entry_preflight() -> None:
    root = Path(__file__).resolve().parents[1]
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    assert "read the whole project first" in agents.lower()
    assert "active_agent_count" in agents
    assert "AI-COLLAB" in agents
    assert "combined diff" in agents
