from __future__ import annotations

import importlib.util
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest


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
bootstrap_preconditions = _coordination.bootstrap_preconditions
bootstrap_records = _coordination.bootstrap_records
collaboration_allows_overlap = _coordination.collaboration_allows_overlap
declared_scope_overlap = _coordination.declared_scope_overlap
has_coordination_override = _coordination.has_coordination_override
lease_for_branch = _coordination.lease_for_branch
overlap_paths = _coordination.overlap_paths
parse_agent_record = _coordination.parse_agent_record
pr_coordination_errors = _coordination.pr_coordination_errors
scope_contains_path = _coordination.scope_contains_path
tracked_repository_manifest = _coordination.tracked_repository_manifest


def _comment(body: str, created_at: str = "2026-09-04T01:00:00Z") -> dict[str, str]:
    return {"body": body, "created_at": created_at}


def _bootstrap(branch: str = "feat/x", session_id: str = "session-x") -> dict[str, str]:
    return {
        "agent": "Agent X",
        "session_id": session_id,
        "task": "Coordination test",
        "branch": branch,
        "main_sha": "abc",
        "files": "src/example.py",
        "tracked_files": "120",
        "tracked_bytes": "4567",
        "text_files": "110",
        "binary_files": "10",
        "manifest_sha256": "manifest-abc",
        "observed_active_sessions": "1",
        "status": "complete",
    }


def _lease(
    branch: str = "feat/x",
    base_sha: str = "abc",
    session_id: str = "session-x",
) -> dict[str, str]:
    return {
        "agent": "Agent X",
        "session_id": session_id,
        "task": "Coordination test",
        "branch": branch,
        "base_sha": base_sha,
        "files": "src/example.py",
        "lease_until": "2026-09-04T05:00:00Z",
        "status": "active",
        "bootstrap_main_sha": base_sha,
        "bootstrap_manifest_sha": "manifest-abc",
        "bootstrap_files": "120",
    }


def test_overlap_paths_is_deterministic() -> None:
    assert overlap_paths({"b.py", "a.py"}, {"c.py", "a.py", "b.py"}) == ["a.py", "b.py"]


def test_scope_match_and_declared_overlap_are_conservative() -> None:
    assert scope_contains_path("src/**, docs/", "src/a/b.py") is True
    assert scope_contains_path("src/**, docs/", "tests/a.py") is False
    assert declared_scope_overlap("src/**,docs/a.md", "src/api.py,tests/**") == [
        "src/** <-> src/api.py"
    ]


def test_legacy_override_is_detected_but_not_a_collaboration() -> None:
    body = "Coordination-Override: #65\nReason: reviewed together"
    assert has_coordination_override(body, 65) is True
    allowed, reason = collaboration_allows_overlap(
        [],
        left_branch="feat/a",
        right_branch="feat/b",
        base_sha="abc",
        shared_files=["src/a.py"],
        subject_pr=64,
        subject_head_sha="head-a",
    )
    assert allowed is False
    assert "no bilateral AI-COLLAB" in reason


def test_agent_record_parser_requires_marker() -> None:
    assert parse_agent_record("not-a-lease\nagent: x", "AI-LEASE") is None
    record = parse_agent_record("AI-LEASE\nagent: GPT\nbranch: feat/x", "AI-LEASE")
    assert record == {"agent": "GPT", "branch": "feat/x"}


def test_active_leases_obey_release_and_expiry() -> None:
    comments = [
        _comment(
            "AI-LEASE\nagent: Agent A\nbranch: feat/a\n"
            "lease_until: 2026-09-04T03:00:00Z\nstatus: active"
        ),
        _comment(
            "AI-LEASE\nagent: Agent B\nbranch: feat/b\n"
            "lease_until: 2026-09-04T01:30:00Z\nstatus: active",
            "2026-09-04T01:05:00Z",
        ),
        _comment(
            "AI-RELEASE\nagent: Agent A\nbranch: feat/a\nstatus: completed",
            "2026-09-04T01:10:00Z",
        ),
        _comment(
            "AI-LEASE\nagent: Agent C\nsession_id: c1\nbranch: feat/c\n"
            "lease_until: 2026-09-04T04:00:00Z\nstatus: active",
            "2026-09-04T01:20:00Z",
        ),
    ]
    now = datetime(2026, 9, 4, 2, 0, tzinfo=UTC)
    leases = active_leases(comments, now=now)
    assert [lease["branch"] for lease in leases] == ["feat/c"]


def test_bootstrap_records_keep_latest_branch_session_receipt() -> None:
    comments = [
        _comment(
            "AI-BOOTSTRAP\nbranch: feat/x\nsession_id: s1\nmanifest_sha256: old",
            "2026-09-04T01:00:00Z",
        ),
        _comment(
            "AI-BOOTSTRAP\nbranch: feat/x\nsession_id: s1\nmanifest_sha256: new",
            "2026-09-04T01:01:00Z",
        ),
    ]
    assert bootstrap_records(comments)[0]["manifest_sha256"] == "new"


def test_lease_for_branch_requires_exact_branch() -> None:
    leases = [_lease("feat/a"), _lease("feat/b")]
    assert lease_for_branch(leases, "feat/b") == leases[1]
    assert lease_for_branch(leases, "feat/missing") is None


def test_pr_coordination_requires_bootstrap_backed_lease() -> None:
    legacy = {
        "agent": "Legacy",
        "task": "old",
        "branch": "feat/x",
        "base_sha": "abc",
        "files": "src/example.py",
        "lease_until": "2026-09-04T05:00:00Z",
        "status": "active",
    }
    errors = pr_coordination_errors(
        head_branch="feat/x",
        base_sha="abc",
        live_main_sha="abc",
        leases=[legacy],
        bootstraps=[],
    )
    assert any("session_id" in error for error in errors)
    assert any("no matching AI-BOOTSTRAP" in error for error in errors)


def test_pr_coordination_rejects_stale_base() -> None:
    errors = pr_coordination_errors(
        head_branch="feat/x",
        base_sha="old",
        live_main_sha="new",
        leases=[_lease(base_sha="old")],
        bootstraps=[_bootstrap()],
    )
    assert any("current main SHA" in error for error in errors)


def test_pr_coordination_accepts_matching_bootstrap_receipt() -> None:
    errors = pr_coordination_errors(
        head_branch="feat/x",
        base_sha="abc",
        live_main_sha="abc",
        leases=[_lease()],
        bootstraps=[_bootstrap()],
    )
    assert errors == []


def test_trusted_manifest_rejects_forged_bootstrap_receipt() -> None:
    errors = pr_coordination_errors(
        head_branch="feat/x",
        base_sha="abc",
        live_main_sha="abc",
        leases=[_lease()],
        bootstraps=[_bootstrap()],
        expected_manifest={
            "tracked_files": 120,
            "tracked_bytes": 4567,
            "text_files": 110,
            "binary_files": 10,
            "manifest_sha256": "different-manifest",
        },
    )
    assert errors == [
        "AI-BOOTSTRAP manifest_sha256 does not match the trusted main checkout"
    ]


def test_collaboration_requires_both_acks_and_exact_head_review() -> None:
    comments = [
        _comment(
            "AI-COLLAB\ncollab_id: c1\nbase_sha: abc\n"
            "branches: feat/a, feat/b\nshared_files: src/a.py\nstatus: agreed"
        ),
        _comment(
            "AI-COLLAB-ACK\ncollab_id: c1\nbranch: feat/a\nstatus: accepted",
            "2026-09-04T01:01:00Z",
        ),
        _comment(
            "AI-COLLAB-ACK\ncollab_id: c1\nbranch: feat/b\nstatus: accepted",
            "2026-09-04T01:02:00Z",
        ),
    ]
    allowed, reason = collaboration_allows_overlap(
        comments,
        left_branch="feat/a",
        right_branch="feat/b",
        base_sha="abc",
        shared_files=["src/a.py"],
        subject_pr=10,
        subject_head_sha="head-a",
    )
    assert allowed is False
    assert "exact-head review" in reason

    comments.append(
        _comment(
            "AI-COLLAB-REVIEW\ncollab_id: c1\nreviewer_branch: feat/b\n"
            "subject_pr: 10\nhead_sha: head-a\nstatus: approved",
            "2026-09-04T01:03:00Z",
        )
    )
    allowed, _reason = collaboration_allows_overlap(
        comments,
        left_branch="feat/a",
        right_branch="feat/b",
        base_sha="abc",
        shared_files=["src/a.py"],
        subject_pr=10,
        subject_head_sha="head-a",
    )
    assert allowed is True


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def test_repository_manifest_scans_all_tracked_files_and_is_deterministic(
    tmp_path: Path,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.bin").write_bytes(b"\x00\x01\x02")
    _git(tmp_path, "add", "a.txt", "b.bin")
    _git(tmp_path, "commit", "-m", "fixture")
    first = tracked_repository_manifest(tmp_path)
    second = tracked_repository_manifest(tmp_path)
    assert first == second
    assert first["tracked_files"] == 2
    assert first["text_files"] == 1
    assert first["binary_files"] == 1
    assert first["tracked_bytes"] == 9


def test_bootstrap_preconditions_require_clean_current_task_branch(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-m", "fixture")
    head = _git(tmp_path, "rev-parse", "HEAD")
    _git(tmp_path, "checkout", "-b", "feat/x")
    assert bootstrap_preconditions(tmp_path, head) == ("feat/x", head)
    (tmp_path / "a.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="before edits"):
        bootstrap_preconditions(tmp_path, head)


def test_coordination_workflow_runs_from_trusted_main() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "ai-coordination.yml").read_text(
        encoding="utf-8"
    )
    assert "pull_request_target:" in workflow
    assert "ref: main" in workflow
    assert "persist-credentials: false" in workflow
    assert "if: github.event_name == 'pull_request_target'" in workflow
