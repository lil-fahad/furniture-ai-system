from __future__ import annotations

import base64
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

core = importlib.import_module("ai_coordination")
guard = importlib.import_module("ai_lease_conflict_guard")
remote = importlib.import_module("ai_remote_bootstrap")


def _lease(branch: str, files: str) -> dict[str, str]:
    return {
        "agent": f"agent-{branch}",
        "session_id": f"session-{branch}",
        "task": f"task-{branch}",
        "branch": branch,
        "base_sha": "abc",
        "files": files,
        "lease_until": "2026-09-04T12:00:00Z",
        "status": "active",
    }


def _comment(body: str, created_at: str = "2026-09-04T03:00:00Z") -> dict[str, str]:
    return {"body": body, "created_at": created_at}


def _collaboration_comments() -> list[dict[str, str]]:
    return [
        _comment(
            "AI-COLLAB\n"
            "collab_id: c1\n"
            "base_sha: abc\n"
            "branches: feat/a, feat/b\n"
            "shared_files: src/shared.py\n"
            "status: agreed"
        ),
        _comment(
            "AI-COLLAB-ACK\n"
            "collab_id: c1\n"
            "branch: feat/a\n"
            "status: accepted",
            "2026-09-04T03:00:01Z",
        ),
        _comment(
            "AI-COLLAB-ACK\n"
            "collab_id: c1\n"
            "branch: feat/b\n"
            "status: accepted",
            "2026-09-04T03:00:02Z",
        ),
    ]


def test_active_lease_without_pr_blocks_overlapping_work() -> None:
    current = _lease("feat/a", "src/shared.py")
    peer = _lease("feat/b", "src/shared.py")
    errors = guard.active_lease_conflicts(
        current_files={"src/shared.py"},
        current_lease=current,
        leases=[current, peer],
        comments=[],
        base_sha="abc",
    )
    assert len(errors) == 1
    assert "feat/b" in errors[0]


def test_bilateral_ack_allows_known_active_lease_overlap() -> None:
    current = _lease("feat/a", "src/shared.py")
    peer = _lease("feat/b", "src/shared.py")
    errors = guard.active_lease_conflicts(
        current_files={"src/shared.py"},
        current_lease=current,
        leases=[current, peer],
        comments=_collaboration_comments(),
        base_sha="abc",
    )
    assert errors == []


def test_bootstrap_conflict_cannot_be_ignored() -> None:
    receipt = {"conflict_state": "coordination_required"}
    errors = guard.bootstrap_conflict_errors(
        receipt=receipt,
        unresolved_live_conflicts=["peer still owns shared scope"],
    )
    assert errors
    assert "coordination_required" in errors[0]


def test_bootstrap_conflict_is_clear_after_live_conflict_is_resolved() -> None:
    receipt = {"conflict_state": "coordination_required"}
    assert guard.bootstrap_conflict_errors(
        receipt=receipt,
        unresolved_live_conflicts=[],
    ) == []


def test_bootstrap_requires_explicit_conflict_state() -> None:
    errors = guard.bootstrap_conflict_errors(
        receipt={"status": "complete"},
        unresolved_live_conflicts=[],
    )
    assert errors == [
        "AI-BOOTSTRAP conflict_state must be clear or coordination_required"
    ]


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def test_remote_manifest_matches_local_canonical_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    text = b"alpha\n"
    binary = b"\x00\x01\x02"
    (tmp_path / "a.txt").write_bytes(text)
    (tmp_path / "b.bin").write_bytes(binary)
    _git(tmp_path, "add", "a.txt", "b.bin")
    _git(tmp_path, "commit", "-m", "fixture")
    local = core.tracked_repository_manifest(tmp_path)

    blobs = {"blob-a": text, "blob-b": binary}

    def fake_get(_repository: str, endpoint: str) -> object:
        if endpoint == "/git/commits/abc":
            return {"tree": {"sha": "tree-1"}}
        if endpoint == "/git/trees/tree-1?recursive=1":
            return {
                "truncated": False,
                "tree": [
                    {
                        "path": "a.txt",
                        "type": "blob",
                        "sha": "blob-a",
                        "size": len(text),
                    },
                    {
                        "path": "b.bin",
                        "type": "blob",
                        "sha": "blob-b",
                        "size": len(binary),
                    },
                ],
            }
        if endpoint.startswith("/git/blobs/"):
            key = endpoint.rsplit("/", 1)[-1]
            data = blobs[key]
            return {
                "encoding": "base64",
                "content": base64.b64encode(data).decode("ascii"),
            }
        raise AssertionError(endpoint)

    monkeypatch.setattr(remote, "github_get", fake_get)
    assert remote.remote_repository_manifest("owner/repo", "abc") == local


def test_remote_manifest_fails_closed_on_truncated_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get(_repository: str, endpoint: str) -> object:
        if endpoint == "/git/commits/abc":
            return {"tree": {"sha": "tree-1"}}
        if endpoint == "/git/trees/tree-1?recursive=1":
            return {"truncated": True, "tree": []}
        raise AssertionError(endpoint)

    monkeypatch.setattr(remote, "github_get", fake_get)
    with pytest.raises(RuntimeError, match="truncated"):
        remote.remote_repository_manifest("owner/repo", "abc")


def test_remote_request_parser_is_strict() -> None:
    record = remote.parse_request(
        "AI-BOOTSTRAP-REQUEST\n"
        "agent: Connector AI\n"
        "session_id: s1\n"
        "task: API integration\n"
        "branch: feat/api\n"
        "files: src/api.py\n"
        "status: requested"
    )
    assert record["session_id"] == "s1"
    with pytest.raises(RuntimeError, match="missing fields"):
        remote.parse_request("AI-BOOTSTRAP-REQUEST\nagent: Connector AI")


def test_remote_bootstrap_workflow_is_trusted_and_authorized() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ai-remote-bootstrap.yml").read_text(
        encoding="utf-8"
    )
    assert "issue_comment:" in workflow
    assert "github.event.issue.number == 66" in workflow
    assert "author_association == 'OWNER'" in workflow
    assert "ref: main" in workflow
    assert "persist-credentials: false" in workflow
    assert "AI_BOOTSTRAP_REQUEST: ${{ github.event.comment.body }}" in workflow
