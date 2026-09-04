#!/usr/bin/env python3
"""FurnitureAI multi-agent coordination and mandatory bootstrap helper.

The helper keeps issue #66 as the live coordination board, verifies that every
future coding session scanned the complete tracked repository at the exact main
SHA it is based on, reports active AI sessions, blocks stale or undeclared pull
requests, and requires bilateral collaboration records for overlapping work.

Only the Python standard library is used so the trusted guard can run before
project dependencies are installed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

COORDINATION_ISSUE = 66
DEFAULT_REPOSITORY = "lil-fahad/furniture-ai-system"
API_ROOT = "https://api.github.com"
REQUIRED_LEASE_FIELDS = (
    "agent",
    "session_id",
    "task",
    "branch",
    "base_sha",
    "files",
    "lease_until",
    "status",
    "bootstrap_main_sha",
    "bootstrap_manifest_sha",
    "bootstrap_files",
)
REQUIRED_BOOTSTRAP_FIELDS = (
    "agent",
    "session_id",
    "task",
    "branch",
    "main_sha",
    "files",
    "tracked_files",
    "tracked_bytes",
    "text_files",
    "binary_files",
    "manifest_sha256",
    "observed_active_sessions",
    "status",
)


def _token() -> str | None:
    value = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return value.strip() if value and value.strip() else None


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, object] | None = None,
    token: str | None = None,
) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "FurnitureAI-AI-Coordination/4",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    active_token = token or _token()
    if active_token:
        headers["Authorization"] = f"Bearer {active_token}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2_000]
        raise RuntimeError(f"GitHub API HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GitHub API unavailable: {exc.reason}") from exc
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


def github_get(repository: str, endpoint: str) -> Any:
    return _request(f"{API_ROOT}/repos/{repository}{endpoint}")


def github_get_pages(repository: str, endpoint: str) -> list[dict[str, Any]]:
    separator = "&" if "?" in endpoint else "?"
    results: list[dict[str, Any]] = []
    for page in range(1, 101):
        url = (
            f"{API_ROOT}/repos/{repository}{endpoint}"
            f"{separator}per_page=100&page={page}"
        )
        payload = _request(url)
        if not isinstance(payload, list):
            raise RuntimeError(f"Expected list from GitHub endpoint {endpoint}")
        results.extend(item for item in payload if isinstance(item, dict))
        if len(payload) < 100:
            return results
    raise RuntimeError(f"GitHub pagination exceeded safety limit for {endpoint}")


def github_post(repository: str, endpoint: str, payload: dict[str, object]) -> Any:
    token = _token()
    if not token:
        raise RuntimeError("GITHUB_TOKEN or GH_TOKEN is required for GitHub writes")
    return _request(
        f"{API_ROOT}/repos/{repository}{endpoint}",
        method="POST",
        payload=payload,
        token=token,
    )


def parse_iso8601(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_agent_record(body: str, marker: str) -> dict[str, str] | None:
    lines = [line.strip() for line in body.strip().splitlines()]
    if not lines or lines[0] != marker:
        return None
    record: dict[str, str] = {}
    for line in lines[1:]:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        record[key.strip().lower()] = value.strip()
    return record


def active_leases(
    comments: list[dict[str, Any]], now: datetime | None = None
) -> list[dict[str, str]]:
    current = now.astimezone(UTC) if now else datetime.now(UTC)
    by_branch: dict[str, dict[str, str]] = {}
    ordered = sorted(comments, key=lambda item: str(item.get("created_at", "")))
    for comment in ordered:
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        lease = parse_agent_record(body, "AI-LEASE")
        if lease is not None:
            branch = lease.get("branch")
            if branch:
                by_branch[branch] = lease
            continue
        release = parse_agent_record(body, "AI-RELEASE")
        if release is not None:
            branch = release.get("branch")
            if branch:
                by_branch.pop(branch, None)

    active: list[dict[str, str]] = []
    for lease in by_branch.values():
        if lease.get("status", "active").lower() != "active":
            continue
        expiry = lease.get("lease_until")
        if not expiry:
            continue
        try:
            if parse_iso8601(expiry) <= current:
                continue
        except ValueError:
            continue
        active.append(lease)
    return sorted(
        active,
        key=lambda item: (
            item.get("agent", ""),
            item.get("session_id", ""),
            item.get("branch", ""),
        ),
    )


def lease_for_branch(
    leases: list[dict[str, str]], branch: str | None
) -> dict[str, str] | None:
    if not branch:
        return None
    for lease in leases:
        if lease.get("branch") == branch:
            return lease
    return None


def bootstrap_records(comments: list[dict[str, Any]]) -> list[dict[str, str]]:
    latest: dict[tuple[str, str], dict[str, str]] = {}
    ordered = sorted(comments, key=lambda item: str(item.get("created_at", "")))
    for comment in ordered:
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        record = parse_agent_record(body, "AI-BOOTSTRAP")
        if record is None:
            continue
        branch = record.get("branch")
        session_id = record.get("session_id")
        if branch and session_id:
            latest[(branch, session_id)] = record
    return list(latest.values())


def bootstrap_for_lease(
    bootstraps: list[dict[str, str]], lease: dict[str, str]
) -> dict[str, str] | None:
    branch = lease.get("branch")
    session_id = lease.get("session_id")
    if not branch or not session_id:
        return None
    for record in bootstraps:
        if record.get("branch") == branch and record.get("session_id") == session_id:
            return record
    return None


def pr_coordination_errors(
    *,
    head_branch: str | None,
    base_sha: str | None,
    live_main_sha: str,
    leases: list[dict[str, str]],
    bootstraps: list[dict[str, str]],
    expected_manifest: dict[str, object] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not head_branch:
        errors.append("PR head branch could not be resolved")
        return errors

    if base_sha != live_main_sha:
        errors.append(
            "PR base SHA is not the current main SHA; rebuild/update the branch from "
            "current main and rerun bootstrap plus exact-head CI"
        )

    lease = lease_for_branch(leases, head_branch)
    if lease is None:
        errors.append(
            f"PR branch `{head_branch}` has no active AI-LEASE on issue "
            f"#{COORDINATION_ISSUE}"
        )
        return errors

    missing = [field for field in REQUIRED_LEASE_FIELDS if not lease.get(field)]
    if missing:
        errors.append("AI-LEASE is missing required fields: " + ", ".join(missing))
    if lease.get("base_sha") != base_sha:
        errors.append(
            "AI-LEASE base_sha does not match the PR base SHA; rerun bootstrap and "
            "renew the lease after updating the branch"
        )

    receipt = bootstrap_for_lease(bootstraps, lease)
    if receipt is None:
        errors.append(
            "AI-LEASE has no matching AI-BOOTSTRAP receipt for the same branch/session"
        )
        return errors
    bootstrap_missing = [field for field in REQUIRED_BOOTSTRAP_FIELDS if not receipt.get(field)]
    if bootstrap_missing:
        errors.append(
            "AI-BOOTSTRAP is missing required fields: " + ", ".join(bootstrap_missing)
        )
    if receipt.get("status", "").lower() != "complete":
        errors.append("AI-BOOTSTRAP status must be complete")
    if receipt.get("main_sha") != base_sha:
        errors.append("AI-BOOTSTRAP main_sha does not match the PR base SHA")
    if lease.get("bootstrap_main_sha") != receipt.get("main_sha"):
        errors.append("AI-LEASE bootstrap_main_sha does not match AI-BOOTSTRAP")
    if lease.get("bootstrap_manifest_sha") != receipt.get("manifest_sha256"):
        errors.append("AI-LEASE bootstrap_manifest_sha does not match AI-BOOTSTRAP")
    if lease.get("bootstrap_files") != receipt.get("tracked_files"):
        errors.append("AI-LEASE bootstrap_files does not match AI-BOOTSTRAP")
    if expected_manifest is not None:
        expected_pairs = {
            "manifest_sha256": "manifest_sha256",
            "tracked_files": "tracked_files",
            "tracked_bytes": "tracked_bytes",
            "text_files": "text_files",
            "binary_files": "binary_files",
        }
        for receipt_field, manifest_field in expected_pairs.items():
            if receipt.get(receipt_field) != str(expected_manifest.get(manifest_field)):
                errors.append(
                    f"AI-BOOTSTRAP {receipt_field} does not match the trusted main checkout"
                )
    return errors


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _scope_prefix(pattern: str) -> str:
    wildcard_positions = [
        pos for token in ("*", "?", "[") if (pos := pattern.find(token)) >= 0
    ]
    if not wildcard_positions:
        return pattern.rstrip("/")
    return pattern[: min(wildcard_positions)].rstrip("/")


def scope_contains_path(scope: str | None, path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("./")
    for pattern in split_csv(scope):
        candidate = pattern.replace("\\", "/").lstrip("./")
        if candidate in {"*", "**", "**/*"}:
            return True
        if fnmatch.fnmatchcase(normalized, candidate):
            return True
        if not any(token in candidate for token in "*?["):
            prefix = candidate.rstrip("/") + "/"
            if normalized.startswith(prefix):
                return True
    return False


def declared_scope_overlap(left: str | None, right: str | None) -> list[str]:
    overlaps: set[str] = set()
    for left_item in split_csv(left):
        for right_item in split_csv(right):
            left_norm = left_item.replace("\\", "/").lstrip("./")
            right_norm = right_item.replace("\\", "/").lstrip("./")
            if left_norm == right_norm:
                overlaps.add(left_norm)
                continue
            if fnmatch.fnmatchcase(left_norm, right_norm) or fnmatch.fnmatchcase(
                right_norm, left_norm
            ):
                overlaps.add(f"{left_norm} <-> {right_norm}")
                continue
            left_prefix = _scope_prefix(left_norm)
            right_prefix = _scope_prefix(right_norm)
            if left_prefix and right_prefix and (
                left_prefix.startswith(right_prefix.rstrip("/") + "/")
                or right_prefix.startswith(left_prefix.rstrip("/") + "/")
            ):
                overlaps.add(f"{left_norm} <-> {right_norm}")
    return sorted(overlaps)


def overlap_paths(left: set[str], right: set[str]) -> list[str]:
    return sorted(left.intersection(right))


def has_coordination_override(body: str | None, other_pr: int) -> bool:
    """Legacy parser retained for diagnostics; override no longer authorizes overlap."""
    if not body:
        return False
    needle = f"coordination-override: #{other_pr}".lower()
    return needle in body.lower()


def collaboration_state(comments: list[dict[str, Any]]) -> dict[str, object]:
    collaborations: dict[str, dict[str, str]] = {}
    acknowledgements: dict[str, dict[str, dict[str, str]]] = {}
    reviews: list[dict[str, str]] = []
    ordered = sorted(comments, key=lambda item: str(item.get("created_at", "")))
    for comment in ordered:
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        collaboration = parse_agent_record(body, "AI-COLLAB")
        if collaboration is not None and collaboration.get("collab_id"):
            collaborations[collaboration["collab_id"]] = collaboration
            continue
        ack = parse_agent_record(body, "AI-COLLAB-ACK")
        if ack is not None and ack.get("collab_id") and ack.get("branch"):
            acknowledgements.setdefault(ack["collab_id"], {})[ack["branch"]] = ack
            continue
        review = parse_agent_record(body, "AI-COLLAB-REVIEW")
        if review is not None:
            reviews.append(review)
    return {
        "collaborations": collaborations,
        "acknowledgements": acknowledgements,
        "reviews": reviews,
    }


def collaboration_allows_overlap(
    comments: list[dict[str, Any]],
    *,
    left_branch: str,
    right_branch: str,
    base_sha: str,
    shared_files: list[str],
    subject_pr: int,
    subject_head_sha: str,
) -> tuple[bool, str]:
    state = collaboration_state(comments)
    collaborations = state["collaborations"]
    acknowledgements = state["acknowledgements"]
    reviews = state["reviews"]
    assert isinstance(collaborations, dict)
    assert isinstance(acknowledgements, dict)
    assert isinstance(reviews, list)

    wanted_branches = {left_branch, right_branch}
    for collab_id, raw in collaborations.items():
        if not isinstance(collab_id, str) or not isinstance(raw, dict):
            continue
        record = {str(k): str(v) for k, v in raw.items()}
        if record.get("status", "").lower() not in {"active", "agreed"}:
            continue
        if record.get("base_sha") != base_sha:
            continue
        if set(split_csv(record.get("branches"))) != wanted_branches:
            continue
        declared_shared = set(split_csv(record.get("shared_files")))
        if not set(shared_files).issubset(declared_shared):
            continue
        raw_acks = acknowledgements.get(collab_id, {})
        if not isinstance(raw_acks, dict):
            continue
        if any(
            not isinstance(raw_acks.get(branch), dict)
            or str(raw_acks[branch].get("status", "")).lower() != "accepted"
            for branch in wanted_branches
        ):
            return False, f"AI-COLLAB {collab_id} is missing accepted ACK from both branches"
        matching_review = any(
            isinstance(review, dict)
            and review.get("collab_id") == collab_id
            and review.get("reviewer_branch") == right_branch
            and review.get("subject_pr") == str(subject_pr)
            and review.get("head_sha") == subject_head_sha
            and review.get("status", "").lower() == "approved"
            for review in reviews
        )
        if not matching_review:
            return (
                False,
                f"AI-COLLAB {collab_id} needs exact-head review from `{right_branch}` "
                f"for PR #{subject_pr} at {subject_head_sha}",
            )
        return True, f"AI-COLLAB {collab_id} approved by both branches and exact-head review"
    return False, "no bilateral AI-COLLAB covers both branches, current base, and shared files"


def main_sha(repository: str) -> str:
    payload = github_get(repository, "/commits/main")
    if not isinstance(payload, dict) or not isinstance(payload.get("sha"), str):
        raise RuntimeError("Could not resolve main SHA")
    return payload["sha"]


def open_pulls(repository: str) -> list[dict[str, Any]]:
    return github_get_pages(repository, "/pulls?state=open")


def pr_files(repository: str, number: int) -> set[str]:
    rows = github_get_pages(repository, f"/pulls/{number}/files")
    files: set[str] = set()
    for row in rows:
        filename = row.get("filename")
        if isinstance(filename, str):
            files.add(filename)
    return files


def coordination_comments(
    repository: str, issue: int = COORDINATION_ISSUE
) -> list[dict[str, Any]]:
    return github_get_pages(repository, f"/issues/{issue}/comments")


def _pull_ref(pr: dict[str, Any], side: str, key: str) -> str | None:
    payload = pr.get(side)
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) else None


def build_snapshot(repository: str, issue: int = COORDINATION_ISSUE) -> dict[str, object]:
    comments = coordination_comments(repository, issue)
    pulls = open_pulls(repository)
    leases = active_leases(comments)
    lease_by_branch = {
        lease["branch"]: lease for lease in leases if isinstance(lease.get("branch"), str)
    }
    open_rows: list[dict[str, object]] = []
    for pr in pulls:
        head = _pull_ref(pr, "head", "ref")
        lease = lease_by_branch.get(head or "")
        open_rows.append(
            {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "draft": bool(pr.get("draft")),
                "head": head,
                "head_sha": _pull_ref(pr, "head", "sha"),
                "base_sha": _pull_ref(pr, "base", "sha"),
                "updated_at": pr.get("updated_at"),
                "coordination_state": "declared" if lease else "undeclared",
                "active_lease": lease,
            }
        )
    sessions = [
        {
            "agent": lease.get("agent"),
            "session_id": lease.get("session_id"),
            "task": lease.get("task"),
            "branch": lease.get("branch"),
            "base_sha": lease.get("base_sha"),
            "files": lease.get("files"),
            "lease_until": lease.get("lease_until"),
        }
        for lease in leases
    ]
    return {
        "schema_version": 4,
        "repository": repository,
        "observed_at": datetime.now(UTC).isoformat(),
        "main_sha": main_sha(repository),
        "coordination_issue": issue,
        "active_session_count": len(sessions),
        "active_sessions": sessions,
        "open_pull_requests": open_rows,
    }


def _git_bytes(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def _git_text(root: Path, *arguments: str) -> str:
    return _git_bytes(root, *arguments).decode("utf-8", errors="replace").strip()


def tracked_repository_manifest(root: Path) -> dict[str, object]:
    root = root.resolve()
    raw_paths = _git_bytes(root, "ls-files", "-z")
    paths = sorted(os.fsdecode(item) for item in raw_paths.split(b"\0") if item)
    manifest = hashlib.sha256()
    total_bytes = 0
    text_files = 0
    binary_files = 0
    for relative in paths:
        file_path = root / relative
        if file_path.is_symlink():
            data = os.fsencode(os.readlink(file_path))
        else:
            data = file_path.read_bytes()
        total_bytes += len(data)
        try:
            decoded = data.decode("utf-8")
            is_text = "\x00" not in decoded
        except UnicodeDecodeError:
            is_text = False
        if is_text:
            text_files += 1
        else:
            binary_files += 1
        file_sha = hashlib.sha256(data).hexdigest()
        manifest.update(os.fsencode(relative))
        manifest.update(b"\0")
        manifest.update(str(len(data)).encode("ascii"))
        manifest.update(b"\0")
        manifest.update(file_sha.encode("ascii"))
        manifest.update(b"\n")
    return {
        "tracked_files": len(paths),
        "tracked_bytes": total_bytes,
        "text_files": text_files,
        "binary_files": binary_files,
        "manifest_sha256": manifest.hexdigest(),
    }


def bootstrap_preconditions(root: Path, live_main_sha: str) -> tuple[str, str]:
    branch = _git_text(root, "branch", "--show-current")
    head = _git_text(root, "rev-parse", "HEAD")
    if not branch or branch == "main":
        raise RuntimeError("bootstrap must run on a dedicated task branch, never directly on main")
    if head != live_main_sha:
        raise RuntimeError(
            f"local HEAD {head} is not current main {live_main_sha}; rebuild the branch first"
        )
    status = _git_text(root, "status", "--porcelain", "--untracked-files=normal")
    if status:
        raise RuntimeError("bootstrap must run before edits on a clean task branch")
    return branch, head


def bootstrap_receipt_body(
    *,
    agent: str,
    session_id: str,
    task: str,
    branch: str,
    live_main_sha: str,
    files: str,
    manifest: dict[str, object],
    active_session_count: int,
    conflicts: list[str],
) -> str:
    return "\n".join(
        [
            "AI-BOOTSTRAP",
            f"agent: {agent}",
            f"session_id: {session_id}",
            f"task: {task}",
            f"branch: {branch}",
            f"main_sha: {live_main_sha}",
            f"files: {files}",
            f"tracked_files: {manifest['tracked_files']}",
            f"tracked_bytes: {manifest['tracked_bytes']}",
            f"text_files: {manifest['text_files']}",
            f"binary_files: {manifest['binary_files']}",
            f"manifest_sha256: {manifest['manifest_sha256']}",
            f"observed_active_sessions: {active_session_count}",
            "conflict_state: " + ("coordination_required" if conflicts else "clear"),
            "status: complete",
        ]
    )


def run_bootstrap(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    live_main = main_sha(args.repo)
    branch, _head = bootstrap_preconditions(root, live_main)
    if args.branch and args.branch != branch:
        raise RuntimeError(
            f"--branch {args.branch!r} does not match current branch {branch!r}"
        )
    manifest = tracked_repository_manifest(root)
    comments = coordination_comments(args.repo, args.issue)
    leases = active_leases(comments)
    conflicts: list[str] = []
    for lease in leases:
        other_branch = lease.get("branch", "")
        if other_branch == branch:
            continue
        shared = declared_scope_overlap(args.files, lease.get("files"))
        if shared:
            conflicts.append(
                f"{other_branch} ({lease.get('task', 'unknown task')}): " + "; ".join(shared)
            )
    receipt = bootstrap_receipt_body(
        agent=args.agent,
        session_id=args.session_id,
        task=args.task,
        branch=branch,
        live_main_sha=live_main,
        files=args.files,
        manifest=manifest,
        active_session_count=len(leases),
        conflicts=conflicts,
    )
    print(
        json.dumps(
            {
                "main_sha": live_main,
                "branch": branch,
                "manifest": manifest,
                "active_session_count": len(leases),
                "active_sessions": leases,
                "scope_conflicts": conflicts,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    print("\n" + receipt)
    if args.post:
        github_post(args.repo, f"/issues/{args.issue}/comments", {"body": receipt})
        print(f"bootstrap receipt posted to issue #{args.issue}")
    if conflicts:
        print(
            "coordination: intended scope overlaps active work; do not edit until an "
            "AI-COLLAB handshake or serialized ownership decision is recorded"
        )
        return 3
    return 0


def check_pr(repository: str, number: int) -> int:
    current = github_get(repository, f"/pulls/{number}")
    if not isinstance(current, dict):
        raise RuntimeError(f"PR #{number} not found")
    current_files = pr_files(repository, number)
    current_draft = bool(current.get("draft"))
    current_head = _pull_ref(current, "head", "ref")
    current_head_sha = _pull_ref(current, "head", "sha") or ""
    current_base = _pull_ref(current, "base", "sha")
    body_value = current.get("body")
    current_body = body_value if isinstance(body_value, str) else ""
    live_main = main_sha(repository)
    comments = coordination_comments(repository)
    leases = active_leases(comments)
    bootstraps = bootstrap_records(comments)
    trusted_manifest = tracked_repository_manifest(Path.cwd())

    print(f"coordination: PR #{number} files={len(current_files)} draft={current_draft}")
    print(f"coordination: head={current_head} base_sha={current_base}")
    print(f"coordination: live_main_sha={live_main} active_sessions={len(leases)}")

    blocking_reasons = pr_coordination_errors(
        head_branch=current_head,
        base_sha=current_base,
        live_main_sha=live_main,
        leases=leases,
        bootstraps=bootstraps,
        expected_manifest=trusted_manifest,
    )
    for reason in blocking_reasons:
        print(f"::error::{reason}")

    current_lease = lease_for_branch(leases, current_head)
    if current_lease is not None:
        outside_scope = sorted(
            path for path in current_files if not scope_contains_path(current_lease.get("files"), path)
        )
        if outside_scope:
            blocking_reasons.append("PR changes files outside its declared AI-LEASE scope")
            print(
                "::error::PR changes files outside AI-LEASE scope: "
                + ", ".join(outside_scope[:30])
            )

    for other in open_pulls(repository):
        other_number = other.get("number")
        if not isinstance(other_number, int) or other_number == number:
            continue
        shared = overlap_paths(current_files, pr_files(repository, other_number))
        if not shared:
            continue
        other_draft = bool(other.get("draft"))
        shown = ", ".join(shared[:20])
        if len(shared) > 20:
            shown += f", ... (+{len(shared) - 20} more)"
        if current_draft or other_draft:
            print(
                f"::warning::PR #{number} overlaps draft/experimental "
                f"PR #{other_number}: {shown}"
            )
            continue
        other_branch = _pull_ref(other, "head", "ref") or ""
        other_lease = lease_for_branch(leases, other_branch)
        if other_lease is None:
            blocking_reasons.append(f"overlap with undeclared PR #{other_number}")
            print(
                f"::error::Overlapping PR #{other_number} branch `{other_branch}` has no "
                "active lease; collaboration cannot be trusted"
            )
            continue
        if has_coordination_override(current_body, other_number):
            print(
                "::warning::Legacy Coordination-Override is present but no longer "
                "authorizes overlapping work"
            )
        allowed, reason = collaboration_allows_overlap(
            comments,
            left_branch=current_head or "",
            right_branch=other_branch,
            base_sha=current_base or "",
            shared_files=shared,
            subject_pr=number,
            subject_head_sha=current_head_sha,
        )
        if not allowed:
            blocking_reasons.append(f"uncoordinated overlap with PR #{other_number}")
            print(
                f"::error::PR #{number} overlaps non-draft PR #{other_number}: {shown}. "
                f"{reason}. Record AI-COLLAB + ACK from both branches + exact-head "
                "AI-COLLAB-REVIEW on issue #66."
            )
        else:
            print(f"coordination: overlap with PR #{other_number} accepted: {reason}")

    for lease in leases:
        print(
            "coordination: session "
            f"agent={lease.get('agent')} session={lease.get('session_id')} "
            f"branch={lease.get('branch')} task={lease.get('task')} files={lease.get('files')}"
        )
    return 2 if blocking_reasons else 0


def notify_main(repository: str, issue: int) -> None:
    sha = main_sha(repository)
    commit = github_get(repository, f"/commits/{sha}")
    message = "unknown"
    if isinstance(commit, dict):
        commit_data = commit.get("commit")
        if isinstance(commit_data, dict) and isinstance(commit_data.get("message"), str):
            message = commit_data["message"].splitlines()[0]
    pulls = open_pulls(repository)
    comments = coordination_comments(repository, issue)
    leases = active_leases(comments)
    lease_branches = {lease.get("branch") for lease in leases}
    ready = [
        f"#{pr.get('number')} {pr.get('title')}"
        for pr in pulls
        if not bool(pr.get("draft"))
    ]
    undeclared = [
        f"#{pr.get('number')} {_pull_ref(pr, 'head', 'ref') or 'unknown-branch'}"
        for pr in pulls
        if not bool(pr.get("draft"))
        and _pull_ref(pr, "head", "ref") not in lease_branches
    ]
    session_rows = [
        f"{lease.get('agent', 'unknown')}[{lease.get('session_id', 'legacy')}] "
        f"branch={lease.get('branch')} task={lease.get('task')} files={lease.get('files')}"
        for lease in leases
    ]
    body = "\n".join(
        [
            "AI-MAIN-UPDATE",
            f"observed_at: {datetime.now(UTC).isoformat()}",
            f"main_sha: {sha}",
            f"commit: {message}",
            f"active_sessions: {len(leases)}",
            "session_details: " + (" | ".join(session_rows) or "none declared"),
            "open_non_draft_prs: " + (" | ".join(ready) or "none"),
            "undeclared_non_draft_prs: " + (" | ".join(undeclared) or "none"),
        ]
    )
    github_post(repository, f"/issues/{issue}/comments", {"body": body})
    print(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="print live coordination state as JSON")
    snapshot.add_argument("--repo", default=DEFAULT_REPOSITORY)
    snapshot.add_argument("--issue", type=int, default=COORDINATION_ISSUE)

    bootstrap = subparsers.add_parser(
        "bootstrap",
        help="scan every tracked file and create a pre-edit AI-BOOTSTRAP receipt",
    )
    bootstrap.add_argument("--repo", default=DEFAULT_REPOSITORY)
    bootstrap.add_argument("--issue", type=int, default=COORDINATION_ISSUE)
    bootstrap.add_argument("--root", type=Path, default=Path("."))
    bootstrap.add_argument("--agent", required=True)
    bootstrap.add_argument("--session-id", required=True)
    bootstrap.add_argument("--task", required=True)
    bootstrap.add_argument("--branch")
    bootstrap.add_argument("--files", required=True)
    bootstrap.add_argument("--post", action="store_true")

    check = subparsers.add_parser("check-pr", help="enforce the PR coordination policy")
    check.add_argument("--repo", default=DEFAULT_REPOSITORY)
    check.add_argument("--pr", type=int, required=True)

    notify = subparsers.add_parser("notify-main", help="publish a main update to the board")
    notify.add_argument("--repo", default=DEFAULT_REPOSITORY)
    notify.add_argument("--issue", type=int, default=COORDINATION_ISSUE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "snapshot":
        print(json.dumps(build_snapshot(args.repo, args.issue), indent=2, sort_keys=True))
        return 0
    if args.command == "bootstrap":
        return run_bootstrap(args)
    if args.command == "check-pr":
        return check_pr(args.repo, args.pr)
    if args.command == "notify-main":
        notify_main(args.repo, args.issue)
        return 0
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
