#!/usr/bin/env python3
"""FurnitureAI multi-agent coordination helper.

Reads live GitHub state, reports active AI leases from issue #66, rejects stale
or undeclared pull requests, detects parallel PR file overlap, requires board-
recorded collaboration for deliberate overlap, and publishes main updates.
Uses only the Python standard library so the guard can run before dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

COORDINATION_ISSUE = 66
DEFAULT_REPOSITORY = "lil-fahad/furniture-ai-system"
API_ROOT = "https://api.github.com"
REQUIRED_LEASE_FIELDS = (
    "agent", "task", "branch", "base_sha", "files", "lease_until", "status"
)
REQUIRED_COLLAB_FIELDS = (
    "agents", "branches", "prs", "files", "integration_owner", "plan", "tests", "ack", "status"
)


def _token() -> str | None:
    value = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    return value.strip() if value and value.strip() else None


def _request(url: str, *, method: str = "GET", payload: dict[str, object] | None = None, token: str | None = None) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "FurnitureAI-AI-Coordination/3",
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
    return None if not raw else json.loads(raw.decode("utf-8"))


def github_get(repository: str, endpoint: str) -> Any:
    return _request(f"{API_ROOT}/repos/{repository}{endpoint}")


def github_get_pages(repository: str, endpoint: str) -> list[dict[str, Any]]:
    separator = "&" if "?" in endpoint else "?"
    results: list[dict[str, Any]] = []
    for page in range(1, 101):
        payload = _request(f"{API_ROOT}/repos/{repository}{endpoint}{separator}per_page=100&page={page}")
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
    return _request(f"{API_ROOT}/repos/{repository}{endpoint}", method="POST", payload=payload, token=token)


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


def active_leases(comments: list[dict[str, Any]], now: datetime | None = None) -> list[dict[str, str]]:
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
        if release is not None and release.get("branch"):
            by_branch.pop(release["branch"], None)
    active: list[dict[str, str]] = []
    for lease in by_branch.values():
        if lease.get("status", "active").lower() != "active" or not lease.get("lease_until"):
            continue
        try:
            if parse_iso8601(lease["lease_until"]) <= current:
                continue
        except ValueError:
            continue
        active.append(lease)
    return sorted(active, key=lambda item: (item.get("agent", ""), item.get("branch", "")))


def active_collaborations(comments: list[dict[str, Any]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for comment in comments:
        body = comment.get("body")
        if not isinstance(body, str):
            continue
        record = parse_agent_record(body, "AI-COLLAB")
        if record and record.get("status", "active").lower() == "active":
            records.append(record)
    return records


def lease_for_branch(leases: list[dict[str, str]], branch: str | None) -> dict[str, str] | None:
    if not branch:
        return None
    return next((lease for lease in leases if lease.get("branch") == branch), None)


def pr_coordination_errors(*, head_branch: str | None, base_sha: str | None, live_main_sha: str, leases: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    if not head_branch:
        return ["PR head branch could not be resolved"]
    if base_sha != live_main_sha:
        errors.append("PR base SHA is not the current main SHA; rebuild/update the branch from current main and rerun exact-head CI")
    lease = lease_for_branch(leases, head_branch)
    if lease is None:
        errors.append(f"PR branch `{head_branch}` has no active AI-LEASE on issue #{COORDINATION_ISSUE}")
        return errors
    missing = [field for field in REQUIRED_LEASE_FIELDS if not lease.get(field)]
    if missing:
        errors.append("AI-LEASE is missing required fields: " + ", ".join(missing))
    if lease.get("base_sha") != base_sha:
        errors.append("AI-LEASE base_sha does not match the PR base SHA; renew the lease after updating the branch")
    return errors


def overlap_paths(left: set[str], right: set[str]) -> list[str]:
    return sorted(left.intersection(right))


def has_coordination_override(body: str | None, other_pr: int) -> bool:
    return bool(body and f"coordination-override: #{other_pr}".lower() in body.lower())


def _split_refs(value: str) -> set[str]:
    return {part.strip().lower() for part in value.replace(",", "|").split("|") if part.strip()}


def collaboration_authorizes(collaborations: list[dict[str, str]], *, current_pr: int, other_pr: int, current_branch: str | None, other_branch: str | None, shared: list[str]) -> bool:
    wanted_prs = {f"#{current_pr}", f"#{other_pr}"}
    wanted_branches = {branch.lower() for branch in (current_branch, other_branch) if branch}
    for record in collaborations:
        if any(not record.get(field) for field in REQUIRED_COLLAB_FIELDS):
            continue
        if not wanted_prs.issubset(_split_refs(record.get("prs", ""))):
            continue
        if not wanted_branches.issubset(_split_refs(record.get("branches", ""))):
            continue
        declared_files = _split_refs(record.get("files", ""))
        if not all(path.lower() in declared_files for path in shared):
            continue
        if len(_split_refs(record.get("agents", ""))) < 2:
            continue
        if len(_split_refs(record.get("ack", ""))) < 2:
            continue
        return True
    return False


def main_sha(repository: str) -> str:
    payload = github_get(repository, "/commits/main")
    if not isinstance(payload, dict) or not isinstance(payload.get("sha"), str):
        raise RuntimeError("Could not resolve main SHA")
    return payload["sha"]


def open_pulls(repository: str) -> list[dict[str, Any]]:
    return github_get_pages(repository, "/pulls?state=open")


def pr_files(repository: str, number: int) -> set[str]:
    return {row["filename"] for row in github_get_pages(repository, f"/pulls/{number}/files") if isinstance(row.get("filename"), str)}


def coordination_comments(repository: str, issue: int = COORDINATION_ISSUE) -> list[dict[str, Any]]:
    return github_get_pages(repository, f"/issues/{issue}/comments")


def _pull_ref(pr: dict[str, Any], side: str, key: str) -> str | None:
    payload = pr.get(side)
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) else None


def build_snapshot(repository: str, issue: int = COORDINATION_ISSUE) -> dict[str, object]:
    pulls = open_pulls(repository)
    comments = coordination_comments(repository, issue)
    leases = active_leases(comments)
    collaborations = active_collaborations(comments)
    lease_by_branch = {lease["branch"]: lease for lease in leases if lease.get("branch")}
    open_rows: list[dict[str, object]] = []
    file_sets: dict[int, set[str]] = {}
    for pr in pulls:
        number = pr.get("number")
        if isinstance(number, int):
            file_sets[number] = pr_files(repository, number)
    for pr in pulls:
        number = pr.get("number")
        head = _pull_ref(pr, "head", "ref")
        lease = lease_by_branch.get(head or "")
        overlaps: list[dict[str, object]] = []
        if isinstance(number, int):
            for other in pulls:
                other_number = other.get("number")
                if not isinstance(other_number, int) or other_number == number:
                    continue
                shared = overlap_paths(file_sets.get(number, set()), file_sets.get(other_number, set()))
                if shared:
                    overlaps.append({"pr": other_number, "draft": bool(other.get("draft")), "files": shared})
        open_rows.append({
            "number": number, "title": pr.get("title"), "draft": bool(pr.get("draft")),
            "head": head, "head_sha": _pull_ref(pr, "head", "sha"), "base_sha": _pull_ref(pr, "base", "sha"),
            "updated_at": pr.get("updated_at"), "coordination_state": "declared" if lease else "undeclared",
            "active_lease": lease, "changed_files": sorted(file_sets.get(number, set())) if isinstance(number, int) else [],
            "overlaps": overlaps,
        })
    return {
        "schema_version": 3, "repository": repository, "observed_at": datetime.now(UTC).isoformat(),
        "main_sha": main_sha(repository), "coordination_issue": issue,
        "active_agent_count": len(leases), "active_agents": leases,
        "active_collaborations": collaborations, "open_pull_request_count": len(open_rows),
        "open_pull_requests": open_rows,
    }


def check_pr(repository: str, number: int) -> int:
    current = github_get(repository, f"/pulls/{number}")
    if not isinstance(current, dict):
        raise RuntimeError(f"PR #{number} not found")
    current_files = pr_files(repository, number)
    current_draft = bool(current.get("draft"))
    current_head = _pull_ref(current, "head", "ref")
    current_base = _pull_ref(current, "base", "sha")
    current_body = current.get("body") if isinstance(current.get("body"), str) else ""
    live_main = main_sha(repository)
    comments = coordination_comments(repository)
    leases = active_leases(comments)
    collaborations = active_collaborations(comments)
    print(f"coordination: PR #{number} files={len(current_files)} draft={current_draft}")
    print(f"coordination: head={current_head} base_sha={current_base}")
    print(f"coordination: live_main_sha={live_main} active_agents={len(leases)}")
    for lease in leases:
        print(f"coordination: active agent={lease.get('agent')} task={lease.get('task')} branch={lease.get('branch')} files={lease.get('files')}")
    blocking_reasons = pr_coordination_errors(head_branch=current_head, base_sha=current_base, live_main_sha=live_main, leases=leases)
    for reason in blocking_reasons:
        print(f"::error::{reason}")
    for other in open_pulls(repository):
        other_number = other.get("number")
        if not isinstance(other_number, int) or other_number == number:
            continue
        shared = overlap_paths(current_files, pr_files(repository, other_number))
        if not shared:
            continue
        shown = ", ".join(shared[:20]) + (f", ... (+{len(shared)-20} more)" if len(shared) > 20 else "")
        if current_draft or bool(other.get("draft")):
            print(f"::warning::PR #{number} overlaps draft/experimental PR #{other_number}: {shown}")
            continue
        other_head = _pull_ref(other, "head", "ref")
        override = has_coordination_override(current_body, other_number)
        authorized = collaboration_authorizes(collaborations, current_pr=number, other_pr=other_number, current_branch=current_head, other_branch=other_head, shared=shared)
        if override and authorized:
            print(f"::warning::Board-recorded collaboration authorizes overlap with PR #{other_number}: {shown}")
            continue
        blocking_reasons.append(f"uncoordinated overlap with PR #{other_number}")
        detail = "override is missing" if not override else "matching AI-COLLAB board evidence is missing/incomplete"
        print(f"::error::Uncoordinated overlap with non-draft PR #{other_number}: {shown}; {detail}. Record joint AI-COLLAB on issue #{COORDINATION_ISSUE} and require combined review/tests.")
    return 2 if blocking_reasons else 0


def notify_main(repository: str, issue: int) -> None:
    snapshot = build_snapshot(repository, issue)
    sha = str(snapshot["main_sha"])
    commit = github_get(repository, f"/commits/{sha}")
    message = "unknown"
    if isinstance(commit, dict) and isinstance(commit.get("commit"), dict) and isinstance(commit["commit"].get("message"), str):
        message = commit["commit"]["message"].splitlines()[0]
    leases = snapshot["active_agents"]
    pulls = snapshot["open_pull_requests"]
    ready = [f"#{pr.get('number')} {pr.get('title')}" for pr in pulls if not pr.get("draft")]
    undeclared = [f"#{pr.get('number')} {pr.get('head') or 'unknown-branch'}" for pr in pulls if not pr.get("draft") and pr.get("coordination_state") == "undeclared"]
    conflicts = [f"#{pr.get('number')} overlaps={len(pr.get('overlaps', []))}" for pr in pulls if pr.get("overlaps")]
    agent_work = [f"{lease.get('agent')} => {lease.get('task')} [{lease.get('branch')}]" for lease in leases]
    body = "\n".join([
        "AI-MAIN-UPDATE", f"observed_at: {datetime.now(UTC).isoformat()}", f"main_sha: {sha}", f"commit: {message}",
        f"active_agent_count: {len(leases)}", "active_agent_work: " + (" | ".join(agent_work) or "none declared"),
        "open_non_draft_prs: " + (" | ".join(ready) or "none"), "undeclared_non_draft_prs: " + (" | ".join(undeclared) or "none"),
        "detected_open_pr_overlaps: " + (" | ".join(conflicts) or "none"),
    ])
    github_post(repository, f"/issues/{issue}/comments", {"body": body})
    print(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot", help="print live coordination state as JSON")
    snapshot.add_argument("--repo", default=DEFAULT_REPOSITORY); snapshot.add_argument("--issue", type=int, default=COORDINATION_ISSUE)
    check = subparsers.add_parser("check-pr", help="enforce the PR coordination policy")
    check.add_argument("--repo", default=DEFAULT_REPOSITORY); check.add_argument("--pr", type=int, required=True)
    notify = subparsers.add_parser("notify-main", help="publish a main update to the board")
    notify.add_argument("--repo", default=DEFAULT_REPOSITORY); notify.add_argument("--issue", type=int, default=COORDINATION_ISSUE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "snapshot":
            print(json.dumps(build_snapshot(args.repo, args.issue), indent=2, sort_keys=True)); return 0
        if args.command == "check-pr":
            return check_pr(args.repo, args.pr)
        if args.command == "notify-main":
            notify_main(args.repo, args.issue); return 0
    except RuntimeError as exc:
        print(f"coordination error: {exc}", file=sys.stderr); return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
