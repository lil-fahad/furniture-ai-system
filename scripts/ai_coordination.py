#!/usr/bin/env python3
"""FurnitureAI multi-agent coordination helper.

Reads live GitHub state, reports active AI leases from issue #66, detects
parallel PR file overlap, and publishes main-branch updates to the coordination
board. The script uses only the Python standard library.
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
        "User-Agent": "FurnitureAI-AI-Coordination/1",
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
        key=lambda item: (item.get("agent", ""), item.get("branch", "")),
    )


def overlap_paths(left: set[str], right: set[str]) -> list[str]:
    return sorted(left.intersection(right))


def has_coordination_override(body: str | None, other_pr: int) -> bool:
    if not body:
        return False
    needle = f"coordination-override: #{other_pr}".lower()
    return needle in body.lower()


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
    pulls = open_pulls(repository)
    leases = active_leases(coordination_comments(repository, issue))
    return {
        "schema_version": 1,
        "repository": repository,
        "observed_at": datetime.now(UTC).isoformat(),
        "main_sha": main_sha(repository),
        "coordination_issue": issue,
        "active_leases": leases,
        "open_pull_requests": [
            {
                "number": pr.get("number"),
                "title": pr.get("title"),
                "draft": bool(pr.get("draft")),
                "head": _pull_ref(pr, "head", "ref"),
                "head_sha": _pull_ref(pr, "head", "sha"),
                "base_sha": _pull_ref(pr, "base", "sha"),
                "updated_at": pr.get("updated_at"),
            }
            for pr in pulls
        ],
    }


def check_pr(repository: str, number: int) -> int:
    current = github_get(repository, f"/pulls/{number}")
    if not isinstance(current, dict):
        raise RuntimeError(f"PR #{number} not found")
    current_files = pr_files(repository, number)
    current_draft = bool(current.get("draft"))
    body_value = current.get("body")
    current_body = body_value if isinstance(body_value, str) else ""
    current_base = _pull_ref(current, "base", "sha")
    live_main = main_sha(repository)

    print(f"coordination: PR #{number} files={len(current_files)} draft={current_draft}")
    print(f"coordination: base_sha={current_base} live_main_sha={live_main}")
    if current_base != live_main:
        print(
            "::warning::PR base SHA is older than current main; review intervening commits "
            "and rerun exact-head CI before merge."
        )

    blocking: list[tuple[int, list[str]]] = []
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
        if has_coordination_override(current_body, other_number):
            print(
                f"::warning::Overlap with PR #{other_number} explicitly acknowledged: "
                f"{shown}"
            )
            continue
        blocking.append((other_number, shared))
        print(
            f"::error::Uncoordinated overlap with non-draft PR #{other_number}: "
            f"{shown}. Coordinate on issue #{COORDINATION_ISSUE} or add "
            f"`Coordination-Override: #{other_number}` with the reason to the PR body."
        )

    leases = active_leases(coordination_comments(repository))
    print(f"coordination: active_leases={len(leases)}")
    for lease in leases:
        print(
            "coordination: lease "
            f"agent={lease.get('agent')} branch={lease.get('branch')} "
            f"task={lease.get('task')} files={lease.get('files')}"
        )
    return 2 if blocking else 0


def notify_main(repository: str, issue: int) -> None:
    sha = main_sha(repository)
    commit = github_get(repository, f"/commits/{sha}")
    message = "unknown"
    if isinstance(commit, dict):
        commit_data = commit.get("commit")
        if isinstance(commit_data, dict) and isinstance(commit_data.get("message"), str):
            message = commit_data["message"].splitlines()[0]
    pulls = open_pulls(repository)
    leases = active_leases(coordination_comments(repository, issue))
    ready = [
        f"#{pr.get('number')} {pr.get('title')}"
        for pr in pulls
        if not bool(pr.get("draft"))
    ]
    agent_names = ", ".join(lease.get("agent", "unknown") for lease in leases)
    body = "\n".join(
        [
            "AI-MAIN-UPDATE",
            f"observed_at: {datetime.now(UTC).isoformat()}",
            f"main_sha: {sha}",
            f"commit: {message}",
            f"active_leases: {len(leases)}",
            "active_agents: " + (agent_names or "none declared"),
            "open_non_draft_prs: " + (" | ".join(ready) or "none"),
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

    check = subparsers.add_parser("check-pr", help="guard a PR against uncoordinated overlap")
    check.add_argument("--repo", default=DEFAULT_REPOSITORY)
    check.add_argument("--pr", type=int, required=True)

    notify = subparsers.add_parser("notify-main", help="publish a main update to the board")
    notify.add_argument("--repo", default=DEFAULT_REPOSITORY)
    notify.add_argument("--issue", type=int, default=COORDINATION_ISSUE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "snapshot":
            print(json.dumps(build_snapshot(args.repo, args.issue), indent=2, sort_keys=True))
            return 0
        if args.command == "check-pr":
            return check_pr(args.repo, args.pr)
        if args.command == "notify-main":
            notify_main(args.repo, args.issue)
            return 0
    except RuntimeError as exc:
        print(f"coordination error: {exc}", file=sys.stderr)
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
