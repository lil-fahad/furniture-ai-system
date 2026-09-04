#!/usr/bin/env python3
"""Create trusted AI bootstrap receipts for connector/API-only coding agents.

The request arrives as an issue #66 comment. This script is intended to run
only from a trusted ``main`` checkout in GitHub Actions. It verifies that the
requested task branch still points at current ``main``, computes the same
canonical content manifest from GitHub Git objects, inventories live work, and
posts the authoritative ``AI-BOOTSTRAP`` receipt as the Actions bot.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
from urllib.parse import quote

from ai_coordination import (
    COORDINATION_ISSUE,
    DEFAULT_REPOSITORY,
    active_leases,
    bootstrap_receipt_body,
    coordination_comments,
    declared_scope_overlap,
    github_get,
    github_post,
    main_sha,
    open_pulls,
    parse_agent_record,
    pr_files,
    scope_contains_path,
)

REQUIRED_REQUEST_FIELDS = ("agent", "session_id", "task", "branch", "files")


def parse_request(body: str) -> dict[str, str]:
    record = parse_agent_record(body, "AI-BOOTSTRAP-REQUEST")
    if record is None:
        raise RuntimeError("comment is not an AI-BOOTSTRAP-REQUEST")
    missing = [field for field in REQUIRED_REQUEST_FIELDS if not record.get(field)]
    if missing:
        raise RuntimeError(
            "AI-BOOTSTRAP-REQUEST is missing fields: " + ", ".join(missing)
        )
    if record.get("status") and record["status"].lower() not in {"requested", "new"}:
        raise RuntimeError("AI-BOOTSTRAP-REQUEST status must be requested/new when present")
    return record


def _blob_bytes(repository: str, sha: str, expected_size: int | None) -> bytes:
    payload = github_get(repository, f"/git/blobs/{sha}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Git blob {sha} could not be read")
    if payload.get("encoding") != "base64" or not isinstance(payload.get("content"), str):
        raise RuntimeError(f"Git blob {sha} did not return canonical base64 content")
    compact = payload["content"].replace("\n", "")
    try:
        data = base64.b64decode(compact, validate=True)
    except ValueError as exc:
        raise RuntimeError(f"Git blob {sha} returned invalid base64") from exc
    if expected_size is not None and len(data) != expected_size:
        raise RuntimeError(
            f"Git blob {sha} size mismatch: API={expected_size}, decoded={len(data)}"
        )
    return data


def remote_repository_manifest(repository: str, commit_sha: str) -> dict[str, object]:
    commit = github_get(repository, f"/git/commits/{commit_sha}")
    if not isinstance(commit, dict):
        raise RuntimeError("Git commit could not be resolved")
    tree_ref = commit.get("tree")
    if not isinstance(tree_ref, dict) or not isinstance(tree_ref.get("sha"), str):
        raise RuntimeError("Git commit did not expose a tree SHA")

    tree = github_get(repository, f"/git/trees/{tree_ref['sha']}?recursive=1")
    if not isinstance(tree, dict) or not isinstance(tree.get("tree"), list):
        raise RuntimeError("Git tree could not be read")
    if tree.get("truncated") is True:
        raise RuntimeError("recursive Git tree was truncated; refusing incomplete bootstrap")

    rows: list[tuple[str, str, int | None]] = []
    for item in tree["tree"]:
        if not isinstance(item, dict) or item.get("type") != "blob":
            continue
        path = item.get("path")
        sha = item.get("sha")
        size = item.get("size")
        if not isinstance(path, str) or not isinstance(sha, str):
            raise RuntimeError("Git tree contains an invalid blob entry")
        rows.append((path, sha, size if isinstance(size, int) else None))

    manifest = hashlib.sha256()
    total_bytes = 0
    text_files = 0
    binary_files = 0
    for path, blob_sha, size in sorted(rows, key=lambda row: row[0]):
        data = _blob_bytes(repository, blob_sha, size)
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
        digest = hashlib.sha256(data).hexdigest()
        manifest.update(os.fsencode(path))
        manifest.update(b"\0")
        manifest.update(str(len(data)).encode("ascii"))
        manifest.update(b"\0")
        manifest.update(digest.encode("ascii"))
        manifest.update(b"\n")

    return {
        "tracked_files": len(rows),
        "tracked_bytes": total_bytes,
        "text_files": text_files,
        "binary_files": binary_files,
        "manifest_sha256": manifest.hexdigest(),
    }


def branch_head_sha(repository: str, branch: str) -> str:
    payload = github_get(repository, f"/branches/{quote(branch, safe='')}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"branch `{branch}` could not be resolved")
    commit = payload.get("commit")
    if not isinstance(commit, dict) or not isinstance(commit.get("sha"), str):
        raise RuntimeError(f"branch `{branch}` did not expose a head SHA")
    return commit["sha"]


def live_scope_conflicts(
    repository: str,
    *,
    branch: str,
    files: str,
    leases: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, object]]]:
    conflicts: set[str] = set()
    for lease in leases:
        other_branch = lease.get("branch", "")
        if not other_branch or other_branch == branch:
            continue
        shared = declared_scope_overlap(files, lease.get("files"))
        if shared:
            conflicts.add(
                f"lease {other_branch} ({lease.get('task', 'unknown task')}): "
                + "; ".join(shared)
            )

    inventory: list[dict[str, object]] = []
    for pr in open_pulls(repository):
        number = pr.get("number")
        if not isinstance(number, int):
            continue
        head = pr.get("head")
        head_branch = head.get("ref") if isinstance(head, dict) else None
        changed = pr_files(repository, number)
        inventory.append(
            {
                "number": number,
                "title": pr.get("title"),
                "draft": bool(pr.get("draft")),
                "head": head_branch,
                "changed_files": sorted(changed),
            }
        )
        if head_branch == branch:
            continue
        exact = sorted(path for path in changed if scope_contains_path(files, path))
        if exact:
            conflicts.add(
                f"PR #{number} {head_branch or 'unknown-branch'}: exact files "
                + ", ".join(exact)
            )
    return sorted(conflicts), inventory


def process_request(
    repository: str,
    issue: int,
    request_body: str,
    *,
    post: bool,
) -> tuple[int, str]:
    request = parse_request(request_body)
    live_main = main_sha(repository)
    branch = request["branch"]
    if branch_head_sha(repository, branch) != live_main:
        raise RuntimeError(
            "requested branch is not a clean pre-edit pointer to current main; "
            "create/rebuild it from live main first"
        )

    manifest = remote_repository_manifest(repository, live_main)
    comments = coordination_comments(repository, issue)
    leases = active_leases(comments)
    conflicts, _inventory = live_scope_conflicts(
        repository,
        branch=branch,
        files=request["files"],
        leases=leases,
    )
    receipt = bootstrap_receipt_body(
        agent=request["agent"],
        session_id=request["session_id"],
        task=request["task"],
        branch=branch,
        live_main_sha=live_main,
        files=request["files"],
        manifest=manifest,
        active_session_count=len(leases),
        conflicts=conflicts,
    )
    receipt += "\nbootstrap_source: github-actions"
    if post:
        github_post(repository, f"/issues/{issue}/comments", {"body": receipt})
    return (3 if conflicts else 0), receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY)
    parser.add_argument("--issue", type=int, default=COORDINATION_ISSUE)
    parser.add_argument("--request-env", default="AI_BOOTSTRAP_REQUEST")
    parser.add_argument("--no-post", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    body = os.environ.get(args.request_env, "")
    if not body:
        raise RuntimeError(f"environment variable {args.request_env} is empty")
    code, receipt = process_request(
        args.repo,
        args.issue,
        body,
        post=not args.no_post,
    )
    print(receipt)
    if code:
        print(
            "coordination: scope conflict detected; receipt recorded but editing must "
            "not begin until bilateral collaboration or serialized ownership resolves it"
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
