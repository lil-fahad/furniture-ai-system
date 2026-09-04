#!/usr/bin/env python3
"""Fail closed when a PR overlaps another active AI lease without collaboration.

This complements the exact open-PR overlap guard in ``ai_coordination.py``.
It deliberately checks active work even when the peer branch has not opened a
pull request yet, closing the race between pre-edit leasing and PR creation.
"""

from __future__ import annotations

import argparse
from typing import Any

from ai_coordination import (
    COORDINATION_ISSUE,
    DEFAULT_REPOSITORY,
    active_leases,
    bootstrap_for_lease,
    bootstrap_records,
    collaboration_state,
    coordination_comments,
    declared_scope_overlap,
    github_get,
    lease_for_branch,
    pr_files,
    scope_contains_path,
    split_csv,
)


def _pull_ref(pr: dict[str, Any], side: str, key: str) -> str | None:
    payload = pr.get(side)
    if not isinstance(payload, dict):
        return None
    value = payload.get(key)
    return value if isinstance(value, str) else None


def bilateral_collaboration_exists(
    comments: list[dict[str, Any]],
    *,
    left_branch: str,
    right_branch: str,
    base_sha: str,
    exact_files: list[str],
    left_scope: str | None,
    right_scope: str | None,
) -> bool:
    """Require both ACKs and collaboration scope that covers this exact conflict."""
    state = collaboration_state(comments)
    collaborations = state.get("collaborations")
    acknowledgements = state.get("acknowledgements")
    if not isinstance(collaborations, dict) or not isinstance(acknowledgements, dict):
        return False

    wanted = {left_branch, right_branch}
    for collab_id, raw in collaborations.items():
        if not isinstance(collab_id, str) or not isinstance(raw, dict):
            continue
        record = {str(key): str(value) for key, value in raw.items()}
        if record.get("status", "").lower() not in {"active", "agreed"}:
            continue
        if record.get("base_sha") != base_sha:
            continue
        if set(split_csv(record.get("branches"))) != wanted:
            continue

        shared_scope = record.get("shared_files")
        if exact_files:
            if not all(scope_contains_path(shared_scope, path) for path in exact_files):
                continue
        else:
            if not declared_scope_overlap(shared_scope, left_scope):
                continue
            if not declared_scope_overlap(shared_scope, right_scope):
                continue

        ack_map = acknowledgements.get(collab_id)
        if not isinstance(ack_map, dict):
            continue
        if all(
            isinstance(ack_map.get(branch), dict)
            and str(ack_map[branch].get("status", "")).lower() == "accepted"
            for branch in wanted
        ):
            return True
    return False


def active_lease_conflicts(
    *,
    current_files: set[str],
    current_lease: dict[str, str],
    leases: list[dict[str, str]],
    comments: list[dict[str, Any]],
    base_sha: str,
) -> list[str]:
    """Describe uncoordinated conflicts with active leases, including peers without PRs."""
    current_branch = current_lease.get("branch", "")
    errors: list[str] = []
    for other in leases:
        other_branch = other.get("branch", "")
        if not other_branch or other_branch == current_branch:
            continue
        exact = sorted(
            path
            for path in current_files
            if scope_contains_path(other.get("files"), path)
        )
        declared = declared_scope_overlap(
            current_lease.get("files"),
            other.get("files"),
        )
        if not exact and not declared:
            continue
        if bilateral_collaboration_exists(
            comments,
            left_branch=current_branch,
            right_branch=other_branch,
            base_sha=base_sha,
            exact_files=exact,
            left_scope=current_lease.get("files"),
            right_scope=other.get("files"),
        ):
            continue
        detail = ", ".join(exact[:20]) if exact else "; ".join(declared[:20])
        errors.append(
            f"active lease overlap with `{other_branch}` is not bilaterally coordinated: "
            f"{detail}"
        )
    return errors


def bootstrap_conflict_errors(
    *,
    receipt: dict[str, str] | None,
    unresolved_live_conflicts: list[str],
) -> list[str]:
    """Require an explicit conflict state and prevent ignored bootstrap conflicts."""
    if receipt is None:
        return ["matching AI-BOOTSTRAP receipt is missing"]
    conflict_state = receipt.get("conflict_state", "").lower()
    if conflict_state not in {"clear", "coordination_required"}:
        return ["AI-BOOTSTRAP conflict_state must be clear or coordination_required"]
    if conflict_state == "coordination_required" and unresolved_live_conflicts:
        return [
            "AI-BOOTSTRAP reported coordination_required and live conflicts remain "
            "without bilateral acknowledgement"
        ]
    return []


def check_pr(repository: str, number: int) -> int:
    current = github_get(repository, f"/pulls/{number}")
    if not isinstance(current, dict):
        raise RuntimeError(f"PR #{number} not found")
    head_branch = _pull_ref(current, "head", "ref")
    base_sha = _pull_ref(current, "base", "sha")
    if not head_branch or not base_sha:
        print("::error::Could not resolve PR branch/base")
        return 2

    comments = coordination_comments(repository, COORDINATION_ISSUE)
    leases = active_leases(comments)
    current_lease = lease_for_branch(leases, head_branch)
    if current_lease is None:
        print(f"::error::No active AI-LEASE for `{head_branch}`")
        return 2

    conflicts = active_lease_conflicts(
        current_files=pr_files(repository, number),
        current_lease=current_lease,
        leases=leases,
        comments=comments,
        base_sha=base_sha,
    )
    receipt = bootstrap_for_lease(bootstrap_records(comments), current_lease)
    errors = conflicts + bootstrap_conflict_errors(
        receipt=receipt,
        unresolved_live_conflicts=conflicts,
    )
    for error in errors:
        print(f"::error::{error}")
    if not errors:
        print(
            "coordination: active-lease guard clear; no uncoordinated live scope "
            "conflicts remain"
        )
    return 2 if errors else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPOSITORY)
    parser.add_argument("--pr", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return check_pr(args.repo, args.pr)


if __name__ == "__main__":
    raise SystemExit(main())
