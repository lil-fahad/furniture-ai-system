# FurnitureAI multi-agent coordination

This file is mandatory reading for every coding agent, AI integration, automation, and human-assisted coding tool that modifies this repository.

## Live source of truth

The coordination board is GitHub issue **#66**:
https://github.com/lil-fahad/furniture-ai-system/issues/66

Do not infer active workers from the GitHub account name. Multiple AI systems may operate through the same account. An agent is considered active only when it has an unexpired `AI-LEASE` on issue #66.

## Required preflight

Before changing any file:

1. Fetch the latest `main` and record its SHA.
2. Read issue #66 and its newest comments.
3. Run, when available:
   `python scripts/ai_coordination.py snapshot --repo lil-fahad/furniture-ai-system`
4. Inspect open PRs and their changed files.
5. Create a task branch from the exact current `main` SHA.
6. Register an `AI-LEASE` comment on issue #66 with agent/tool name, task, branch, base SHA, file scope, expiry, and `status: active` **before editing**.
7. If another unexpired lease or non-draft PR overlaps the intended files, do not begin conflicting edits until the scope is coordinated.

## Lease format

```text
AI-LEASE
agent: <tool/model or integration name>
task: <short task>
branch: <branch>
base_sha: <main SHA the branch is currently based on>
files: <paths/globs>
lease_until: <ISO-8601 UTC>
status: active
```

If `main` advances and the branch is updated/rebuilt, renew the lease with the new `base_sha`. A lease whose `base_sha` no longer matches the PR base is invalid for merge.

When work ends, post:

```text
AI-RELEASE
agent: <tool/model or integration name>
branch: <branch>
status: completed|abandoned
result: <PR/commit or explanation>
```

## Conflict and merge rules

- Never push coding work directly to `main`; use a task branch and PR.
- Every PR must have an active lease whose `branch` exactly matches the PR head branch.
- The PR must be based on the current `main` SHA. A stale base is a blocking error, not a warning.
- One active task owns a file scope at a time unless the owners explicitly coordinate.
- A non-draft PR must not silently overlap another non-draft PR. The AI coordination CI guard treats this as a conflict.
- Draft PR overlaps are warnings, not authorization to overwrite another branch.
- A deliberate file overlap may use `Coordination-Override: #<other PR>` only after the owners record the coordination decision on issue #66. The override never bypasses lease or current-main requirements.
- Before merging, re-check the live main SHA, PR mergeability, required CI, the coordination board, and overlapping PRs.
- If `main` advances, update/rebuild the branch, renew its lease, and rerun exact-head CI.
- Do not close, rewrite, or supersede another agent's work without recording the decision on issue #66.
- Never use a manual merge to bypass a failed coordination check.

## Trusted guard execution

The PR coordination workflow executes the guard implementation checked out from trusted `main`, not from the pull request branch. A PR therefore cannot weaken its own guard by editing `scripts/ai_coordination.py` or `.github/workflows/ai-coordination.yml` in the same change.

For hard enforcement at the GitHub merge-button level, repository settings should require the `ai-coordination / pr-conflict-guard` status check and require branches to be up to date before merge. Agents must treat those rules as mandatory even if repository permissions temporarily allow a manual bypass.

## Current status

Never copy a static "current agent" list from this file. Use issue #66 or the `snapshot` command. They show the latest `main` SHA, active leases, open work, whether each open PR has a declared lease, and overlap/staleness information.

## Agent-specific instruction files

- GitHub Copilot: `.github/copilot-instructions.md`
- Claude-compatible tools: `CLAUDE.md`

Those files point back here. This file and issue #66 are authoritative.
