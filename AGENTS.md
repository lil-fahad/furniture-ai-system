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
5. Register an `AI-LEASE` comment on issue #66 with agent/tool name, task, branch, base SHA, file scope, expiry, and `status: active`.
6. If another unexpired lease or non-draft PR overlaps the intended files, do not begin conflicting edits until the scope is coordinated.

## Lease format

```text
AI-LEASE
agent: <tool/model or integration name>
task: <short task>
branch: <branch>
base_sha: <main SHA when work started>
files: <paths/globs>
lease_until: <ISO-8601 UTC>
status: active
```

Renew the lease before it expires if work continues. When work ends, post:

```text
AI-RELEASE
agent: <tool/model or integration name>
branch: <branch>
status: completed|abandoned
result: <PR/commit or explanation>
```

## Conflict rules

- Never push coding work directly to `main`; use a task branch and PR.
- One active task owns a file scope at a time unless the owners explicitly coordinate.
- A non-draft PR must not silently overlap another non-draft PR. The AI coordination CI guard treats this as a conflict.
- Draft PR overlaps are warnings, not authorization to overwrite another branch.
- Before merging, re-check the live main SHA, PR mergeability, required CI, the coordination board, and overlapping PRs.
- If `main` advanced after work began, review the intervening commits and rerun tests on the exact merge candidate.
- Do not close, rewrite, or supersede another agent's work without recording the decision on issue #66.

## Current status

Never copy a static "current agent" list from this file. Use issue #66 or the `snapshot` command; they are designed to show the latest main SHA, active leases, open work, and overlap warnings.

## Agent-specific instruction files

- GitHub Copilot: `.github/copilot-instructions.md`
- Claude-compatible tools: `CLAUDE.md`

Those files point back here. This file and issue #66 are authoritative.
