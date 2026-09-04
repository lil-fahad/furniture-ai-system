# FurnitureAI multi-agent coordination

This file is mandatory reading for every coding agent, AI integration, automation, and human-assisted coding tool that modifies this repository.

## Live source of truth

The coordination board is GitHub issue **#66**:
https://github.com/lil-fahad/furniture-ai-system/issues/66

Do not infer active workers from the GitHub account name. Multiple AI systems may operate through the same account. An agent is considered active only when it has an unexpired `AI-LEASE` on issue #66.

## Mandatory entry protocol — read the whole project first

**No AI may edit code immediately after entering the project.** Every new AI session/agent must first build a complete repository map and understand the live parallel work.

Before changing any file, the agent must:

1. Fetch the exact latest `main` SHA and use that SHA as the project baseline.
2. Read this `AGENTS.md`, `README.md`, `SECURITY.md`, `CLAUDE.md`, `.github/copilot-instructions.md`, `docs/AI_COORDINATION.md`, all workflow files under `.github/workflows/`, and the repository tree recursively.
3. Inspect **every tracked project path** in the recursive tree. Read all human-readable source/config/docs/test files relevant to understanding architecture, contracts, training, data/provenance, deployment, security, CI, and the intended task. Large/binary/model/data artifacts must at minimum be inventoried by path/metadata; do not pretend to have semantically read opaque binary bytes.
4. Read issue #66 and its newest comments, then run when available:
   `python scripts/ai_coordination.py snapshot --repo lil-fahad/furniture-ai-system`
5. From the live snapshot, explicitly determine: current `main`, `active_agent_count`, each active agent identity, task, branch, base SHA and file scope, every open PR, and every detected overlap/conflict.
6. Inspect changed files for all open PRs that are active, non-draft, undeclared, or potentially related to the intended task. Do not rely only on PR titles.
7. Only after steps 1–6, choose a task branch from the exact current `main` SHA and register an `AI-LEASE` on issue #66 **before editing**.

If the tool cannot access enough repository state to complete this preflight, it must stop and report the missing visibility instead of editing from assumptions.

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

## Conflict means collaborate, not overwrite

- Never push coding work directly to `main`; use a task branch and PR.
- Every PR must have an active lease whose `branch` exactly matches the PR head branch.
- The PR must be based on the current `main` SHA. A stale base is a blocking error, not a warning.
- One active task owns a file scope at a time unless the owners explicitly coordinate.
- A non-draft PR must not silently overlap another non-draft PR. The AI coordination CI guard treats this as a conflict.
- Draft PR overlaps are warnings, not authorization to overwrite another branch.
- When two agents need the same file or coupled subsystem, neither agent may independently overwrite or supersede the other. They must record a shared `AI-COLLAB` decision on issue #66 containing both branches/agents, the overlapping files, integration owner, review/test plan, and both agents' acknowledgement references.
- `Coordination-Override: #<other PR>` is valid only when issue #66 contains a matching acknowledged `AI-COLLAB` record. The guard must fail closed if the PR merely contains the override text without the board evidence.
- Collaborative overlap requires joint review of the combined diff plus the relevant unit/integration/regression tests. Passing Git mergeability alone is not proof that the combined behavior is correct.
- Before merging, re-check live `main`, active agents, PR mergeability, exact-head CI, the coordination board, and overlaps. If `main` advances, rebuild/update, renew the lease, and rerun exact-head CI.
- Do not close, rewrite, or supersede another agent's work without recording the decision on issue #66.
- Never use a manual merge to bypass a failed coordination check.

### Collaboration record

```text
AI-COLLAB
agents: <agent A> | <agent B>
branches: <branch A> | <branch B>
prs: #<A> | #<B>
files: <overlapping paths>
integration_owner: <agent/branch responsible for combined branch>
plan: <how the implementations will be reconciled>
tests: <joint tests/CI required before merge>
ack: <explicit acknowledgement identifiers from both sides>
status: active
```

The collaboration record is not permission to skip tests or stale-base rules. It only authorizes a deliberate overlap after both pieces of work have been inspected together.

## Trusted guard execution

The PR coordination workflow executes the guard implementation checked out from trusted `main`, not from the pull request branch. A PR therefore cannot weaken its own guard by editing `scripts/ai_coordination.py` or `.github/workflows/ai-coordination.yml` in the same change.

For hard enforcement at the GitHub merge-button level, repository settings should require the `ai-coordination / pr-conflict-guard` status check and require branches to be up to date before merge. Agents must treat those rules as mandatory even if repository permissions temporarily allow a manual bypass.

## Current status

Never copy a static current-agent list from this file. Use issue #66 or `python scripts/ai_coordination.py snapshot --repo lil-fahad/furniture-ai-system`. The snapshot must report the live active-agent count and work inventory every time an AI enters.

## Agent-specific instruction files

- GitHub Copilot: `.github/copilot-instructions.md`
- Claude-compatible tools: `CLAUDE.md`

Those files point back here. This file and issue #66 are authoritative.
