# FurnitureAI AI coordination

FurnitureAI can be modified by more than one AI/coding tool at the same time. GitHub authentication alone cannot identify which model or external AI is behind a branch because multiple tools may use the same repository account. The coordination layer therefore uses explicit leases plus live GitHub state.

## Authoritative surfaces

- Root `AGENTS.md` — mandatory operating rules and entry protocol for all coding agents.
- GitHub issue #66 — live lease/collaboration registry and chronological `main` update feed.
- `scripts/ai_coordination.py` — machine-readable active-work snapshot and fail-closed PR guard.
- `.github/workflows/ai-coordination.yml` — trusted-base conflict checking and main-update publishing.

## Mandatory AI entry sequence

Every AI must begin by understanding the repository rather than immediately editing a file. It must read the recursive repository tree; inspect the architecture, source, tests, configuration, CI/workflows, security, training, data/provenance and deployment documentation relevant to the system; inventory opaque binary/model/data artifacts; read the live coordination board; and inspect all open work that may interact with its intended task.

The requirement to read the project "fully" means complete project awareness, not pretending binary model weights or huge datasets are prose. Every tracked path must be inventoried; human-readable code/config/docs/tests needed to understand behavior and interfaces must be read; opaque artifacts are identified by metadata/path and verified with their existing provenance/checksum mechanisms where applicable.

Before editing, run:

```bash
python scripts/ai_coordination.py snapshot --repo lil-fahad/furniture-ai-system
```

Schema v3 reports `main_sha`, `active_agent_count`, each active agent's task/branch/base/file scope, active collaboration records, every open PR, exact changed-file lists, and detected PR overlaps. This is the required answer to "how many AIs are working now and what are they doing?" Do not use a static list.

If repository or coordination visibility is incomplete, the AI must stop rather than code from assumptions.

## Identity and active work

An AI becomes active by posting an unexpired `AI-LEASE` on issue #66. GitHub cannot truthfully infer an external model identity from a shared account, so an open PR without a matching lease is `undeclared` and is blocked rather than assigned a guessed identity.

## Fail-closed PR policy

For every PR targeting `main`, the guard requires:

1. active branch-matching `AI-LEASE` with all required fields;
2. lease `base_sha` matching the PR base SHA;
3. PR base equal to the live current `main` SHA;
4. no uncoordinated exact-file overlap with another non-draft PR;
5. deliberate overlap to have both a PR `Coordination-Override: #<other PR>` and matching complete `AI-COLLAB` evidence on issue #66.

A stale base, undeclared agent, or overlap without joint evidence is blocking.

## Collaborative conflict protocol

When two agents need the same file or tightly coupled subsystem, the goal is not to make one overwrite the other. They collaborate explicitly.

Issue #66 must contain:

```text
AI-COLLAB
agents: <agent A> | <agent B>
branches: <branch A> | <branch B>
prs: #<A> | #<B>
files: <exact overlapping files separated by |>
integration_owner: <agent/branch responsible for combined integration>
plan: <reconciliation approach>
tests: <joint unit/integration/regression/CI plan>
ack: <acknowledgement A> | <acknowledgement B>
status: active
```

The guard verifies that both PRs/branches and every exact overlapping file are represented, that at least two agents and two acknowledgements exist, and that the PR explicitly references the other PR with `Coordination-Override`. A bare override is rejected.

The integration owner must inspect both diffs, preserve compatible intent from both branches, resolve semantic disagreements deliberately, and run the combined relevant test suite plus exact-head CI. Git's ability to auto-merge is not enough: a semantic conflict can exist even when different lines merge cleanly.

## Trusted guard execution

The PR workflow uses `pull_request_target` with read-only permissions and checks out `main`, not PR-controlled code. A PR cannot weaken its own coordination policy and pass itself.

## Main update notifications

Every push to `main` publishes an `AI-MAIN-UPDATE` containing the current SHA, active agent count, each active agent's task and branch, open non-draft PRs, undeclared PRs, and detected open-PR overlaps. New AI sessions must read the latest update plus the live snapshot because the update is historical and the snapshot is current.

## Merge gate

Before production merge: current-main base, valid lease, trusted coordination check, exact-head CI, relevant combined tests, overlap/collaboration evidence when needed, and a final atomic re-check of `main` are mandatory. After merge, verify the new `AI-MAIN-UPDATE` and release the lease.

## GitHub repository-setting enforcement

The workflow fails closed, but administrator-level prevention of manual bypass depends on GitHub repository settings. `main` should require `ai-coordination / pr-conflict-guard` and up-to-date branches. Agents must never manually bypass these rules even when account permissions technically allow it.
