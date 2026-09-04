# FurnitureAI AI coordination

FurnitureAI can be modified by more than one AI/coding tool at the same time. GitHub authentication alone cannot identify which model or external AI is behind a branch because multiple tools may use the same repository account. The coordination layer therefore uses explicit leases plus live GitHub state.

## Authoritative surfaces

- Root `AGENTS.md` — mandatory operating rules for all coding agents.
- GitHub issue #66 — live lease registry and chronological `main` update feed.
- `scripts/ai_coordination.py` — machine-readable snapshot and fail-closed PR guard.
- `.github/workflows/ai-coordination.yml` — trusted-base conflict checking and main-update publishing.

## See where development stopped

Run:

```bash
python scripts/ai_coordination.py snapshot --repo lil-fahad/furniture-ai-system
```

The JSON output includes the current `main` SHA, observation time, active declared leases, and all open PRs with head/base SHAs. Each PR also reports `coordination_state` and its matching active lease when one exists. The current main SHA is the authoritative answer to "where did the latest merged development stop?"

## See who is active and what they are doing

An AI agent becomes visible by posting an `AI-LEASE` comment on issue #66. The snapshot command parses leases, removes released or expired entries, and reports the remaining active agents with their task, branch, base SHA, and file scope.

GitHub cannot truthfully report a live AI/model identity when an external tool has not declared itself. The system therefore treats an open PR whose branch has no matching active lease as **undeclared** rather than guessing an identity. Undeclared PRs are surfaced in snapshots/main-update comments and are blocked by the PR guard.

## Fail-closed PR policy

For every PR targeting `main`, the coordination guard requires all of the following:

1. The PR head branch has an active `AI-LEASE` on issue #66.
2. The lease contains `agent`, `task`, `branch`, `base_sha`, `files`, `lease_until`, and `status`.
3. The lease branch exactly matches the PR head branch.
4. The lease `base_sha` matches the PR base SHA.
5. The PR base SHA is the current live `main` SHA.
6. There is no unacknowledged exact-file overlap with another non-draft PR.

A stale base or missing/mismatched lease is a blocking error. It cannot be bypassed with a coordination overlap override.

Two non-draft PRs that modify the same file are blocked by default. An overlap involving a draft PR is a warning because draft work may be exploratory. A deliberate overlap can be acknowledged only by adding `Coordination-Override: #<other PR>` to the PR body after the owners record the decision on issue #66.

This guard does not replace Git merge-conflict detection. It catches semantic collision earlier, including cases where Git could technically auto-merge two agents editing different lines of the same file.

## Trusted guard execution

The PR workflow uses `pull_request_target` with read-only permissions and checks out `main`, not the PR branch, before running `scripts/ai_coordination.py`. No PR-controlled code is executed by the coordination job. This prevents a PR from weakening its own coordination script or workflow and then passing the modified guard.

The coordination token is read-only for PR checks. Only the `push`-to-`main` notification job receives `issues: write`.

## Main update notifications

After every push to `main`, the coordination workflow posts an `AI-MAIN-UPDATE` comment to issue #66 containing:

- timestamp;
- new main SHA;
- first line of the merged commit message;
- number and names of active declared agents;
- currently open non-draft PRs;
- non-draft PRs that have no active declared lease.

Agents starting later should read the newest update comment before doing work.

## Lease lifecycle

Leases are intentionally time-bounded. An abandoned agent cannot lock a file forever. Agents should renew before expiry and release immediately after completion or abandonment. If `main` advances and a branch is updated/rebuilt, the agent must renew its lease using the new `base_sha`. File scopes may be paths or globs for human readability; the PR guard independently checks exact changed files from GitHub.

## Merge gate

A production merge should satisfy all of the following:

1. PR is non-draft and mergeable.
2. Its active AI lease matches the branch and current base SHA.
3. Required CI/checks pass on the exact head commit.
4. Coordination guard has no stale-base error or unacknowledged non-draft overlap.
5. The branch is based on the latest `main`.
6. The resulting main push appears on issue #66 as an `AI-MAIN-UPDATE`.
7. The agent posts `AI-RELEASE` after completion.

## GitHub repository-setting enforcement

The workflow itself fails closed, but preventing a repository administrator from manually merging a failed PR is a GitHub repository setting. `main` should require `ai-coordination / pr-conflict-guard` and require branches to be up to date before merging. If an integration cannot read or change Branch Protection/Rulesets, agents must not claim those settings are enabled; they should still treat the checks as mandatory and never manually bypass them.
