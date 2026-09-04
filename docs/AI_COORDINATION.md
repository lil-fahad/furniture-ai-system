# FurnitureAI AI coordination

FurnitureAI can be modified by more than one AI/coding tool at the same time. GitHub authentication alone cannot identify which model or external AI is behind a branch because multiple tools may use the same repository account. The coordination layer therefore uses explicit leases plus live GitHub state.

## Authoritative surfaces

- Root `AGENTS.md` — mandatory operating rules for all coding agents.
- GitHub issue #66 — live lease registry and chronological `main` update feed.
- `scripts/ai_coordination.py` — machine-readable snapshot and PR overlap guard.
- `.github/workflows/ai-coordination.yml` — automatic conflict checking and main-update publishing.

## See where development stopped

Run:

```bash
python scripts/ai_coordination.py snapshot --repo lil-fahad/furniture-ai-system
```

The JSON output includes the current `main` SHA, observation time, active declared leases, and all open PRs with head/base SHAs. The current main SHA is the authoritative answer to "where did the latest merged development stop?"

## See who is active and what they are doing

An AI agent becomes visible by posting an `AI-LEASE` comment on issue #66. The snapshot command parses leases, removes released or expired entries, and reports the remaining active agents with their task, branch, base SHA, and file scope.

GitHub cannot truthfully report a live AI/model identity when an external tool has not declared itself. Such work can still be observed indirectly as branches/PRs, but the model name must be treated as unknown until it registers a lease.

## Preventing task conflicts

For every pull request event, the coordination workflow compares the exact changed-file set with other open PRs.

- Two non-draft PRs that modify the same file are blocked by default.
- An overlap involving a draft PR is reported as a warning because draft work may be exploratory.
- A deliberate overlap can be acknowledged only by adding `Coordination-Override: #<other PR>` to the PR body after the owners coordinate on issue #66.
- A PR based on an older `main` SHA receives a warning and must review intervening changes and rerun exact-head CI before merge.

This guard does not replace Git merge-conflict detection. It catches semantic collision earlier, including cases where Git could technically auto-merge two agents editing different lines of the same file.

## Main update notifications

After every push to `main`, the coordination workflow posts an `AI-MAIN-UPDATE` comment to issue #66 containing:

- timestamp;
- new main SHA;
- first line of the merged commit message;
- number and names of active declared agents;
- currently open non-draft PRs.

Agents starting later should read the newest update comment before doing work.

## Lease lifecycle

Leases are intentionally time-bounded. An abandoned agent cannot lock a file forever. Agents should renew before expiry and release immediately after completion or abandonment. File scopes may be paths or globs for human readability; the PR guard independently checks exact changed files from GitHub.

## Merge gate

A production merge should satisfy all of the following:

1. PR is non-draft and mergeable.
2. Required CI/checks pass on the exact head commit.
3. Coordination guard has no unacknowledged non-draft overlap.
4. Latest main changes since the branch base were reviewed.
5. Relevant active leases are released or explicitly handed off.
6. The resulting main push appears on issue #66 as an `AI-MAIN-UPDATE`.
