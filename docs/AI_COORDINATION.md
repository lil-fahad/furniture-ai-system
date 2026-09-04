# FurnitureAI AI coordination v4

FurnitureAI can be modified by multiple AI/coding tools at the same time. The coordination layer is designed to make every future coding session enter through the same observable process: full tracked-project bootstrap, live session census, scoped lease, conflict detection, and bilateral review when work genuinely overlaps.

## Authoritative surfaces

- `/AGENTS.md` — mandatory first-action and merge rules for every coding AI.
- GitHub issue #66 — chronological `AI-BOOTSTRAP`, `AI-LEASE`, collaboration, release, and `AI-MAIN-UPDATE` registry.
- `scripts/ai_coordination.py` — bootstrap, live snapshot, manifest verification, and fail-closed PR guard.
- `.github/workflows/ai-coordination.yml` — executes the guard from trusted `main` and publishes main updates.
- `CLAUDE.md` and `.github/copilot-instructions.md` — agent-specific entry pointers back to `/AGENTS.md`.

## What "read the whole project" means operationally

Before an AI edits anything, `bootstrap` runs on a clean task branch whose `HEAD` exactly equals live `main`. It enumerates every path returned by `git ls-files`, reads every tracked Git blob byte-for-byte, counts tracked/text/binary files and bytes, and creates a deterministic SHA-256 manifest.

The PR guard later recomputes the manifest from its trusted `main` checkout. A receipt with a fabricated hash/count or a receipt from an older main SHA is rejected.

The machine check proves complete tracked-byte coverage. It cannot prove a model's internal semantic comprehension. `/AGENTS.md` therefore separately requires the AI to inspect project structure and consume source/configuration text necessary to understand architecture and dependencies; large text must be processed in chunks rather than silently skipped.

Untracked local datasets, model binaries, secrets, caches, ignored files, and external systems are deliberately outside this repository-manifest claim.

## Mandatory bootstrap

From a clean task branch created at current main:

```bash
python scripts/ai_coordination.py bootstrap \
  --repo lil-fahad/furniture-ai-system \
  --agent "<model/tool>" \
  --session-id "<unique-session-id>" \
  --task "<task>" \
  --files "<planned paths/globs>" \
  --post
```

The command fails if the branch is `main`, local HEAD is stale, or the checkout is already dirty. It prints:

- current main SHA;
- repository manifest and file/byte counts;
- `active_session_count`;
- each active session's agent label, session ID, task, branch, and scope;
- planned-scope conflicts;
- the exact `AI-BOOTSTRAP` receipt.

If GitHub write credentials are unavailable locally, omit `--post` and post the emitted receipt through the connected GitHub integration.

## Live census

Use:

```bash
python scripts/ai_coordination.py snapshot --repo lil-fahad/furniture-ai-system
```

The JSON schema exposes `active_session_count` and `active_sessions`. A session is an unexpired lease, not a GitHub username. This avoids pretending that one GitHub account equals one AI model.

## Receipt-backed leases

After bootstrap and before editing, every session posts the `AI-LEASE` fields documented in `/AGENTS.md`, including:

- unique `session_id`;
- exact `base_sha`;
- intended `files` scope;
- `bootstrap_main_sha`;
- `bootstrap_manifest_sha`;
- `bootstrap_files`.

The trusted guard requires a matching `AI-BOOTSTRAP` for the same branch/session and validates its SHA/counts against the trusted current-main repository manifest. The guard also rejects PR files outside the declared lease scope.

## Conflict detection happens twice

### Before editing

Bootstrap compares the intended file/path scope with every active lease. If a collision is found it reports `coordination_required`. The AI must not edit the shared scope until ownership is serialized or collaboration is agreed.

### At PR time

The guard queries exact changed filenames for all open PRs. Exact overlap between two non-draft PRs is blocking by default even when Git could technically merge different lines.

Draft overlap remains a warning because draft work may be exploratory; it is not ownership authorization.

## Bilateral collaboration protocol

`Coordination-Override` is deprecated and no longer authorizes overlap.

If two active sessions really must work on the same files, issue #66 needs:

1. `AI-COLLAB` identifying a `collab_id`, the same current `base_sha`, exactly two participating branches, exact `shared_files`, integration plan, and agreed status.
2. `AI-COLLAB-ACK` from **both branches** for that collaboration.
3. Before an overlapping PR passes, `AI-COLLAB-REVIEW` from the other branch for the current PR number and its **exact current head SHA**.

The exact-head requirement means any new commit invalidates the prior cross-review. Both PRs are checked independently, so both directions receive review if both proceed toward merge.

If collaboration is not essential, serialization is safer: one session releases/removes the shared scope and the other proceeds.

## Fail-closed PR policy

For a non-draft PR targeting `main`, the trusted guard requires all of the following:

1. PR base SHA equals live `main` SHA.
2. PR branch has an active matching `AI-LEASE`.
3. Lease contains all v4 session/bootstrap fields.
4. Matching `AI-BOOTSTRAP` exists for the same branch + session.
5. Bootstrap main SHA equals PR base/current main.
6. Bootstrap manifest SHA and file/byte/text/binary counts equal the manifest recomputed from trusted `main`.
7. PR changed files are inside the lease scope.
8. Any exact non-draft overlap either disappears through serialization or satisfies bilateral collaboration + both ACKs + other-session exact-head review.

A stale branch must be rebuilt/updated from current main, bootstrapped again, re-leased, and re-tested. A collaboration record never bypasses stale-main/bootstrap requirements.

## Trusted workflow execution

The PR workflow uses `pull_request_target`, read-only PR/issue/content permissions, and explicitly checks out `main`. It never executes coordination code from the PR branch. This prevents a PR from weakening its own guard and passing itself.

The manifest is recomputed inside that trusted checkout, not accepted merely because an AI wrote a hash into issue #66.

## Main update feed

After every main push, the workflow posts `AI-MAIN-UPDATE` with:

- new main SHA and commit;
- exact active session count;
- session details: agent label, session ID, branch, task, and file scope;
- open non-draft PRs;
- undeclared non-draft PRs.

## Repository-setting limitation

Repository policy currently relies on workflow checks, while GitHub branch protection/rulesets are a separate administrative setting. If `main` is not protected, an administrator can technically press a merge button despite a failing check. Coding agents must never use that bypass.

For hard merge-button enforcement, GitHub should require `ai-coordination / pr-conflict-guard`, project CI, and branches to be up to date. An integration that only has read access to protection/rulesets must not claim those settings are enabled.
