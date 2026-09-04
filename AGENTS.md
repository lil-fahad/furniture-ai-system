# FurnitureAI mandatory AI entry and multi-agent coordination

This file is mandatory for every coding agent, AI integration, automation, and human-assisted coding tool that modifies this repository.

## Absolute first-action rule

**Do not edit any repository file immediately after entering FurnitureAI.**

Every new AI coding session must first establish an exact, current view of the whole tracked project and the other active AI sessions. The live coordination board is GitHub issue **#66**.

GitHub account names are not AI identities. Several models/tools may use the same account. A live coding session is identified by its unique `session_id` + task branch + unexpired `AI-LEASE`.

## Mandatory entry/bootstrap sequence

Before the first edit of a session:

1. Fetch or resolve current `main` and record its exact SHA.
2. Read this file, `CLAUDE.md` when applicable, `.github/copilot-instructions.md` when applicable, the repository tree/architecture needed for the task, and the newest issue #66 coordination records.
3. Create a dedicated task branch **at exactly the current `main` SHA**. Never start by editing `main`.
4. Obtain a trusted bootstrap receipt using one of the two supported paths below.

### Path A — local checkout

From a clean unmodified task branch:

```bash
python scripts/ai_coordination.py bootstrap \
  --repo lil-fahad/furniture-ai-system \
  --agent "<tool/model>" \
  --session-id "<unique-session-id>" \
  --task "<task>" \
  --files "<planned paths/globs>" \
  --post
```

If the environment cannot post to GitHub, run it without `--post` and post the exact emitted `AI-BOOTSTRAP` record to issue #66 using the connected GitHub tool.

### Path B — connector/API-only AI

An AI that can use GitHub but has no local shell must **not** skip bootstrap. After creating its task branch at current `main`, post this request to issue #66:

```text
AI-BOOTSTRAP-REQUEST
agent: <tool/model or integration name>
session_id: <unique session id>
task: <short task>
branch: <task branch already pointing at current main>
files: <planned paths/globs>
status: requested
```

The trusted `ai-remote-bootstrap` GitHub Actions workflow runs code from `main`, validates the request and branch, computes the canonical manifest from Git commit/tree/blob objects, inventories active leases and open-PR changed files, and posts the authoritative `AI-BOOTSTRAP` receipt as `github-actions[bot]`. Requests are accepted only on issue #66 from repository OWNER/MEMBER/COLLABORATOR identities.

5. Bootstrap covers **every Git-tracked blob byte-for-byte**, classifies text/binary files, counts bytes/files, and computes a deterministic repository manifest SHA-256. It also reads the live coordination board, reports the exact number of active AI sessions, what each session is doing, its branch, and its declared file scope.
6. Bootstrap must happen before edits and against a task branch whose head equals live `main`. Local bootstrap additionally requires a clean checkout. Stale SHA, truncated Git tree, unreadable blob, or incomplete repository visibility is a hard failure.
7. Byte coverage proves tracked-file coverage/integrity, not semantic comprehension. The AI must still inspect the project structure and consume the tracked text/source/configuration needed to understand architecture, contracts, tests, security, deployment, data/provenance, and dependencies relevant to its task. Large files must be processed in chunks rather than silently skipped.
8. If bootstrap reports `conflict_state: coordination_required`, **do not edit**. Either serialize ownership (one session releases/removes the shared scope) or establish the bilateral collaboration protocol below.
9. Only after a valid `AI-BOOTSTRAP` record exists and any conflict is resolved/coordinated may the session register its `AI-LEASE` and begin editing.

## AI-BOOTSTRAP receipt

A future PR is rejected unless its lease points to a matching bootstrap receipt for the same branch/session/current-main SHA. The trusted guard recomputes the manifest and rejects forged or stale receipt values. The supplemental active-lease guard also requires an explicit `conflict_state` and refuses an ignored unresolved conflict.

```text
AI-BOOTSTRAP
agent: <tool/model or integration name>
session_id: <unique session id>
task: <short task>
branch: <task branch>
main_sha: <exact current main SHA>
files: <planned paths/globs>
tracked_files: <count>
tracked_bytes: <count>
text_files: <count>
binary_files: <count>
manifest_sha256: <full repository manifest SHA-256>
observed_active_sessions: <count>
conflict_state: clear|coordination_required
status: complete
```

## Required lease format

After bootstrap and before editing, post:

```text
AI-LEASE
agent: <tool/model or integration name>
session_id: <same session id as AI-BOOTSTRAP>
task: <short task>
branch: <task branch>
base_sha: <same current main SHA>
files: <paths/globs>
lease_until: <ISO-8601 UTC>
status: active
bootstrap_main_sha: <AI-BOOTSTRAP main_sha>
bootstrap_manifest_sha: <AI-BOOTSTRAP manifest_sha256>
bootstrap_files: <AI-BOOTSTRAP tracked_files>
```

The PR guard rejects changed files outside the lease scope. If `main` advances, rebuild/update the task branch, rerun bootstrap on the new base, and renew the lease with the new receipt before continuing toward merge.

When work ends, post:

```text
AI-RELEASE
agent: <tool/model or integration name>
session_id: <session id>
branch: <branch>
status: completed|abandoned
result: <PR/commit or explanation>
```

## Live census: how many AIs are working and what are they doing?

Run locally when possible:

```bash
python scripts/ai_coordination.py snapshot --repo lil-fahad/furniture-ai-system
```

Connector-only agents must read issue #66 plus open PRs/changed files directly. The result/state exposes the active session count and each agent/tool label, unique session id, task, branch, base SHA, file scope, lease expiry, open PR inventory, and detected overlaps. Always use live state; never copy a static agent count into documentation.

## Conflict policy: serialize or collaborate

One active session owns a file scope at a time unless a real collaboration is established. Conflict is checked at bootstrap, against **all active leases even when the peer has no PR**, and again against exact changed files at PR time.

A unilateral `Coordination-Override` **does not by itself authorize overlap**.

If two sessions genuinely need the same files, they must record one collaboration on issue #66:

```text
AI-COLLAB
collab_id: <unique id>
base_sha: <current main SHA>
branches: <branch-a>, <branch-b>
shared_files: <exact shared paths>
mode: pair-review
plan: <how responsibilities are split and integration order>
status: agreed
```

Each branch/session must independently acknowledge it:

```text
AI-COLLAB-ACK
collab_id: <same id>
branch: <its branch>
agent: <tool/model>
session_id: <its session id>
status: accepted
```

Before an overlapping PR can pass, the **other branch/session** must review the exact current head and post:

```text
AI-COLLAB-REVIEW
collab_id: <same id>
reviewer_branch: <other branch>
reviewer_session_id: <other session id>
subject_pr: <PR number>
head_sha: <exact current PR head SHA>
status: approved
```

If the PR head changes after review, the approval is stale and another exact-head review is required. Each overlapping PR is checked independently, so both directions receive review when both are candidates for merge.

If collaboration is unnecessary, the safer path is serialization: one AI releases or removes the shared file scope, then the other proceeds.

## Merge rules

- Never push coding work directly to `main`.
- Every PR needs an active branch-matching lease backed by a trusted-manifest bootstrap receipt after this policy migration is installed.
- PR base SHA must equal current live `main`.
- Changed files must stay inside the lease scope.
- Any live active-lease scope conflict is blocked unless bilaterally acknowledged, even if the peer has not opened a PR.
- Non-draft exact-file overlap with another non-draft PR requires bilateral collaboration + both ACKs + other-session exact-head review.
- Draft overlap is a warning only; it is not permission to overwrite another branch.
- Before merge, re-check live `main`, exact PR head, mergeability, CI, issue #66, and overlap state atomically.
- If `main` advances, rebuild/update, rerun bootstrap, renew the lease, and rerun exact-head CI.
- Do not close, rewrite, or supersede another session's work without recording the decision on issue #66.
- Never manually bypass a failed coordination check.

## Trusted guard execution

The PR coordination workflow uses `pull_request_target` and checks out trusted `main`, never PR-controlled coordination code. It recomputes the repository manifest from that trusted checkout and validates bootstrap/lease/current-main state. A second trusted guard checks active leases that do not yet have PRs.

The current repository-level GitHub branch protection may not be enabled. Agents must still treat `ai-coordination / pr-conflict-guard`, current-main freshness, and CI as mandatory. A repository administrator should configure GitHub to require those checks so the merge button cannot bypass policy.

## Agent-specific entry files

- GitHub Copilot: `.github/copilot-instructions.md`
- Claude-compatible tools: `CLAUDE.md`

Those files must direct the agent here and to the applicable local or remote bootstrap path. This file plus issue #66 are authoritative.
