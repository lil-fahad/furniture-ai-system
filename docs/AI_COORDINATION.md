# FurnitureAI AI coordination v4

FurnitureAI can be modified by multiple AI/coding tools at the same time. The coordination layer makes every future coding session enter through the same observable process: tracked-project bootstrap, live session census, scoped lease, pre-edit conflict detection, and bilateral review when work genuinely overlaps.

## Authoritative surfaces

- `/AGENTS.md` — mandatory first-action and merge rules for every coding AI.
- GitHub issue #66 — chronological `AI-BOOTSTRAP`, `AI-LEASE`, collaboration, release, and `AI-MAIN-UPDATE` registry.
- `scripts/ai_coordination.py` — local bootstrap, live snapshot, manifest verification, and fail-closed PR guard.
- `scripts/ai_remote_bootstrap.py` + `.github/workflows/ai-remote-bootstrap.yml` — trusted bootstrap for GitHub/API/connector-only agents that have no local shell.
- `scripts/ai_lease_conflict_guard.py` — blocks overlap with active leases even before the peer creates a PR.
- `.github/workflows/ai-coordination.yml` — executes trusted guards from `main` and publishes main updates.
- `CLAUDE.md` and `.github/copilot-instructions.md` — agent-specific entry pointers back to `/AGENTS.md`.

## What "read the whole project" means operationally

A bootstrap is tied to a dedicated task branch whose head exactly equals live `main`. The canonical manifest enumerates every tracked Git blob, reads its canonical bytes, counts tracked/text/binary files and bytes, and creates a deterministic SHA-256 manifest based on path, length, and per-file SHA-256.

The local path uses `git ls-files` + `git show HEAD:<path>`. The remote path uses trusted Git commit/tree/blob API objects. A regression test requires both paths to produce the same canonical manifest for the same commit. A truncated recursive Git tree or unreadable/size-mismatched blob is a hard failure.

The PR guard later recomputes the manifest from its trusted `main` checkout. A fabricated hash/count or a receipt from an older main SHA is rejected.

The machine check proves complete tracked-byte coverage. It cannot prove a model's internal semantic comprehension. `/AGENTS.md` therefore separately requires the AI to inspect project structure and consume source/configuration/test/security/deployment text needed to understand architecture and dependencies; large text must be processed in chunks rather than silently skipped.

Untracked local datasets, model binaries, secrets, caches, ignored files, and external systems are deliberately outside the repository-manifest claim.

## Mandatory bootstrap: two trusted entry paths

### Local checkout

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

The command fails if the branch is `main`, local HEAD is stale, or the checkout is already dirty.

### GitHub/API/connector-only agent

An AI with GitHub access but no shell must not skip bootstrap. It creates a task branch at exact current `main` and posts on issue #66:

```text
AI-BOOTSTRAP-REQUEST
agent: <tool/model>
session_id: <unique session id>
task: <task>
branch: <task branch at current main>
files: <planned paths/globs>
status: requested
```

The trusted `ai-remote-bootstrap` workflow only handles issue #66 comments from repository OWNER/MEMBER/COLLABORATOR identities. It checks out trusted `main` with credentials not persisted, resolves the requested branch, requires its head to equal current `main`, computes the canonical manifest from Git objects, reads live leases and open PR changed files, and posts the authoritative `AI-BOOTSTRAP` receipt as GitHub Actions.

The request body is passed as data through an environment variable and parsed as strict record fields; it is never evaluated as shell code.

## Bootstrap output and live census

The receipt/state includes:

- current main SHA;
- repository manifest and file/byte counts;
- `active_session_count` / observed active sessions;
- each active session's agent label, session ID, task, branch, and scope;
- planned-scope conflicts;
- `conflict_state: clear|coordination_required`.

For local sessions, use:

```bash
python scripts/ai_coordination.py snapshot --repo lil-fahad/furniture-ai-system
```

Connector-only agents read issue #66 and open PR changed-file state through GitHub. A session is an unexpired lease, not a GitHub username; this avoids pretending one shared account equals one AI.

## Receipt-backed leases

After a valid bootstrap and before editing, every session posts the `AI-LEASE` fields documented in `/AGENTS.md`, including unique `session_id`, exact `base_sha`, intended scope, and bootstrap SHA/manifest/file-count references.

The trusted guard requires a matching `AI-BOOTSTRAP` for the same branch/session, validates its `conflict_state`, SHA/counts against current `main`, and rejects changed files outside the declared lease scope.

## Conflict detection happens before and after PR creation

### Before editing

Bootstrap compares intended scope with every active lease and open PR changed-file inventory. If a collision is found it emits `conflict_state: coordination_required`; the AI must not edit shared scope until ownership is serialized or collaboration is agreed.

### While another session only has a lease

`ai_lease_conflict_guard.py` compares the current PR's exact changed files and lease scope with **every other active lease**, including a peer branch that has not opened a PR. This closes the race where one AI could previously ignore a bootstrap warning and merge before the peer published its PR.

An old `coordination_required` receipt does not permanently poison a branch: if the peer releases/removes the conflicting scope, live conflict disappears and the guard can proceed. If both sessions still need the scope, bilateral collaboration must exist.

### At PR time

The core guard queries exact changed filenames for all open PRs. Exact overlap between two non-draft PRs is blocking by default even when Git could technically merge different lines. Draft overlap is a warning, not ownership authorization.

## Bilateral collaboration protocol

`Coordination-Override` is diagnostic/legacy text and never authorizes overlap by itself.

If two active sessions genuinely need the same files, issue #66 needs:

1. `AI-COLLAB` identifying one `collab_id`, the same current `base_sha`, exactly the participating branches, shared files, integration plan, and agreed/active status.
2. `AI-COLLAB-ACK` from both branches/sessions.
3. For exact non-draft PR overlap, `AI-COLLAB-REVIEW` from the other branch for the current PR number and **exact current head SHA**.

Any new commit invalidates the old exact-head peer review. Each overlapping PR is checked independently. If collaboration is unnecessary, serialization is safer: one session releases/removes shared scope and the other proceeds.

## Fail-closed PR policy

For a non-draft PR targeting `main`, the trusted policy requires:

1. PR base SHA equals live `main` SHA.
2. PR branch has an active matching lease/session.
3. Lease contains all required bootstrap/session fields.
4. Matching bootstrap exists for the same branch/session.
5. Bootstrap main SHA equals current main/PR base.
6. Bootstrap manifest SHA and file/byte/text/binary counts equal the trusted current-main manifest.
7. Bootstrap has explicit valid `conflict_state`.
8. PR changed files are inside lease scope.
9. No unresolved overlap with another active lease, even if the peer has no PR.
10. Any exact non-draft PR overlap satisfies bilateral collaboration + ACKs + other-session exact-head review.

A stale branch must be rebuilt/updated from current main, bootstrapped again, re-leased, and re-tested. Collaboration never bypasses current-main/bootstrap/CI requirements.

## Trusted workflow execution

The PR workflow uses `pull_request_target`, read-only PR/issue/content permissions, and explicitly checks out `main`; it never executes coordination code from the candidate PR. The remote bootstrap workflow likewise checks out trusted `main` and fails closed on incomplete Git API state.

## Main update feed

After every main push, the coordination workflow posts `AI-MAIN-UPDATE` with new main SHA/commit, exact active session count/details, open non-draft PRs, and undeclared non-draft PRs. New agents must still read live state because an update comment is historical while leases/PRs can change afterward.

## Repository-setting limitation

Workflow checks fail closed, while GitHub branch protection/rulesets are a separate administrative setting. Coding agents must never use administrator/manual merge as a bypass. For hard merge-button enforcement, GitHub should require `ai-coordination / pr-conflict-guard`, project CI, and branches to be up to date.
