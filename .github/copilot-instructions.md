# FurnitureAI Copilot instructions

Your **first repository action** is coordination bootstrap, not editing.

- Read `/AGENTS.md` completely and follow it as the authoritative policy.
- Resolve current `main`, create a dedicated task branch at that exact SHA, and inspect the newest GitHub issue #66 coordination records plus relevant open PR changed-file scopes.
- Before proposing or applying repository changes, obtain a trusted bootstrap receipt using the applicable `/AGENTS.md` path:
  - with a local checkout, run `python scripts/ai_coordination.py bootstrap ... --post`;
  - with GitHub/connector access but no shell, post `AI-BOOTSTRAP-REQUEST` to issue #66 and use the receipt posted by the trusted `ai-remote-bootstrap` workflow.
- Bootstrap must cover the tracked project, verify current main, and report the live active-AI session census and conflicts. Register the receipt-backed `AI-LEASE` only after the receipt is valid.
- If `conflict_state: coordination_required` or another active lease overlaps your scope, do not edit shared files until ownership is serialized or both sessions complete `AI-COLLAB` and `AI-COLLAB-ACK`. This applies even before the peer creates a PR.
- Exact overlapping PRs additionally require exact-head review by the other session; changing the PR head invalidates the old approval.
- `Coordination-Override` is diagnostic/legacy text and never replaces bilateral collaboration evidence. Never bypass a failed coordination check or merge stale work.

Use issue #66 and, when a shell exists, `python scripts/ai_coordination.py snapshot` for the current active-session count, tasks, branches, scopes, open PR files, and overlaps.
