# FurnitureAI instructions for Claude-compatible agents

Your **first repository action** is coordination bootstrap, not editing.

1. Read `/AGENTS.md` completely. It is authoritative.
2. Resolve the latest `main`, create a dedicated task branch at that exact SHA, and read the newest records on GitHub issue #66 plus relevant open PR changed-file scopes.
3. Before changing any file, obtain a trusted bootstrap receipt using the applicable `/AGENTS.md` path:
   - local checkout: `python scripts/ai_coordination.py bootstrap ... --post`;
   - connector/API-only session: post `AI-BOOTSTRAP-REQUEST` to issue #66 and use the receipt produced by the trusted `ai-remote-bootstrap` GitHub Actions workflow.
4. Bootstrap must cover the tracked project, report the live active-session census and conflicts, and be tied to your exact current-main task branch. Register a receipt-backed `AI-LEASE` only after the bootstrap is valid.
5. If `conflict_state: coordination_required` or another active lease overlaps your scope, do not edit the shared files until ownership is serialized or both sessions complete the bilateral `AI-COLLAB` / `AI-COLLAB-ACK` handshake. This rule applies even when the peer has not opened a PR.
6. Overlapping PRs additionally require exact-head review from the other collaborating session. A changed head invalidates the old review.
7. Never use `Coordination-Override` or a manual merge as a substitute for the collaboration evidence and tests.

Do not treat this file as a separate policy. `/AGENTS.md`, issue #66, and the trusted coordination guards are authoritative.
