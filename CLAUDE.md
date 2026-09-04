# FurnitureAI instructions for Claude-compatible agents

Your **first repository action** is coordination bootstrap, not editing.

1. Read `/AGENTS.md` completely. It is authoritative.
2. Fetch the latest `main`, create a dedicated clean task branch at that exact SHA, and read the newest records on GitHub issue #66.
3. Before changing any file, run `python scripts/ai_coordination.py bootstrap` with your unique `session_id`, task, and intended file scope as specified in `/AGENTS.md`.
4. The bootstrap must scan every Git-tracked file and report the live number of active AI sessions, their tasks/branches/scopes, and planned conflicts. Post its `AI-BOOTSTRAP` receipt to issue #66, then register a receipt-backed `AI-LEASE` before editing.
5. If another session overlaps your intended scope, do not edit the shared files until work is serialized or both sessions complete the `AI-COLLAB` / `AI-COLLAB-ACK` handshake. Overlapping non-draft PRs additionally require exact-head review from the other session.
6. Never use `Coordination-Override` or a manual merge to bypass the guard.

Do not treat this file as a separate policy. `/AGENTS.md`, issue #66, and the trusted `ai-coordination` guard are authoritative.
