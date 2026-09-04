# FurnitureAI Copilot instructions

Your **first repository action** is coordination bootstrap, not editing.

- Read `/AGENTS.md` completely and follow it as the authoritative policy.
- Fetch current `main`, create a clean dedicated task branch at that exact SHA, and inspect the newest GitHub issue #66 coordination records.
- Before proposing or applying repository changes, run `python scripts/ai_coordination.py bootstrap` with a unique `session_id`, the task, and intended file scope. The bootstrap scans every Git-tracked file, verifies the current main SHA, reports the live active-AI session census and conflicts, and produces the required `AI-BOOTSTRAP` receipt.
- Post the receipt to issue #66, then register the receipt-backed `AI-LEASE` described in `/AGENTS.md` before editing.
- If scope conflicts exist, do not edit shared files until ownership is serialized or both sessions complete `AI-COLLAB` and `AI-COLLAB-ACK`. Exact overlapping non-draft PRs also require exact-head review by the other session.
- `Coordination-Override` is legacy text only and never authorizes overlap. Never bypass a failed coordination check or merge stale work.

Use issue #66 and `python scripts/ai_coordination.py snapshot` for the current number of active AI sessions and what each one is doing.
