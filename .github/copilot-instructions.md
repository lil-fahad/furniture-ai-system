# FurnitureAI Copilot instructions

Before proposing or applying repository changes, read `/AGENTS.md` and follow its **mandatory full-project entry protocol**.

Do not begin by editing the requested file in isolation. First inventory the recursive repository tree, read the relevant human-readable architecture/source/config/tests/docs/workflows/security/training/deployment surfaces, then use issue #66 plus the coordination snapshot to determine the current `main`, `active_agent_count`, every active agent's task/branch/file scope, open PR changed files, and overlaps.

Register an `AI-LEASE` only after the preflight. If work overlaps another agent, use the board-recorded `AI-COLLAB` protocol and combined review/tests; never rely on a bare `Coordination-Override` or silently overwrite another PR.

Never bypass the trusted coordination CI guard or stale-main requirement. `/AGENTS.md` and issue #66 are authoritative.
