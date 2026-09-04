# FurnitureAI instructions for Claude-compatible agents

Read and follow `/AGENTS.md` **before making any change**. Its mandatory entry protocol requires a complete repository map/read of relevant human-readable project sources before coding, followed by a live coordination snapshot.

The live coordination board is GitHub issue #66. Before editing, determine the current `main` SHA, `active_agent_count`, what every active agent is doing, all open PR changed-file scopes, and detected overlaps. Register your own `AI-LEASE` only after that preflight.

If your intended work overlaps another agent, do not overwrite it. Follow the `AI-COLLAB` protocol in `/AGENTS.md`: inspect both implementations, record joint integration/test evidence on issue #66, and require combined regression/CI validation.

Do not treat this file as a separate policy. `/AGENTS.md` and issue #66 are authoritative.
