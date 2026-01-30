
# Architecture — Control4 MCP Server

This document is a quick-reference for the **runtime architecture** and the **non-negotiable layering constraints**.

For the full tool catalog and detailed behavior notes, see:

- [docs/project_overview.md](docs/project_overview.md)
- [docs/context_pack.md](docs/context_pack.md)

## Strict layering (must follow)

1) **Tool surface (sync):** `app.py`
2) **Adapter (sync passthrough):** `control4_adapter.py`
3) **Gateway (async orchestration):** `control4_gateway.py`

Rules:

- `app.py` must not import/talk to pyControl4/Director directly.
- `control4_adapter.py` must stay thin (no orchestration/business logic).
- All Control4 network I/O runs only on the gateway’s single background asyncio loop thread.
- Tool names/signatures are stable; add tools instead of changing existing contracts.

## Transports

- **HTTP MCP:** `GET /mcp/list`, `POST /mcp/call`
- **STDIO MCP:** JSON-RPC shim (`claude_stdio_server.py`) for clients like Claude Desktop

## Integration boundary

## Non-negotiable system rules

1) **c4-mcp must always be decoupled from c4-mcp-app.**
	- `c4-mcp` is a standalone service with a stable MCP tool surface.
	- Clients integrate only via HTTP (`/mcp/list`, `/mcp/call`) or STDIO; there is no shared-code contract.
	- `c4-mcp` must not take a dependency on `c4-mcp-app` (packages, modules, repo layout, deployment scripts, etc.).

2) **Command interpretation belongs to the client (AI/app), not the MCP server.**
	- The client (e.g., `c4-mcp-app` using Gemini) decides: user intent → tool selection → arguments → sequencing.
	- `c4-mcp` focuses on: validation, guardrails, deterministic execution, ambiguity reporting, and structured results.

- `c4-mcp` is a standalone service: clients integrate only via HTTP or STDIO.
- `c4-mcp-app` (and similar apps) should treat `c4-mcp` as an external dependency and configure the base URL (no shared code required).

## Performance & safety primitives

- **Inventory caching:** Director item inventory is cached for a short TTL (`C4_ITEMS_CACHE_TTL_S`) to speed name/room resolution.
- **Safe-by-default writes:** writes are blocked unless explicitly enabled; allow/deny lists can restrict state-changing tools.
- **Accepted vs confirmed:** write tools report whether the command was accepted vs whether state was confirmed via polling (best-effort + time-bounded).

## Troubleshooting checklist (Windows)

- Prefer `tools/run_e2e.py` (or `npm run e2e`) to start the server, wait for readiness, and capture logs.
- If `/mcp/list` fails or hangs, check `logs/http_server_err.txt` (and ensure only one server instance is running).
