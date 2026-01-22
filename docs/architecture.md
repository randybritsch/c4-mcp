
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

## Performance & safety primitives

- **Inventory caching:** Director item inventory is cached for a short TTL (`C4_ITEMS_CACHE_TTL_S`) to speed name/room resolution.
- **Safe-by-default writes:** writes are blocked unless explicitly enabled; allow/deny lists can restrict state-changing tools.
- **Accepted vs confirmed:** write tools report whether the command was accepted vs whether state was confirmed via polling (best-effort + time-bounded).

## Troubleshooting checklist (Windows)

- Prefer `tools/run_e2e.py` (or `npm run e2e`) to start the server, wait for readiness, and capture logs.
- If `/mcp/list` fails or hangs, check `logs/http_server_err.txt` (and ensure only one server instance is running).
