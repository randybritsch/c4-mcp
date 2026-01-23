# PROJECT BOOTSTRAP SUMMARY — c4-mcp (Control4 MCP Server)

**Last Updated:** January 23, 2026

**1) One-line purpose**
Expose Control4 home automation as **safe-by-default MCP tools** over **HTTP** and **STDIO**, so MCP clients (Claude Desktop, scripts, agents) can discover devices and run automations with guardrails.

**2) Architecture overview (3–6 bullets)**
- **Two transports**: HTTP (`app.py` via `/mcp/list`, `/mcp/call`) and STDIO JSON-RPC (`claude_stdio_server.py`).
- **Strict layering**: Flask/MCP tool surface → sync adapter → async gateway (no direct Director calls from `app.py`).
- **Single asyncio loop thread** in `control4_gateway.py` performs all Control4 I/O (cloud auth + Director calls).
- **Inventory caching**: short TTL caching speeds room/device resolution and list calls (`C4_ITEMS_CACHE_TTL_S`).
- **Safety gates**: writes are disabled by default; allow/deny lists constrain state-changing tools.

**3) Key modules and roles (bullet list)**
- `app.py`: HTTP server + MCP tool registration + request middleware and guardrails.
- `claude_stdio_server.py`: STDIO MCP shim (stdout JSON-RPC only; stderr logs); supports tool filtering modes.
- `control4_adapter.py`: sync facade used by the tool layer (keeps request handlers simple).
- `control4_gateway.py`: async Control4 orchestration (timeouts, caching, confirmation polling, driver quirks).
- `session_memory.py`: lightweight session memory for follow-ups (e.g., “dim those lights”).
- `tools/`: validators and diagnostics (`run_e2e.py`, protocol validators, inspectors).

**4) Data & contracts (top 3–5 only)**
- MCP HTTP contract: `POST /mcp/call` body `{ kind:"tool", name:"<tool>", args:{...} }`.
- Tool catalog contract: `GET /mcp/list` returns a **map** of tool name → JSON schema (PowerShell: use `tools.PSObject.Properties`).
- Control4 “items/variables/commands” shapes returned from Director (driver-dependent; best-effort).
- Ambiguity/disambiguation: name-based resolvers can return multiple candidates (client should re-call with a refined scope).

**5) APIs (key endpoints only)**
- HTTP: `GET /mcp/list`, `POST /mcp/call` (default `http://127.0.0.1:3333`).
- STDIO: JSON-RPC `initialize`, `tools/list`, `tools/call` (for Claude Desktop / stdio MCP clients).

**6) Coding conventions (AI must always follow)**
- Preserve layering: `app.py` must not talk to Director/pyControl4 directly.
- All Control4 I/O must run on the gateway’s asyncio loop thread.
- Do not break existing tool names/signatures; add new tools instead.
- Keep stdout clean for STDIO mode (JSON-RPC only); logs go to stderr.
- Never commit secrets; assume `config.json` is local-only and gitignored.

**7) Current priorities (Top 5)**
1. Keep “public repo safe” posture: no credentials in tree/history; clear config error messages.
2. Maintain safe-by-default writes + guardrails (and keep scheduler writes additionally gated).
3. Preserve protocol stability (HTTP + STDIO) with deterministic validators.
4. Keep performance predictable (inventory caching; bounded timeouts for slow projects).
5. Document the decoupled integration model (works with any client via HTTP/STDIO; no coupling to `c4-mcp-app`).

**8) Open risks/unknowns (Top 5)**
1. Driver variability: variables/commands differ across hardware/OS/driver versions.
2. “Accepted” vs “confirmed” device state can diverge (confirmation is best-effort).
3. Python packaging compatibility on Windows (prefer Python 3.12 for `pydantic-core` wheels).
4. Exposing port 3333 beyond LAN is risky; LAN-only firewall/VPN recommended.
5. Multiple local server instances (port conflicts) can confuse testing and clients.

**9) Links/paths to full docs**
- `README.md`
- `docs/project_overview.md`
- `docs/architecture.md`
- `docs/context_pack.md`
- `tools/run_e2e.py`
- `Dockerfile`, `docker-compose.yml`
- `server.json`, `pyproject.toml`
