# PROJECT BOOTSTRAP SUMMARY — c4-mcp (Control4 MCP Server)

**Last Updated:** February 5, 2026

**1) One-line purpose**
Expose a Control4 system as **safe-by-default, schema-driven MCP tools** (HTTP + STDIO) with guardrails, deterministic ambiguity handling, and session-aware follow-ups.

**2) Architecture overview (3–6 bullets)**
- **Two transports**: HTTP MCP (`app.py`: `GET /mcp/list`, `POST /mcp/call`) and STDIO JSON-RPC (`claude_stdio_server.py`).
- **Strict layering**: tool surface → `control4_adapter.py` (sync facade) → `control4_gateway.py` (async orchestration + Director/cloud I/O).
- **Single asyncio loop thread** owns all Control4 network I/O to avoid async deadlocks and flaky request behavior.
- **Short TTL inventory cache** (items/rooms/devices) accelerates resolution while staying fresh.
- **Ambiguity-first contract**: name-based tools return candidates; clients re-call with stable IDs.
- **Session memory** keyed by `X-Session-Id` powers `*_last` follow-ups (TV remote/off, “those lights”, etc.).

**3) Key modules and roles (bullet list)**
- `app.py`: MCP tool registry + schema enforcement, HTTP server, structured results, session memory plumbing.
- `control4_adapter.py`: sync-only facade used by tools (no orchestration logic).
- `control4_gateway.py`: async Control4 core (timeouts, retries, confirmation polling, driver quirks, caching).
- `session_memory.py`: per-session “last referenced” TV/media + lights context for follow-ups.
- `claude_stdio_server.py`: STDIO MCP shim (stdout JSON-RPC only; logs to stderr).

**4) Data & contracts (top 3–5 only)**
- Tool list: `GET /mcp/list` returns the authoritative tool schema map.
- Tool call: `POST /mcp/call` with `{ "kind": "tool", "name": "<tool>", "args": { ... } }`.
- Strict schemas: required args must be present (clients should validate/normalize before calling).
- Ambiguity: tools may return `ok:false` with candidates; clients must clarify then retry deterministically.
- Follow-ups: clients should send a stable `X-Session-Id` so `c4_tv_remote_last`, `c4_tv_off_last`, `c4_lights_set_last`, etc. work across calls.

**5) APIs (key endpoints only)**
- HTTP: `GET /mcp/list`, `POST /mcp/call` (container default `:3333`; NAS often publishes `:3334 → :3333`).
- STDIO: JSON-RPC `initialize`, `tools/list`, `tools/call`.

**6) Coding conventions (only the rules the AI must always follow)**
- Preserve the layering/threading model: Control4 I/O remains in `control4_gateway.py` on the gateway’s loop thread.
- Never break tool names/schemas; evolve by **adding** tools/args or adding backward-compatible shims.
- Prefer stable IDs (`room_id`, `device_id`) after resolution; avoid repeated fuzzy matching in follow-ups.
- Keep writes safe-by-default (guardrails on; writes time-bounded; explicit enable flags).
- Keep secrets out of git/logs; treat `config.json` as local-only and never print credentials.

**7) Current priorities (Top 5)**
1. Keep the tool surface strict and stable; improve client-facing errors for schema violations/ambiguity.
2. Make TV/Watch source selection robust (room signals + `SELECT_VIDEO_DEVICE` fallback when listings are incomplete).
3. Keep follow-ups reliable (`X-Session-Id` semantics + `*_last` tools).
4. Maintain low latency without stale-state surprises (cache TTLs + explicit timeouts).
5. Keep diagnostics operationally useful (structured logs, correlation/session identifiers, server info).

**8) Open risks/unknowns (Top 5)**
1. Driver variability across installs (commands/variables differ by hardware/driver versions).
2. “Accepted” vs “confirmed” divergence (cloud drivers can actuate while variables stay stale).
3. Media/Watch inventory endpoints can be incomplete, requiring command-based probing fallbacks.
4. Cross-deploy drift (tool schema changes vs clients) can surface as runtime errors if clients aren’t validating.
5. Exposure risk if reachable beyond LAN (firewall/VPN strongly recommended).

**9) Links/paths to the full docs**
- `README.md`
- `docs/project_overview.md`
- `docs/architecture.md`
- `docs/context_pack.md`
- `tools/`
- `Dockerfile`, `docker-compose.yml`
- `requirements.txt`, `pyproject.toml`
