# PROJECT BOOTSTRAP SUMMARY — c4-mcp (Control4 MCP Server)

**Last Updated:** January 26, 2026

**1) One-line purpose**
Expose a Control4 system as **safe-by-default MCP tools** (HTTP + STDIO) for lights, media/TV, locks, thermostats, and room Watch/Listen with a stable tool surface and guardrails.

**2) Architecture overview (3–6 bullets)**
- **Two transports**: HTTP (`app.py` via `/mcp/list`, `/mcp/call`) and STDIO JSON-RPC (`claude_stdio_server.py`).
- **Strict layering**: tool surface → `control4_adapter.py` → `control4_gateway.py` (no direct Director/pyControl4 from `app.py`).
- **Single asyncio loop thread** owns Control4 I/O (cloud auth, Director calls, polling/confirmation) to avoid deadlocks.
- **Short TTL inventory caching** accelerates name resolution + list calls while staying fresh.
- **Ambiguity-first UX**: name-based tools return structured candidates; clients re-call with refined args.
- **Session follow-ups**: in-process memory keyed by `X-Session-Id` powers “those lights” / “turn off the TV” style follow-ups.

**3) Key modules and roles (bullet list)**
- `app.py`: MCP tool registry, HTTP server, guardrails, ambiguity formatting, session memory integration.
- `claude_stdio_server.py`: STDIO MCP shim (stdout JSON-RPC only; logs to stderr).
- `control4_adapter.py`: synchronous facade used by tool handlers.
- `control4_gateway.py`: async orchestration (timeouts, caching, retries, confirmation polling, driver quirks).
- `session_memory.py`: session-scoped “last” context for lights + TV/media.
- `tools/`: local validators/regression scripts (HTTP + STDIO).

**4) Data & contracts (top 3–5 only)**
- MCP HTTP call: `POST /mcp/call` body `{ "kind": "tool", "name": "<tool>", "args": { ... } }`.
- Tool catalog: `GET /mcp/list` returns a **map** of tool name → schema/metadata.
- Ambiguity contract: `ok:false` with `details.error='ambiguous'` and `details.matches|candidates=[...]`.
- Follow-ups: clients pass a stable `X-Session-Id` so `*_last` tools work across calls.
- Results separate **accepted** vs best-effort **confirmed** (driver-dependent).

**5) APIs (key endpoints only)**
- HTTP: `GET /mcp/list`, `POST /mcp/call` (default binds `:3333`).
- STDIO: JSON-RPC `initialize`, `tools/list`, `tools/call`.

**6) Coding conventions (only the rules the AI must always follow)**
- Preserve layering/threading: Control4 I/O stays in `control4_gateway.py` on its loop thread.
- Never break existing tool names/arg schemas; evolve by addition.
- Keep STDIO stdout clean (JSON-RPC only); logs go to stderr.
- Keep writes safe-by-default and time-bounded; keep scheduler/macros extra gated.
- Never commit secrets; treat `config.json` as local-only.

**7) Current priorities (Top 5)**
1. Reduce by-name ambiguity for TV/media flows (room-scoped source viability checks).
2. Keep follow-ups reliable (session memory + `*_last` tools).
3. Maintain protocol stability (HTTP + STDIO) and strict schema validation.
4. Keep latency bounded (caching + explicit timeouts).
5. Improve operational diagnostics (server/process info, clearer errors).

**8) Open risks/unknowns (Top 5)**
1. Driver variability across installs (commands/variables differ by hardware/driver versions).
2. “Accepted” vs “confirmed” divergence (cloud drivers can actuate while variables stay stale).
3. Media inventory endpoints can be incomplete (requiring command-based fallbacks).
4. Exposure risk if HTTP is reachable beyond LAN (firewall/VPN strongly recommended).
5. Multi-instance confusion during dev (stale processes/ports masking active code).

**9) Links/paths to the full docs**
- `README.md`
- `docs/project_overview.md`
- `docs/architecture.md`
- `docs/context_pack.md`
- `tools/run_e2e.py`
- `Dockerfile`, `docker-compose.yml`
- `server.json`, `pyproject.toml`
