# PROJECT BOOTSTRAP SUMMARY — c4-mcp (Control4 MCP Server)

**Last Updated:** January 23, 2026

**1) One-line purpose**
Expose a Control4 system as **safe-by-default MCP tools** (HTTP + STDIO) so clients can reliably control lights, media/TV, locks, thermostats, and room Watch/Listen without coupling to any specific UI.

**2) Architecture overview (3–6 bullets)**
- **Two transports**: HTTP (`app.py` via `/mcp/list`, `/mcp/call`) and STDIO JSON-RPC (`claude_stdio_server.py`).
- **Strict layering**: tool surface → `control4_adapter.py` → `control4_gateway.py` (tool handlers never call Director/pyControl4 directly).
- **Single asyncio loop thread** owns all Control4 I/O (cloud auth, Director calls, polling/confirmation) to avoid deadlocks.
- **Short TTL inventory caching** speeds name resolution and list calls (rooms/devices) while keeping results fresh.
- **Ambiguity-first UX**: name-based tools return structured candidates when resolution isn’t unique; clients re-call with refined args.
- **Session follow-ups**: per-session memory (via `X-Session-Id`) supports “turn it off / turn it down” using last referenced context tools.

**3) Key modules and roles (bullet list)**
- `app.py`: HTTP server, tool registry, request guardrails, ambiguity formatting.
- `claude_stdio_server.py`: STDIO MCP shim (stdout JSON-RPC only; stderr for logs).
- `control4_adapter.py`: synchronous facade for tool handlers.
- `control4_gateway.py`: async orchestration, caching, retries/timeouts, best-effort confirmations, driver quirks.
- `session_memory.py`: session-scoped context for follow-ups (lights + TV/media).
- `tools/`: local validators and regression helpers (protocol + e2e scripts).

**4) Data & contracts (top 3–5 only)**
- MCP HTTP call: `POST /mcp/call` body `{ "kind":"tool", "name":"<tool>", "args":{...} }`.
- Tool catalog: `GET /mcp/list` returns a **map** of tool name → schema/metadata (PowerShell often needs `tools.PSObject.Properties`).
- Ambiguity contract: tool returns `ok:false` with `details.error='ambiguous'` and `details.matches|candidates=[...]`.
- Follow-up contract: client passes stable `X-Session-Id` so “last” tools work across requests.
- Results distinguish **accepted** vs best-effort **confirmed** state (driver-dependent).

**5) APIs (key endpoints only)**
- HTTP: `GET /mcp/list`, `POST /mcp/call` (default binds `:3333`).
- STDIO: JSON-RPC `initialize`, `tools/list`, `tools/call`.

**6) Coding conventions (only the rules the AI must always follow)**
- Preserve layering and threading: Control4 I/O stays in `control4_gateway.py` on its loop thread.
- Don’t break tool names or argument schemas; add new tools rather than changing existing ones.
- Keep STDIO mode stdout clean (JSON-RPC only); write logs to stderr.
- Guardrail posture: write tools must remain safe-by-default; keep “powerful” domains (scheduler/macros) extra gated.
- Never commit secrets; treat `config.json` as local-only.

**7) Current priorities (Top 5)**
1. Keep by-name tools accurate (device-scoped room disambiguation for TV/media/listen flows).
2. Preserve follow-up reliability (session memory tools for lights + TV/media).
3. Maintain protocol stability across HTTP and STDIO transports.
4. Keep latency predictable via caching + bounded timeouts.
5. Improve operational diagnostics (server/process info, clearer config errors).

**8) Open risks/unknowns (Top 5)**
1. Driver variability across installs (commands/variables differ by hardware/driver version).
2. “Accepted” vs “confirmed” mismatches (cloud drivers can actuate while variables stay stale).
3. Name ambiguity in real homes (many rooms/devices share tokens like “TV”, “Basement”, “Roku”).
4. Exposure risk if HTTP port is reachable beyond LAN (firewall/VPN strongly recommended).
5. Multi-instance confusion during dev (stale processes/ports can mask which code is running).

**9) Links/paths to the full docs**
- `README.md`
- `docs/project_overview.md`
- `docs/architecture.md`
- `docs/context_pack.md`
- `tools/run_e2e.py`
- `Dockerfile`, `docker-compose.yml`
- `server.json`, `pyproject.toml`
