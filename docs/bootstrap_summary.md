# PROJECT BOOTSTRAP SUMMARY — c4-mcp (Control4 MCP Server)

**Last Updated:** January 29, 2026

**1) One-line purpose**
Expose a Control4 system as **safe-by-default MCP tools** (HTTP + STDIO) for lights, media/TV, locks, thermostats, and room Watch/Listen — with stable schemas, guardrails, and session-aware follow-ups.

**2) Architecture overview (3–6 bullets)**
- **Two transports**: HTTP (`app.py` via `/mcp/list`, `/mcp/call`) and STDIO JSON-RPC (`claude_stdio_server.py`).
- **Strict layering**: tool surface → `control4_adapter.py` → `control4_gateway.py` (no direct Director I/O from `app.py`).
- **Single asyncio loop thread** owns Control4 I/O (auth, Director calls, polling/confirmation) to avoid deadlocks.
- **Short TTL inventory caching** accelerates name resolution while staying fresh.
- **Ambiguity-first contract**: name-based tools return structured candidates; clients re-call with stable IDs.
- **Session follow-ups**: in-process memory keyed by `X-Session-Id` powers “those lights” / “turn it off” flows.

**3) Key modules and roles (bullet list)**
- `app.py`: MCP tool registry + HTTP server, schema validation, ambiguity shaping, structured logging, session memory integration.
- `claude_stdio_server.py`: STDIO MCP shim (stdout JSON-RPC only; logs to stderr).
- `control4_adapter.py`: synchronous facade used by tool handlers.
- `control4_gateway.py`: async orchestration (timeouts, retries, caching, confirmation polling, driver quirks).
- `session_memory.py`: session-scoped “last referenced” context (TV/media + lights).
- `tools/`: debug/regression scripts.

**4) Data & contracts (top 3–5 only)**
- MCP list: `GET /mcp/list` returns tool schemas/metadata.
- MCP call: `POST /mcp/call` body `{ "kind": "tool", "name": "<tool>", "args": { ... } }`.
- Ambiguity: tool results may return `ok:false` plus candidate sets; clients must clarify and retry.
- Follow-ups: clients should send a stable `X-Session-Id` header so `*_last` tools work across calls.
- Execution semantics: results separate **accepted** vs best-effort **confirmed** (driver-dependent).

**5) APIs (key endpoints only)**
- HTTP: `GET /mcp/list`, `POST /mcp/call` (default container port `3333`).
- STDIO: JSON-RPC `initialize`, `tools/list`, `tools/call`.

**6) Coding conventions (only the rules the AI must always follow)**
- Preserve layering/threading: Control4 I/O stays in `control4_gateway.py` on its loop thread.
- Never break tool names/schemas; evolve by **adding** args/tools and keeping backward-compatible shims.
- Prefer stable identifiers (`room_id`, `device_id`) over names once resolved.
- Keep writes safe-by-default and time-bounded; keep guardrails enabled unless intentionally changing safety posture.
- Keep secrets out of git and logs; treat `config.json` as local-only.

**7) Current priorities (Top 5)**
1. Keep tool schemas resilient to client drift (back-compat shims for common arg variants).
2. Improve TV/media Watch flows (room-scoped source viability; `SELECT_VIDEO_DEVICE` fallbacks).
3. Keep follow-ups reliable (`*_last` tools + clear session handling).
4. Maintain low latency (caching + explicit timeouts) without stale-state surprises.
5. Improve operational diagnostics (clearer errors, request IDs, server/process info).

**8) Open risks/unknowns (Top 5)**
1. Driver variability across installs (commands/variables differ by hardware/driver versions).
2. “Accepted” vs “confirmed” divergence (cloud drivers can actuate while variables stay stale).
3. Media inventory endpoints can be incomplete, requiring command-based fallbacks.
4. Tool/schema drift across deployments (especially when using hotfix overrides).
5. Exposure risk if reachable beyond LAN (firewall/VPN strongly recommended).

**9) Links/paths to the full docs**
- `README.md`
- `docs/project_overview.md`
- `docs/architecture.md`
- `docs/context_pack.md`
- `tools/`
- `Dockerfile`, `docker-compose.yml`
- `pyproject.toml`, `requirements.txt`
