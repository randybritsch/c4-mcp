# PROJECT BOOTSTRAP SUMMARY - Control4 MCP Server

## **1) One-line purpose**
Expose Control4 home automation as **safe, local MCP tools** (HTTP + STDIO) so MCP clients (Claude Desktop, scripts, agents) can discover devices and run automations with guardrails.

## **2) Architecture overview (3-6 bullets)**
- **Two transports**: **HTTP** (`/mcp/list`, `/mcp/call`) and **STDIO JSON-RPC** (Claude Desktop) via a dedicated shim.
- **Three strict layers**: tool surface (sync) -> adapter (sync passthrough) -> gateway (async).
- **Single background asyncio loop thread** (gateway-owned) performs *all* Control4 I/O (pyControl4 + Director HTTP).
- **Performance**: Director item inventory cached with short TTL (`C4_ITEMS_CACHE_TTL_S`) to speed name/room resolution.
- **Write semantics**: returns structured JSON and separates **accepted vs confirmed** (confirmation is best-effort + time-bounded polling).
- **Safe-by-default**: writes are blocked unless explicitly enabled; allow/deny lists restrict state-changing tools.

## **3) Key modules and roles**
- **app.py** — MCP tool registration + HTTP endpoints + safety guardrails.
- **claude_stdio_server.py** — STDIO JSON-RPC shim (stdout = JSON-RPC only; stderr = logs); supports tool filtering modes.
- **control4_adapter.py** — thin sync facade over the gateway; no orchestration.
- **control4_gateway.py** — async orchestration: caching, batching, retries, confirmation polling; owns the event loop thread.
- **server.json / pyproject.toml** — registry + packaging metadata (console entrypoints).
- **tools/** — validators and on-box diagnostics (E2E runner, protocol validators, device inspectors).

## **4) Data & contracts (top 3-5 only)**
- **Director items**: room/device metadata used for discovery + resolution.
- **Variables**: `{ varName, value }` snapshots used for confirmation/telemetry (driver-dependent).
- **Commands**: named command invocations (e.g., `SET_LEVEL`, `ROOM_OFF`, `UNLOCK`, Roku `LaunchApp`).
- **Tool results**: JSON dicts with `ok` + details; failures are structured and actionable.
- **Config diagnostics**: non-fatal "can I connect?" status is exposed (so tool listing doesn't crash when config is missing).

## **5) APIs (key endpoints/tools only)**
- **HTTP MCP**: `GET /mcp/list`, `POST /mcp/call`.
- **STDIO MCP**: `initialize`, `tools/list`, `tools/call`.
- **Core discovery/inspection**: `ping`, `c4_server_info`, `c4_list_rooms`, `c4_list_devices(category)`, `c4_item_variables`, `c4_item_commands`, `c4_item_bindings`.
- **Lighting (speed-critical)**: `c4_light_set_by_name`, `c4_room_lights_set`, `c4_light_get_level`.
- **Scenes/macros/scheduler**: `c4_scene_set_state_by_name`, `c4_macro_execute_by_name`, `c4_scheduler_set_enabled`.
- **Watch/Listen**: `c4_room_watch_status`, `c4_room_list_video_devices`, `c4_room_listen_status`, `c4_room_listen`.
- **Locks + thermostat**: `c4_lock_lock`/`c4_lock_unlock`, `c4_thermostat_*` setters/getters.

## **6) Coding conventions (only the rules the AI must always follow)**
- Preserve strict layering: `app.py` must not talk to Director/pyControl4 directly.
- Gateway owns the asyncio loop; all Control4 I/O runs only on that loop.
- Adapter stays passthrough-only (no business logic).
- Don’t break existing tool names/signatures; add new tools instead.
- STDIO shim: stdout must be JSON-RPC only; logs go to stderr; keep output encoding safe for Windows.

## **7) Current priorities (Top 5)**
1. Keep tool listing + startup robust (no import-time crashes; clear config warnings).
2. Maintain safe-by-default writes and enforce allow/deny guardrails.
3. Keep fast-path tools fast (minimize round trips; cache inventory safely).
4. Preserve Watch/Listen and Roku watch+launch reliability across driver variance.
5. Keep regressions easy: E2E + protocol validators should stay deterministic.

## **8) Open risks/unknowns (Top 5)**
1. Driver variability: variable names/behaviors differ by device and OS build.
2. "Accepted" commands may not reflect physical state; confirmation is best-effort.
3. Some Director endpoints are environment-dependent (e.g., room video device lists may be empty).
4. Tool catalog filtering (compact vs all) can surprise clients/tests if not controlled.
5. Windows process/logging pitfalls (multiple server instances, buffering/encoding) can cause hard-to-debug client symptoms.

## **9) Links/paths to full docs**
- docs/project_overview.md
- docs/context_pack.md
- docs/architecture.md
- README.md
- app.py, claude_stdio_server.py, control4_adapter.py, control4_gateway.py
- server.json, pyproject.toml
- tools/run_e2e.py

---

## From-scratch runbook (Windows, PowerShell)

These are the most reliable steps to avoid “is the server running?” and PowerShell JSON-shape issues.

1) Bootstrap and run end-to-end (recommended)

- `npm run setup`
- `npm run e2e`

`tools/run_e2e.py` starts `app.py` in safe-by-default mode (guardrails on, writes off), waits for `/mcp/list`, then runs HTTP + STDIO validators and writes logs under `./logs/`.

2) Manual HTTP smoke test (correct payloads)

- List tools: `GET http://127.0.0.1:3333/mcp/list`
- Call a tool: `POST http://127.0.0.1:3333/mcp/call` with body `{"kind":"tool","name":"c4_list_rooms","args":{}}`

PowerShell note: `/mcp/list` returns a tool *map*; count tools via:

- `($r.tools.PSObject.Properties.Count)`

3) Avoid multiple instances

- Check port: `Test-NetConnection 127.0.0.1 -Port 3333 | Select-Object TcpTestSucceeded`
- Prefer starting the server via `tools/run_e2e.py` (it manages lifecycle + logs)

4) Scheduler safety

- Even if you enable writes for lights/locks, **Scheduler Agent writes stay disabled by default**.
- `c4_scheduler_set_enabled` requires `C4_SCHEDULER_WRITES_ENABLED=true`.
