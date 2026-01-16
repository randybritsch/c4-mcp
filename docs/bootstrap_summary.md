# PROJECT BOOTSTRAP SUMMARY — Control4 MCP Server

## **1) One-line purpose**
Expose Control4 home automation as stable local MCP tools, optimized for fast “common actions” (especially lighting) while keeping a strict sync/async boundary.

## **2) Architecture overview (3–6 bullets)**
- **Three strict layers**: MCP/Flask tools (sync) → adapter (sync passthrough) → gateway (async).
- **One background asyncio loop thread** (gateway-owned) performs *all* Control4 I/O (pyControl4 + Director HTTP).
- **Performance**: item inventory is cached with short TTL (`C4_ITEMS_CACHE_TTL_S`) to speed name resolution.
- **Write semantics**: tools return structured JSON and prefer **accepted vs confirmed** (confirm via follow-up reads when possible).
- MCP surface: **GET /mcp/list** and **POST /mcp/call**.

## **3) Key modules and roles**
- **app.py** — MCP tool registration + validation + guardrails; includes registry patch to avoid tool-arg name collisions.
- **control4_adapter.py** — thin sync facade; no business logic.
- **control4_gateway.py** — owns the async loop; implements retries, confirmation polling, batching, and caching.
- **tools/** — validators and operational scripts; `tools/validate_mcp_e2e.py` is the main regression entry.

## **4) Data & contracts (top 3–5 only)**
- **Item inventory** (Director): device/room metadata used for discovery and name resolution.
- **Variables**: `{ varName, value }` snapshots used for confirmation/telemetry.
- **Commands**: named invocations (e.g., `SET_LEVEL`, `RAMP_TO_LEVEL`, `UNLOCK`, `LaunchApp`).
- **Tool results**: JSON dicts with `ok` + details; avoid returning bare primitives.

## **5) APIs (key endpoints/tools only)**
- **MCP HTTP**: `GET /mcp/list`, `POST /mcp/call`.
- **Discovery**: `c4_list_rooms`, `c4_list_devices(category)`, `resolve_room`, `resolve_device`.
- **Lighting (fast path)**:
	- `c4_light_set_by_name` — resolve + set `level`/`state` + optional ramp/confirm in one call.
	- `c4_room_lights_set` — set all lights in a room (exclude/include lists, ramp, bounded concurrency, optional confirm).
	- `c4_light_get_level`, `c4_light_set_level`, `c4_light_ramp`.
	- Confirmation is best-effort and time-bounded, but now uses multiple signals (brightness + `LIGHT_STATE` when present) and returns richer diagnostics in `execute`: `confirm_reason`, `observed_state`, `confirm_trace`, and optional `confirm_fallback`.
- **Scenes (fast path)**: `c4_scene_set_state_by_name` (SetState On/Off + best-effort confirm).
- **Media**: `c4_media_watch_launch_app`, `c4_media_watch_launch_app_by_name`, `c4_media_remote(_sequence)`, `c4_media_roku_list_apps`.
- **Locks/thermostats**: `c4_lock_get_state`, `c4_lock_unlock`, `c4_lock_lock`, `c4_lock_set_by_name`; `c4_thermostat_get_state`, `c4_thermostat_set_target_f`.
- **Debug/inspection**: `c4_item_variables`, `c4_item_commands`, `c4_item_bindings`, `c4_debug_trace_command`.

## **6) Coding conventions (must-follow rules)**
- Gateway owns the asyncio loop; **no async Control4 code in app.py**.
- **No pyControl4 imports outside control4_gateway.py**.
- Adapter stays **passthrough-only**.
- Every network/Director operation has **explicit timeouts** and structured failures.
- Don’t break existing MCP tool names/signatures; add new tools instead.

## **7) Current priorities (Top 5)**
1. Make lighting actions fast: single-call by-name + room batching + confirm when feasible.
2. Keep inventory caching safe and effective (no deadlocks; bounded TTL; predictable latency).
3. Preserve strict layering (all Director I/O in gateway loop thread).
4. Maintain write guardrails (dry-run options; clear planned vs executed behavior).
5. Keep validators current (`tools/validate_mcp_e2e.py` as the baseline regression).

## **8) Open risks/unknowns (Top 5)**
1. Driver variability: different variable names/behaviors across Control4 builds and drivers.
2. “Accepted” commands may not reflect real-world state changes; confirmation is best-effort.
3. Batch lighting can stress Director/mesh; concurrency needs tuning per site.
4. Windows can run multiple `app.py` processes; stale process can expose stale tool registry.
5. Some device types (locks/scheduler) can be flaky/stale at the Director API layer.

## **9) Links/paths to full docs**
- docs/project_overview.md
- docs/architecture.md
- docs/context_pack.md
- docs/project_spec.md
- app.py, control4_adapter.py, control4_gateway.py
- tools/validate_mcp_e2e.py
