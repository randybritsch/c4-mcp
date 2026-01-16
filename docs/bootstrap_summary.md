# PROJECT BOOTSTRAP SUMMARY — Control4 MCP Server

## **1) One-line purpose**
Expose Control4 home automation as stable local MCP tools over HTTP, optimized for fast “common actions” while preserving a strict sync/async boundary.

## **2) Architecture overview (3–6 bullets)**
- **Three strict layers**: MCP/Flask tools (sync) → adapter (sync passthrough) → gateway (async).
- **One background asyncio loop thread** (gateway-owned) performs *all* Control4 I/O (pyControl4 + Director HTTP).
- **Performance**: Director item inventory is cached with short TTL (`C4_ITEMS_CACHE_TTL_S`) to speed resolution.
- **Write semantics**: tools return structured JSON and report **accepted vs confirmed** (confirmation is best-effort + time-bounded).
- **MCP HTTP**: `GET /mcp/list` and `POST /mcp/call`.

## **3) Key modules and roles**
- **app.py** — MCP tool registration + validation + guardrails.
- **control4_adapter.py** — thin sync facade; no business logic.
- **control4_gateway.py** — async orchestration: retries, confirmation polling, batching, caching; owns the event loop thread.
- **tools/** — validators and operational scripts; use for regressions and on-box diagnostics.

## **4) Data & contracts (top 3–5 only)**
- **Item inventory** (Director): device/room metadata used for discovery and name resolution.
- **Variables**: `{ varName, value }` snapshots used for confirmation/telemetry.
- **Commands**: named command invocations (e.g., `SET_LEVEL`, `ROOM_OFF`, `UNLOCK`, `LaunchApp`).
- **Tool results**: JSON dicts with `ok` and details; avoid returning bare primitives.

## **5) APIs (key endpoints/tools only)**
- **Discovery**: `c4_list_rooms`, `c4_list_devices(category)`, `c4_find_rooms`.
- **Lighting (fast path)**: `c4_light_set_by_name`, `c4_room_lights_set`, `c4_light_get_level`.
- **Scenes/macros**: `c4_scene_set_state_by_name`, `c4_macro_execute_by_name`.
- **Media (Watch)**: `c4_room_select_video_device`, `c4_media_watch_launch_app`, `c4_media_watch_launch_app_by_name`, `c4_media_roku_list_apps`.
- **Watch diagnostics (read-only)**: `c4_room_watch_status`, `c4_room_list_video_devices` (may return empty depending on Director/UI config).
- **Listen**: `c4_room_listen_status`, `c4_room_listen`, `c4_room_off` (restore).
- **Debug/inspection**: `c4_item_variables`, `c4_item_commands`, `c4_item_bindings`.

## **6) Coding conventions (must-follow rules)**
- Gateway owns the asyncio loop; **no async Control4 code in app.py**.
- **No pyControl4 imports outside control4_gateway.py**.
- Adapter stays **passthrough-only**.
- Every Director/network operation has explicit timeouts and structured failures.
- Don’t break existing MCP tool names/signatures; add new tools instead.

## **7) Current priorities (Top 5)**
1. Keep “fast path” tools fast: minimize round trips, confirm when feasible.
2. Preserve strict layering and single-loop concurrency.
3. Maintain reliable Watch/Listen flows (including Roku app launching).
4. Keep validators usable as regressions (sensible defaults; safe writes).
5. Improve operational ergonomics (avoid stale `app.py` processes on Windows).

## **8) Open risks/unknowns (Top 5)**
1. Driver variability: variable names/behaviors differ across devices and OS builds.
2. “Accepted” commands may not reflect physical state; confirmation is best-effort.
3. Some Director endpoints are environment-dependent (e.g., room `video_devices` may be empty).
4. Batch operations can stress Director/mesh; concurrency needs site-specific tuning.
5. Windows can run multiple `app.py` processes; stale processes can expose stale tool registries.

## **9) Links/paths to full docs**
- docs/project_overview.md
- docs/architecture.md
- docs/context_pack.md
- docs/project_spec.md
- app.py, control4_adapter.py, control4_gateway.py
- tools/validate_mcp_e2e.py
- tools/validate_listen.py
