# PROJECT BOOTSTRAP SUMMARY — Control4 MCP Server

## **1) One-line purpose**
Expose Control4 automation (lights, locks, thermostats, media/AV, macros, scheduler, announcements) as stable local MCP tools via a strict sync/async gateway.

## **2) Architecture overview (3–6 bullets)**
- **Three strict layers**: MCP/Flask tool handlers (sync) -> adapter (sync) -> gateway (async).
- **One background asyncio loop** (owned by the gateway) performs *all* Control4 I/O (pyControl4 + Director HTTP).
- Tools are **bounded-time** (explicit timeouts) and return **structured JSON** failures (no hangs).
- Write tools use **accepted vs confirmed**: do not claim state changed unless observed via a follow-up read.
- MCP is mounted at **/mcp**; the public surface used by validators is **GET /mcp/list** and **POST /mcp/call**.
- Windows ops note: stale `app.py` processes can expose an outdated tool registry; use `c4_server_info` to verify PID/tool_count.
- Windows ops note: run the server with the project venv interpreter (e.g., `.venv\\Scripts\\python.exe app.py`) to ensure dependencies like `flask_mcp_server` are available.

## **3) Key modules and roles**
- **app.py** — Flask app + MCP tool registration; input validation; response shaping; patches MCP registry dispatch to avoid argument-name collisions.
- **control4_adapter.py** — Thin synchronous facade; no business logic; delegates to gateway.
- **control4_gateway.py** — Async orchestrator; owns loop + pyControl4; implements retries/confirmation logic (locks, Roku launches, scheduler writes).
- **tools/validate_mcp_e2e.py** — End-to-end validator for **/mcp/call**; read-only by default; write tests gated by flags.
- **tools/validate_scheduler_toggle.py** — Safe scheduler toggle+restore validator; defaults to dry-run.
- **config.json** — Local runtime config (Director host + credentials); treat as uncommitted secret-ish state.

## **4) Data & contracts (top 3–5 only)**
- **Item**: `{ id, typeName, name, control, parentId, URIs }` (device metadata from Director).
- **Variable**: `{ varName, value }` (e.g., Roku `CURRENT_APP_ID`, lock `LockStatus`).
- **Command**: named command invocation (`"ON"`, `"OFF"`, `"SET_LEVEL"`, `"UNLOCK"`, `"LaunchApp"`, etc.).
- **Accepted vs confirmed**: “accepted” = Director acknowledged request; “confirmed” = reread shows desired state.
- **Tool result shape**: tool results are JSON dicts (typically include `ok` plus details); avoid returning bare primitives.

## **5) APIs (key endpoints/tools only)**
- **MCP HTTP**: `GET /mcp/list`, `POST /mcp/call` (body: `{kind, name, args}`).
- **Core discovery/diagnostics**: `ping`, `c4_server_info`, `c4_list_rooms`, `c4_list_devices(category)`.
- **Low-level inspection**: `c4_item_variables`, `c4_item_commands`, `c4_item_bindings`, `c4_item_send_command`, `c4_debug_trace_command`.
- **Rooms/media**: `c4_room_select_video_device`, `c4_room_list_commands`, `c4_room_send_command`, `c4_media_watch_launch_app`, `c4_media_roku_list_apps`, `c4_media_remote(_sequence)`, `c4_media_now_playing`, `c4_room_now_playing`, `c4_room_off`.
- **Rooms/audio (Listen)**: `c4_room_listen_status` (discover sources), `c4_room_listen` / `c4_room_select_audio_device` (start Listen), `c4_room_now_playing` (best-effort room-scoped now playing).
- **Macros/scheduler/announcements**: `c4_macro_list`, `c4_macro_execute_by_name`; `c4_scheduler_list`, `c4_scheduler_get`, `c4_scheduler_set_enabled` (best-effort); `c4_announcement_list`, `c4_announcement_execute_by_name`.
- **Thermostats/lights/locks**: `c4_thermostat_get_state`, `c4_thermostat_set_target_f`; `c4_light_get_state`, `c4_light_set_level`, `c4_light_ramp`; `c4_lock_get_state`, `c4_lock_unlock`, `c4_lock_lock`.

## **6) Coding conventions (must-follow rules)**
- Gateway owns the only asyncio loop; **no async Control4 code in app.py**.
- **No pyControl4 imports outside control4_gateway.py**.
- Adapter remains a **thin pass-through** (no business logic).
- Every Director/network operation has **explicit timeouts** and returns structured failures.
- **Do not change MCP tool names/signatures**; add new tools instead.
- Preserve **accepted vs confirmed** semantics for stateful actions.
- Dispatcher safety: tools may accept an arg named `name`; avoid keyword collisions in MCP call plumbing.

## **7) Current priorities (Top 5)**
1. Keep Roku watch+launch reliable (Watch selection + retries + confirm via `CURRENT_APP_ID`).
2. Keep dispatcher/tool-call stability (no arg-name collisions; maintain tool registry correctness).
3. Improve scheduler write reliability (more confirmed success, clearer diagnostics when not).
4. Maintain safe write patterns (timeouts + confirmation + restore in validation scripts).
5. Improve room-scoped audio telemetry (Listen + best-effort now-playing by room/device variables) and keep docs + tooling in sync after each debugging session.

## **8) Open risks/unknowns (Top 5)**
1. Driver state can be stale even when actions physically occur (locks especially).
2. Scheduler writes are unreliable on some Director builds (400 “Timeout Modifying Scheduled Event” or 200 no-op); always check `confirmed`.
3. Media visibility depends on room Watch/input state (actions can be accepted but not visible).
4. Windows can run multiple `app.py` processes; wrong PID can yield missing tools or stale behavior.
5. Director/driver variability across versions (variable names, endpoints, behavior).

## **9) Links/paths to full docs**
- docs/project_overview.md
- docs/context_pack.md
- docs/project_spec.md
- docs/architecture.md
- app.py, control4_adapter.py, control4_gateway.py
- tools/validate_mcp_e2e.py, tools/validate_scheduler_toggle.py
