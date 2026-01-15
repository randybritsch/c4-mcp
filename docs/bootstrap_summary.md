# PROJECT BOOTSTRAP SUMMARY — Control4 MCP Server

## **1) One-line purpose**
Expose Control4 home automation (lights, locks, thermostats, media/AV) as stable, local MCP tools with reliable confirmations and strict sync/async layering.

## **2) Architecture overview (3–6 bullets)**
- **Three strict layers**: MCP/Flask tool handlers (sync) -> Adapter (sync) -> Gateway (async).
- **Single asyncio event loop** lives in the gateway and runs forever in one background thread.
- All Control4 I/O (pyControl4 + any HTTP polling/fallback) runs **only** on that gateway loop.
- Tools favor **bounded-time** behavior (timeouts + structured error results; no hangs).
- For flaky/stale drivers, results separate **"accepted" (Director ack)** from **"confirmed" (observed state)**.
- Media visibility requires room context: ensure the room is actively **Watch-ing** the correct input before app launch.

## **3) Key modules and roles**
- **app.py** — Flask + MCP tool registration; validates inputs; adds small response summaries (e.g., `summary_text` for watch+launch); never does Control4 I/O.
- **control4_adapter.py** — Thin sync facade; delegates to gateway; no business logic.
- **control4_gateway.py** — Async orchestrator; owns loop + pyControl4; implements retries/confirmation logic (locks, Roku app launches, thermostat setpoints).
- **tools/** — Local scripts for inspection/diagnostics and quick E2E checks (e.g., Roku current app, basement app launch).
- **config.json** — Local, uncommitted runtime config (Director host + credentials).

## **4) Data & contracts (top 3–5 only)**
- **Item**: `{ id, typeName, name, control, roomId/parentId, URIs }` (device metadata from Director).
- **Variable**: `{ varName, value }` (e.g., Roku `CURRENT_APP_ID`, lock `LockStatus`, thermostat setpoints).
- **Command**: named command invocation (`"ON"`, `"OFF"`, `"SET_LEVEL"`, `"UNLOCK"`, `"LaunchApp"`, etc.).
- **Accepted vs confirmed**: write tools must not claim confirmation unless an observed state matches within a timeout window.
- **Tool result shape**: tools return JSON dicts (typically with `ok`, plus details); avoid returning bare primitives.

## **5) APIs (key tools/endpoints only)**
- **Core discovery/diagnostics**: `ping`, `c4_server_info`, `c4_list_rooms`, `c4_list_devices(category)`.
- **Low-level inspection**: `c4_item_variables`, `c4_item_commands`, `c4_item_bindings`, `c4_item_send_command`, `c4_debug_trace_command`.
- **Media/AV**: `c4_media_get_state`, `c4_media_now_playing`, `c4_media_remote`, `c4_media_remote_sequence`, `c4_media_launch_app`, `c4_media_roku_list_apps`.
- **Room source selection**: `c4_room_select_video_device`.
- **High-level Roku helper**: `c4_media_watch_launch_app` (ensures Watch/input selection, launches app, confirms via Roku variables, returns `summary` + `summary_text`).
- **Thermostats**: `c4_thermostat_get_state`, `c4_thermostat_set_hvac_mode`, `c4_thermostat_set_fan_mode`, `c4_thermostat_set_hold_mode`, `c4_thermostat_set_target_f` (mode-safe target setpoint).
- **Lights/Locks**: `c4_light_get_state`, `c4_light_set_level`, `c4_light_ramp`; `c4_lock_get_state`, `c4_lock_unlock`, `c4_lock_lock`.

## **6) Coding conventions (must-follow rules)**
- Gateway owns the only asyncio loop; **no async Control4 code in app.py**.
- **No pyControl4 imports outside control4_gateway.py**.
- Adapter must remain a **thin pass-through** (no business logic).
- Every network/Director action has **explicit timeouts** and returns structured failures.
- **Do not change MCP tool names/signatures**; add new tools instead.
- For stateful devices, preserve **accepted vs confirmed** semantics.
- Prefer **deterministic, idempotent** writes and add auto-restore in validation scripts.

## **7) Current priorities (Top 5)**
1. Keep Roku watch+launch reliable: Watch selection + retries + confirmation via `CURRENT_APP_ID`.
2. Expand room-level media control (room off/end session, volume/input) without breaking layering.
3. Maintain thermostat write safety: mode-aware target logic + confirmation/restore patterns.
4. Keep tooling robust on Windows: detect/avoid multiple `app.py` processes (use `c4_server_info`).
5. Keep docs in sync with code (overview/context pack/bootstrap summary).

## **8) Open risks / unknowns (Top 5)**
1. Some drivers physically actuate but Director variables remain stale (locks especially).
2. Media actions can be "accepted" but not visible if the room is not actively on the correct Watch input.
3. Roku appears as multiple proxy items; command routing must stay robust across those IDs.
4. Multi-process/stale server instances on Windows can present a wrong tool registry.
5. pyControl4/Director quirks and variable naming differences across driver versions.

## **9) Links / paths to full docs**
- docs/project_overview.md
- docs/context_pack.md
- docs/architecture.md
- docs/project_spec.md
- app.py, control4_adapter.py, control4_gateway.py
- tools/get_roku_current_app.py, tools/test_paramount_basement.py
