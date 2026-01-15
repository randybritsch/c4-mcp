# CONTEXT PACK — Control4 MCP Server (Jan 2026)

## Mini Executive Summary (≤120 words)

This project exposes Control4 automation (lights, locks, thermostats, media/AV) as MCP tools via a local Flask server. It enforces a strict 3-layer design: synchronous MCP/Flask entrypoints -> sync adapter pass-through -> async gateway that owns all Control4 I/O on a single background asyncio loop thread. Some cloud lock drivers can physically actuate while Director variables remain stale; lock actions report "accepted" (Director ack) separately from "confirmed" (observed state change) and may include a best-effort estimate. Roku app launching is made reliable by selecting the room Watch source when needed, broadcasting LaunchApp across the Roku protocol group, and confirming success by polling Roku variables (notably CURRENT_APP_ID).

## Critical Architecture (≤6 bullets)

- app.py: MCP tool surface (Flask) + small response shaping (e.g., media watch+launch summaries).
- control4_adapter.py: synchronous facade; minimal logic; hands work to the gateway.
- control4_gateway.py: async orchestrator; owns the single background asyncio loop and all Control4 I/O.
- "accepted" vs "confirmed": preserve this split for actions where Director state may lag (locks; Roku app changes).
- Media visibility: ensure the room is "watching" the Roku input before launching apps (SELECT_VIDEO_DEVICE).
- Roku reliability: broadcast LaunchApp across the Roku protocol group and confirm via CURRENT_APP_ID polling.

## Current Working Set (3–7 files)

- ../app.py — MCP tools + summaries.
- ../control4_adapter.py — sync wrappers and gateway lifecycle.
- ../control4_gateway.py — async Control4 integration; media watch+launch; thermostat safe set-target; lock semantics.
- project_overview.md — tool list + decisions.
- ../tools/get_roku_current_app.py — quick confirmation helper.
- ../tools/test_paramount_basement.py — E2E watch+launch validation.

## Interfaces / Contracts That Must Not Break

- Discovery/debug: ping, c4_server_info, c4_list_rooms, c4_list_devices, c4_item_variables, c4_item_commands, c4_item_bindings, c4_item_send_command, c4_debug_trace_command
- Rooms/media: c4_room_select_video_device, c4_media_watch_launch_app, c4_media_launch_app, c4_media_roku_list_apps, c4_media_remote, c4_media_remote_sequence, c4_media_now_playing, c4_media_get_state
- Thermostats: c4_thermostat_get_state, c4_thermostat_set_hvac_mode, c4_thermostat_set_fan_mode, c4_thermostat_set_hold_mode, c4_thermostat_set_target_f, c4_thermostat_set_heat_setpoint_f, c4_thermostat_set_cool_setpoint_f
- Lights: c4_light_get_state, c4_light_get_level, c4_light_set_level, c4_light_ramp
- Locks: c4_lock_get_state, c4_lock_unlock, c4_lock_lock (must keep accepted/confirmed semantics)

## Today’s Objectives + Acceptance Criteria

- Media (Basement Roku): c4_media_watch_launch_app(..., room_id=456, app="Netflix"|"Paramount+") returns ok=true and summary_text shows Watch active and CURRENT_APP_ID changing to the expected value.
- Room off: room-off command returns accepted=true and the room Watch becomes inactive best-effort.
- Thermostat safety: c4_thermostat_set_target_f chooses the correct setpoint (heat vs cool) based on mode and confirms when possible; no exceptions due to mode mismatch.
- Reliability: no MCP tool call hangs; timeouts return structured results.

## Guardrails (Conventions & Constraints)

- Keep strict layering: app.py must not talk to Control4 directly.
- All Control4 I/O must run on the gateway's single asyncio loop thread.
- Do not change MCP tool names/signatures without updating docs and clients.
- Write operations should be safe: prefer idempotent commands; when validating with real devices, use auto-restore patterns.
- Roku: do not assume a single device id; LaunchApp must work from any Roku-related item id (protocol root / media_service / media_player / avswitch).
- Windows ops: multiple app.py processes can cause stale tool registries; use c4_server_info to confirm which process is serving.

## Links / Paths for Deeper Docs

- project_overview.md
- architecture.md
- project_spec.md
- ../tools/watch_lock_activity.py (lock-state tracing)
- ../logs/ (recent traces)

## Next Prompt to Paste

```text
Load context from docs/project_overview.md and docs/context_pack.md.

Goal: run an end-to-end validation of Basement Roku watch+launch and one thermostat write-test with auto-restore.

Use room_id=456 and Roku-related device ids 2074/2075/2076/2077.

Steps:
1) Call c4_media_watch_launch_app(device_id=2074, room_id=456, app="Netflix") and report summary_text.
2) Call c4_media_roku_list_apps(device_id=2075, search="Paramount") then launch Paramount+ via c4_media_watch_launch_app.
3) Pick one thermostat, call c4_thermostat_get_state then c4_thermostat_set_target_f(+1F) with confirm, then restore.

Constraints: keep strict layering; no signature changes; add explicit timeouts; report accepted vs confirmed clearly.
```
