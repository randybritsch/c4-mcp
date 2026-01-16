# CONTEXT PACK — Control4 MCP Server (Jan 2026)

## Mini Executive Summary (≤120 words)

This project exposes Control4 automation (lights, locks, thermostats, media/AV, room Watch/Listen) as MCP tools via a local Flask server. It enforces a strict 3-layer design: synchronous MCP/Flask entrypoints -> sync adapter pass-through -> async gateway that owns all Control4 I/O on a single background asyncio loop thread. Some cloud lock drivers can physically actuate while Director variables remain stale; lock actions report "accepted" (Director ack) separately from "confirmed" (observed state change) and may include a best-effort estimate. Roku app launching is made reliable by selecting the room Watch source when needed, broadcasting LaunchApp across the Roku protocol group, and confirming success by polling Roku variables (notably CURRENT_APP_ID). Room-level “what’s playing” is best-effort and is derived from device variables (which vary by driver) rather than UI configuration.

## Critical Architecture (≤6 bullets)

- app.py: MCP tool surface (Flask) + small response shaping (e.g., media watch+launch summaries).
- control4_adapter.py: synchronous facade; minimal logic; hands work to the gateway.
- control4_gateway.py: async orchestrator; owns the single background asyncio loop and all Control4 I/O.
- "accepted" vs "confirmed": preserve this split for actions where Director state may lag (locks; Roku app changes).
- Media visibility: ensure the room is "watching" the Roku input before launching apps (SELECT_VIDEO_DEVICE).
- Roku reliability: broadcast LaunchApp across the Roku protocol group and confirm via CURRENT_APP_ID polling.
- Audio now-playing: UI configuration lists sources + active, but does not include selected source/track; use device variables and room-scoped probing (`c4_room_now_playing`).

## Current Working Set (3–7 files)

- ../app.py — MCP tools + summaries.
- ../control4_adapter.py — sync wrappers and gateway lifecycle.
- ../control4_gateway.py — async Control4 integration; media watch+launch; thermostat safe set-target; lock semantics.
- ../control4_gateway.py — includes best-effort now-playing normalization (driver variable-name variants) and room-scoped now-playing probe.
- project_overview.md — tool list + decisions.
- ../tools/get_roku_current_app.py — quick confirmation helper.
- ../tools/test_paramount_basement.py — E2E watch+launch validation.

## Interfaces / Contracts That Must Not Break

- Discovery/debug: ping, c4_server_info, c4_list_rooms, c4_list_devices, c4_item_variables, c4_item_commands, c4_item_bindings, c4_item_send_command, c4_debug_trace_command
- Discovery/power: c4_capabilities_report
- Rooms/media: c4_room_list_commands, c4_room_send_command, c4_room_select_video_device, c4_media_watch_launch_app, c4_media_launch_app, c4_media_roku_list_apps, c4_media_remote, c4_media_remote_sequence, c4_media_now_playing, c4_media_get_state
- Rooms/audio: c4_room_select_audio_device, c4_room_listen, c4_room_listen_status, c4_room_now_playing
- Scenes/UI Buttons: c4_uibutton_list, c4_uibutton_activate, c4_scene_list, c4_scene_activate
- Contacts/sensors: c4_contact_list, c4_contact_get_state
- Motion sensors: c4_motion_list, c4_motion_get_state
- Keypads: c4_keypad_list, c4_keypad_buttons, c4_keypad_button_action, c4_control_keypad_list, c4_control_keypad_send_command
- Fans: c4_fan_list, c4_fan_get_state, c4_fan_set_speed, c4_fan_set_power
- Outlets: c4_outlet_list, c4_outlet_get_state, c4_outlet_set_power
- Intercom: c4_intercom_list, c4_intercom_touchscreen_set_feature, c4_intercom_touchscreen_screensaver, c4_doorstation_set_led, c4_doorstation_set_external_chime, c4_doorstation_set_raw_setting
- Macros: c4_macro_list, c4_macro_list_commands, c4_macro_execute, c4_macro_execute_by_name
- Scheduler: c4_scheduler_list, c4_scheduler_get, c4_scheduler_list_commands, c4_scheduler_set_enabled
- Announcements: c4_announcement_list, c4_announcement_list_commands, c4_announcement_execute, c4_announcement_execute_by_name
- TV (room-based): c4_tv_list, c4_tv_remote, c4_tv_watch, c4_tv_off
- Thermostats: c4_thermostat_get_state, c4_thermostat_set_hvac_mode, c4_thermostat_set_fan_mode, c4_thermostat_set_hold_mode, c4_thermostat_set_target_f, c4_thermostat_set_heat_setpoint_f, c4_thermostat_set_cool_setpoint_f
- Lights: c4_light_get_state, c4_light_get_level, c4_light_set_level, c4_light_ramp
- Locks: c4_lock_get_state, c4_lock_unlock, c4_lock_lock (must keep accepted/confirmed semantics)

## Today’s Objectives + Acceptance Criteria

- Media (Roku): c4_media_watch_launch_app(..., room_id=<ROOM_ID>, app="Netflix"|"Paramount+") returns ok=true and summary_text shows Watch active and CURRENT_APP_ID changing to the expected value.
- Room off: room-off command returns accepted=true and the room Watch becomes inactive best-effort.
- Thermostat safety: c4_thermostat_set_target_f chooses the correct setpoint (heat vs cool) based on mode and confirms when possible; no exceptions due to mode mismatch.
- Reliability: no MCP tool call hangs; timeouts return structured results.
- Scheduler writes: c4_scheduler_set_enabled is best-effort; always check confirmed (some Director builds return 400 "Timeout Modifying Scheduled Event" or 200 no-op responses).
- Audio: c4_room_listen_status(room_id) returns active+sources; c4_room_now_playing(room_id) returns ok=true plus a best-effort `best` result when Listen is active.

## Guardrails (Conventions & Constraints)

- Keep strict layering: app.py must not talk to Control4 directly.
- All Control4 I/O must run on the gateway's single asyncio loop thread.
- Do not change MCP tool names/signatures without updating docs and clients.
- Write operations should be safe: prefer idempotent commands; when validating with real devices, use auto-restore patterns.
- Roku: do not assume a single device id; LaunchApp must work from any Roku-related item id (protocol root / media_service / media_player / avswitch).
- TVs: prefer room-level commands for universal control across TV drivers (c4_tv_remote uses /rooms/{room_id}/commands).
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

Goal: run an end-to-end validation of Roku watch+launch and one thermostat write-test with auto-restore.

Use your own ids (discover them via c4_list_rooms / c4_list_devices).

Steps:
1) Call c4_media_watch_launch_app(device_id=<ROKU_DEVICE_ID>, room_id=<ROOM_ID>, app="Netflix") and report summary_text.
2) Call c4_media_roku_list_apps(device_id=<ROKU_DEVICE_ID>, search="Paramount") then launch Paramount+ via c4_media_watch_launch_app.
3) Pick one thermostat, call c4_thermostat_get_state then c4_thermostat_set_target_f(+1F) with confirm, then restore.
4) Pick one room that is currently listening and call c4_room_now_playing(room_id=<ROOM_ID>) to report the station/title (if present).

Constraints: keep strict layering; no signature changes; add explicit timeouts; report accepted vs confirmed clearly.
```

## New Chat Boot Prompt

```text
You are joining an ongoing software project. Load context strictly from the following pasted summaries and references.

[Paste the latest BOOTSTRAP SUMMARY]
[Paste the latest CONTEXT PACK]

Your tasks:
1) Acknowledge understanding of architecture and constraints
2) Ask only the 1–2 highest leverage clarifying questions
3) Begin executing on “Today’s objectives” using the Guardrails
4) Before making changes that could break contracts, propose a minimal plan
5) Proceed one step at a time, testing after each step

Do NOT re-architect unless asked. Be concise and code-first.
```
