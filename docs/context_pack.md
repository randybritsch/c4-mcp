# CONTEXT PACK — Control4 MCP Server (Jan 2026)

## Mini executive summary (≤120 words)

This project exposes Control4 home automation as stable MCP tools via a local Flask server. It enforces a strict 3-layer design: sync MCP/Flask entrypoints → sync adapter passthrough → async gateway that owns all Control4/Director I/O on a single background asyncio loop thread. The current focus is speed of “common actions”, especially lighting, by (1) caching Director item inventory with a short TTL (`C4_ITEMS_CACHE_TTL_S`) and (2) providing single-call, fast-path tools (`c4_light_set_by_name`, `c4_room_lights_set`) to reduce round trips. Write results are structured and use “accepted vs confirmed” semantics when confirmation is best-effort. The system also supports reliable Roku app launching (Watch selection + protocol-group broadcast + variable confirmation).

## Critical architecture (≤6 bullets)

- Three strict layers: `app.py` (sync MCP tools) → `control4_adapter.py` (sync passthrough) → `control4_gateway.py` (async gateway).
- Gateway owns exactly one background asyncio loop thread; all Control4 I/O runs only on that loop.
- No pyControl4 imports outside `control4_gateway.py`.
- Inventory caching: `/api/v1/items` reads are cached with short TTL for fast name/room resolution.
- Fast paths: prefer “resolve + execute + optional confirm” in one MCP call for latency.
- Write semantics: report accepted vs confirmed; confirmations are polling-based and time-bounded.

## Current working set (3–7 files)

- `app.py` — MCP tool surface + validation/guardrails.
- `control4_adapter.py` — sync wrappers only; no business logic.
- `control4_gateway.py` — async orchestration: caching, batching, confirmation polling.
- `docs/project_overview.md` — single source of truth for architecture/tool surface.
- `docs/bootstrap_summary.md` — fast reload summary (keep aligned).
- `docs/context_pack.md` — this file; kept current after sessions.
- `tools/validate_mcp_e2e.py` — baseline regression entry (read-only by default).

## Interfaces / contracts that must not break

Keep tool names and signatures stable; add new tools instead of changing old ones.

**Core discovery/inspection**

- `ping`, `c4_server_info`
- `c4_list_rooms`, `c4_list_devices(category)`
- `c4_item_variables(device_id)`, `c4_item_commands(device_id)`, `c4_item_bindings(device_id)`

**Lighting (speed-critical)**

- `c4_light_set_by_name` (fast-path)
- `c4_room_lights_set` (fast-path batch)
- `c4_light_get_level`, `c4_light_set_level`, `c4_light_ramp`

**Lighting confirmation diagnostics (non-breaking additions)**

- Fast-path write tools return an `execute` object that may include:
	- `before_level`, `before_state`, `before_observed`
	- `observed_level`, `observed_state`, `observed`
	- `confirm_reason` (e.g., `level_match`, `state_match_no_level`, `timeout`)
	- `confirm_trace` (short time-series of observations)
	- `confirm_fallback` (records any best-effort ON/OFF fallback attempts)

**Scenes/macros/scheduler**

- `c4_scene_set_state_by_name` (fast-path)
- `c4_macro_execute_by_name`
- `c4_scheduler_set_enabled` (best-effort; always re-read to confirm)

**Locks**

- `c4_lock_get_state`, `c4_lock_unlock`, `c4_lock_lock`, `c4_lock_set_by_name`

**Media (Roku watch+launch)**

- `c4_room_select_video_device`, `c4_media_watch_launch_app`, `c4_media_watch_launch_app_by_name`, `c4_media_roku_list_apps`

## Today’s objectives + acceptance criteria

**Objective A — Validate lighting fast paths and batching**

- `c4_light_set_by_name` sets a named light to a target `level` or `state` in one call.
- With `confirm=true`, results show a bounded confirm loop (no hangs) and clearly report confirmed/not confirmed.
- `c4_room_lights_set` supports `dry_run=true` to preview targets and supports `exclude_names`/`include_names`.
- Batch executes with bounded concurrency and returns per-device results + elapsed time.

Acceptance details for confirm:
- Confirm remains best-effort + time-bounded (`confirm_timeout_s`, `poll_interval_s`) but now reports `confirm_reason` and may include `confirm_trace` / `confirm_fallback`.

**Objective B — Keep caching safe and effective**

- Repeated name/device discovery is faster with `C4_ITEMS_CACHE_TTL_S` enabled.
- No deadlocks: inventory calls must not wait on the gateway loop from inside the loop.

## Guardrails (must-follow)

- Strict layering: `app.py` must not talk to Director/pyControl4 directly.
- All Control4 network I/O must execute on the gateway’s single asyncio loop thread.
- Adapter stays passthrough-only; put orchestration (caching/batching/confirm loops) in the gateway.
- Every tool returns structured JSON and uses explicit timeouts.
- For writes, support safe patterns: dry-run where appropriate; “accepted vs confirmed”; avoid unbounded polling.

## Links / paths for deeper docs

- `docs/project_overview.md`
- `docs/bootstrap_summary.md`
- `docs/architecture.md`
- `tools/validate_mcp_e2e.py`

## Next prompt to paste

```text
Load context from docs/project_overview.md and docs/context_pack.md.

Goal: validate lighting speed paths and inventory caching.

Steps:
1) Use c4_list_rooms to pick a room.
2) Run c4_room_lights_set(room_name=<ROOM>, state="off", dry_run=true) and show the planned target list.
3) Run c4_room_lights_set(room_name=<ROOM>, state="off", confirm=false, ramp_ms=0, concurrency=6) and report elapsed + failures.
4) Pick one light by friendly name and run c4_light_set_by_name(name=<LIGHT>, state="on", confirm=true, confirm_timeout_s=8).
5) Repeat step 4 twice and compare latency to confirm caching helps.

Constraints: keep strict layering; do not change existing tool names/signatures; every call must have bounded timeouts; report accepted vs confirmed.
```
