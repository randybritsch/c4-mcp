```markdown
# CONTEXT PACK — Control4 MCP Server (Jan 15, 2026)

## Mini Executive Summary (≤120 words)

This project exposes Control4 automation as MCP tools via a local Flask server. It enforces a strict 3-layer design: synchronous MCP/Flask entrypoints → sync adapter pass-through → async gateway that owns all Control4 I/O on a single background asyncio loop thread. We discovered some cloud lock drivers can physically actuate while Director variables remain stale; therefore lock actions report “accepted” (Director ack) separately from “confirmed” (observed state change) and include a best-effort estimate. We recently fixed a regression where light reads worked but light writes were missing; light `OFF/ON/SET_LEVEL/RAMP_TO_LEVEL` now succeed and were validated on real devices.

## Critical Architecture (≤6 bullets)

- [app.py](../app.py): Flask + MCP tool registration only; synchronous handlers.
- [control4_adapter.py](../control4_adapter.py): sync-only pass-through; no business logic.
- [control4_gateway.py](../control4_gateway.py): all Control4 I/O; runs on one background asyncio loop thread.
- pyControl4 is the primary Director integration; gateway may use HTTP fallback for select endpoints (e.g., bindings).
- Result semantics: separate “Director accepted” from “state confirmed”; do not trust lock variables alone.
- Debug tooling exists to inspect items, commands, variables, and trace activity.

## Current Working Set (3–7 files)

- [control4_gateway.py](../control4_gateway.py): lock semantics (accepted/confirmed/estimate), bindings/variables timeouts, light get/set/ramp implementations.
- [control4_adapter.py](../control4_adapter.py): sync facade and pass-throughs (including timeouts for variables/bindings).
- [app.py](../app.py): MCP tool surface; lock result augmentation fields; light tools wiring.
- [tools/watch_lock_activity.py](../tools/watch_lock_activity.py): variable/activity watcher and logging improvements.
- [tools/inspect_item.py](../tools/inspect_item.py): quick inspection of item info/commands/variables.
- [docs/project_overview.md](project_overview.md): “project brain” updated to reflect current architecture/tools.

## Interfaces / Contracts That Must Not Break

- MCP tool names and parameter shapes (external contract), including:
  - `c4_lock_get_state`, `c4_lock_unlock`, `c4_lock_lock`
  - `c4_light_get_state`, `c4_light_get_level`, `c4_light_set_level`, `c4_light_ramp`
  - discovery/debug tools like `c4_list_rooms`, `c4_list_devices`, `c4_item_variables`, `c4_item_commands`, `c4_item_bindings`, `c4_item_send_command`, `c4_debug_trace_command`
- Layering contract: no Control4 I/O or async work in `app.py` or `control4_adapter.py`.
- Lock response semantics: preserve `accepted` vs `confirmed`; only add fields (don’t rename/remove existing keys).

## Today’s Objectives + Acceptance Criteria

- Objective: Validate “no regressions” for lights + keep lock behavior safe under stale state.
- Acceptance criteria:
  - Light device can `OFF`, `ON`, `SET_LEVEL 30`, `SET_LEVEL 100` with read-back matching (`LIGHT_STATE`/brightness).
  - Lock actions always send commands (no “already locked” short-circuit) and never claim confirmed unless an actual change is observed.
  - Tools return bounded-time responses (timeouts handled as structured results, not hangs).
  - Docs reflect current repo layout and tool surface.

## Guardrails (Conventions & Constraints)

- Python 3.11+ (validated on Python 3.14); keep type hints.
- Explicit timeouts on all network/Director operations.
- Keep the strict layering: `app.py` sync MCP only → adapter pass-through → gateway async I/O only.
- Prefer command-by-name (`"OFF"`, `"ON"`, `"SET_LEVEL"`, `"RAMP_TO_LEVEL"`, `"LOCK"`, `"UNLOCK"`) over command IDs when possible.
- Do not “confirm” actions based on stale Director variables; treat confirmation as best-effort.
- Make minimal, surgical changes; preserve existing tool signatures and outputs.

## Links / Paths for Deeper Docs

- Project brain: [docs/project_overview.md](project_overview.md)
- Architecture notes: [docs/architecture.md](architecture.md)
- Additional context/specs: [docs/project_spec.md](project_spec.md), [docs/context_pack.md](context_pack.md)

## Next Prompt to Paste

```text
Load context from docs/project_overview.md and focus only on the current working set (control4_gateway.py, control4_adapter.py, app.py, tools/watch_lock_activity.py).
Goal: verify lights and locks end-to-end through MCP (not just direct gateway calls).
Constraints: keep strict layering; do not change MCP tool signatures; prefer minimal changes; add explicit timeouts.
Plan a small validation script or MCP call sequence, run it, and report results with accepted vs confirmed semantics.
```
  - Never hang or deadlock MCP requests
