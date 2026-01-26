# Context Pack — c4-mcp (Control4 MCP Server)

**Last Updated:** January 23, 2026

## Mini executive summary (≤120 words)

c4-mcp exposes Control4 home automation as safe-by-default MCP tools over HTTP and STDIO. It enforces strict layering: sync tool handlers (`app.py`) → sync facade (`control4_adapter.py`) → async gateway (`control4_gateway.py`) that owns all Director I/O on a single background asyncio loop thread (avoids async deadlocks and flaky behavior). It’s optimized for “common actions” with short-TTL item caching (`C4_ITEMS_CACHE_TTL_S`) and fast-path tools like `c4_light_set_by_name` and `c4_room_lights_set`. Writes remain gated/guardrailed and confirmations are best-effort and time-bounded. Lightweight per-session memory keyed by `X-Session-Id` enables follow-ups for both lights and TV/media (e.g., “turn it back on”, “turn off the TV”, “mute it”) via the `*_get_last` / `*_last` helper tools.

## Critical architecture (≤6 bullets)

- Strict layers: `app.py` → `control4_adapter.py` → `control4_gateway.py`.
- All Control4 I/O runs only on the gateway’s single asyncio loop thread.
- No pyControl4 imports/calls outside `control4_gateway.py`.
- Cache Director item inventory with a short TTL for fast resolution (`C4_ITEMS_CACHE_TTL_S`).
- Write semantics are bounded: “accepted vs confirmed” with time-bounded polling.
- Session context is keyed by `X-Session-Id` (HTTP) to support follow-ups.

## Current working set (3–7 files/modules)

- `app.py` — MCP tool surface, validation, guardrails, transport glue.
- `control4_gateway.py` — async orchestration: timeouts, caching, confirmations, driver quirks.
- `session_memory.py` — session-scoped memory and “last” helpers for lights + TV/media follow-ups.
- `tools/run_e2e.py` — fast regression runner (HTTP + STDIO coverage).
- `tools/validate_http_session_memory.py` — validates `X-Session-Id` memory behavior.
- `docs/project_overview.md` — source of truth for architecture/tool surface.

## Interfaces/contracts that must not break

**Transports**

- HTTP discovery: `GET /mcp/list`
- HTTP execution: `POST /mcp/call` body `{ "kind": "tool", "name": "<tool>", "args": { ... } }`
- Session header: `X-Session-Id: <stable-id>` (clients should keep this stable per device/user session)

**Tool surface (stability rule)**

- Do not rename or change existing tool signatures; add new tools instead.
- Follow-up memory tools (required for “turn it back on / turn it off / turn it down” style commands):
	- Lights: `c4_lights_get_last`, `c4_lights_set_last`
	- TV/media: `c4_tv_get_last`, `c4_tv_off_last`, `c4_tv_remote_last`

## Today’s objectives and acceptance criteria

**Objective A — Keep follow-ups reliable (session memory)**

- Memory is scoped by `X-Session-Id` and survives multiple sequential tool calls.
- `c4_lights_set_last` and TV/media “last” tools apply the prior target deterministically for the same session.

**Objective B — Prevent hangs and deadlocks**

- Every tool has explicit timeouts; no unbounded polling loops.
- Inventory/cache operations never deadlock the gateway loop.

**Objective C — Maintain fast, safe lighting UX**

- `c4_light_set_by_name` and `c4_room_lights_set` stay fast via caching and resolve+execute fast paths.
- With confirmation enabled, results remain time-bounded and clearly report confirmed/not-confirmed.

## Guardrails (from conventions)

- Preserve layering; never call Director/pyControl4 from `app.py`.
- All Control4 I/O must run on the gateway’s asyncio loop thread.
- Never commit secrets; `config.json` is local-only and gitignored.
- Keep all writes safe-by-default and time-bounded; prefer dry-run where appropriate.
- Preserve tool names/signatures; contracts evolve by addition.

## Links/paths for deeper docs

- `README.md`
- `docs/project_overview.md`
- `docs/architecture.md`
- `docs/bootstrap_summary.md`
- `tools/run_e2e.py`
- `tools/validate_http_session_memory.py`

## Next Prompt to Paste

```text
Load context from docs/project_overview.md and docs/context_pack.md.

Today’s goal: validate session memory + prevent hangs.

Steps:
1) Verify tool discovery: GET /mcp/list.
2) Run tools/validate_http_session_memory.py against the active server.
3) Run tools/run_e2e.py (HTTP + STDIO). Record any failures and the exact tool name.
4) Validate TV/media follow-ups: run a TV command (watch/off/remote) then exercise a `*_last` TV tool in the same `X-Session-Id`.

Constraints: preserve strict layering; don’t change existing tool names/signatures; all operations must have explicit, bounded timeouts; do not introduce new secrets into the repo.
```
