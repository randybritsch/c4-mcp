Below is a **complete, ready-to-commit `PROJECT_OVERVIEW.md`**, filled in for your **Control4 MCP Server** project based on everything we’ve built and debugged together. You can tweak names or wording later, but this is a solid “project brain” starting point.

---

```markdown
# PROJECT_OVERVIEW.md

## Project name
**Control4 MCP Server**

---

## 1. Executive summary

The Control4 MCP Server is a local integration layer that exposes Control4 home automation capabilities (lights, locks, thermostats, media/AV, room Watch/Listen, plus macros/scheduler/announcements) as Model Context Protocol (MCP) tools. It runs entirely on the local network, mediates all Control4 interactions through a carefully controlled gateway (single background asyncio loop) to avoid async deadlocks and flaky HTTP behavior, and provides a clean, synchronous tool interface for AI agents and other MCP clients. It is optimized for fast “common actions” by caching Director item inventory with a short TTL and offering single-call tools like `c4_light_set_by_name` and `c4_room_lights_set` to reduce round trips. Some devices (notably certain cloud lock drivers) can physically actuate while Director variables remain stale; tool results separate “Director accepted” from “state confirmed” and provide a best-effort estimate. For Roku app launching, the gateway routes commands across Roku proxy items and confirms success by polling Roku variables (e.g., `CURRENT_APP_ID`). For room-level audio now-playing, the server provides best-effort metadata derived from device variables (driver-dependent), exposed both by device id and by room id.

---

## 2. Architecture

### Logical architecture

```

[MCP Clients / AI Agents]
|
v
Flask + MCP
(app.py)
|
v
Sync Adapter Layer
(control4_adapter.py)
|
v
Async Control4 Gateway
(control4_gateway.py)
|
v
pyControl4
|
v
Control4 Director
(local controller + cloud)

````

### Runtime architecture

- **Single asyncio event loop** runs forever in a background thread.
- All Control4 I/O (cloud auth, Director calls, command execution) occurs **only** on that loop.
- Flask handlers never run async Control4 code directly; they delegate to the adapter.
- Flask tools run synchronously; operations execute in a small threadpool to avoid blocking request handling.
- Long-running or flaky operations (e.g., cloud lock drivers, bindings endpoints) are isolated from Flask request threads.

### External dependencies

- Control4 Director (local controller)
- Control4 cloud services (authentication, cloud drivers)
- pyControl4 library
- aiohttp (HTTP fallback + polling, e.g., Roku app confirmation)
- flask-mcp-server (MCP protocol implementation)

---

## 3. Directory structure

```text
.
├── app.py                  # Flask app + MCP tool registration
├── control4_gateway.py     # Async Control4 integration core (single event loop)
├── control4_adapter.py     # Thin synchronous facade over the gateway
├── config.json             # Local config (host, credentials) – not committed
├── docs/                   # Project documentation (architecture/spec/context)
├── tools/                  # Local debug / inspection scripts
├── logs/                   # Local logs from debug tools (not committed)
└── README.md               # Basic setup and usage instructions
````

**Key principles**

* `app.py` = MCP + HTTP only
* `control4_adapter.py` = sync-only, no logic
* `control4_gateway.py` = async-only, owns Control4 and pyControl4

---

## 4. Module inventory

| Module                | Purpose                    | Inputs                            | Outputs                      | Boundaries                                      |
| --------------------- | -------------------------- | --------------------------------- | ---------------------------- | ----------------------------------------------- |
| `app.py`              | Expose MCP tools over HTTP | MCP requests                      | JSON responses               | No async Control4 logic                         |
| `control4_adapter.py` | Sync API for app layer     | Primitive types                   | Dicts / primitives           | Sync-only pass-through; no business logic       |
| `control4_gateway.py` | Control4 orchestration     | Device IDs, commands, timeouts    | Structured results / booleans | Owns asyncio loop thread + pyControl4 + HTTP fallback |
| `pyControl4`          | Control4 protocol client   | Tokens, commands                  | Director responses           | External dependency                             |

---

## 5. Data & schemas

### Key data shapes

#### Item (from Director)

```json
{
  "id": 1234,
  "typeName": "device",
  "name": "Example Lock",
  "control": "lock",
  "parentId": 1233,
  "URIs": {
    "commands": "/api/v1/items/1234/commands",
    "variables": "/api/v1/items/1234/variables"
  }
}
```

#### Variable

```json
{
  "varName": "LockStatus",
  "value": "locked"
}
```

#### Command

```json
{
  "id": 0,
  "command": "UNLOCK",
  "display": "Unlock Front Door"
}
```

#### Command response (Director acknowledgement)

```json
{
  "name": "SendToDevice",
  "result": 1,
  "seq": "..."
}
```

### Versioning strategy

* No persistent storage yet
* Tool contracts are versioned implicitly via MCP tool names
* Backward compatibility preserved by adding tools, not changing signatures

---

## 6. API surface

### MCP tools (external)

* `ping`
* `c4_server_info` (debug)
* `c4_list_rooms`
* `c4_list_typenames`
* `c4_list_controls`
* `c4_list_devices(category)` (lights/locks/thermostat/media)
* `c4_director_methods` (debug)
* `c4_item_variables(device_id)`
* `c4_item_bindings(device_id)`
* `c4_item_commands(device_id)`
* `c4_item_execute_command(device_id, command_id)`
* `c4_item_send_command(device_id, command, params)` (debug)
* `c4_debug_trace_command(device_id, command, params, ...)` (debug)
* `c4_room_select_video_device(room_id, device_id, deselect)`

**Watch diagnostics (read-only)**

* `c4_room_watch_status(room_id)`
* `c4_room_list_video_devices(room_id)`

**Audio (Room-based / Listen)**

* `c4_room_select_audio_device(room_id, source_device_id, deselect)`
* `c4_room_listen(room_id, source_device_id, confirm_timeout_s)`
* `c4_room_listen_status(room_id)` (read-only; discover available Listen sources)
* `c4_room_now_playing(room_id, max_sources)` (read-only; best-effort “what’s playing” for the room)

**Media / AV**

* `c4_media_get_state(device_id)`
* `c4_media_send_command(device_id, command, params)`
* `c4_media_remote(device_id, button, press)`
* `c4_media_remote_sequence(device_id, buttons, press, delay_ms)`
* `c4_media_now_playing(device_id)`
* `c4_room_now_playing(room_id, max_sources)`
* `c4_media_launch_app(device_id, app)`
* `c4_media_watch_launch_app(device_id, app, room_id, pre_home)` (returns `summary` + `summary_text`)
* `c4_media_watch_launch_app_by_name(device_name, app, room_name|room_id, pre_home, ...)` (resolves ids, then calls watch+launch)
* `c4_media_roku_list_apps(device_id, search)`

**Thermostats**

* `c4_thermostat_get_state(device_id)`
* `c4_thermostat_set_hvac_mode(device_id, mode, confirm_timeout_s)`
* `c4_thermostat_set_fan_mode(device_id, mode, confirm_timeout_s)`
* `c4_thermostat_set_hold_mode(device_id, mode, confirm_timeout_s)`
* `c4_thermostat_set_heat_setpoint_f(device_id, setpoint_f, confirm_timeout_s)`
* `c4_thermostat_set_cool_setpoint_f(device_id, setpoint_f, confirm_timeout_s)`
* `c4_thermostat_set_target_f(device_id, target_f, confirm_timeout_s, deadband_f)`

**Lights**

* `c4_light_get_state`
* `c4_light_get_level`
* `c4_light_set_level`
* `c4_light_ramp`
* `c4_light_set_by_name` (fast-path: resolve by name and set level/state in one call)
* `c4_room_lights_set` (fast-path: set all lights in a room; optional exclude/include, ramp, confirm)

**Locks**

* `c4_lock_get_state`
* `c4_lock_unlock`
* `c4_lock_lock`
* `c4_lock_set_by_name(lock_name, state, room_name|room_id, ...)` (resolve + lock/unlock)

**Macros / Scheduler / Announcements**

* `c4_macro_list`
* `c4_macro_list_commands`
* `c4_macro_execute`
* `c4_macro_execute_by_name`
* `c4_scheduler_list`
* `c4_scheduler_get`
* `c4_scheduler_list_commands`
* `c4_scheduler_set_enabled` (best-effort; always check confirmed)
* `c4_announcement_list`
* `c4_announcement_list_commands`
* `c4_announcement_execute`
* `c4_announcement_execute_by_name`

### Authentication

* MCP auth handled by `flask-mcp-server` middleware
* Control4 auth handled internally via pyControl4 (cloud + Director tokens)

---

## 7. Decision log (ADR-style)

* **Single asyncio loop**
  → Prevents deadlocks and race conditions with pyControl4.

* **Strict layer separation**
  → Keeps Flask simple and recoverable; isolates Control4 quirks.

* **pyControl4 for all Director access**
  → Avoids 401s and protocol mismatches seen with raw REST calls.

* **Command execution by name, not ID**
  → Director REST rejects `commandId`; named commands are accepted.

* **Separate “accepted” vs “confirmed”**
  → Some devices physically actuate without reliable Director variable updates; tools report Director acknowledgement separately from observed state change.

* **Roku app launching: route + confirm**
  → Roku devices appear as multiple Control4 proxy items; LaunchApp is broadcast across the Roku protocol group and success is confirmed via polling `CURRENT_APP_ID`.

* **Room Watch before visible media actions**
  → Selecting the room’s active video device (Watch/HDMI) is required for app launching to be reliably visible on-screen.

* **Item inventory cache (short TTL)**
  → Name resolution and list endpoints rely on `/api/v1/items`; caching with `C4_ITEMS_CACHE_TTL_S` reduces latency dramatically for repeated actions.

* **Fast-path lighting tools**
  → Common actions (set a specific light, set all room lights) are implemented as single MCP calls to minimize round trips and keep the cache warm.

* **Best-effort state estimate when stale**
  → The gateway tracks recent lock intent and returns an estimate when confirmation via variables is unreliable.

* **Scheduler writes are best-effort; confirm via reread**
  → Some Director builds return 400 “Timeout Modifying Scheduled Event” or 200 no-op responses; tools must report accepted vs confirmed.

* **MCP dispatch guards against arg-name collisions**
  → Tools may accept an argument named `name`; MCP registry call plumbing is patched to avoid `TypeError: got multiple values for argument 'name'`.

---

## 8. Non-functional requirements

### Performance

* Local network latency preferred
* Gateway calls must return within bounded timeouts

### Security

* No public exposure by default (localhost binding)
* Credentials stored locally only (not in repo)

### Scalability

* Single-user / single-controller scope
* Designed for correctness over throughput

### Reliability

* Retries and backoff on cloud auth
* Timeouts around all async calls

### Observability

* Structured return objects for debugging
* Debug tools expose raw variables and commands

---

## 9. Testing strategy

### Unit tests

* Adapter passthroughs
* Variable parsing logic

### Integration tests

* Live Director connection
* Command execution paths (lights, locks)

### End-to-end

* MCP client → Flask → Adapter → Gateway → Director
* Scripted validator: `tools/validate_mcp_e2e.py` (read-only by default; write tests gated by flags)
* Scheduler toggle validator: `tools/validate_scheduler_toggle.py` (defaults to dry-run; supports toggle+restore)

### Coverage goals

* Gateway logic: ~70%
* Adapter/app wiring: ~80%

---

## 10. Operational runbook

### Environments

* Local dev (Windows/macOS/Linux)
* Same config for prod (on-prem)

### Configuration

* `config.json`:

  * `host`
  * `username`
  * `password`

### Secrets

* Never committed
* Rotated manually

### Alerts / SLIs

* Startup failures
* Repeated auth failures
* Excessive command timeouts

---

## 11. Coding conventions

* Python 3.11+ (validated on Python 3.14)
* Type hints everywhere
* Explicit timeouts on I/O
* No silent exception swallowing
* Logging via structured dicts (not print)

---

## 12. Current risks & unknowns

* Cloud lock drivers may ignore or delay commands
* Director state/variables may remain stale even when the device physically actuates (locks observed)
* Scheduler enable/disable writes can be unreliable (400 server-side timeouts or 200 no-op); treat as best-effort and always check confirmed
* Media/app actions can be accepted but not visible if the room is not actively “watching” the correct input
* Multiple/stale `app.py` processes on Windows can present an incorrect tool registry (use `c4_server_info` to confirm PID/tool_count)
* pyControl4 API differences across versions
* Limited documentation from Control4

---

## 13. Roadmap

### Short-term (0–2 weeks)

* Harden lock semantics and monitoring (stale state, intent estimates, tracing)
* Continue hardening media reliability (Watch/HDMI selection + app launch confirmation)
* Expand debug tooling for item inspection, variable/binding tracing, and process/tool-registry diagnostics

### Recently completed

* Implement light write-path (`OFF`/`ON`/`SET_LEVEL`/`RAMP_TO_LEVEL`) and validate against real devices
* Add fast-path lighting tools: `c4_light_set_by_name` and `c4_room_lights_set` (batch + optional confirm)
* Add short-TTL Director item inventory cache (`C4_ITEMS_CACHE_TTL_S`) to speed device discovery and by-name actions
* Add thermostat end-to-end support (read + safe write/confirm patterns)
* Add media/AV tooling (remote/navigation, now-playing best-effort, app launching)
* Make Roku app launching reliable (Watch/HDMI selection + protocol-group routing + `CURRENT_APP_ID` confirmation)
* Add server/process diagnostic tool to detect stale/multiple `app.py` instances on Windows

### Mid-term (2–8 weeks)

* Room-based commands (e.g., “lock all doors”)
* Friendly name resolution (room + device)
* Expand room-level media control (power off / end session, volume, input)
* Cache observability and tuning (hit-rate + per-category caches + safe invalidation)

---

## 14. Glossary

* **MCP** – Model Context Protocol
* **Director** – Control4 controller API
* **Gateway** – Async Control4 integration layer
* **Adapter** – Sync facade for app layer
* **Cloud driver** – Device driver that relies on vendor cloud APIs
* **Confirmed** – State change observed
* **Accepted** – Command acknowledged by Director

---

```

If you want, next we can:
- Split this into **`ARCHITECTURE.md` + `RUNBOOK.md`**
- Add **sequence diagrams** (ASCII or Mermaid)
- Turn the ADR section into separate numbered ADR files
```
