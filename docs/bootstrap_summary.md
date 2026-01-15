# PROJECT BOOTSTRAP SUMMARY — Control4 MCP Server

---

## **1) One-line purpose**
Expose Control4 home automation (lights, locks, thermostats, later media) as **stable, local MCP tools** using a strict async gateway + sync facade architecture that avoids deadlocks and flaky HTTP behavior.

---

## **2) Architecture overview**
- **Three strict layers**: MCP/App → Sync Adapter → Async Gateway  
- **Single asyncio event loop** owned by the gateway, running forever in one background thread  
- **All Control4 / pyControl4 calls live in the gateway only**  
- Flask handlers never call async code directly (thread offloading only)  
- **pyControl4 Director protocol preferred**; raw REST avoided unless proven safe  
- Commands are **accepted vs confirmed** (cloud drivers may lag state updates)

---

## **3) Key modules and roles**
- **`control4_gateway.py`**  
  Async core. Owns asyncio loop, Control4 auth, Director client, retries, command execution, polling, parsing.
- **`control4_adapter.py`**  
  Thin sync-only facade. No async, no pyControl4, no logic.
- **`app.py`**  
  Flask + MCP only. Tool registration, validation, timeouts, formatting.
- **`config.json`** (local, uncommitted)  
  Control4 host + credentials.
- **`PROJECT_OVERVIEW.md`**  
  Full project brain / living documentation.

---

## **4) Data & contracts (top essentials)**
- **Item**: `{ id, typeName, name, control, parentId, URIs }`
- **Variable**: `{ varName, value }` (e.g., `LockStatus`, `BATTERY_LEVEL`)
- **Command**: `{ id, command, display }` (e.g., `UNLOCK`, `LOCK`)
- **Lock state contract**:  
  - `accepted`: command sent successfully  
  - `confirmed`: state observed changing within window  
- **Tool result shape**: always JSON dict, never raw lists

---

## **5) APIs (key endpoints/tools)**
**MCP tools (external):**
- `ping`
- `c4_list_rooms`
- `c4_list_devices(category)`
- `c4_item_variables(device_id)`
- `c4_item_commands(device_id)`
- `c4_item_execute_command(device_id, command_id)`
- **Lights**: `c4_light_get_state`, `c4_light_get_level`, `c4_light_set_level`, `c4_light_ramp`
- **Locks**: `c4_lock_get_state`, `c4_lock_unlock`, `c4_lock_lock`

**Internal (gateway-only):**
- `getItemVariables`
- `getItemCommands`
- `sendPostRequest` (signature-safe wrapper)

---

## **6) Coding conventions (must-follow rules)**
- **One event loop, one thread — gateway owns it**
- **No asyncio in Flask handlers**
- **No pyControl4 imports outside the gateway**
- Adapter contains **no logic**, only delegation
- Every I/O call has **explicit timeouts**
- Never change tool signatures; add new tools instead
- Prefer **named commands** over commandId unless proven safe
- Always return structured dicts (`ok`, `error`, `details`), never bare values

---

## **7) Current priorities (Top 5)**
1. Finalize lock semantics (`accepted` vs `confirmed`) for cloud drivers  
2. Add thermostat tools (get state, set heat/cool, read temperature)  
3. Improve lock confirmation via additional variables (e.g., `LastActionDescription`)  
4. Add debug tool for full variable dump on any item  
5. Light metadata caching to reduce repeated Director calls

---

## **8) Open risks / unknowns (Top 5)**
1. Cloud lock drivers may **accept commands but delay or skip state updates**
2. pyControl4 API differences across versions
3. Control4 Director undocumented quirks
4. Variable naming inconsistencies across drivers
5. No official Control4 contract stability guarantees

---

## **9) Full docs & references**
- **`PROJECT_OVERVIEW.md`** — complete architecture & runbook  
- **`control4_gateway.py`** — source of truth for async + Control4 logic  
- **`control4_adapter.py`** — sync API surface  
- **`app.py`** — MCP tools + Flask wiring  
- (Future) `/docs/architecture.md`, `/docs/runbook.md`, `/docs/adr/`

---