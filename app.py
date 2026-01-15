# app.py (recovered + reformatted)

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException
from flask_mcp_server import Mcp, mount_mcp
from flask_mcp_server.http_integrated import mw_auth, mw_cors, mw_ratelimit

from control4_adapter import (
    gateway as adapter_gateway,
    debug_trace_command,
    get_all_items,
    item_execute_command,
    item_get_bindings,
    item_get_commands,
    item_get_variables,
    item_send_command,
    light_get_level,
    light_get_state,
    light_ramp,
    light_set_level,
    list_rooms,
    lock_get_state,
    lock_lock,
    lock_unlock,
    thermostat_get_state,
    thermostat_set_cool_setpoint_f,
    thermostat_set_fan_mode,
    thermostat_set_heat_setpoint_f,
    thermostat_set_hold_mode,
    thermostat_set_hvac_mode,
    thermostat_set_target_f,
)

# ---------- App / Gateway ----------
app = Flask(__name__)

# Locks can block (cloud/driver latency); run them in a small thread pool
_lock_pool = ThreadPoolExecutor(max_workers=4)


def _augment_lock_result(result: dict, desired_locked: bool | None = None) -> dict:
    """Add derived fields without changing existing semantics."""
    accepted = bool(result.get("accepted"))
    confirmed = bool(result.get("confirmed"))

    estimate = result.get("estimate") if isinstance(result.get("estimate"), dict) else None
    est_locked = estimate.get("locked") if isinstance(estimate, dict) else None

    success_likely = accepted and (confirmed or (desired_locked is not None and est_locked == desired_locked))
    result["success_likely"] = bool(success_likely)

    # Provide a single "best guess" state for consumers when Director state is stale.
    locked = result.get("locked")
    if locked in (True, False):
        result["effective_state"] = "locked" if locked else "unlocked"
    elif isinstance(result.get("after"), dict) and result["after"].get("locked") in (True, False):
        result["effective_state"] = "locked" if result["after"].get("locked") else "unlocked"
    elif est_locked in (True, False):
        result["effective_state"] = "locked" if est_locked else "unlocked"
    else:
        result["effective_state"] = result.get("state") or "unknown"

    return result

# Return JSON errors, but preserve correct HTTP status codes (e.g., 404).
@app.errorhandler(HTTPException)
def _handle_http_exception(e: HTTPException):
    return (
        jsonify(
            {
                "ok": False,
                "error": e.name,
                "status": int(getattr(e, "code", 500) or 500),
                "details": str(getattr(e, "description", "")) or None,
            }
        ),
        int(getattr(e, "code", 500) or 500),
    )


@app.errorhandler(Exception)
def _handle_any_exception(e: Exception):
    return jsonify({"ok": False, "error": repr(e)}), 500


# ---------- MCP tools (REGISTER ON GLOBAL REGISTRY via Mcp.tool) ----------

@Mcp.tool(name="ping", description="Health check tool to verify the MCP server is reachable.")
def ping() -> dict:
    return {"ok": True}


@Mcp.tool(name="c4_director_methods", description="List callable methods on the Director object (debug).")
def c4_director_methods() -> dict:
    d = adapter_gateway._loop_thread.run(adapter_gateway._director_async(), timeout_s=10)
    names = sorted([n for n in dir(d) if callable(getattr(d, n, None)) and not n.startswith("_")])
    return {"ok": True, "methods": names}


@Mcp.tool(name="c4_item_variables", description="Get raw Director variables for an item (debug).")
def c4_item_variables(device_id: str) -> dict:
    vars_ = item_get_variables(int(device_id))
    return {"ok": True, "device_id": str(device_id), "variables": vars_}


@Mcp.tool(name="c4_item_bindings", description="Get Director bindings for an item (debug).")
def c4_item_bindings(device_id: str) -> dict:
    result = item_get_bindings(int(device_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(name="c4_item_commands", description="Get available Director commands for an item (debug).")
def c4_item_commands(device_id: str) -> dict:
    result = item_get_commands(int(device_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(name="c4_item_execute_command", description="Execute a specific Director command by command_id (debug).")
def c4_item_execute_command(device_id: str, command_id: int) -> dict:
    result = item_execute_command(int(device_id), int(command_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_item_send_command",
    description="Send a named Director command to an item (debug). Example: command='UNLOCK' or 'CLOSE'.",
)
def c4_item_send_command(device_id: str, command: str, params: dict | None = None) -> dict:
    result = item_send_command(int(device_id), str(command or ""), params)
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_debug_trace_command",
    description=(
        "Force-send a named Director command and poll for variable/state changes (debug). "
        "Useful when cached lock state is stale."
    ),
)
def c4_debug_trace_command(
    device_id: str,
    command: str,
    params: dict | None = None,
    watch_var_names: list[str] | None = None,
    poll_interval_s: float = 0.5,
    timeout_s: float = 30.0,
) -> dict:
    result = debug_trace_command(
        int(device_id),
        str(command or ""),
        params,
        watch_var_names=watch_var_names,
        poll_interval_s=float(poll_interval_s),
        timeout_s=float(timeout_s),
    )
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(name="c4_list_rooms", description="List rooms from Control4 (live).")
def c4_list_rooms() -> dict:
    return {"ok": True, "rooms": list_rooms()}


@Mcp.tool(name="c4_list_typenames", description="List Control4 item typeName values and counts (discovery).")
def c4_list_typenames() -> dict:
    items = get_all_items()
    counts = Counter(i.get("typeName") for i in items if isinstance(i, dict))
    return {
        "ok": True,
        "typeNames": [
            {"typeName": k, "count": counts[k]}
            for k in sorted(counts.keys(), key=lambda x: (-(counts[x] or 0), str(x)))
        ],
    }


@Mcp.tool(name="c4_list_controls", description="List Control4 item control values and counts (discovery).")
def c4_list_controls() -> dict:
    items = get_all_items()
    counts = Counter(
        (i.get("control") or "UNKNOWN")
        for i in items
        if isinstance(i, dict) and i.get("typeName") == "device"
    )
    return {
        "ok": True,
        "controls": [
            {"control": k, "count": counts[k]}
            for k in sorted(counts.keys(), key=lambda x: (-(counts[x] or 0), str(x)))
        ],
    }


@Mcp.tool(name="c4_list_devices", description="List Control4 devices by category (lights, locks, thermostat, media).")
def c4_list_devices(category: str) -> dict:
    category = (category or "").lower().strip()

    category_controls = {
        "lights": {"light_v2", "control4_lights_gen3", "outlet_light", "outlet_module_v2"},
        # Locks may appear either as a lock proxy (control=lock) or as a relay-style door lock proxy.
        "locks": {"lock", "control4_relaysingle"},
        "thermostat": {"thermostatV2"},
        "media": {
            "media_player",
            "media_service",
            "receiver",
            "tv",
            "dvd",
            "tuner",
            "satellite",
            "avswitch",
            "av_gen",
            "control4_digitalaudio",
        },
    }

    if category not in category_controls:
        return {
            "ok": False,
            "error": f"Unknown category '{category}'. Use one of: {sorted(category_controls.keys())}",
        }

    items = get_all_items()
    rooms_by_id = {
        str(i.get("id")): i.get("name")
        for i in items
        if isinstance(i, dict) and i.get("typeName") == "room"
    }

    allowed = category_controls[category]
    devices = []

    for i in items:
        if not isinstance(i, dict):
            continue
        if i.get("typeName") != "device":
            continue
        control = (i.get("control") or "")
        categories = i.get("categories")
        is_lock_category = category == "locks" and isinstance(categories, list) and any(
            str(c).lower() == "locks" for c in categories
        )
        if control not in allowed and not is_lock_category:
            continue

        room_id = i.get("roomId")
        parent_id = i.get("parentId")
        resolved_room_id = room_id if room_id is not None else parent_id
        resolved_room_name = i.get("roomName") or (rooms_by_id.get(str(resolved_room_id)) if resolved_room_id is not None else None)
        devices.append(
            {
                "id": str(i.get("id")),
                "name": i.get("name"),
                "control": i.get("control"),
                "roomId": str(resolved_room_id) if resolved_room_id is not None else None,
                "roomName": resolved_room_name,
                "uris": i.get("URIs") or {},
            }
        )

    devices.sort(key=lambda d: ((d.get("roomName") or ""), (d.get("name") or "")))
    return {"ok": True, "category": category, "devices": devices}


# ---- Thermostats ----

@Mcp.tool(name="c4_thermostat_get_state", description="Get current state for a Control4 thermostat.")
def c4_thermostat_get_state_tool(device_id: str) -> dict:
    result = thermostat_get_state(int(device_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(name="c4_thermostat_set_hvac_mode", description="Set HVAC mode (Off/Heat/Cool/Auto) on a Control4 thermostat.")
def c4_thermostat_set_hvac_mode_tool(device_id: str, mode: str, confirm_timeout_s: float = 8.0) -> dict:
    result = thermostat_set_hvac_mode(int(device_id), str(mode or ""), float(confirm_timeout_s))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(name="c4_thermostat_set_fan_mode", description="Set fan mode (On/Auto/Circulate) on a Control4 thermostat.")
def c4_thermostat_set_fan_mode_tool(device_id: str, mode: str, confirm_timeout_s: float = 8.0) -> dict:
    result = thermostat_set_fan_mode(int(device_id), str(mode or ""), float(confirm_timeout_s))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(name="c4_thermostat_set_hold_mode", description="Set hold mode (Off/2 Hours/Next Event/Permanent/Hold Until) on a Control4 thermostat.")
def c4_thermostat_set_hold_mode_tool(device_id: str, mode: str, confirm_timeout_s: float = 8.0) -> dict:
    result = thermostat_set_hold_mode(int(device_id), str(mode or ""), float(confirm_timeout_s))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(name="c4_thermostat_set_heat_setpoint_f", description="Set heat setpoint (F) on a Control4 thermostat.")
def c4_thermostat_set_heat_setpoint_f_tool(device_id: str, setpoint_f: float, confirm_timeout_s: float = 8.0) -> dict:
    result = thermostat_set_heat_setpoint_f(int(device_id), float(setpoint_f), float(confirm_timeout_s))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(name="c4_thermostat_set_cool_setpoint_f", description="Set cool setpoint (F) on a Control4 thermostat.")
def c4_thermostat_set_cool_setpoint_f_tool(device_id: str, setpoint_f: float, confirm_timeout_s: float = 8.0) -> dict:
    result = thermostat_set_cool_setpoint_f(int(device_id), float(setpoint_f), float(confirm_timeout_s))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_thermostat_set_target_f",
    description=(
        "Set a target temperature (F) without changing HVAC mode. "
        "Heat sets heat setpoint; Cool sets cool setpoint; Auto sets heat=target and cool=target+deadband."
    ),
)
def c4_thermostat_set_target_f_tool(
    device_id: str,
    target_f: float,
    confirm_timeout_s: float = 10.0,
    deadband_f: float | None = None,
) -> dict:
    result = thermostat_set_target_f(
        int(device_id),
        float(target_f),
        float(confirm_timeout_s),
        (float(deadband_f) if deadband_f is not None else None),
    )
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Lights ----

@Mcp.tool(name="c4_light_get_state", description="Get current on/off state of a Control4 light.")
def c4_light_get_state_tool(device_id: str) -> dict:
    state = light_get_state(int(device_id))
    return {"ok": True, "device_id": str(device_id), "state": bool(state)}


@Mcp.tool(name="c4_light_get_level", description="Get current brightness level (0-100) of a Control4 light.")
def c4_light_get_level_tool(device_id: str) -> dict:
    result = light_get_level(int(device_id))
    if isinstance(result, int):
        return {"ok": True, "device_id": str(device_id), "level": result}
    return {"ok": True, "device_id": str(device_id), "variables": result}


@Mcp.tool(name="c4_light_set_level", description="Set a Control4 light level (0-100).")
def c4_light_set_level_tool(device_id: str, level: int) -> dict:
    level = int(level)
    if level < 0 or level > 100:
        return {"ok": False, "error": "level must be 0-100"}
    state = light_set_level(int(device_id), level)
    return {"ok": True, "device_id": str(device_id), "level": level, "state": bool(state)}


@Mcp.tool(name="c4_light_ramp", description="Ramp a Control4 light to a level over time_ms.")
def c4_light_ramp_tool(device_id: str, level: int, time_ms: int) -> dict:
    level = int(level)
    time_ms = int(time_ms)
    if level < 0 or level > 100:
        return {"ok": False, "error": "level must be 0-100"}
    if time_ms < 0:
        return {"ok": False, "error": "time_ms must be >= 0"}
    state = light_ramp(int(device_id), level, time_ms)
    return {"ok": True, "device_id": str(device_id), "level": level, "time_ms": time_ms, "state": bool(state)}


# ---- Locks ----

@Mcp.tool(name="c4_lock_get_state", description="Get current lock state (locked/unlocked) for a Control4 lock.")
def c4_lock_get_state_tool(device_id: str) -> dict:
    try:
        fut = _lock_pool.submit(lock_get_state, int(device_id))
        result = fut.result(timeout=20)
        if isinstance(result, dict):
            return _augment_lock_result(result, desired_locked=None)
        return {"ok": True, "result": result}
    except FutureTimeout:
        return {"ok": False, "device_id": int(device_id), "error": "tool timeout (20s)"}
    except Exception as e:
        return {"ok": False, "device_id": int(device_id), "error": repr(e)}


@Mcp.tool(name="c4_lock_unlock", description="Unlock a Control4 lock.")
def c4_lock_unlock_tool(device_id: str) -> dict:
    try:
        fut = _lock_pool.submit(lock_unlock, int(device_id))
        result = fut.result(timeout=20)
        if isinstance(result, dict):
            return _augment_lock_result(result, desired_locked=False)
        return {"ok": True, "result": result}
    except FutureTimeout:
        return {"ok": False, "device_id": int(device_id), "error": "tool timeout (20s)"}
    except Exception as e:
        return {"ok": False, "device_id": int(device_id), "error": repr(e)}


@Mcp.tool(name="c4_lock_lock", description="Lock a Control4 lock.")
def c4_lock_lock_tool(device_id: str) -> dict:
    try:
        fut = _lock_pool.submit(lock_lock, int(device_id))
        result = fut.result(timeout=20)
        if isinstance(result, dict):
            return _augment_lock_result(result, desired_locked=True)
        return {"ok": True, "result": result}
    except FutureTimeout:
        return {"ok": False, "device_id": int(device_id), "error": "tool timeout (20s)"}
    except Exception as e:
        return {"ok": False, "device_id": int(device_id), "error": repr(e)}


# ✅ In 0.6.1: mount without passing a registry object or Mcp() instance
mount_mcp(app, url_prefix="/mcp", middlewares=[mw_auth, mw_ratelimit, mw_cors])


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3333, debug=False, use_reloader=False, threaded=True)