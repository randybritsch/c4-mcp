from flask import Flask, jsonify
from flask_mcp_server import mount_mcp, Mcp
from flask_mcp_server.http_integrated import mw_auth, mw_ratelimit, mw_cors

from collections import Counter

from control4_adapter import (
    list_rooms,
    get_all_items,
    item_get_variables,
    light_get_state,
    light_get_level,
    light_set_level,
    light_ramp,
    lock_get_state,
    lock_lock,
    lock_unlock,
)

app = Flask(__name__)

# Return JSON for any unhandled error so PowerShell doesn't hide it
@app.errorhandler(Exception)
def _handle_any_exception(e):
    return jsonify({"ok": False, "error": repr(e)}), 500


# ---------- MCP tools (REGISTER ON GLOBAL REGISTRY via Mcp.tool) ----------

@Mcp.tool(name="ping", description="Health check tool to verify the MCP server is reachable.")
def ping() -> dict:
    return {"ok": True}


@Mcp.tool(name="c4_item_variables", description="Get raw Director variables for an item (debug).")
def c4_item_variables(device_id: str) -> dict:
    vars_ = item_get_variables(int(device_id))
    return {"ok": True, "device_id": str(device_id), "variables": vars_}


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
        "locks": {"lock"},
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
        if (i.get("control") or "") not in allowed:
            continue

        parent_id = i.get("parentId")
        devices.append(
            {
                "id": str(i.get("id")),
                "name": i.get("name"),
                "control": i.get("control"),
                "roomId": str(parent_id) if parent_id is not None else None,
                "roomName": rooms_by_id.get(str(parent_id)) if parent_id is not None else None,
                "uris": i.get("URIs") or {},
            }
        )

    devices.sort(key=lambda d: ((d.get("roomName") or ""), (d.get("name") or "")))
    return {"ok": True, "category": category, "devices": devices}


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
    result = lock_get_state(int(device_id))
    return {"ok": True, **result}


@Mcp.tool(name="c4_lock_lock", description="Lock a Control4 lock.")
def c4_lock_lock_tool(device_id: str) -> dict:
    result = lock_lock(int(device_id))
    return {"ok": True, **result}


@Mcp.tool(name="c4_lock_unlock", description="Unlock a Control4 lock.")
def c4_lock_unlock_tool(device_id: str) -> dict:
    result = lock_unlock(int(device_id))
    return {"ok": True, **result}


# ✅ In 0.6.1: mount without passing a registry object or Mcp() instance
mount_mcp(app, url_prefix="/mcp", middlewares=[mw_auth, mw_ratelimit, mw_cors])

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3333, debug=False, use_reloader=False)