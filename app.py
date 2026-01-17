# app.py (recovered + reformatted)

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
import time
import uuid

from flask import Flask, jsonify, request, g, has_request_context
from werkzeug.exceptions import HTTPException
from flask_mcp_server import Mcp, mount_mcp
from flask_mcp_server.http_integrated import mw_auth, mw_cors, mw_ratelimit

import flask_mcp_server

from session_memory import SessionStore, extract_lights_from_call, is_last_lights_token

from control4_adapter import (
    gateway as adapter_gateway,
    announcement_execute,
    announcement_execute_by_name,
    announcement_list,
    announcement_list_commands,
    capabilities_report,
    control_keypad_list,
    control_keypad_send_command,
    contact_get_state,
    doorstation_set_external_chime,
    doorstation_set_led,
    doorstation_set_raw_setting,
    debug_trace_command,
    fan_get_state,
    fan_list,
    fan_set_power,
    fan_set_speed,
    find_devices,
    find_rooms,
    get_all_items,
    intercom_list,
    intercom_touchscreen_screensaver,
    intercom_touchscreen_set_feature,
    item_execute_command,
    item_get_bindings,
    item_get_commands,
    item_get_variables,
    item_send_command,
    item_set_state,
    keypad_button_action,
    keypad_get_buttons,
    keypad_list,
    light_get_level,
    light_get_state,
    light_ramp,
    light_set_level,
    light_set_level_ex,
    list_rooms,
    lock_get_state,
    lock_lock,
    lock_unlock,
    macro_execute,
    macro_execute_by_name,
    macro_list,
    macro_list_commands,
    scheduler_get,
    scheduler_list,
    scheduler_list_commands,
    scheduler_set_enabled,
    motion_get_state,
    motion_list,
    alarm_list,
    alarm_get_state,
    alarm_set_mode,
    resolve_device,
    resolve_named_candidates,
    resolve_room,
    resolve_room_and_device,
    room_off,
    room_list_commands,
    room_list_video_devices,
    room_remote,
    room_send_command,
    uibutton_activate,
    media_get_state,
    media_get_now_playing,
    media_remote,
    media_remote_sequence,
    media_send_command,
    media_launch_app,
        media_watch_launch_app,
        room_select_video_device,
        room_watch_status,
    media_roku_list_apps,
    thermostat_get_state,
    thermostat_set_cool_setpoint_f,
    thermostat_set_fan_mode,
    thermostat_set_heat_setpoint_f,
    thermostat_set_hold_mode,
    thermostat_set_hvac_mode,
    thermostat_set_target_f,
    room_select_audio_device,
    room_listen,
    room_listen_status,
    room_now_playing,
    room_lights_set,
    shade_close,
    shade_get_state,
    shade_list,
    shade_open,
    shade_set_position,
    shade_stop,
)

# ---------- App / Gateway ----------
app = Flask(__name__)


def _setup_logging() -> logging.Logger:
    """Configure structured logging for the MCP server.

    This is intentionally lightweight and safe:
    - No contract changes to MCP responses
    - Avoids logging sensitive fields by default
    """

    logger = logging.getLogger("c4-mcp")
    if getattr(logger, "_c4_configured", False):
        return logger

    level_name = str(os.getenv("C4_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    os.makedirs("logs", exist_ok=True)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = RotatingFileHandler(os.path.join("logs", "mcp_server.log"), maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.propagate = False
    logger._c4_configured = True
    return logger


_log = _setup_logging()


def _safe_json(obj: object) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return json.dumps({"_error": "json-encode-failed"})


_SENSITIVE_ARG_KEYS = {
    "password",
    "pass",
    "token",
    "api_key",
    "apikey",
    "secret",
}


def _env_truthy(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_csv(name: str) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return []
    parts = [p.strip() for p in str(raw).split(",")]
    return [p for p in parts if p]


# ---------- Session memory (in-process, per client session when possible) ----------

_SESSION_STORE = SessionStore(
    max_sessions=int(os.getenv("C4_SESSION_MAX", "200") or "200"),
    ttl_s=float(os.getenv("C4_SESSION_TTL_S", str(2 * 60 * 60)) or str(2 * 60 * 60)),
)

_PROCESS_SESSION_ID = str(os.getenv("C4_SESSION_ID") or "").strip() or str(uuid.uuid4())
os.environ.setdefault("C4_SESSION_ID", _PROCESS_SESSION_ID)


def _current_session_id(explicit: str | None = None) -> str:
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    # HTTP clients can supply a stable session id header.
    if has_request_context():
        for hdr in ("X-Session-Id", "X-MCP-Session-Id", "X-MCP-Session", "X-C4-Session"):
            v = request.headers.get(hdr)
            if v and str(v).strip():
                return str(v).strip()

    # STDIO clients can set this once at process start (Claude shim does this on initialize).
    v = os.getenv("C4_SESSION_ID")
    if v and str(v).strip():
        return str(v).strip()

    return _PROCESS_SESSION_ID


def _remember_tool_call(tool_name: str, args: dict | None, result: Any) -> None:
    try:
        sid = _current_session_id(None)
        mem = _SESSION_STORE.get(sid, create=True)
        mem.last_tool = str(tool_name or "")
        mem.last_tool_args = dict(args or {})
        mem.last_tool_at_s = time.time()

        lights = extract_lights_from_call(str(tool_name or ""), dict(args or {}), result)
        if lights:
            mem.add_last_lights(lights, window_s=5.0)
    except Exception:
        # Never let memory tracking break tools.
        return


# Best-effort classification of tools that mutate state.
# This list is intentionally conservative; you can refine over time.
_WRITE_TOOL_NAMES = {
    # Generic / debug
    "c4_item_send_command",
    "c4_item_execute_command",
    # Lighting
    "c4_light_set_level",
    "c4_light_ramp",
    "c4_light_set_by_name",
    "c4_room_lights_set",
    "c4_lights_set_last",
    # Locks
    "c4_lock_lock",
    "c4_lock_unlock",
    "c4_lock_set_by_name",
    # Thermostat
    "c4_thermostat_set_target_f",
    "c4_thermostat_set_heat_setpoint_f",
    "c4_thermostat_set_cool_setpoint_f",
    "c4_thermostat_set_hvac_mode",
    "c4_thermostat_set_fan_mode",
    "c4_thermostat_set_hold_mode",
    # Room / UI actions
    "c4_room_send_command",
    "c4_room_remote",
    "c4_room_off",
    "c4_room_select_video_device",
    "c4_tv_watch_by_name",
    "c4_room_select_audio_device",
    "c4_room_listen",
    "c4_room_listen_by_name",
    "c4_uibutton_activate",
    "c4_scene_activate",
    "c4_scene_activate_by_name",
    "c4_scene_set_state_by_name",
    # Media
    "c4_media_send_command",
    "c4_media_remote",
    "c4_media_remote_sequence",
    "c4_media_launch_app",
    "c4_media_watch_launch_app",
    "c4_media_watch_launch_app_by_name",
    # Scheduler
    "c4_scheduler_set_enabled",
    # Macros / announcements
    "c4_macro_execute",
    "c4_macro_execute_by_name",
    "c4_announcement_execute",
    "c4_announcement_execute_by_name",
    # Intercom / doorstation
    "c4_intercom_touchscreen_screensaver",
    "c4_intercom_touchscreen_set_feature",
    "c4_doorstation_set_led",
    "c4_doorstation_set_external_chime",
    "c4_doorstation_set_raw_setting",
    # Fans
    "c4_fan_set_power",
    "c4_fan_set_speed",
    # Keypads (actions)
    "c4_keypad_button_action",
    "c4_control_keypad_send_command",
}


def _is_write_tool(tool_name: str) -> bool:
    if not tool_name:
        return False
    name = str(tool_name)
    if name in _WRITE_TOOL_NAMES:
        return True
    # Heuristic fallback for common write naming patterns.
    lowered = name.lower()
    return any(
        lowered.startswith(prefix)
        for prefix in (
            "c4_set_",
            "c4_light_set_",
            "c4_light_ramp",
            # Locks: avoid blocking read-only tools like c4_lock_get_state.
            "c4_lock_set_",
            # Shades: avoid blocking read-only tools like c4_shade_list/get_state.
            "c4_shade_open",
            "c4_shade_close",
            "c4_shade_stop",
            "c4_shade_set_",
            # Scenes/UI buttons: avoid blocking read-only list tools.
            "c4_scene_activate",
            "c4_scene_set_",
            "c4_thermostat_set_",
            "c4_media_remote",
            "c4_media_send_",
            "c4_room_send_",
        )
    )


def _write_guardrails_enabled() -> bool:
    # Opt-in only: leaving this off preserves current behavior.
    return _env_truthy("C4_WRITE_GUARDRAILS", default=False)


def _writes_enabled() -> bool:
    # Only relevant when guardrails are enabled.
    return _env_truthy("C4_WRITES_ENABLED", default=False)


def _write_allowed(tool_name: str) -> tuple[bool, str | None]:
    deny = {s.lower() for s in _env_csv("C4_WRITE_DENYLIST")}
    allow = {s.lower() for s in _env_csv("C4_WRITE_ALLOWLIST")}

    tn = str(tool_name or "")
    tnl = tn.lower()

    if tnl in deny:
        return False, "write denied by C4_WRITE_DENYLIST"
    if allow and tnl not in allow:
        return False, "write not present in C4_WRITE_ALLOWLIST"
    return True, None


@app.before_request
def _c4_before_request() -> None:
    g._c4_start = time.perf_counter()
    g.request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    g.session_id = _current_session_id(None)

    # Opt-in: allow operators to run the server in a safe, read-only mode.
    # This is enforced at the HTTP boundary so we don't have to thread flags through tool code.
    try:
        if not _write_guardrails_enabled():
            return

        if request.path.endswith("/mcp/call") and request.method.upper() == "POST" and request.is_json:
            body = request.get_json(silent=True) or {}
            if isinstance(body, dict) and body.get("kind") == "tool":
                tool_name = str(body.get("name") or "")
                if _is_write_tool(tool_name):
                    if not _writes_enabled():
                        _log.warning(
                            _safe_json(
                                {
                                    "event": "write_blocked",
                                    "request_id": getattr(g, "request_id", None),
                                    "reason": "C4_WRITES_ENABLED is not true",
                                    "tool": tool_name,
                                }
                            )
                        )
                        # Match our general JSON error shape; keep it simple.
                        return (
                            jsonify(
                                {
                                    "ok": False,
                                    "error": "writes_disabled",
                                    "details": "Write tools are blocked (set C4_WRITES_ENABLED=true or disable C4_WRITE_GUARDRAILS).",
                                    "request_id": getattr(g, "request_id", None),
                                }
                            ),
                            403,
                        )

                    allowed, why = _write_allowed(tool_name)
                    if not allowed:
                        _log.warning(
                            _safe_json(
                                {
                                    "event": "write_blocked",
                                    "request_id": getattr(g, "request_id", None),
                                    "reason": why,
                                    "tool": tool_name,
                                }
                            )
                        )
                        return (
                            jsonify(
                                {
                                    "ok": False,
                                    "error": "write_not_allowed",
                                    "details": why,
                                    "request_id": getattr(g, "request_id", None),
                                }
                            ),
                            403,
                        )
    except Exception:
        # Never break requests due to guardrails parsing.
        return


@app.after_request
def _c4_after_request(resp):
    try:
        resp.headers["X-Request-Id"] = getattr(g, "request_id", "")
        resp.headers["X-Session-Id"] = getattr(g, "session_id", "")

        start = getattr(g, "_c4_start", None)
        duration_ms = None
        if isinstance(start, (int, float)):
            duration_ms = round((time.perf_counter() - float(start)) * 1000.0, 2)

        fields: dict[str, object] = {
            "event": "http_request",
            "request_id": getattr(g, "request_id", None),
            "method": request.method,
            "path": request.path,
            "status": int(getattr(resp, "status_code", 0) or 0),
            "duration_ms": duration_ms,
        }

        # MCP-specific context (best-effort). Avoid logging full args.
        if request.path.endswith("/mcp/call"):
            body = request.get_json(silent=True) if request.is_json else None
            if isinstance(body, dict):
                fields["mcp_kind"] = body.get("kind")
                fields["mcp_name"] = body.get("name")
                args = body.get("args")
                if isinstance(args, dict):
                    lowered = {str(k).lower() for k in args.keys()}
                    fields["arg_count"] = len(args)
                    fields["arg_redacted"] = any(k in _SENSITIVE_ARG_KEYS for k in lowered)
                    if not fields["arg_redacted"]:
                        keys = [str(k) for k in args.keys()]
                        fields["arg_keys"] = keys[:25]

        _log.info(_safe_json(fields))
    except Exception:
        # Never let logging break a request.
        pass
    return resp

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


@Mcp.tool(
    name="c4_memory_get",
    description=(
        "Return this server's in-process session memory for the current MCP client session. "
        "For HTTP clients, provide/echo a stable X-Session-Id header to persist context."
    ),
)
def c4_memory_get_tool(session_id: str | None = None) -> dict:
    sid = _current_session_id(session_id)
    mem = _SESSION_STORE.get(sid, create=True)
    return {"ok": True, "session_id": sid, "memory": mem.snapshot()}


@Mcp.tool(
    name="c4_memory_clear",
    description=(
        "Clear this server's in-process session memory for the current MCP client session. "
        "Useful if the model got confused about what 'those lights' refers to."
    ),
)
def c4_memory_clear_tool(session_id: str | None = None) -> dict:
    sid = _current_session_id(session_id)
    _SESSION_STORE.clear(sid)
    return {"ok": True, "session_id": sid, "cleared": True}


@Mcp.tool(
    name="c4_lights_get_last",
    description=(
        "Return the last referenced light devices in this session (for follow-ups like 'turn off those lights')."
    ),
)
def c4_lights_get_last_tool(session_id: str | None = None) -> dict:
    sid = _current_session_id(session_id)
    mem = _SESSION_STORE.get(sid, create=True)
    return {"ok": True, "session_id": sid, "count": len(mem.last_lights), "lights": list(mem.last_lights)}


@Mcp.tool(
    name="c4_lights_set_last",
    description=(
        "Set the state/level of the last referenced lights in this session (the safe way to implement 'those lights'). "
        "Provide exactly one of: state ('on'/'off') or level (0-100)."
    ),
)
def c4_lights_set_last_tool(
    state: str | None = None,
    level: int | None = None,
    ramp_ms: int | None = None,
    session_id: str | None = None,
) -> dict:
    if (state is None) == (level is None):
        return {"ok": False, "error": "provide exactly one of: state or level"}

    target_level: int
    if state is not None:
        s = str(state or "").strip().lower()
        if s not in {"on", "off"}:
            return {"ok": False, "error": "state must be 'on' or 'off'"}
        target_level = 100 if s == "on" else 0
    else:
        target_level = int(level)  # type: ignore[arg-type]
        if target_level < 0 or target_level > 100:
            return {"ok": False, "error": "level must be 0-100"}

    sid = _current_session_id(session_id)
    mem = _SESSION_STORE.get(sid, create=True)
    if not mem.last_lights:
        return {"ok": False, "error": "no remembered lights in this session yet", "session_id": sid}

    results: list[dict] = []
    for row in list(mem.last_lights):
        did = row.get("device_id")
        if did is None:
            continue
        try:
            did_i = int(did)
        except Exception:
            continue
        try:
            if ramp_ms is not None:
                rr = light_ramp(int(did_i), int(target_level), int(ramp_ms))
                results.append({"device_id": did_i, "ok": True, "ramped": True, "result": rr, "name": row.get("name")})
            else:
                rr = light_set_level(int(did_i), int(target_level))
                results.append({"device_id": did_i, "ok": True, "state": bool(rr), "name": row.get("name")})
        except Exception as e:
            results.append({"device_id": did_i, "ok": False, "error": repr(e), "name": row.get("name")})

    out = {
        "ok": all(r.get("ok") is True for r in results),
        "session_id": sid,
        "count": len(results),
        "target_level": int(target_level),
        "ramp_ms": (int(ramp_ms) if ramp_ms is not None else None),
        "results": results,
    }
    _remember_tool_call("c4_lights_set_last", {"state": state, "level": level, "ramp_ms": ramp_ms}, out)
    return out


@Mcp.tool(
    name="c4_server_info",
    description=(
        "Return process/runtime info for the running MCP server (PID, exe, cwd, argv) plus a tool-registry summary. "
        "Useful for diagnosing multiple/stale app.py processes on Windows."
    ),
)
def c4_server_info_tool() -> dict:
    reg = getattr(flask_mcp_server, "default_registry", None)
    tools_dict = None
    if reg is not None:
        for attr in ("tools", "_tools", "tool_map", "_tool_map", "_tools_by_name"):
            v = getattr(reg, attr, None)
            if isinstance(v, dict):
                tools_dict = v
                break

    tool_names = sorted(list(tools_dict.keys())) if isinstance(tools_dict, dict) else []
    return {
        "ok": True,
        "pid": os.getpid(),
        "ppid": os.getppid() if hasattr(os, "getppid") else None,
        "python_executable": sys.executable,
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "app_file": __file__,
        "registry": {
            "tool_count": len(tool_names),
            "has_media_remote": "c4_media_remote" in tool_names,
            "has_media_now_playing": "c4_media_now_playing" in tool_names,
            "sample_tools": tool_names[:50],
        },
    }


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
    name="c4_room_select_video_device",
    description=(
        "Select a room's active video device (i.e., trigger the Control4 Watch flow for a given HDMI/source device). "
        "This is often required before launching Roku apps so the TV is on the correct input."
    ),
)
def c4_room_select_video_device(room_id: str, device_id: str, deselect: bool = False) -> dict:
    result = room_select_video_device(int(room_id), int(device_id), bool(deselect))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_room_off",
    description=(
        "Turn off all Audio/Video in a room (ROOM_OFF) and best-effort confirm Watch becomes inactive. "
        "Returns accepted/confirmed semantics."
    ),
)
def c4_room_off_tool(room_id: str, confirm_timeout_s: float = 10.0) -> dict:
    result = room_off(int(room_id), float(confirm_timeout_s))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_room_list_commands",
    description=(
        "List available room-level commands (GET /rooms/{room_id}/commands). "
        "This is the most universal way to control AV/TV, audio, and navigation in Control4 rooms."
    ),
)
def c4_room_list_commands_tool(room_id: str, search: str | None = None) -> dict:
    result = room_list_commands(int(room_id), (str(search) if search is not None else None))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_room_list_video_devices",
    description=(
        "List the selectable video devices (sources) for a room (GET /locations/rooms/{room_id}/video_devices). "
        "Use these device ids with c4_tv_watch or c4_room_select_video_device."
    ),
)
def c4_room_list_video_devices_tool(room_id: str) -> dict:
    result = room_list_video_devices(int(room_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_room_watch_status",
    description=(
        "Get best-effort Watch UI status for a room via UI configuration agent. "
        "Returns active flag and the current configured sources when available."
    ),
)
def c4_room_watch_status_tool(room_id: str) -> dict:
    result = room_watch_status(int(room_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_room_send_command",
    description=(
        "Send a named room-level command to a room (POST /rooms/{room_id}/commands). "
        "Use c4_room_list_commands to discover valid command strings and required params."
    ),
)
def c4_room_send_command_tool(room_id: str, command: str, params: dict | None = None) -> dict:
    result = room_send_command(int(room_id), str(command or ""), params)
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


@Mcp.tool(name="c4_find_rooms", description="Find rooms by name (case-insensitive, fuzzy).")
def c4_find_rooms_tool(search: str, limit: int = 10, include_raw: bool = False) -> dict:
    return find_rooms(str(search or ""), limit=int(limit), include_raw=bool(include_raw))


@Mcp.tool(
    name="c4_resolve_room",
    description=(
        "Resolve a room name to a single room_id (best-effort). Returns candidates when ambiguous. "
        "Use c4_find_rooms if you want to pick manually."
    ),
)
def c4_resolve_room_tool(name: str, require_unique: bool = True, include_candidates: bool = True) -> dict:
    return resolve_room(str(name or ""), require_unique=bool(require_unique), include_candidates=bool(include_candidates))


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


@Mcp.tool(
    name="c4_capabilities_report",
    description=(
        "Summarize your Control4 inventory by control/proxy/driver filename/room. "
        "Useful for figuring out what else is available to automate next."
    ),
)
def c4_capabilities_report_tool(top_n: int = 20, include_examples: bool = False, max_examples_per_bucket: int = 3) -> dict:
    result = capabilities_report(int(top_n), bool(include_examples), int(max_examples_per_bucket))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- UI Buttons / Scenes (best-effort) ----


@Mcp.tool(
    name="c4_uibutton_list",
    description=(
        "List UI Button (uibutton) devices. These often represent Navigator shortcuts (mini-apps) "
        "and are a good proxy for 'scenes' or automations that users can trigger."
    ),
)
def c4_uibutton_list_tool() -> dict:
    items = get_all_items()
    rooms_by_id = {
        str(i.get("id")): i.get("name")
        for i in items
        if isinstance(i, dict) and i.get("typeName") == "room"
    }

    buttons = []
    for i in items:
        if not isinstance(i, dict) or i.get("typeName") != "device":
            continue
        if str(i.get("proxy") or "").lower() != "uibutton":
            continue
        room_id = i.get("roomId") or i.get("parentId")
        resolved_room_name = i.get("roomName") or (rooms_by_id.get(str(room_id)) if room_id is not None else None)
        buttons.append(
            {
                "device_id": str(i.get("id")),
                "name": i.get("name"),
                "room_id": str(room_id) if room_id is not None else None,
                "room_name": resolved_room_name,
            }
        )

    buttons.sort(key=lambda d: ((d.get("room_name") or ""), (d.get("name") or "")))
    return {"ok": True, "count": len(buttons), "uibuttons": buttons}


@Mcp.tool(
    name="c4_uibutton_activate",
    description=(
        "Activate a UI Button device. By default this sends the best-known activation command (usually 'Select'). "
        "Use dry_run=true to see what would be sent."
    ),
)
def c4_uibutton_activate_tool(device_id: str, command: str | None = None, dry_run: bool = False) -> dict:
    result = uibutton_activate(int(device_id), (str(command) if command is not None else None), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# Convenience aliases (many users think of these as scenes)


@Mcp.tool(name="c4_scene_list", description="Alias of c4_uibutton_list.")
def c4_scene_list_tool() -> dict:
    return c4_uibutton_list_tool()


@Mcp.tool(name="c4_scene_activate", description="Alias of c4_uibutton_activate.")
def c4_scene_activate_tool(device_id: str, command: str | None = None, dry_run: bool = False) -> dict:
    return c4_uibutton_activate_tool(device_id=device_id, command=command, dry_run=bool(dry_run))


@Mcp.tool(
    name="c4_scene_activate_by_name",
    description=(
        "Resolve and activate a scene by name (best-effort). Uses UI Button devices as a proxy for scenes. "
        "Optionally scope the search by room_name."
    ),
)
def c4_scene_activate_by_name_tool(
    scene_name: str,
    room_name: str | None = None,
    require_unique: bool = True,
    include_candidates: bool = True,
    command: str | None = None,
    dry_run: bool = False,
) -> dict:
    resolved_room_id: int | None = None
    resolved_room_name: str | None = None

    if room_name is not None and str(room_name).strip():
        rr = resolve_room(
            str(room_name),
            require_unique=bool(require_unique),
            include_candidates=bool(include_candidates),
        )
        if not isinstance(rr, dict) or not rr.get("ok"):
            return {"ok": False, "error": "could not resolve room", "details": rr}
        try:
            resolved_room_id = int(rr.get("room_id"))
        except Exception:
            resolved_room_id = None
        resolved_room_name = str(rr.get("name")) if rr.get("name") is not None else None

    rd = resolve_device(
        str(scene_name),
        category="scenes",
        room_id=resolved_room_id,
        require_unique=bool(require_unique),
        include_candidates=bool(include_candidates),
    )
    if not isinstance(rd, dict) or not rd.get("ok"):
        return {"ok": False, "error": "could not resolve scene", "details": rd}

    device_id = rd.get("device_id")
    if device_id is None:
        return {"ok": False, "error": "resolve_device returned no device_id", "details": rd}

    exec_res = uibutton_activate(int(device_id), (str(command) if command is not None else None), bool(dry_run))
    return {
        "ok": bool(exec_res.get("ok")) if isinstance(exec_res, dict) else True,
        "scene_name": str(scene_name),
        "room_id": (str(resolved_room_id) if resolved_room_id is not None else None),
        "room_name": resolved_room_name,
        "device_id": str(device_id),
        "resolve": rd,
        "execute": exec_res,
    }


# ---- Alarm / Security (best-effort) ----


@Mcp.tool(
    name="c4_alarm_list",
    description=(
        "List alarm/security panel-like devices (best-effort discovery). "
        "Returns an empty list when no alarm panel is present."
    ),
)
def c4_alarm_list_tool(limit: int = 200) -> dict:
    return alarm_list(int(limit))


@Mcp.tool(
    name="c4_alarm_get_state",
    description=(
        "Get best-effort alarm/security state from device variables (armed/mode/alarm_active/ready/trouble when available)."
    ),
)
def c4_alarm_get_state_tool(device_id: str, timeout_s: float = 8.0) -> dict:
    return alarm_get_state(int(device_id), timeout_s=float(timeout_s))


@Mcp.tool(
    name="c4_alarm_set_mode",
    description=(
        "Arm/disarm an alarm/security panel (best-effort). Mode must be one of: disarmed, away, stay, night. "
        "Optionally pass code (PIN/user code) if the driver requires it. Supports dry_run and returns accepted/confirmed."
    ),
)
def c4_alarm_set_mode_tool(
    device_id: str,
    mode: str,
    code: str | None = None,
    confirm_timeout_s: float = 12.0,
    dry_run: bool = False,
) -> dict:
    return alarm_set_mode(
        int(device_id),
        str(mode or ""),
        (str(code) if code is not None else None),
        confirm_timeout_s=float(confirm_timeout_s),
        dry_run=bool(dry_run),
    )


@Mcp.tool(name="c4_alarm_disarm", description="Disarm an alarm/security panel (best-effort).")
def c4_alarm_disarm_tool(device_id: str, code: str | None = None, confirm_timeout_s: float = 12.0, dry_run: bool = False) -> dict:
    return c4_alarm_set_mode_tool(device_id=device_id, mode="disarmed", code=code, confirm_timeout_s=float(confirm_timeout_s), dry_run=bool(dry_run))


@Mcp.tool(name="c4_alarm_arm_away", description="Arm an alarm/security panel in Away mode (best-effort).")
def c4_alarm_arm_away_tool(device_id: str, code: str | None = None, confirm_timeout_s: float = 12.0, dry_run: bool = False) -> dict:
    return c4_alarm_set_mode_tool(device_id=device_id, mode="away", code=code, confirm_timeout_s=float(confirm_timeout_s), dry_run=bool(dry_run))


@Mcp.tool(name="c4_alarm_arm_stay", description="Arm an alarm/security panel in Stay mode (best-effort).")
def c4_alarm_arm_stay_tool(device_id: str, code: str | None = None, confirm_timeout_s: float = 12.0, dry_run: bool = False) -> dict:
    return c4_alarm_set_mode_tool(device_id=device_id, mode="stay", code=code, confirm_timeout_s=float(confirm_timeout_s), dry_run=bool(dry_run))


@Mcp.tool(
    name="c4_scene_set_state_by_name",
    description=(
        "Fast-path: resolve a scene (UI Button) by name and set its on/off state in a single call. "
        "This is ideal for devices like 'Space Heater' that expose SetState(State=On|Off). "
        "Optionally scope the search by room_name. Best-effort confirmation polls the STATE variable."
    ),
)
def c4_scene_set_state_by_name_tool(
    scene_name: str,
    state: str,
    room_name: str | None = None,
    require_unique: bool = True,
    include_candidates: bool = True,
    confirm_timeout_s: float = 2.0,
    dry_run: bool = False,
) -> dict:
    state_norm = str(state or "").strip().lower()
    if state_norm not in {"on", "off"}:
        return {"ok": False, "error": "state must be 'on' or 'off'"}

    resolved_room_id: int | None = None
    resolved_room_name: str | None = None

    if room_name is not None and str(room_name).strip():
        rr = resolve_room(
            str(room_name),
            require_unique=bool(require_unique),
            include_candidates=bool(include_candidates),
        )
        if not isinstance(rr, dict) or not rr.get("ok"):
            return {"ok": False, "error": "could not resolve room", "details": rr}
        try:
            resolved_room_id = int(rr.get("room_id"))
        except Exception:
            resolved_room_id = None
        resolved_room_name = str(rr.get("name")) if rr.get("name") is not None else None

    rd = resolve_device(
        str(scene_name),
        category="scenes",
        room_id=resolved_room_id,
        require_unique=bool(require_unique),
        include_candidates=bool(include_candidates),
    )
    if not isinstance(rd, dict) or not rd.get("ok"):
        return {"ok": False, "error": "could not resolve scene", "details": rd}

    device_id = rd.get("device_id")
    if device_id is None:
        return {"ok": False, "error": "resolve_device returned no device_id", "details": rd}

    planned = {
        "device_id": str(device_id),
        "command": "SetState",
        "params": {"State": ("On" if state_norm == "on" else "Off")},
        "confirm_timeout_s": float(confirm_timeout_s),
    }

    if bool(dry_run):
        return {
            "ok": True,
            "scene_name": str(scene_name),
            "state": str(state),
            "room_id": (str(resolved_room_id) if resolved_room_id is not None else None),
            "room_name": resolved_room_name,
            "device_id": str(device_id),
            "resolve": rd,
            "planned": planned,
            "dry_run": True,
        }

    exec_res = item_set_state(int(device_id), state_norm, confirm_timeout_s=float(confirm_timeout_s))
    return {
        "ok": bool(exec_res.get("ok")) if isinstance(exec_res, dict) else True,
        "scene_name": str(scene_name),
        "state": str(state),
        "room_id": (str(resolved_room_id) if resolved_room_id is not None else None),
        "room_name": resolved_room_name,
        "device_id": str(device_id),
        "resolve": rd,
        "execute": exec_res,
    }


# ---- Contacts / Sensors (best-effort) ----


@Mcp.tool(
    name="c4_contact_list",
    description=(
        "List contact/sensor-style devices. Currently focuses on Card Access wireless contact/motion drivers "
        "(control='cardaccess_wirelesscontact')."
    ),
)
def c4_contact_list_tool() -> dict:
    items = get_all_items()
    rooms_by_id = {
        str(i.get("id")): i.get("name")
        for i in items
        if isinstance(i, dict) and i.get("typeName") == "room"
    }

    devices = []
    for i in items:
        if not isinstance(i, dict) or i.get("typeName") != "device":
            continue
        if str(i.get("control") or "").lower() != "cardaccess_wirelesscontact":
            continue
        room_id = i.get("roomId") or i.get("parentId")
        resolved_room_name = i.get("roomName") or (rooms_by_id.get(str(room_id)) if room_id is not None else None)
        devices.append(
            {
                "device_id": str(i.get("id")),
                "name": i.get("name"),
                "room_id": str(room_id) if room_id is not None else None,
                "room_name": resolved_room_name,
            }
        )

    devices.sort(key=lambda d: ((d.get("room_name") or ""), (d.get("name") or "")))
    return {"ok": True, "count": len(devices), "contacts": devices}


@Mcp.tool(
    name="c4_contact_get_state",
    description=(
        "Get best-effort state for a contact/motion sensor device. Returns raw variables plus parsed fields "
        "(battery_level, temperature, etc.)."
    ),
)
def c4_contact_get_state_tool(device_id: str, timeout_s: float = 6.0) -> dict:
    result = contact_get_state(int(device_id), float(timeout_s))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Motion sensors (best-effort) ----


@Mcp.tool(
    name="c4_motion_list",
    description=(
        "List motion sensor devices (best-effort). Currently includes contactsingle_motionsensor and wireless PIR proxies."
    ),
)
def c4_motion_list_tool() -> dict:
    result = motion_list()
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_motion_get_state",
    description=(
        "Get best-effort motion state for a motion sensor device. Returns raw variables plus parsed fields and motion_detected."
    ),
)
def c4_motion_get_state_tool(device_id: str, timeout_s: float = 6.0) -> dict:
    result = motion_get_state(int(device_id), float(timeout_s))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Intercom (best-effort) ----


@Mcp.tool(
    name="c4_intercom_list",
    description=(
        "List intercom-capable devices (best-effort; proxy contains 'intercom'). "
        "Includes touchscreens and door stations where present."
    ),
)
def c4_intercom_list_tool() -> dict:
    result = intercom_list()
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_intercom_touchscreen_set_feature",
    description=(
        "Enable/disable a touchscreen intercom feature. feature must be one of: autobrightness, proximity, alexa. "
        "Uses the device command strings exposed by c4_item_commands."
    ),
)
def c4_intercom_touchscreen_set_feature_tool(device_id: str, feature: str, enabled: bool, dry_run: bool = False) -> dict:
    result = intercom_touchscreen_set_feature(int(device_id), str(feature or ""), bool(enabled), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_intercom_touchscreen_screensaver",
    description=(
        "Control a touchscreen screensaver: optionally set mode, set start_time_s, and/or action enter/exit. "
        "You may combine multiple operations in one call."
    ),
)
def c4_intercom_touchscreen_screensaver_tool(
    device_id: str,
    action: str | None = None,
    mode: str | None = None,
    start_time_s: int | None = None,
    dry_run: bool = False,
) -> dict:
    result = intercom_touchscreen_screensaver(
        int(device_id),
        (str(action) if action is not None else None),
        (str(mode) if mode is not None else None),
        (int(start_time_s) if start_time_s is not None else None),
        bool(dry_run),
    )
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_doorstation_set_led",
    description=("Enable/disable the LED indicator on a Control4 door station (intercom proxy)."),
)
def c4_doorstation_set_led_tool(device_id: str, enabled: bool, dry_run: bool = False) -> dict:
    result = doorstation_set_led(int(device_id), bool(enabled), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_doorstation_set_external_chime",
    description=("Enable/disable the external chime on a Control4 door station (intercom proxy)."),
)
def c4_doorstation_set_external_chime_tool(device_id: str, enabled: bool, dry_run: bool = False) -> dict:
    result = doorstation_set_external_chime(int(device_id), bool(enabled), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_doorstation_set_raw_setting",
    description=(
        "Set a raw key/value setting on a Control4 door station via the 'Set Raw Settings' command. "
        "This is driver-specific; use cautiously."
    ),
)
def c4_doorstation_set_raw_setting_tool(device_id: str, key: str, value: str, dry_run: bool = False) -> dict:
    result = doorstation_set_raw_setting(int(device_id), str(key or ""), str(value or ""), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Macros (Agent) ----


@Mcp.tool(
    name="c4_macro_list",
    description=("List macros configured in Control4 (agents/macros)."),
)
def c4_macro_list_tool() -> dict:
    result = macro_list()
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_macro_list_commands",
    description=("List available macros agent commands (discovery/debug)."),
)
def c4_macro_list_commands_tool() -> dict:
    result = macro_list_commands()
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_macro_execute",
    description=("Execute a configured Control4 macro by id. Supports dry_run."),
)
def c4_macro_execute_tool(macro_id: int, dry_run: bool = False) -> dict:
    result = macro_execute(int(macro_id), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_macro_execute_by_name",
    description=(
        "Execute a configured Control4 macro by exact name (case-insensitive exact match). "
        "If the name is missing/ambiguous, returns suggestions and does not execute. Supports dry_run."
    ),
)
def c4_macro_execute_by_name_tool(name: str, dry_run: bool = False) -> dict:
    result = macro_execute_by_name(str(name or ""), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Scheduler (Agent) ----


@Mcp.tool(
    name="c4_scheduler_list",
    description=("List scheduled events configured in Control4 (agents/scheduler)."),
)
def c4_scheduler_list_tool(search: str | None = None) -> dict:
    result = scheduler_list((str(search) if search is not None else None))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_scheduler_get",
    description=("Get details for a scheduler event by event_id (agents/scheduler/{event_id})."),
)
def c4_scheduler_get_tool(event_id: int) -> dict:
    result = scheduler_get(int(event_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_scheduler_list_commands",
    description=("List available scheduler agent commands (discovery/debug)."),
)
def c4_scheduler_list_commands_tool() -> dict:
    result = scheduler_list_commands()
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_scheduler_set_enabled",
    description=(
        "Enable/disable a scheduler event by event_id. Supports dry_run. "
        "Returns accepted/confirmed based on a best-effort reread."
    ),
)
def c4_scheduler_set_enabled_tool(event_id: int, enabled: bool, dry_run: bool = False) -> dict:
    result = scheduler_set_enabled(int(event_id), bool(enabled), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Announcements (Agent) ----


@Mcp.tool(
    name="c4_announcement_list",
    description=("List announcements configured in Control4 (agents/announcements)."),
)
def c4_announcement_list_tool() -> dict:
    result = announcement_list()
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_announcement_list_commands",
    description=("List available announcements agent commands (discovery/debug)."),
)
def c4_announcement_list_commands_tool() -> dict:
    result = announcement_list_commands()
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_announcement_execute",
    description=("Execute a configured Control4 announcement by id. Supports dry_run."),
)
def c4_announcement_execute_tool(announcement_id: int, dry_run: bool = False) -> dict:
    result = announcement_execute(int(announcement_id), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_announcement_execute_by_name",
    description=(
        "Execute a configured Control4 announcement by exact name (case-insensitive exact match). "
        "If the name is missing/ambiguous, returns suggestions and does not execute. Supports dry_run."
    ),
)
def c4_announcement_execute_by_name_tool(name: str, dry_run: bool = False) -> dict:
    result = announcement_execute_by_name(str(name or ""), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Keypads (best-effort) ----


@Mcp.tool(
    name="c4_keypad_list",
    description=(
        "List physical keypad_proxy devices (keypads/dimmers with programmable buttons). "
        "Use c4_keypad_buttons and c4_keypad_button_action for button-based interaction."
    ),
)
def c4_keypad_list_tool() -> dict:
    result = keypad_list()
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_keypad_buttons",
    description=(
        "List button IDs and names for a keypad_proxy device (best-effort; derived from KEYPAD_BUTTON_* command metadata)."
    ),
)
def c4_keypad_buttons_tool(device_id: str) -> dict:
    result = keypad_get_buttons(int(device_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_keypad_button_action",
    description=(
        "Perform a keypad button action on a keypad_proxy device. "
        "action='tap' sends press+release; action can also be 'press' or 'release'."
    ),
)
def c4_keypad_button_action_tool(
    device_id: str,
    button_id: int,
    action: str = "tap",
    tap_ms: int = 200,
    dry_run: bool = False,
) -> dict:
    result = keypad_button_action(int(device_id), int(button_id), str(action or ""), int(tap_ms), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_control_keypad_list",
    description=(
        "List room_control_keypad devices (programmed 'control buttons' that can trigger presets/lights/room-off, etc.)."
    ),
)
def c4_control_keypad_list_tool() -> dict:
    result = control_keypad_list()
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_control_keypad_send_command",
    description=(
        "Trigger a command on a room_control_keypad device by command name (exact string from c4_item_commands or the list)."
    ),
)
def c4_control_keypad_send_command_tool(device_id: str, command: str, dry_run: bool = False) -> dict:
    result = control_keypad_send_command(int(device_id), str(command or ""), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Fans ----


@Mcp.tool(name="c4_fan_list", description="List fan devices (proxy='fan').")
def c4_fan_list_tool() -> dict:
    result = fan_list()
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(name="c4_fan_get_state", description="Get current fan power/speed (best-effort).")
def c4_fan_get_state_tool(device_id: str) -> dict:
    result = fan_get_state(int(device_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_fan_set_speed",
    description=(
        "Set fan speed. speed may be 0-4 or a name: off/low/medium/medium high/high. Returns accepted/confirmed."
    ),
)
def c4_fan_set_speed_tool(device_id: str, speed: str | int, confirm_timeout_s: float = 4.0, dry_run: bool = False) -> dict:
    result = fan_set_speed(int(device_id), speed, float(confirm_timeout_s), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_fan_set_power",
    description=("Set fan power: power must be one of on/off/toggle. Returns accepted/confirmed."),
)
def c4_fan_set_power_tool(device_id: str, power: str, confirm_timeout_s: float = 4.0, dry_run: bool = False) -> dict:
    result = fan_set_power(int(device_id), str(power or ""), float(confirm_timeout_s), bool(dry_run))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Outlets (as lights) ----


@Mcp.tool(
    name="c4_outlet_list",
    description=(
        "List outlet-controlled loads (control='outlet_light', proxy='light_v2'). "
        "These are typically the controllable outlets for outlet switch modules."
    ),
)
def c4_outlet_list_tool() -> dict:
    items = get_all_items()
    rooms_by_id = {
        str(i.get("id")): i.get("name")
        for i in items
        if isinstance(i, dict) and i.get("typeName") == "room"
    }

    outlets = []
    for i in items:
        if not isinstance(i, dict) or i.get("typeName") != "device":
            continue
        if str(i.get("control") or "").lower() != "outlet_light":
            continue
        room_id = i.get("roomId") or i.get("parentId")
        resolved_room_name = i.get("roomName") or (rooms_by_id.get(str(room_id)) if room_id is not None else None)
        outlets.append(
            {
                "device_id": str(i.get("id")),
                "name": i.get("name"),
                "room_id": str(room_id) if room_id is not None else None,
                "room_name": resolved_room_name,
            }
        )

    outlets.sort(key=lambda d: ((d.get("room_name") or ""), (d.get("name") or "")))
    return {"ok": True, "count": len(outlets), "outlets": outlets}


@Mcp.tool(name="c4_outlet_get_state", description="Get current outlet state (as a light).")
def c4_outlet_get_state_tool(device_id: str) -> dict:
    state = light_get_state(int(device_id))
    level = light_get_level(int(device_id))
    out = {"ok": True, "device_id": str(device_id), "state": bool(state)}
    if isinstance(level, int):
        out["level"] = level
    return out


@Mcp.tool(
    name="c4_outlet_set_power",
    description=("Turn an outlet load on/off (implemented via light level 0/100)."),
)
def c4_outlet_set_power_tool(device_id: str, on: bool, level_on: int = 100) -> dict:
    level_on = int(level_on)
    if level_on < 1 or level_on > 100:
        return {"ok": False, "error": "level_on must be 1-100"}
    level = level_on if bool(on) else 0
    state = light_set_level(int(device_id), int(level))
    return {"ok": True, "device_id": str(device_id), "on": bool(on), "level": int(level), "state": bool(state)}


@Mcp.tool(name="c4_list_devices", description="List Control4 devices by category (lights, locks, thermostat, media, scenes).")
def c4_list_devices(category: str) -> dict:
    category = (category or "").lower().strip()

    category_controls = {
        "lights": {"light_v2", "control4_lights_gen3", "outlet_light", "outlet_module_v2"},
        # Locks may appear either as a lock proxy (control=lock) or as a relay-style door lock proxy.
        "locks": {"lock", "control4_relaysingle"},
        "thermostat": {"thermostatV2"},
        # Shades vary wildly by driver; this bucket is discovered by proxy/control/category heuristics.
        "shades": set(),
        # Scenes are usually exposed as UI Button devices.
        "scenes": set(),
        # Alarm/security varies wildly by driver; discover by heuristics.
        "alarm": set(),
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
        if category == "shades":
            proxy_l = str(i.get("proxy") or "").lower()
            control_l = str(control or "").lower()
            cat_l = [str(c).lower() for c in categories] if isinstance(categories, list) else []
            if not (
                any(t in proxy_l for t in ("shade", "blind", "drape", "curtain", "screen"))
                or any(t in control_l for t in ("shade", "blind", "drape", "curtain", "screen"))
                or any(any(t in c for t in ("shade", "blind", "drape", "curtain", "screen")) for c in cat_l)
            ):
                continue
        elif category == "alarm":
            proxy_l = str(i.get("proxy") or "").lower()
            control_l = str(control or "").lower()
            protocol_l = str(i.get("protocolFilename") or "").lower()
            cat_l = [str(c).lower() for c in categories] if isinstance(categories, list) else []

            if proxy_l in {"uibutton", "voice-scene"}:
                continue

            # Avoid common false positives.
            if "keypad" in proxy_l or "keypad" in control_l:
                continue
            if proxy_l in {"light", "light_v2", "thermostat", "tv", "receiver", "media_player"}:
                continue

            token_sources = " ".join([proxy_l, control_l, protocol_l, " ".join(cat_l)])
            name_l = str(i.get("name") or "").lower()

            security_tokens = (
                "security",
                "alarm",
                "dsc",
                "honeywell",
                "vista",
                "ademco",
                "elk",
                "elkm1",
                "paradox",
                "qolsys",
                "2gig",
            )
            has_security = any(t in token_sources for t in security_tokens) or any(t in name_l for t in ("security", "alarm"))
            has_panelish = any(t in token_sources for t in ("panel", "partition"))
            name_panelish = ("panel" in name_l and ("alarm" in name_l or "security" in name_l))

            if not (name_panelish or has_security or (has_panelish and ("alarm" in name_l or "security" in name_l))):
                continue
        elif category == "scenes":
            proxy_l = str(i.get("proxy") or "").lower()
            control_l = str(control or "").lower()
            name_l = str(i.get("name") or "").lower()
            if not (
                proxy_l in {"uibutton", "voice-scene"}
                or control_l in {"uibutton", "voice-scene"}
                or "scene" in name_l
            ):
                continue
        else:
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


# ---- Shades ----


@Mcp.tool(name="c4_shade_list", description="List shade/blind-like devices (best-effort discovery).")
def c4_shade_list_tool(limit: int = 200) -> dict:
    return shade_list(int(limit))


@Mcp.tool(name="c4_shade_get_state", description="Get shade/blind state (best-effort). Returns position 0-100 when available.")
def c4_shade_get_state_tool(device_id: str) -> dict:
    return shade_get_state(int(device_id))


@Mcp.tool(
    name="c4_shade_open",
    description="Open/raise a shade/blind (best-effort). Returns accepted/confirmed semantics when position is available.",
)
def c4_shade_open_tool(device_id: str, confirm_timeout_s: float = 6.0, dry_run: bool = False) -> dict:
    return shade_open(int(device_id), confirm_timeout_s=float(confirm_timeout_s), dry_run=bool(dry_run))


@Mcp.tool(
    name="c4_shade_close",
    description="Close/lower a shade/blind (best-effort). Returns accepted/confirmed semantics when position is available.",
)
def c4_shade_close_tool(device_id: str, confirm_timeout_s: float = 6.0, dry_run: bool = False) -> dict:
    return shade_close(int(device_id), confirm_timeout_s=float(confirm_timeout_s), dry_run=bool(dry_run))


@Mcp.tool(name="c4_shade_stop", description="Stop shade/blind movement (best-effort).")
def c4_shade_stop_tool(device_id: str, dry_run: bool = False) -> dict:
    return shade_stop(int(device_id), dry_run=bool(dry_run))


@Mcp.tool(
    name="c4_shade_set_position",
    description=(
        "Set shade/blind position to 0-100 (best-effort). Uses available item commands when possible and confirms by reading position when available."
    ),
)
def c4_shade_set_position_tool(device_id: str, position: int, confirm_timeout_s: float = 8.0, dry_run: bool = False) -> dict:
    return shade_set_position(int(device_id), int(position), confirm_timeout_s=float(confirm_timeout_s), dry_run=bool(dry_run))


@Mcp.tool(
    name="c4_find_devices",
    description=(
        "Find devices by name (case-insensitive, fuzzy). Optional filters: category in {lights, locks, thermostat, media, scenes, shades, alarm} and room_id."
    ),
)
def c4_find_devices_tool(
    search: str | None = None,
    category: str | None = None,
    room_id: str | None = None,
    limit: int = 20,
    include_raw: bool = False,
) -> dict:
    rid = int(room_id) if room_id is not None and str(room_id).strip() else None
    return find_devices(
        (str(search) if search is not None else None),
        (str(category) if category is not None else None),
        room_id=rid,
        limit=int(limit),
        include_raw=bool(include_raw),
    )


@Mcp.tool(
    name="c4_resolve_device",
    description=(
        "Resolve a device name to a single device_id (best-effort). Optional filters: category (lights, locks, thermostat, media, scenes, shades, alarm) and room_id. Returns candidates when ambiguous."
    ),
)
def c4_resolve_device_tool(
    name: str,
    category: str | None = None,
    room_id: str | None = None,
    require_unique: bool = True,
    include_candidates: bool = True,
) -> dict:
    rid = int(room_id) if room_id is not None and str(room_id).strip() else None
    return resolve_device(
        str(name or ""),
        category=(str(category) if category is not None else None),
        room_id=rid,
        require_unique=bool(require_unique),
        include_candidates=bool(include_candidates),
    )


@Mcp.tool(
    name="c4_resolve",
    description=(
        "Resolve room and/or device names to ids in one call (best-effort). "
        "If room_name is provided, device resolution is scoped to that room."
    ),
)
def c4_resolve_tool(
    room_name: str | None = None,
    device_name: str | None = None,
    category: str | None = None,
    require_unique: bool = True,
    include_candidates: bool = True,
) -> dict:
    return resolve_room_and_device(
        (str(room_name) if room_name is not None else None),
        (str(device_name) if device_name is not None else None),
        (str(category) if category is not None else None),
        require_unique=bool(require_unique),
        include_candidates=bool(include_candidates),
    )


# ---- TV / Room-level control ----


@Mcp.tool(
    name="c4_tv_list",
    description=(
        "List TV devices in Control4 (control='tv'). Returns tv_device_id plus room_id for universal room-based control."
    ),
)
def c4_tv_list_tool() -> dict:
    items = get_all_items()
    rooms_by_id = {
        str(i.get("id")): i.get("name")
        for i in items
        if isinstance(i, dict) and i.get("typeName") == "room"
    }

    tvs = []
    for i in items:
        if not isinstance(i, dict):
            continue
        if i.get("typeName") != "device":
            continue
        if str(i.get("control") or "").lower() != "tv":
            continue

        room_id = i.get("roomId") or i.get("parentId")
        resolved_room_name = i.get("roomName") or (rooms_by_id.get(str(room_id)) if room_id is not None else None)
        tvs.append(
            {
                "tv_device_id": str(i.get("id")),
                "name": i.get("name"),
                "room_id": str(room_id) if room_id is not None else None,
                "room_name": resolved_room_name,
            }
        )

    tvs.sort(key=lambda t: ((t.get("room_name") or ""), (t.get("name") or "")))
    return {"ok": True, "count": len(tvs), "tvs": tvs}


@Mcp.tool(
    name="c4_tv_remote",
    description=(
        "Send a universal room-level remote command for the TV in that room (UP/DOWN/ENTER/BACK/MENU/INFO/EXIT, volume, channel, etc). "
        "This is room-based so it works with any TV driver in Control4."
    ),
)
def c4_tv_remote_tool(room_id: str, button: str, press: str | None = None) -> dict:
    result = room_remote(int(room_id), str(button or ""), press)
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_tv_watch",
    description=(
        "Start/ensure a Watch session in a room by selecting a video device (source) for that room. "
        "This is the reliable way to 'turn on the TV' in Control4."
    ),
)
def c4_tv_watch_tool(room_id: str, source_device_id: str, deselect: bool = False) -> dict:
    result = room_select_video_device(int(room_id), int(source_device_id), bool(deselect))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_tv_watch_by_name",
    description=(
        "Resolve a room by name and a video source device by name, then start/ensure a Watch session in that room. "
        "If resolution is ambiguous, returns candidates and does not execute."
    ),
)
def c4_tv_watch_by_name_tool(
    room_name: str,
    source_device_name: str,
    room_id: str | None = None,
    require_unique: bool = True,
    include_candidates: bool = True,
    deselect: bool = False,
    dry_run: bool = False,
) -> dict:
    resolved_room_id: int | None = None
    rr: dict | None = None

    if room_id is not None and str(room_id).strip():
        try:
            resolved_room_id = int(room_id)
        except Exception:
            resolved_room_id = None
    else:
        rr = resolve_room(
            str(room_name or ""),
            require_unique=bool(require_unique),
            include_candidates=bool(include_candidates),
        )
        if not isinstance(rr, dict) or not rr.get("ok"):
            return {"ok": False, "error": "could not resolve room", "details": rr}
        try:
            resolved_room_id = int(rr.get("room_id"))
        except Exception:
            resolved_room_id = None

    if resolved_room_id is None:
        return {"ok": False, "error": "room_id could not be resolved", "details": rr}

    # For Watch, we resolve the source device within the room to avoid cross-room ambiguity.
    rd = resolve_device(
        str(source_device_name or ""),
        category="media",
        room_id=resolved_room_id,
        require_unique=bool(require_unique),
        include_candidates=bool(include_candidates),
    )
    if not isinstance(rd, dict) or not rd.get("ok"):
        return {"ok": False, "error": "could not resolve video source device", "details": rd}

    source_device_id = rd.get("device_id")
    if source_device_id is None:
        return {"ok": False, "error": "resolve_device returned no device_id", "details": rd}

    planned = {
        "room_id": str(resolved_room_id),
        "source_device_id": str(source_device_id),
        "deselect": bool(deselect),
    }

    if bool(dry_run):
        return {
            "ok": True,
            "dry_run": True,
            "planned": planned,
            "room_name": str(room_name or ""),
            "room_id": str(resolved_room_id),
            "resolve_room": rr,
            "source_device_name": str(source_device_name or ""),
            "resolve_source": rd,
        }

    result = room_select_video_device(int(resolved_room_id), int(source_device_id), bool(deselect))
    return {
        "ok": bool(result.get("ok")) if isinstance(result, dict) else True,
        "planned": planned,
        "room_name": str(room_name or ""),
        "room_id": str(resolved_room_id),
        "resolve_room": rr,
        "source_device_name": str(source_device_name or ""),
        "resolve_source": rd,
        "result": result,
    }


@Mcp.tool(
    name="c4_tv_off",
    description=(
        "Turn off the room's Audio/Video session (ROOM_OFF). Best-effort confirms Watch becomes inactive."
    ),
)
def c4_tv_off_tool(room_id: str, confirm_timeout_s: float = 10.0) -> dict:
    result = room_off(int(room_id), float(confirm_timeout_s))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Audio (Room-based) ----

@Mcp.tool(
    name="c4_room_select_audio_device",
    description=(
        "Select a room's audio source (SELECT_AUDIO_DEVICE). "
        "Use c4_room_listen_status to discover valid source_device_id values for that room."
    ),
)
def c4_room_select_audio_device_tool(room_id: str, source_device_id: str, deselect: bool = False) -> dict:
    result = room_select_audio_device(int(room_id), int(source_device_id), bool(deselect))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_room_listen",
    description=(
        "Start a room 'Listen' session by selecting an audio source (SELECT_AUDIO_DEVICE) and confirming activation best-effort. "
        "Use c4_room_listen_status to find available sources; this is the audio equivalent of c4_tv_watch."
    ),
)
def c4_room_listen_tool(room_id: str, source_device_id: str, confirm_timeout_s: float = 10.0) -> dict:
    result = room_listen(int(room_id), int(source_device_id), float(confirm_timeout_s))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_room_listen_by_name",
    description=(
        "Resolve a room by name and a Listen source device by name, then start a room Listen session. "
        "Uses the same safe name resolution as c4_light_set_by_name and c4_media_watch_launch_app_by_name. "
        "If resolution is ambiguous, returns candidates and does not execute."
    ),
)
def c4_room_listen_by_name_tool(
    room_name: str,
    source_device_name: str,
    room_id: str | None = None,
    require_unique: bool = True,
    include_candidates: bool = True,
    confirm_timeout_s: float = 10.0,
    dry_run: bool = False,
) -> dict:
    resolved_room_id: int | None = None
    rr: dict | None = None

    if room_id is not None and str(room_id).strip():
        try:
            resolved_room_id = int(room_id)
        except Exception:
            resolved_room_id = None
    else:
        rr = resolve_room(
            str(room_name or ""),
            require_unique=bool(require_unique),
            include_candidates=bool(include_candidates),
        )
        if not isinstance(rr, dict) or not rr.get("ok"):
            return {"ok": False, "error": "could not resolve room", "details": rr}
        try:
            resolved_room_id = int(rr.get("room_id"))
        except Exception:
            resolved_room_id = None

    if resolved_room_id is None:
        return {"ok": False, "error": "room_id could not be resolved", "details": rr}

    # Resolve the source from the room's actual available Listen sources.
    ls_raw = room_listen_status(int(resolved_room_id))
    ls = ls_raw if isinstance(ls_raw, dict) else {"ok": True, "result": ls_raw}
    listen = ls.get("listen") if isinstance(ls.get("listen"), dict) else {}
    sources = listen.get("sources") if isinstance(listen.get("sources"), list) else []

    source_rows: list[dict] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        sid = None
        for k in ("deviceid", "deviceId", "id"):
            if s.get(k) is None:
                continue
            try:
                sid = int(s.get(k))
                break
            except Exception:
                continue
        if sid is None or sid <= 0:
            continue
        label = None
        for k in ("name", "label", "display", "title"):
            v = s.get(k)
            if isinstance(v, str) and v.strip():
                label = v.strip()
                break
        source_rows.append({"id": int(sid), "name": str(label or sid)})

    if not source_rows:
        return {
            "ok": False,
            "error": "no listen sources found for room",
            "room_id": str(resolved_room_id),
            "listen_status": ls,
        }

    resolved_src = resolve_named_candidates(
        str(source_device_name or ""),
        source_rows,
        entity="listen_source",
        name_key="name",
        id_key="id",
        max_candidates=10,
    )
    if not isinstance(resolved_src, dict) or not resolved_src.get("ok"):
        return {
            "ok": False,
            "error": "could not resolve listen source device",
            "room_id": str(resolved_room_id),
            "source_device_name": str(source_device_name or ""),
            "resolve_source": resolved_src,
            "listen_sources": source_rows[:15],
        }

    source_device_id = resolved_src.get("id")
    if source_device_id is None:
        return {
            "ok": False,
            "error": "resolved listen source missing id",
            "resolve_source": resolved_src,
        }

    planned = {
        "room_id": str(resolved_room_id),
        "source_device_id": str(source_device_id),
        "confirm_timeout_s": float(confirm_timeout_s),
    }

    if bool(dry_run):
        return {
            "ok": True,
            "dry_run": True,
            "planned": planned,
            "room_name": str(room_name or ""),
            "room_id": str(resolved_room_id),
            "resolve_room": rr,
            "source_device_name": str(source_device_name or ""),
            "resolve_source": resolved_src,
            "listen_status": ls,
        }

    result = room_listen(int(resolved_room_id), int(source_device_id), float(confirm_timeout_s))
    return {
        "ok": bool(result.get("ok")) if isinstance(result, dict) else True,
        "planned": planned,
        "room_name": str(room_name or ""),
        "room_id": str(resolved_room_id),
        "resolve_room": rr,
        "source_device_name": str(source_device_name or ""),
        "resolve_source": resolved_src,
        "listen_status": ls,
        "result": result,
    }


@Mcp.tool(
    name="c4_room_listen_status",
    description=(
        "Read-only: return the room's current Listen status and available Listen sources (best-effort), "
        "from UI configuration. Use this to find valid source device ids for c4_room_listen."
    ),
)
def c4_room_listen_status_tool(room_id: str) -> dict:
    result = room_listen_status(int(room_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_room_now_playing",
    description=(
        "Read-only: best-effort room-scoped now playing. Probes the room's Listen sources and returns the first "
        "device that exposes usable now-playing metadata (normalized when available), plus probe diagnostics."
    ),
)
def c4_room_now_playing_tool(room_id: str, max_sources: int = 30) -> dict:
    result = room_now_playing(int(room_id), int(max_sources))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


# ---- Media / AV ----

@Mcp.tool(name="c4_media_get_state", description="Get current state for a Control4 media/AV device (best-effort).")
def c4_media_get_state_tool(device_id: str) -> dict:
    result = media_get_state(int(device_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_media_send_command",
    description=(
        "Send a named command to a Control4 media/AV device. "
        "Use c4_item_commands(device_id) to discover supported command names and params."
    ),
)
def c4_media_send_command_tool(device_id: str, command: str, params: dict | None = None) -> dict:
    result = media_send_command(int(device_id), str(command or ""), params)
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_media_remote",
    description=(
        "Send a basic remote/navigation action to a media device (up/down/left/right/select/menu/home/playpause). "
        "Uses the device's transport proxy commands (when available)."
    ),
)
def c4_media_remote_tool(device_id: str, button: str, press: str = "Tap") -> dict:
    result = media_remote(int(device_id), str(button or ""), str(press or "Tap"))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_media_remote_sequence",
    description=(
        "Send a sequence of remote actions to a media device (e.g., ['home','down','down','select']). "
        "Useful for navigation macros."
    ),
)
def c4_media_remote_sequence_tool(device_id: str, buttons: list[str], press: str = "Tap", delay_ms: int = 250) -> dict:
    result = media_remote_sequence(int(device_id), list(buttons), str(press or "Tap"), int(delay_ms))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_media_now_playing",
    description=(
        "Best-effort 'now playing' for a media device. Returns normalized fields when present, plus candidate variables."
    ),
)
def c4_media_now_playing_tool(device_id: str) -> dict:
    result = media_get_now_playing(int(device_id))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


@Mcp.tool(
    name="c4_media_launch_app",
    description=(
        "Launch an app on a media device (primarily Roku). Uses the driver's LaunchApp command when available. "
        "Example app: 'Netflix' or 'Home'."
    ),
)
def c4_media_launch_app_tool(device_id: str, app: str) -> dict:
    result = media_launch_app(int(device_id), str(app or ""))
    return result if isinstance(result, dict) else {"ok": True, "result": result}

@Mcp.tool(
    name="c4_media_watch_launch_app",
    description=(
        "High-level helper: select the room video source for the given media device (Watch/HDMI) and then launch an app. "
        "This makes app launches reliably visible by ensuring the room is on the correct video input first."
    ),
)
def c4_media_watch_launch_app(device_id: str, app: str, room_id: str | None = None, pre_home: bool = True) -> dict:
    rid = int(room_id) if room_id is not None and str(room_id).strip() else None
    result = media_watch_launch_app(int(device_id), str(app or ""), room_id=rid, pre_home=bool(pre_home))
    if not isinstance(result, dict):
        return {"ok": True, "result": result}

    # Add a small, consistent summary for LLM/tool consumers.
    watch = result.get("watch") if isinstance(result.get("watch"), dict) else {}
    before_watch = watch.get("before") if isinstance(watch.get("before"), dict) else {}
    after_select = watch.get("after_select_video") if isinstance(watch.get("after_select_video"), dict) else {}
    after_launch = watch.get("after_launch") if isinstance(watch.get("after_launch"), dict) else {}

    launch = result.get("launch") if isinstance(result.get("launch"), dict) else {}
    profile = launch.get("profile")
    resolved = launch.get("resolved") if isinstance(launch.get("resolved"), dict) else None
    roku = launch.get("roku") if isinstance(launch.get("roku"), dict) else None

    select_video_ok = bool((result.get("select_video") or {}).get("ok")) if isinstance(result.get("select_video"), dict) else False
    launch_ok = bool(launch.get("ok"))

    summary: dict = {
        "ok": bool(result.get("ok")),
        "select_video_ok": select_video_ok,
        "watch_active_before": (before_watch.get("active") if isinstance(before_watch, dict) else None),
        "watch_active_after_select": (after_select.get("active") if isinstance(after_select, dict) else None),
        "watch_active_after_launch": (after_launch.get("active") if isinstance(after_launch, dict) else None),
        "launch_ok": launch_ok,
        "launch_profile": profile,
        "requested_app": result.get("app"),
    }

    if resolved is not None:
        summary["resolved"] = resolved

    if isinstance(roku, dict):
        before = roku.get("before") if isinstance(roku.get("before"), dict) else None
        after = roku.get("after") if isinstance(roku.get("after"), dict) else None
        summary["roku"] = {
            "expected_app_id": roku.get("expected_app_id"),
            "before_app": (before or {}).get("CURRENT_APP") if isinstance(before, dict) else None,
            "before_app_id": (before or {}).get("CURRENT_APP_ID") if isinstance(before, dict) else None,
            "after_app": (after or {}).get("CURRENT_APP") if isinstance(after, dict) else None,
            "after_app_id": (after or {}).get("CURRENT_APP_ID") if isinstance(after, dict) else None,
        }

    # Human-readable one-liner to make results easy to scan.
    try:
        watch_before = summary.get("watch_active_before")
        watch_after = summary.get("watch_active_after_select")
        if isinstance(summary.get("roku"), dict):
            r = summary["roku"]
            result["summary_text"] = (
                f"watch {watch_before}->{watch_after}; launch ok={launch_ok}; "
                f"roku {r.get('before_app')}({r.get('before_app_id')}) -> {r.get('after_app')}({r.get('after_app_id')}), expected {r.get('expected_app_id')}"
            )
        else:
            result["summary_text"] = f"watch {watch_before}->{watch_after}; launch ok={launch_ok}"
    except Exception:
        pass

    result["summary"] = summary
    return result


@Mcp.tool(
    name="c4_media_watch_launch_app_by_name",
    description=(
        "One-call helper: resolve a media device by name (optionally scoped by room) and then run c4_media_watch_launch_app. "
        "Use this to say things like 'Watch Netflix on <Roku Name> in <Room Name>' without looking up ids. "
        "Returns resolution details and preserves accepted/confirmed semantics from the underlying watch+launch flow."
    ),
)
def c4_media_watch_launch_app_by_name_tool(
    device_name: str,
    app: str,
    room_name: str | None = None,
    room_id: str | None = None,
    pre_home: bool = True,
    require_unique: bool = True,
    include_candidates: bool = True,
    dry_run: bool = False,
) -> dict:
    resolved_room_id: int | None = None
    resolved_room_name: str | None = None
    rr: dict | None = None

    if room_id is not None and str(room_id).strip():
        try:
            resolved_room_id = int(room_id)
        except Exception:
            resolved_room_id = None
    elif room_name is not None and str(room_name).strip():
        rr = resolve_room(
            str(room_name),
            require_unique=bool(require_unique),
            include_candidates=bool(include_candidates),
        )
        if not isinstance(rr, dict) or not rr.get("ok"):
            return {"ok": False, "error": "could not resolve room", "details": rr}
        try:
            resolved_room_id = int(rr.get("room_id"))
        except Exception:
            resolved_room_id = None
        resolved_room_name = str(rr.get("name")) if rr.get("name") is not None else None

    rd = resolve_device(
        str(device_name or ""),
        category="media",
        room_id=resolved_room_id,
        require_unique=bool(require_unique),
        include_candidates=bool(include_candidates),
    )
    if not isinstance(rd, dict) or not rd.get("ok"):
        return {"ok": False, "error": "could not resolve media device", "details": rd}

    # If the caller didn't specify a room scope, keep the resolved device's room
    # for better targeting + more informative output.
    if resolved_room_id is None and rd.get("room_id") is not None:
        try:
            resolved_room_id = int(rd.get("room_id"))
        except Exception:
            resolved_room_id = None
    if resolved_room_name is None and rd.get("room_name") is not None:
        resolved_room_name = str(rd.get("room_name"))

    device_id = rd.get("device_id")
    if device_id is None:
        return {"ok": False, "error": "resolve_device returned no device_id", "details": rd}

    planned = {
        "device_id": str(device_id),
        "app": str(app or ""),
        "room_id": (str(resolved_room_id) if resolved_room_id is not None else None),
        "pre_home": bool(pre_home),
    }

    if bool(dry_run):
        return {
            "ok": True,
            "device_name": str(device_name),
            "room_id": (str(resolved_room_id) if resolved_room_id is not None else None),
            "room_name": resolved_room_name,
            "resolve_room": rr,
            "resolve": rd,
            "planned": planned,
            "dry_run": True,
        }

    # Reuse the existing tool function so the output includes summary_text/summary.
    res = c4_media_watch_launch_app(
        device_id=str(device_id),
        app=str(app or ""),
        room_id=(str(resolved_room_id) if resolved_room_id is not None else None),
        pre_home=bool(pre_home),
    )
    ok = bool(res.get("ok")) if isinstance(res, dict) else bool(res)

    if isinstance(res, dict):
        res["device_name"] = str(device_name)
        res["resolve"] = rd
        if rr is not None:
            res["resolve_room"] = rr
        if resolved_room_name is not None and res.get("room_id") is not None:
            res["room_name"] = resolved_room_name

    return res if isinstance(res, dict) else {"ok": ok, "result": res, "planned": planned, "resolve": rd}


@Mcp.tool(
    name="c4_media_roku_list_apps",
    description=(
        "List Roku app options for the given Roku device by reading universal mini-app variables (APP_NAME/UM_ROKU) in the same room. "
        "Use this to find the exact app name/id to pass to c4_media_launch_app."
    ),
)
def c4_media_roku_list_apps_tool(device_id: str, search: str | None = None) -> dict:
    result = media_roku_list_apps(int(device_id), (str(search) if search is not None else None))
    return result if isinstance(result, dict) else {"ok": True, "result": result}


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
    out = {"ok": True, "device_id": str(device_id), "state": bool(state)}
    _remember_tool_call("c4_light_get_state", {"device_id": str(device_id)}, out)
    return out


@Mcp.tool(name="c4_light_get_level", description="Get current brightness level (0-100) of a Control4 light.")
def c4_light_get_level_tool(device_id: str) -> dict:
    result = light_get_level(int(device_id))
    if isinstance(result, int):
        out = {"ok": True, "device_id": str(device_id), "level": result}
        _remember_tool_call("c4_light_get_level", {"device_id": str(device_id)}, out)
        return out
    out = {"ok": True, "device_id": str(device_id), "variables": result}
    _remember_tool_call("c4_light_get_level", {"device_id": str(device_id)}, out)
    return out


@Mcp.tool(name="c4_light_set_level", description="Set a Control4 light level (0-100).")
def c4_light_set_level_tool(device_id: str, level: int) -> dict:
    level = int(level)
    if level < 0 or level > 100:
        return {"ok": False, "error": "level must be 0-100"}

    # Convenience: allow passing a token like "__last_lights__" to apply to remembered lights.
    if is_last_lights_token(device_id):
        return c4_lights_set_last_tool(level=int(level))

    state = light_set_level(int(device_id), level)
    out = {"ok": True, "device_id": str(device_id), "level": level, "state": bool(state)}
    _remember_tool_call("c4_light_set_level", {"device_id": device_id, "level": level}, out)
    return out


@Mcp.tool(name="c4_light_ramp", description="Ramp a Control4 light to a level over time_ms.")
def c4_light_ramp_tool(device_id: str, level: int, time_ms: int) -> dict:
    level = int(level)
    time_ms = int(time_ms)
    if level < 0 or level > 100:
        return {"ok": False, "error": "level must be 0-100"}
    if time_ms < 0:
        return {"ok": False, "error": "time_ms must be >= 0"}

    if is_last_lights_token(device_id):
        return c4_lights_set_last_tool(level=int(level), ramp_ms=int(time_ms))

    state = light_ramp(int(device_id), level, time_ms)
    out = {"ok": True, "device_id": str(device_id), "level": level, "time_ms": time_ms, "state": bool(state)}
    _remember_tool_call("c4_light_ramp", {"device_id": device_id, "level": level, "time_ms": time_ms}, out)
    return out


@Mcp.tool(
    name="c4_light_set_by_name",
    description=(
        "Fast-path: resolve a light by name (optionally scoped by room) and set level/on/off in a single call. "
        "Uses inventory caching for fast resolution. Optionally ramps and best-effort confirms final level."
    ),
)
def c4_light_set_by_name_tool(
    device_name: str,
    level: int | None = None,
    state: str | None = None,
    room_name: str | None = None,
    room_id: int | None = None,
    on_level: int = 100,
    ramp_ms: int | None = None,
    require_unique: bool = True,
    include_candidates: bool = True,
    confirm_timeout_s: float = 1.5,
    poll_interval_s: float = 0.2,
    dry_run: bool = False,
) -> dict:
    if (level is None) == (state is None):
        return {"ok": False, "error": "provide exactly one of: level or state"}

    target_level: int
    if state is not None:
        state_norm = str(state or "").strip().lower()
        if state_norm not in {"on", "off"}:
            return {"ok": False, "error": "state must be 'on' or 'off'"}
        on_level = max(0, min(100, int(on_level)))
        target_level = on_level if state_norm == "on" else 0
    else:
        target_level = int(level)  # type: ignore[arg-type]
        if target_level < 0 or target_level > 100:
            return {"ok": False, "error": "level must be 0-100"}

    resolved_room_id: int | None = None
    resolved_room_name: str | None = None

    if room_id is not None:
        try:
            resolved_room_id = int(room_id)
        except Exception:
            resolved_room_id = None
    elif room_name is not None and str(room_name).strip():
        rr = resolve_room(
            str(room_name),
            require_unique=bool(require_unique),
            include_candidates=bool(include_candidates),
        )
        if not isinstance(rr, dict) or not rr.get("ok"):
            return {"ok": False, "error": "could not resolve room", "details": rr}
        try:
            resolved_room_id = int(rr.get("room_id"))
        except Exception:
            resolved_room_id = None
        resolved_room_name = str(rr.get("name")) if rr.get("name") is not None else None

    rd = resolve_device(
        str(device_name),
        category="lights",
        room_id=resolved_room_id,
        require_unique=bool(require_unique),
        include_candidates=bool(include_candidates),
    )
    if not isinstance(rd, dict) or not rd.get("ok"):
        return {"ok": False, "error": "could not resolve light", "details": rd}

    device_id = rd.get("device_id")
    if device_id is None:
        return {"ok": False, "error": "resolve_device returned no device_id", "details": rd}

    planned = {
        "device_id": str(device_id),
        "target_level": int(target_level),
        "ramp_ms": (int(ramp_ms) if ramp_ms is not None else None),
        "confirm_timeout_s": float(confirm_timeout_s),
        "poll_interval_s": float(poll_interval_s),
        "tolerance": 1,
    }

    if bool(dry_run):
        return {
            "ok": True,
            "device_name": str(device_name),
            "room_id": (str(resolved_room_id) if resolved_room_id is not None else None),
            "room_name": resolved_room_name,
            "resolve": rd,
            "planned": planned,
            "dry_run": True,
        }

    exec_res = light_set_level_ex(
        int(device_id),
        int(target_level),
        (int(ramp_ms) if ramp_ms is not None else None),
        float(confirm_timeout_s),
        float(poll_interval_s),
        1,
    )

    ok = bool(exec_res.get("ok")) if isinstance(exec_res, dict) else bool(exec_res)
    out = {
        "ok": ok,
        "device_name": str(device_name),
        "room_id": (str(resolved_room_id) if resolved_room_id is not None else None),
        "room_name": resolved_room_name,
        "device_id": str(device_id),
        "resolve": rd,
        "execute": exec_res,
    }
    _remember_tool_call(
        "c4_light_set_by_name",
        {
            "device_name": device_name,
            "level": level,
            "state": state,
            "room_name": room_name,
            "room_id": room_id,
        },
        out,
    )
    return out


@Mcp.tool(
    name="c4_room_lights_set",
    description=(
        "Fast-path: set all lights in a room to a level (or on/off) in a single call. "
        "Optionally exclude/include by device name, ramp, and best-effort confirm each light."
    ),
)
def c4_room_lights_set_tool(
    room_id: int | None = None,
    room_name: str | None = None,
    level: int | None = None,
    state: str | None = None,
    on_level: int = 100,
    exclude_names: list[str] | None = None,
    include_names: list[str] | None = None,
    ramp_ms: int | None = None,
    confirm_timeout_s: float = 0.8,
    poll_interval_s: float = 0.2,
    concurrency: int = 3,
    require_unique: bool = True,
    include_candidates: bool = True,
    dry_run: bool = False,
) -> dict:
    if (room_id is None) == (room_name is None):
        return {"ok": False, "error": "provide exactly one of: room_id or room_name"}
    if (level is None) == (state is None):
        return {"ok": False, "error": "provide exactly one of: level or state"}

    resolved_room_id: int | None = None
    resolved_room_name: str | None = None
    if room_id is not None:
        try:
            resolved_room_id = int(room_id)
        except Exception:
            resolved_room_id = None
    else:
        rr = resolve_room(
            str(room_name),
            require_unique=bool(require_unique),
            include_candidates=bool(include_candidates),
        )
        if not isinstance(rr, dict) or not rr.get("ok"):
            return {"ok": False, "error": "could not resolve room", "details": rr}
        try:
            resolved_room_id = int(rr.get("room_id"))
        except Exception:
            resolved_room_id = None
        resolved_room_name = str(rr.get("name")) if rr.get("name") is not None else None

    if resolved_room_id is None:
        return {"ok": False, "error": "invalid room_id"}

    target_level: int
    if state is not None:
        state_norm = str(state or "").strip().lower()
        if state_norm not in {"on", "off"}:
            return {"ok": False, "error": "state must be 'on' or 'off'"}
        on_level = max(0, min(100, int(on_level)))
        target_level = on_level if state_norm == "on" else 0
    else:
        target_level = int(level)  # type: ignore[arg-type]
        if target_level < 0 or target_level > 100:
            return {"ok": False, "error": "level must be 0-100"}

    planned = {
        "room_id": int(resolved_room_id),
        "target_level": int(target_level),
        "exclude_names": list(exclude_names or []),
        "include_names": list(include_names or []),
        "ramp_ms": (int(ramp_ms) if ramp_ms is not None else None),
        "confirm_timeout_s": float(confirm_timeout_s),
        "poll_interval_s": float(poll_interval_s),
        "tolerance": 1,
        "concurrency": int(concurrency),
    }

    if bool(dry_run):
        preview = find_devices(search=None, category="lights", room_id=int(resolved_room_id), limit=200, include_raw=False)
        return {
            "ok": True,
            "room_id": str(resolved_room_id),
            "room_name": resolved_room_name,
            "preview": preview,
            "planned": planned,
            "dry_run": True,
        }

    exec_res = room_lights_set(
        int(resolved_room_id),
        int(target_level),
        exclude_names=list(exclude_names or []),
        include_names=list(include_names or []),
        ramp_ms=(int(ramp_ms) if ramp_ms is not None else None),
        confirm_timeout_s=float(confirm_timeout_s),
        poll_interval_s=float(poll_interval_s),
        tolerance=1,
        concurrency=int(concurrency),
        dry_run=False,
    )
    ok = bool(exec_res.get("ok")) if isinstance(exec_res, dict) else bool(exec_res)
    out = {
        "ok": ok,
        "room_id": str(resolved_room_id),
        "room_name": resolved_room_name,
        "planned": planned,
        "execute": exec_res,
    }
    _remember_tool_call(
        "c4_room_lights_set",
        {
            "room_id": room_id,
            "room_name": room_name,
            "level": level,
            "state": state,
            "exclude_names": exclude_names,
            "include_names": include_names,
            "ramp_ms": ramp_ms,
        },
        out,
    )
    return out


# ---- Locks ----


def _parse_lock_desired_locked(state: str | None) -> bool | None:
    s = str(state or "").strip().lower()
    if s in {"lock", "locked", "on", "true", "1", "yes"}:
        return True
    if s in {"unlock", "unlocked", "off", "false", "0", "no"}:
        return False
    return None

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


@Mcp.tool(
    name="c4_lock_set_by_name",
    description=(
        "Fast-path: resolve a lock by name (optionally scoped by room) and lock/unlock in one call. "
        "Returns accepted/confirmed semantics and includes resolution details."
    ),
)
def c4_lock_set_by_name_tool(
    lock_name: str,
    state: str,
    room_name: str | None = None,
    room_id: str | None = None,
    require_unique: bool = True,
    include_candidates: bool = True,
    dry_run: bool = False,
) -> dict:
    desired_locked = _parse_lock_desired_locked(state)
    if desired_locked is None:
        return {"ok": False, "error": "state must be lock/unlock (locked/unlocked)", "state": state}

    resolved_room_id: int | None = None
    resolved_room_name: str | None = None
    rr: dict | None = None

    if room_id is not None and str(room_id).strip():
        try:
            resolved_room_id = int(room_id)
        except Exception:
            resolved_room_id = None
    elif room_name is not None and str(room_name).strip():
        rr = resolve_room(
            str(room_name),
            require_unique=bool(require_unique),
            include_candidates=bool(include_candidates),
        )
        if not isinstance(rr, dict) or not rr.get("ok"):
            return {"ok": False, "error": "could not resolve room", "details": rr}
        try:
            resolved_room_id = int(rr.get("room_id"))
        except Exception:
            resolved_room_id = None
        resolved_room_name = str(rr.get("name")) if rr.get("name") is not None else None

    rd = resolve_device(
        str(lock_name or ""),
        category="locks",
        room_id=resolved_room_id,
        require_unique=bool(require_unique),
        include_candidates=bool(include_candidates),
    )
    if not isinstance(rd, dict) or not rd.get("ok"):
        return {"ok": False, "error": "could not resolve lock", "details": rd}

    device_id = rd.get("device_id")
    if device_id is None:
        return {"ok": False, "error": "resolve_device returned no device_id", "details": rd}

    if resolved_room_id is None and rd.get("room_id") is not None:
        try:
            resolved_room_id = int(rd.get("room_id"))
        except Exception:
            resolved_room_id = None
    if resolved_room_name is None and rd.get("room_name") is not None:
        resolved_room_name = str(rd.get("room_name"))

    planned = {
        "device_id": str(device_id),
        "lock_name": str(lock_name),
        "desired_locked": bool(desired_locked),
        "state": "locked" if desired_locked else "unlocked",
        "room_id": (str(resolved_room_id) if resolved_room_id is not None else None),
    }

    if bool(dry_run):
        return {
            "ok": True,
            "dry_run": True,
            "planned": planned,
            "resolve": rd,
            "resolve_room": rr,
            "room_id": planned["room_id"],
            "room_name": resolved_room_name,
        }

    try:
        if desired_locked:
            fut = _lock_pool.submit(lock_lock, int(device_id))
            result = fut.result(timeout=20)
        else:
            fut = _lock_pool.submit(lock_unlock, int(device_id))
            result = fut.result(timeout=20)

        if isinstance(result, dict):
            out = _augment_lock_result(result, desired_locked=bool(desired_locked))
            out["lock_name"] = str(lock_name)
            out["resolve"] = rd
            if rr is not None:
                out["resolve_room"] = rr
            out["room_id"] = planned["room_id"]
            out["room_name"] = resolved_room_name
            return out

        return {
            "ok": True,
            "result": result,
            "planned": planned,
            "resolve": rd,
            "resolve_room": rr,
            "room_id": planned["room_id"],
            "room_name": resolved_room_name,
        }
    except FutureTimeout:
        return {
            "ok": False,
            "device_id": int(device_id),
            "error": "tool timeout (20s)",
            "planned": planned,
            "resolve": rd,
            "resolve_room": rr,
            "room_id": planned["room_id"],
            "room_name": resolved_room_name,
        }
    except Exception as e:
        return {
            "ok": False,
            "device_id": int(device_id),
            "error": repr(e),
            "planned": planned,
            "resolve": rd,
            "resolve_room": rr,
            "room_id": planned["room_id"],
            "room_name": resolved_room_name,
        }


# ✅ In 0.6.1: mount without passing a registry object or Mcp() instance
mount_mcp(app, url_prefix="/mcp", middlewares=[mw_auth, mw_ratelimit, mw_cors])


def _patch_mcp_registry_name_collisions() -> None:
    """Avoid keyword collisions in flask-mcp-server's registry call helpers.

    flask-mcp-server's integrated HTTP handler calls:
      reg.call_tool(name, caller_roles=roles, **args)

    If a tool itself has an argument named 'name' (e.g., execute_by_name tools), Python raises:
      TypeError: call_tool() got multiple values for argument 'name'

    Fix: monkey-patch the *instance methods* on the default registry so they do not
    use a parameter named 'name' (or 'caller_roles') in their signature, then perform
    the same work internally.
    """

    reg = getattr(flask_mcp_server, "default_registry", None)
    if reg is None:
        return

    if getattr(reg, "_c4_name_collision_patch", False):
        return

    import types

    def call_tool_patched(self, tool_name: str, **kwargs):
        caller_roles = kwargs.pop("caller_roles", None)

        if tool_name not in self.tools:
            raise KeyError(f"Tool '{tool_name}' not found")

        item = self.tools[tool_name]
        if not self._permits(item.get("roles", []), caller_roles or []):
            raise PermissionError("Access forbidden: insufficient roles")

        ttl = item.get("ttl")
        if ttl:
            cache_key = self._cache_key("tool:" + tool_name, kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            result = item["callable"](**kwargs)
            self.cache.set(cache_key, result, ttl)
            return result

        return item["callable"](**kwargs)

    def get_resource_patched(self, resource_name: str, **kwargs):
        caller_roles = kwargs.pop("caller_roles", None)

        if resource_name not in self.resources:
            raise KeyError(f"Resource '{resource_name}' not found")

        item = self.resources[resource_name]
        if not self._permits(item.get("roles", []), caller_roles or []):
            raise PermissionError("Access forbidden: insufficient roles")

        ttl = item.get("ttl")
        if ttl:
            cache_key = self._cache_key("resource:" + resource_name, kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            result = item["getter"](**kwargs)
            self.cache.set(cache_key, result, ttl)
            return result

        return item["getter"](**kwargs)

    def get_prompt_patched(self, prompt_name: str, **kwargs):
        caller_roles = kwargs.pop("caller_roles", None)

        if prompt_name not in self.prompts:
            raise KeyError(f"Prompt '{prompt_name}' not found")

        item = self.prompts[prompt_name]
        if not self._permits(item.get("roles", []), caller_roles or []):
            raise PermissionError("Access forbidden: insufficient roles")

        return item["provider"](**kwargs)

    def complete_patched(self, completion_name: str, **kwargs):
        # caller_roles accepted for compatibility; currently not enforced by upstream.
        kwargs.pop("caller_roles", None)

        if completion_name not in self.completions:
            raise KeyError(f"Completion provider '{completion_name}' not found")

        return self.completions[completion_name](**kwargs)

    reg.call_tool = types.MethodType(call_tool_patched, reg)
    reg.get_resource = types.MethodType(get_resource_patched, reg)
    reg.get_prompt = types.MethodType(get_prompt_patched, reg)
    reg.complete = types.MethodType(complete_patched, reg)
    reg._c4_name_collision_patch = True


_patch_mcp_registry_name_collisions()


def main() -> None:
    app.run(host="127.0.0.1", port=3333, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()