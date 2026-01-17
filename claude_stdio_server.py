r"""Claude Desktop STDIO MCP server shim.

Why: Claude Desktop speaks the official MCP JSON-RPC method names like:
- initialize
- tools/list
- tools/call

The vendored flask-mcp-server STDIO implementation in this repo speaks a
different method surface (mcp.list / mcp.call). This shim adapts Claude's
protocol to the existing tool registry built in app.py.

Run (for Claude):
    .venv\\Scripts\\python.exe claude_stdio_server.py

Notes:
- All logs go to stderr (stdout is reserved for JSON-RPC responses).
- Tools are sourced from flask_mcp_server.registry.default_registry.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, Optional
import uuid


def _debug_enabled() -> bool:
    v = os.getenv("C4_STDIO_DEBUG")
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _dprint(msg: str) -> None:
    if _debug_enabled():
        _eprint(msg)


def _json_dumps(obj: Any) -> str:
    # Keep stdout strictly ASCII-safe JSON so Windows host encodings
    # (cp1252/etc) can't break JSON-RPC with UnicodeEncodeError.
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), default=str)


def _parse_csv_env(name: str) -> list[str]:
    v = os.getenv(name)
    if not v:
        return []
    parts = [p.strip() for p in v.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _tool_mode() -> str:
    # all: expose every tool (large tools/list)
    # compact: expose a curated high-level set (best for Claude Desktop stability)
    v = (os.getenv("C4_STDIO_TOOL_MODE") or "all").strip().lower()
    return v if v in {"all", "compact"} else "all"


def _compact_allowlist() -> set[str]:
    # Curated, high-level tools intended to cover common usage without sending
    # a massive schema payload in tools/list.
    return {
        "ping",
        "c4_server_info",
        "c4_find_rooms",
        "c4_find_devices",
        "c4_resolve",
        "c4_resolve_room",
        "c4_resolve_device",
        "c4_list_rooms",
        "c4_list_devices",
        "c4_light_set_by_name",
        "c4_room_lights_set",
        "c4_lights_get_last",
        "c4_lights_set_last",
        "c4_lock_set_by_name",
        "c4_scene_activate_by_name",
        "c4_scene_set_state_by_name",
        "c4_macro_execute_by_name",
        "c4_announcement_execute_by_name",
        "c4_media_watch_launch_app_by_name",
        "c4_room_list_commands",
        "c4_room_send_command",
        "c4_tv_watch_by_name",
        "c4_tv_remote",
        "c4_thermostat_set_target_f",
        "c4_shade_set_position",
        "c4_alarm_set_mode",
    }


def _select_tools(all_tools: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    # Optional explicit allow/deny lists.
    allow = set(_parse_csv_env("C4_STDIO_TOOL_ALLOWLIST"))
    deny = set(_parse_csv_env("C4_STDIO_TOOL_DENYLIST"))

    if not allow and _tool_mode() == "compact":
        allow = _compact_allowlist()

    items: list[tuple[str, dict[str, Any]]] = sorted(all_tools.items())
    if allow:
        items = [(name, t) for (name, t) in items if name in allow]
    if deny:
        items = [(name, t) for (name, t) in items if name not in deny]

    return [_tool_to_mcp(t) for _, t in items]


def _send_result(request_id: Any, result: Any) -> None:
    resp = {"jsonrpc": "2.0", "id": request_id, "result": result}
    sys.stdout.write(_json_dumps(resp) + "\n")
    sys.stdout.flush()


def _send_error(request_id: Any, code: int, message: str, data: Optional[dict[str, Any]] = None) -> None:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    resp = {"jsonrpc": "2.0", "id": request_id, "error": err}
    sys.stdout.write(_json_dumps(resp) + "\n")
    sys.stdout.flush()


def _tool_to_mcp(tool: Dict[str, Any]) -> dict[str, Any]:
    description = (tool.get("description") or "").strip()
    if description:
        # Keep tools/list responses reasonably small; Claude does not need full docstrings here.
        description = description.replace("\r\n", "\n")
        if "\n\n" in description:
            description = description.split("\n\n", 1)[0].strip()
        if len(description) > 320:
            description = description[:317].rstrip() + "..."

    mcp_tool: dict[str, Any] = {
        "name": tool.get("name"),
        "description": description,
        "inputSchema": tool.get("input_schema") or {"type": "object", "properties": {}},
    }

    # Newer MCP clients (including Claude Desktop) may expect outputSchema.
    output_schema = tool.get("output_schema")
    if isinstance(output_schema, dict) and output_schema:
        mcp_tool["outputSchema"] = output_schema

    return mcp_tool


def _wrap_tool_result(value: Any) -> dict[str, Any]:
    # Claude expects a content array. Keep it simple and return JSON as text.
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return {
        "content": [{"type": "text", "text": text}],
        "isError": False,
    }


def _wrap_tool_error(message: str, data: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    if data is not None:
        try:
            details = json.dumps(data, ensure_ascii=False, indent=2, default=str)
            message = f"{message}\n\n{details}"
        except Exception:
            pass
    return {
        "content": [{"type": "text", "text": message}],
        "isError": True,
    }


def _reset_session_id() -> str:
    sid = str(uuid.uuid4())
    os.environ["C4_SESSION_ID"] = sid
    return sid


def main() -> int:
    # Import app.py for side effects: it registers all @Mcp.tool functions.
    # This must happen before accessing default_registry.
    import app  # noqa: F401

    # Pull guardrail helpers from app.py so STDIO is also safe-by-default.
    from app import _is_write_tool, _write_allowed, _write_guardrails_enabled, _writes_enabled

    from flask_mcp_server.registry import default_registry

    server_info = {
        "name": "c4-mcp",
        "version": os.getenv("C4_VERSION", "dev"),
    }

    # Best-effort: if stdout is text-mode and reconfigurable, prefer UTF-8.
    # (We still emit ASCII-safe JSON via ensure_ascii=True.)
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    _eprint("Claude STDIO shim starting (c4-mcp)")
    _eprint(f"Debug enabled: {_debug_enabled()}")
    try:
        _eprint(f"stdout encoding: {getattr(sys.stdout, 'encoding', None)}")
    except Exception:
        pass
    try:
        _eprint(f"argv: {sys.argv}")
        _eprint(f"cwd: {os.getcwd()}")
        _eprint(f"tool_mode: {_tool_mode()}")
        _eprint(f"allowlist size: {len(_parse_csv_env('C4_STDIO_TOOL_ALLOWLIST'))}")
        _eprint(f"denylist size: {len(_parse_csv_env('C4_STDIO_TOOL_DENYLIST'))}")
    except Exception:
        pass
    _eprint(f"Registered tools: {len(default_registry.tools)}")

    session_id = _reset_session_id()
    _eprint(f"Session id: {session_id}")

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        request_id: Any = None
        try:
            req = json.loads(raw_line)
            request_id = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}

            _dprint(f"recv id={request_id} method={method}")

            # JSON-RPC notifications: no id => no response.
            if request_id is None:
                continue

            if method == "initialize":
                # Treat initialize as the start of a new client session.
                session_id = _reset_session_id()
                # Be permissive; echo back a protocol version and minimal capabilities.
                _send_result(
                    request_id,
                    {
                        "protocolVersion": params.get("protocolVersion") or "2024-11-05",
                        "capabilities": {
                            "experimental": {},
                            "tools": {"listChanged": False},
                            "resources": {"subscribe": False, "listChanged": False},
                            "prompts": {"listChanged": False},
                        },
                        "serverInfo": {**server_info, "session_id": session_id},
                    },
                )
                _dprint(f"sent initialize result id={request_id}")
                continue

            if method in ("notifications/initialized", "initialized"):
                # Some clients may send this as a request; acknowledge.
                _send_result(request_id, {})
                continue

            if method == "tools/list":
                tools = _select_tools(default_registry.tools)
                _send_result(request_id, {"tools": tools})
                _dprint(f"sent tools/list result id={request_id} tools={len(tools)}")
                continue

            if method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if not name:
                    _send_error(request_id, -32602, "Missing params.name")
                    continue
                if not isinstance(arguments, dict):
                    _send_error(request_id, -32602, "params.arguments must be an object")
                    continue

                try:
                    # Respect write guardrails in STDIO (HTTP middleware does not apply here).
                    if _write_guardrails_enabled() and _is_write_tool(str(name)):
                        if not _writes_enabled():
                            _send_result(
                                request_id,
                                _wrap_tool_error(
                                    "Writes are disabled (set C4_WRITES_ENABLED=true or disable C4_WRITE_GUARDRAILS).",
                                    {"tool": str(name)},
                                ),
                            )
                            continue
                        allowed, why = _write_allowed(str(name))
                        if not allowed:
                            _send_result(request_id, _wrap_tool_error(f"Write not allowed: {why}", {"tool": str(name)}))
                            continue

                    value = default_registry.call_tool(str(name), **arguments)
                    _send_result(request_id, _wrap_tool_result(value))
                except Exception as e:
                    # concurrent.futures.TimeoutError often has an empty message.
                    msg = str(e).strip()
                    if not msg:
                        msg = f"{e.__class__.__name__}"
                    _eprint(f"Tool error in {name}: {msg}")
                    _eprint(traceback.format_exc())
                    _send_result(request_id, _wrap_tool_error(f"Tool error: {msg}"))
                continue

            if method == "resources/list":
                _send_result(request_id, {"resources": []})
                _dprint(f"sent resources/list result id={request_id}")
                continue

            if method == "prompts/list":
                _send_result(request_id, {"prompts": []})
                _dprint(f"sent prompts/list result id={request_id}")
                continue

            # Unknown method
            _send_error(request_id, -32601, f"Method not found: {method}")

        except json.JSONDecodeError as e:
            if request_id is not None:
                _send_error(request_id, -32700, f"Invalid JSON: {e}")
        except Exception as e:
            _eprint(f"Unhandled error: {e}")
            _eprint(traceback.format_exc())
            if request_id is not None:
                _send_error(request_id, -32603, str(e))

    _eprint("Claude STDIO shim exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
