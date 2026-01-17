r"""Claude Desktop STDIO MCP server shim.

Why: Claude Desktop speaks the official MCP JSON-RPC method names like:
- initialize
- tools/list
- tools/call

The vendored flask-mcp-server STDIO implementation in this repo speaks a
different method surface (mcp.list / mcp.call). This shim adapts Claude's
protocol to the existing tool registry built in app.py.

Run (for Claude):
    .\.venv\Scripts\python.exe claude_stdio_server.py

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


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


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
    return {
        "name": tool.get("name"),
        "description": tool.get("description") or "",
        "inputSchema": tool.get("input_schema") or {"type": "object", "properties": {}},
    }


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

    _eprint("Claude STDIO shim starting (c4-mcp)")
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
                            "tools": {},
                            "resources": {},
                            "prompts": {},
                        },
                        "serverInfo": {**server_info, "session_id": session_id},
                    },
                )
                continue

            if method in ("notifications/initialized", "initialized"):
                # Some clients may send this as a request; acknowledge.
                _send_result(request_id, {})
                continue

            if method == "tools/list":
                tools = [_tool_to_mcp(t) for _, t in sorted(default_registry.tools.items())]
                _send_result(request_id, {"tools": tools})
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
                continue

            if method == "prompts/list":
                _send_result(request_id, {"prompts": []})
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
