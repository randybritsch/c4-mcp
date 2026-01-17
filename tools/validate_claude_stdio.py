"""Validate the Claude Desktop MCP method surface over STDIO.

This simulates what Claude Desktop does (initialize -> tools/list -> tools/call).

Usage:
    ./.venv/Scripts/python.exe tools/validate_claude_stdio.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_exe = os.path.join(repo_root, ".venv", "Scripts", "python.exe")
    server_py = os.path.join(repo_root, "claude_stdio_server.py")

    # Run unbuffered to better match Claude Desktop behavior on Windows.
    cmd = [python_exe, "-u", server_py]
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "validator"}},
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "ping", "arguments": {}}},
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "c4_find_rooms", "arguments": {"search": "basement", "limit": 10}},
        },
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "c4_find_devices", "arguments": {"category": "lights", "limit": 50}},
        },
    ]
    input_text = "\n".join(json.dumps(r) for r in requests) + "\n"

    env = os.environ.copy()
    # Ensure tests don't fail due to inherited tool filtering.
    # This validator is specifically validating the Claude method surface, not tool-filter policy.
    env["C4_STDIO_TOOL_MODE"] = "all"
    env.pop("C4_STDIO_TOOL_ALLOWLIST", None)
    env.pop("C4_STDIO_TOOL_DENYLIST", None)

    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=35)
        responses = {}
        for line in (stdout or "").splitlines():
            try:
                msg = json.loads(line)
            except Exception:
                continue
            if isinstance(msg, dict) and "id" in msg:
                responses[msg["id"]] = msg

        if 1 not in responses or "error" in responses[1]:
            raise RuntimeError(f"initialize failed: {responses.get(1)}\n{stderr}")
        if 2 not in responses or "error" in responses[2]:
            raise RuntimeError(f"tools/list failed: {responses.get(2)}\n{stderr}")

        tools = (responses[2].get("result") or {}).get("tools") or []
        if not any(t.get("name") == "ping" for t in tools if isinstance(t, dict)):
            raise RuntimeError("tools/list did not include ping")
        if not any(t.get("name") == "c4_memory_get" for t in tools if isinstance(t, dict)):
            raise RuntimeError("tools/list did not include c4_memory_get (session memory)")

        if 3 not in responses or "error" in responses[3]:
            raise RuntimeError(f"tools/call failed: {responses.get(3)}\n{stderr}")

        # Smoke-test real inventory calls. Claude often uses these first.
        for rid in (4, 5):
            if rid not in responses:
                raise RuntimeError(f"missing response id={rid}\n{stderr}")
            if "error" in responses[rid]:
                raise RuntimeError(f"tools/call id={rid} failed: {responses.get(rid)}\n{stderr}")

            # Claude-style tools/call responses are wrapped as { result: { content:[{text:...}], isError: bool } }
            wrapped = (responses[rid].get("result") or {})
            if wrapped.get("isError") is True:
                content = wrapped.get("content") or []
                text = None
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    text = content[0].get("text")
                raise RuntimeError(f"tools/call id={rid} returned isError=true: {text or wrapped}\n{stderr}")

        print("OK: Claude-style initialize/tools/list/tools/call works")
        return 0

    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
