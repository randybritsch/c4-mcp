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

    cmd = [python_exe, server_py]
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "validator"}},
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "ping", "arguments": {}}},
    ]
    input_text = "\n".join(json.dumps(r) for r in requests) + "\n"

    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=20)
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

        print("OK: Claude-style initialize/tools/list/tools/call works")
        return 0

    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
