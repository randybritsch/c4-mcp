"""Smoke-test the MCP server in STDIO mode.

Claude Desktop launches MCP servers over STDIO. This script spawns the server
subprocess and performs a minimal JSON-RPC exchange using flask-mcp-server's
STDIO protocol:
- mcp.list
- mcp.call (tool: ping)

Usage (PowerShell):
    ./.venv/Scripts/python.exe tools/validate_mcp_stdio.py

Exit code:
  0 = success
  1 = failure
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    python_exe = os.path.join(repo_root, ".venv", "Scripts", "python.exe")

    if not os.path.exists(python_exe):
        print(f"ERROR: venv python not found at {python_exe}", file=sys.stderr)
        return 1

    cmd = [
        python_exe,
        os.path.join(repo_root, "mcp_cli.py"),
        "serve-stdio",
        "--module",
        "app",
    ]

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "mcp.list", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "mcp.call",
            "params": {"kind": "tool", "name": "ping", "args": {}},
        },
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
        stdout, stderr = proc.communicate(input=input_text, timeout=15)

        responses: dict[int, dict[str, Any]] = {}
        for raw_line in (stdout or "").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                msg = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and isinstance(msg.get("id"), int):
                responses[msg["id"]] = msg

        resp1 = responses.get(1)
        resp2 = responses.get(2)

        if not resp1:
            raise RuntimeError(f"No response for mcp.list. stderr=\n{stderr}")
        if "error" in resp1:
            raise RuntimeError(f"mcp.list error: {resp1['error']}")

        tools = (resp1.get("result") or {}).get("tools") or {}
        if not isinstance(tools, dict) or not tools:
            raise RuntimeError("mcp.list returned no tools")
        if "ping" not in tools:
            raise RuntimeError("mcp.list did not include expected tool: ping")

        if not resp2:
            raise RuntimeError(f"No response for ping call. stderr=\n{stderr}")
        if "error" in resp2:
            raise RuntimeError(f"ping call error: {resp2['error']}")

        print("OK: STDIO server responded to mcp.list and ping")
        return 0

    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    finally:
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
