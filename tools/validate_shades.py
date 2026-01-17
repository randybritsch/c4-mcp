r"""Validate shade discovery/state (best-effort).

Usage:
    \.venv\Scripts\python.exe tools\validate_shades.py
    \.venv\Scripts\python.exe tools\validate_shades.py --limit 20

This script is read-only. It helps you:
- Confirm the server can discover shade/blind-like devices
- Inspect their command surfaces and current position variables

If your project currently has no shades, this will simply report 0 candidates.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from control4_adapter import item_get_commands
from control4_adapter import shade_get_state, shade_list


Json = Dict[str, Any]


def _http_post_json(url: str, body: Json, timeout_s: float) -> Json:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        raw = resp.read()
        payload = json.loads(raw.decode("utf-8")) if raw else {}
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected response payload: {payload!r}")
        return payload


def _mcp_call(base_url: str, tool_name: str, args: Optional[Json], timeout_s: float) -> Json:
    url = base_url.rstrip("/") + "/mcp/call"
    try:
        return _http_post_json(url, {"kind": "tool", "name": tool_name, "args": args or {}}, timeout_s=timeout_s)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else None
        except Exception:
            payload = {"_raw": raw.decode("utf-8", errors="replace")}
        raise RuntimeError(f"HTTP {e.code} calling {tool_name}: {payload}") from e


def _j(obj) -> str:
    return json.dumps(obj, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-url",
        default=None,
        help="If provided, validate via MCP HTTP (/mcp/call) instead of calling the gateway directly.",
    )
    ap.add_argument("--timeout-s", type=float, default=25.0)
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    if args.base_url:
        payload = _mcp_call(str(args.base_url), "c4_shade_list", {"limit": int(args.limit)}, timeout_s=float(args.timeout_s))
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or not result.get("ok"):
            print("Unexpected c4_shade_list response:")
            print(_j(payload)[:4000])
            return 2
        shades = result.get("shades")
        if not isinstance(shades, list):
            print("Unexpected c4_shade_list.shades:")
            print(_j(payload)[:4000])
            return 2

        print(f"Found {len(shades)} shade candidates")
        if not shades:
            return 0

        for s in shades:
            if not isinstance(s, dict):
                continue
            did = s.get("id")
            name = s.get("name")
            print("\n===", did, name, "(room:", s.get("roomName"), ")")

            try:
                state_payload = _mcp_call(str(args.base_url), "c4_shade_get_state", {"device_id": str(did)}, timeout_s=float(args.timeout_s))
                state = state_payload.get("result") if isinstance(state_payload, dict) else state_payload
            except Exception as e:
                state = {"ok": False, "error": repr(e)}
            print("state:")
            print(_j(state)[:2000])

            try:
                cmds_payload = _mcp_call(str(args.base_url), "c4_item_commands", {"device_id": str(did)}, timeout_s=float(args.timeout_s))
                cmds = cmds_payload.get("result") if isinstance(cmds_payload, dict) else cmds_payload
            except Exception as e:
                cmds = {"ok": False, "error": repr(e)}
            print("commands (truncated):")
            print(_j(cmds)[:2500])

        return 0

    # Legacy/local mode: direct gateway access
    listing = shade_list(limit=int(args.limit))
    shades = listing.get("shades") if isinstance(listing, dict) else None
    if not isinstance(shades, list):
        print("Unexpected shade_list response:")
        print(_j(listing)[:2000])
        return 2

    print(f"Found {len(shades)} shade candidates")
    if not shades:
        return 0

    for s in shades:
        if not isinstance(s, dict):
            continue
        did = s.get("id")
        name = s.get("name")
        print("\n===", did, name, "(room:", s.get("roomName"), ")")

        try:
            state = shade_get_state(int(did))
        except Exception as e:
            state = {"ok": False, "error": repr(e)}
        print("state:")
        print(_j(state)[:2000])

        try:
            cmds = item_get_commands(int(did))
        except Exception as e:
            cmds = {"ok": False, "error": repr(e)}
        print("commands (truncated):")
        print(_j(cmds)[:2500])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
