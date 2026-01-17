r"""Validate scene discovery (UI Buttons) and show command surfaces (read-only).

Usage:
  \.venv\Scripts\python.exe tools\validate_scenes.py
  \.venv\Scripts\python.exe tools\validate_scenes.py --limit 25
  \.venv\Scripts\python.exe tools\validate_scenes.py --limit 10 --show-commands

Notes:
- Control4 lighting scenes are not standardized across projects.
- This project treats UI Button devices (proxy/control='uibutton') as a best-effort proxy for scenes.
- This script is read-only (it does not execute commands).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

# Allow running this script directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from control4_adapter import get_all_items, item_get_commands  # noqa: E402


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
    ap.add_argument("--show-commands", action="store_true")
    args = ap.parse_args()

    limit = max(1, min(2000, int(args.limit)))

    if args.base_url:
        payload = _mcp_call(str(args.base_url), "c4_scene_list", {}, timeout_s=float(args.timeout_s))
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict) or not result.get("ok"):
            print("Unexpected c4_scene_list response:")
            print(_j(payload)[:4000])
            return 2

        scenes = result.get("uibuttons")
        if not isinstance(scenes, list):
            print("Unexpected c4_scene_list.uibuttons:")
            print(_j(payload)[:4000])
            return 2

        scenes = scenes[:limit]
        print(f"Found {len(scenes)} scene candidates")
        for s in scenes:
            if not isinstance(s, dict):
                continue
            print(f"- [{s.get('device_id')}] {s.get('room_name') or ''} :: {s.get('name')}")

        if args.show_commands and scenes:
            print("\n--- Commands (first 10) ---")
            for s in scenes[:10]:
                did = s.get("device_id")
                if did is None:
                    continue
                cmds_payload = _mcp_call(str(args.base_url), "c4_item_commands", {"device_id": str(did)}, timeout_s=float(args.timeout_s))
                cmds = cmds_payload.get("result") if isinstance(cmds_payload, dict) else None
                print(f"\n=== device_id={did} name={s.get('name')} ===")
                print(_j(cmds)[:4000])
        return 0

    # Legacy/local mode: direct gateway access
    items = get_all_items()
    rooms_by_id = {
        str(i.get("id")): i.get("name")
        for i in items
        if isinstance(i, dict) and i.get("typeName") == "room" and i.get("id") is not None
    }

    scenes: list[dict] = []
    for i in items:
        if not isinstance(i, dict) or i.get("typeName") != "device":
            continue

        proxy_l = str(i.get("proxy") or "").lower()
        control_l = str(i.get("control") or "").lower()
        name_l = str(i.get("name") or "").lower()

        if not (
            proxy_l in {"uibutton", "voice-scene"}
            or control_l in {"uibutton", "voice-scene"}
            or "scene" in name_l
        ):
            continue

        room_id = i.get("roomId") or i.get("parentId")
        room_name = i.get("roomName") or (rooms_by_id.get(str(room_id)) if room_id is not None else None)

        scenes.append(
            {
                "device_id": str(i.get("id")),
                "name": i.get("name"),
                "room_id": (str(room_id) if room_id is not None else None),
                "room_name": room_name,
                "control": i.get("control"),
                "proxy": i.get("proxy"),
            }
        )

    scenes.sort(key=lambda s: ((s.get("room_name") or ""), (s.get("name") or "")))
    scenes = scenes[:limit]

    print(f"Found {len(scenes)} scene candidates")
    for s in scenes:
        print(f"- [{s.get('device_id')}] {s.get('room_name') or ''} :: {s.get('name')}")

    if args.show_commands and scenes:
        print("\n--- Commands (first 10) ---")
        for s in scenes[:10]:
            did = int(s["device_id"])
            cmds = item_get_commands(did)
            print(f"\n=== device_id={did} name={s.get('name')} ===")
            print(
                _j(
                    {
                        "ok": cmds.get("ok"),
                        "count": len(cmds.get("commands") or []),
                        "commands": cmds.get("commands"),
                    }
                )[:4000]
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
