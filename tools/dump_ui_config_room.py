"""Dump Control4 UI configuration for a specific room.

Usage:
    python tools/dump_ui_config_room.py --room-id <ROOM_ID>

This is a diagnostic utility; it prints a trimmed JSON view of the matching room node.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path when running from tools/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control4_gateway import Control4Gateway


def _find_room(obj, room_id: int):
    if isinstance(obj, dict):
        # common shapes
        if obj.get("room_id") == room_id or obj.get("roomId") == room_id:
            return obj
        if obj.get("id") == room_id and obj.get("type") in {"room", "ROOM"}:
            return obj
        for v in obj.values():
            r = _find_room(v, room_id)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for it in obj:
            r = _find_room(it, room_id)
            if r is not None:
                return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room-id", type=int, required=True)
    ap.add_argument("--max-chars", type=int, default=12000)
    args = ap.parse_args()

    g = Control4Gateway()
    res = g._loop_thread.run(g._director_http_get("/api/v1/agents/ui_configuration"), timeout_s=25)
    print("ok", res.get("ok"), "status", res.get("status"), "url", res.get("url"))
    if not res.get("ok"):
        print((res.get("text") or "")[: args.max_chars])
        return 1

    payload = res.get("json")
    room = _find_room(payload, int(args.room_id))
    print("found_room", bool(room))
    if room is None:
        # quick hint
        if isinstance(payload, dict):
            print("top_keys", list(payload.keys())[:50])
        return 2

    out = json.dumps(room, indent=2)
    print(out[: args.max_chars])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
