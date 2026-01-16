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

# Allow running this script directly from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from control4_adapter import get_all_items, item_get_commands  # noqa: E402


def _j(obj) -> str:
    return json.dumps(obj, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--show-commands", action="store_true")
    args = ap.parse_args()

    limit = max(1, min(2000, int(args.limit)))

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
