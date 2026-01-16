"""Test: switch a TV room to a Roku input and launch an app.

Usage:
    python tools/test_paramount_basement.py --room-id <ROOM_ID> --device-id <ROKU_DEVICE_ID> --player-id <ROKU_PLAYER_ID>

Optional env vars:
    - C4_ROOM_ID
    - C4_ROKU_DEVICE_ID
    - C4_ROKU_PLAYER_ID

This is a generic smoke test for the watch+launch flow.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control4_adapter import item_get_variables, media_watch_launch_app


def _get_var(variables: list[dict], name: str):
    target = name.strip().upper()
    for row in variables:
        if isinstance(row, dict) and str(row.get("varName", "")).strip().upper() == target:
            return row.get("value")
    return None


def main() -> int:
    import os

    ap = argparse.ArgumentParser()
    ap.add_argument("--room-id", type=int, default=int(os.environ.get("C4_ROOM_ID") or 0) or 0)
    ap.add_argument("--device-id", type=int, default=int(os.environ.get("C4_ROKU_DEVICE_ID") or 0) or 0)
    ap.add_argument("--player-id", type=int, default=int(os.environ.get("C4_ROKU_PLAYER_ID") or 0) or 0)
    ap.add_argument("--app", type=str, default="Paramount+")
    ap.add_argument("--pre-home", action="store_true", default=True)
    args = ap.parse_args()

    missing = [k for k, v in {"--room-id": args.room_id, "--device-id": args.device_id, "--player-id": args.player_id}.items() if not v]
    if missing:
        print("ERROR: Missing required ids: " + ", ".join(missing), file=sys.stderr)
        return 2

    print(f"Launching {args.app!r} with Watch flow (room_id={args.room_id}, device_id={args.device_id})...")
    result = media_watch_launch_app(int(args.device_id), str(args.app), room_id=int(args.room_id), pre_home=bool(args.pre_home))
    print(json.dumps(result, indent=2)[:4000])

    time.sleep(1.5)

    v = item_get_variables(int(args.player_id))
    variables = v.get("variables", []) if isinstance(v, dict) else []
    print("Roku Player CURRENT_APP:", _get_var(variables, "CURRENT_APP"))
    print("Roku Player CURRENT_APP_ID:", _get_var(variables, "CURRENT_APP_ID"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
