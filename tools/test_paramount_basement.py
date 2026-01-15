"""Test: switch TV Room to Basement Roku input and launch Paramount+.

Usage:
  python tools/test_paramount_basement.py

This is a one-off smoke test for the 'Turn on Paramount+ in the basement' flow.
"""

from __future__ import annotations

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
    ROOM_ID = 456
    ROKU_DEVICE_ID = 2076
    ROKU_PLAYER_ID = 2074

    print("Launching Paramount+ on Basement Roku with Watch flow...")
    result = media_watch_launch_app(ROKU_DEVICE_ID, "Paramount+", room_id=ROOM_ID, pre_home=True)
    print(json.dumps(result, indent=2)[:4000])

    time.sleep(1.5)

    v = item_get_variables(ROKU_PLAYER_ID)
    variables = v.get("variables", []) if isinstance(v, dict) else []
    print("Roku Player CURRENT_APP:", _get_var(variables, "CURRENT_APP"))
    print("Roku Player CURRENT_APP_ID:", _get_var(variables, "CURRENT_APP_ID"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
