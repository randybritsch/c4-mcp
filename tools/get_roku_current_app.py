"""Print CURRENT_APP/CURRENT_APP_ID for a Roku player item.

Usage:
    python tools/get_roku_current_app.py --device-id <ROKU_PLAYER_ITEM_ID>

You can also set C4_ROKU_PLAYER_ID instead of passing --device-id.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control4_adapter import item_get_variables


def _get_var(variables: list[dict], name: str):
    target = name.strip().upper()
    for row in variables:
        if isinstance(row, dict) and str(row.get("varName", "")).strip().upper() == target:
            return row.get("value")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device-id",
        type=int,
        default=int((__import__("os").environ.get("C4_ROKU_PLAYER_ID") or "0") or 0),
        help="Roku player item id (env: C4_ROKU_PLAYER_ID)",
    )
    args = ap.parse_args()

    if not args.device_id:
        print("ERROR: Provide --device-id or set C4_ROKU_PLAYER_ID", file=sys.stderr)
        return 2

    v = item_get_variables(int(args.device_id))
    variables = v.get("variables", []) if isinstance(v, dict) else []
    print("CURRENT_APP:", _get_var(variables, "CURRENT_APP"))
    print("CURRENT_APP_ID:", _get_var(variables, "CURRENT_APP_ID"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
