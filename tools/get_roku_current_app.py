"""Print CURRENT_APP/CURRENT_APP_ID for the basement Roku player item.

Usage:
  python tools/get_roku_current_app.py
"""

from __future__ import annotations

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
    ROKU_PLAYER_ID = 2074
    v = item_get_variables(ROKU_PLAYER_ID)
    variables = v.get("variables", []) if isinstance(v, dict) else []
    print("CURRENT_APP:", _get_var(variables, "CURRENT_APP"))
    print("CURRENT_APP_ID:", _get_var(variables, "CURRENT_APP_ID"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
