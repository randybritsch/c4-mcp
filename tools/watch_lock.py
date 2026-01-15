"""Watch lock-related items for state/variable changes.

Usage:
  python tools/watch_lock.py 2214 2215 --interval 0.5

Tip:
  Run this, then lock/unlock from the Control4 app to see which item/vars actually change.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Allow running as `python tools/watch_lock.py ...` by adding repo root to sys.path
REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from control4_adapter import item_get_variables, lock_get_state


def _extract_vars(vars_payload: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rows = vars_payload.get("variables")
    if not isinstance(rows, list):
        return out
    wanted = {n.lower() for n in names}
    for row in rows:
        if not isinstance(row, dict):
            continue
        var = str(row.get("varName") or "").strip()
        if var.lower() in wanted:
            out[var] = row.get("value")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("item_ids", nargs="+", type=int)
    p.add_argument("--interval", type=float, default=0.75)
    p.add_argument("--pretty", action="store_true")
    args = p.parse_args()

    watch_vars = [
        "LockStatus",
        "BatteryStatus",
        "LastActionDescription",
        "BATTERY_LEVEL",
        "LAST_UNLOCK_USER",
        "RelayState",
        "StateVerified",
        "LastActionTime",
    ]

    last: Dict[int, Dict[str, Any]] = {}
    print(f"Watching {args.item_ids} every {args.interval:.2f}s. Ctrl+C to stop.")

    try:
        while True:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            for item_id in args.item_ids:
                state = lock_get_state(item_id)
                vars_payload = item_get_variables(item_id)
                vars_sel = _extract_vars(vars_payload if isinstance(vars_payload, dict) else {}, watch_vars)

                snapshot = {
                    "state": state,
                    "vars": vars_sel,
                }

                if last.get(item_id) != snapshot:
                    last[item_id] = snapshot
                    print(f"\n[{now}] item {item_id}")
                    if args.pretty:
                        print(json.dumps(snapshot, indent=2))
                    else:
                        locked = state.get("locked") if isinstance(state, dict) else None
                        src = state.get("source") if isinstance(state, dict) else None
                        print(f"  locked={locked} source={src}")
                        for k, v in vars_sel.items():
                            print(f"  {k}={v}")

            time.sleep(max(args.interval, 0.1))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
