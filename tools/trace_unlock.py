"""Trace lock unlock behavior by snapshotting and polling key variables.

Usage:
  python tools/trace_unlock.py --target 2214 --also 2215 --poll 0.5 --seconds 15

This will:
  1) Print a snapshot for each item
  2) Send unlock to --target
  3) Poll each item for changes and print deltas
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from control4_adapter import item_get_variables, lock_get_state, lock_unlock


WATCH_VARS = [
    "LockStatus",
    "BatteryStatus",
    "LastActionDescription",
    "BATTERY_LEVEL",
    "LAST_UNLOCK_USER",
    "RelayState",
    "StateVerified",
    "LastActionTime",
]


def _vars_map(device_id: int) -> Dict[str, Any]:
    payload = item_get_variables(device_id)
    rows = payload.get("variables") if isinstance(payload, dict) else None
    out: Dict[str, Any] = {}
    if not isinstance(rows, list):
        return out
    wanted = {v.lower() for v in WATCH_VARS}
    for r in rows:
        if not isinstance(r, dict):
            continue
        k = str(r.get("varName") or "").strip()
        if k.lower() in wanted:
            out[k] = r.get("value")
    return out


def snapshot(device_id: int) -> Dict[str, Any]:
    return {
        "device_id": device_id,
        "state": lock_get_state(device_id),
        "vars": _vars_map(device_id),
    }


def _print_snapshot(title: str, snap: Dict[str, Any]) -> None:
    print(f"\n=== {title} item {snap.get('device_id')} ===")
    print(json.dumps(snap, indent=2))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, required=True, help="Device id to send unlock to")
    p.add_argument("--also", type=int, nargs="*", default=[], help="Other device ids to watch")
    p.add_argument("--poll", type=float, default=0.5)
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--no-send", action="store_true", help="Do not send unlock; only watch/poll for changes")
    args = p.parse_args()

    watch_ids: Iterable[int] = [args.target, *args.also]

    before = {i: snapshot(i) for i in watch_ids}
    for i, s in before.items():
        _print_snapshot("BEFORE", s)

    if not args.no_send:
        print(f"\nSending unlock to {args.target}...")
        unlock_result = lock_unlock(args.target)
        print("unlock_result:")
        print(json.dumps(unlock_result, indent=2))
    else:
        print(f"\nNot sending unlock (watch-only mode). Use the Control4 app now.")

    last = {i: before[i] for i in before}
    start = time.time()
    deadline = start + float(args.seconds)
    print(
        f"\nPolling every {float(args.poll):.2f}s for {float(args.seconds):.1f}s "
        f"(until {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(deadline))})..."
    )
    while time.time() < deadline:
        time.sleep(max(float(args.poll), 0.1))
        for i in watch_ids:
            now = snapshot(i)
            if now != last[i]:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{ts}] CHANGE item {i}")
                print(json.dumps(now, indent=2))
                last[i] = now

    elapsed = time.time() - start
    print(f"\nDone. Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
