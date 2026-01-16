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


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from control4_adapter import item_get_commands
from control4_adapter import shade_get_state, shade_list


def _j(obj) -> str:
    return json.dumps(obj, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

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
