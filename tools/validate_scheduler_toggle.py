from __future__ import annotations

import argparse
from datetime import datetime
import os
import sys
from typing import Any

# Ensure repo root is importable when running as a script from tools/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from control4_adapter import scheduler_get, scheduler_list, scheduler_set_enabled


def _minutes_until_next_occurrence(event_row: dict[str, Any], now: datetime) -> float | None:
    no = event_row.get("next_occurrence")
    if not isinstance(no, dict):
        return None

    try:
        dt = datetime(
            int(no["year"]),
            int(no["month"]),
            int(no["day"]),
            int(no.get("hour") or 0),
            int(no.get("min") or 0),
        )
    except Exception:
        return None

    return (dt - now).total_seconds() / 60.0


def _pick_safe_event(events: list[dict[str, Any]], min_minutes: float) -> dict[str, Any] | None:
    now = datetime.now()
    candidates: list[tuple[float | None, dict[str, Any]]] = []

    for e in events:
        if not isinstance(e, dict):
            continue

        if e.get("locked") is True:
            continue
        if e.get("hidden") is True:
            continue
        if e.get("user_hidden") is True:
            continue

        mu = _minutes_until_next_occurrence(e, now)
        if mu is not None and mu >= 0 and mu < float(min_minutes):
            continue

        candidates.append((mu, e))

    if not candidates:
        return None

    candidates.sort(key=lambda t: (t[0] is None, t[0] if t[0] is not None else 1e18))
    return candidates[0][1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate scheduler enable/disable end-to-end by toggling a 'safe' event and restoring it. "
            "Default behavior is dry-run (no changes)."
        )
    )
    parser.add_argument("--event-id", type=int, default=None, help="Specific eventId to target.")
    parser.add_argument("--search", type=str, default=None, help="Search filter to narrow events.")
    parser.add_argument(
        "--min-minutes",
        type=float,
        default=10.0,
        help="Avoid events with next occurrence within this many minutes.",
    )
    parser.add_argument(
        "--doit",
        action="store_true",
        help="Actually perform the toggle + restore. Without this flag, only shows the planned writes.",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Do not restore the original enabled state (NOT recommended).",
    )

    args = parser.parse_args()

    lst = scheduler_list(args.search)
    events = lst.get("events") if isinstance(lst, dict) else None
    if not isinstance(events, list):
        print("scheduler_list did not return an events list")
        return 2

    events_norm = [e for e in events if isinstance(e, dict)]

    if args.event_id is not None:
        selected = next((e for e in events_norm if int(e.get("eventId") or 0) == int(args.event_id)), None)
        if selected is None:
            print(f"eventId {args.event_id} not found in scheduler_list results")
            return 2
    else:
        selected = _pick_safe_event(events_norm, float(args.min_minutes))
        if selected is None:
            print("No safe candidate event found (try lowering --min-minutes or providing --event-id)")
            return 2

    eid = int(selected.get("eventId") or 0)
    display = str(selected.get("display") or "")
    mu = _minutes_until_next_occurrence(selected, datetime.now())

    print(f"Selected: eventId={eid} display={display!r} minutes_until={None if mu is None else round(mu, 1)}")

    before = scheduler_get(eid)
    before_event = before.get("event") if isinstance(before, dict) else None
    if not isinstance(before_event, dict):
        print("scheduler_get did not return an event dict")
        return 2

    orig_enabled = before_event.get("enabled")
    if orig_enabled not in (True, False):
        print(f"Unexpected enabled value: {orig_enabled!r}")
        return 2

    desired = not bool(orig_enabled)
    print(f"Before enabled={orig_enabled} -> toggling to {desired}")

    if not args.doit:
        dry = scheduler_set_enabled(eid, desired, dry_run=True)
        print("DRY RUN only (no changes). Planned attempts:")
        planned = dry.get("planned") if isinstance(dry, dict) else None
        if isinstance(planned, list):
            for p in planned:
                print(" -", p)
        return 0

    t1 = scheduler_set_enabled(eid, desired, dry_run=False)
    print("Toggle result:", {k: t1.get(k) for k in ("ok", "accepted", "confirmed", "event_id", "enabled")})
    if not t1.get("accepted") or not t1.get("confirmed"):
        attempts = t1.get("attempts") if isinstance(t1, dict) else None
        if isinstance(attempts, list) and attempts:
            print("Toggle attempts:")
            for a in attempts:
                if not isinstance(a, dict):
                    continue
                http = a.get("http") if isinstance(a.get("http"), dict) else {}
                print(
                    " -",
                    {
                        "path": a.get("path"),
                        "status": http.get("status"),
                        "ok": http.get("ok"),
                        "error": http.get("error"),
                        "text": (str(http.get("text"))[:200] if http.get("text") is not None else None),
                    },
                )

    mid = scheduler_get(eid)
    mid_event = mid.get("event") if isinstance(mid, dict) else None
    print("Mid enabled:", (mid_event.get("enabled") if isinstance(mid_event, dict) else None))

    if not args.no_restore:
        print(f"Restoring enabled={orig_enabled}")
        t2 = scheduler_set_enabled(eid, bool(orig_enabled), dry_run=False)
        print("Restore result:", {k: t2.get(k) for k in ("ok", "accepted", "confirmed", "event_id", "enabled")})
        if not t2.get("accepted") or not t2.get("confirmed"):
            attempts = t2.get("attempts") if isinstance(t2, dict) else None
            if isinstance(attempts, list) and attempts:
                print("Restore attempts:")
                for a in attempts:
                    if not isinstance(a, dict):
                        continue
                    http = a.get("http") if isinstance(a.get("http"), dict) else {}
                    print(
                        " -",
                        {
                            "path": a.get("path"),
                            "status": http.get("status"),
                            "ok": http.get("ok"),
                            "error": http.get("error"),
                            "text": (str(http.get("text"))[:200] if http.get("text") is not None else None),
                        },
                    )

        after = scheduler_get(eid)
        after_event = after.get("event") if isinstance(after, dict) else None
        print("After enabled:", (after_event.get("enabled") if isinstance(after_event, dict) else None))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
