"""Watch *all* variable changes for items connected to a lock via bindings.

Why:
    You toggled the lock in the Control4 app and neither the lock proxy nor the relay proxy
  variables changed. This script builds a binding-connected set of items and watches their variables
  to discover what actually changes when the app acts.

Usage:
    python tools/watch_lock_activity.py --start <LOCK_PROXY_ID> <RELAY_PROXY_ID> --depth 3 --seconds 120 --interval 0.5

    # Or widen the net to all lock-related items (recommended if Control4 behavior is sporadic):
    python tools/watch_lock_activity.py --all-locks --seconds 180 --interval 0.5

    # Or only locks in a given room:
    python tools/watch_lock_activity.py --all-locks --room-id <ROOM_ID> --seconds 180 --interval 0.5

Run it, then unlock/lock from the Control4 app during the window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import IO
from typing import Any, Dict, Iterable, List, Set, Tuple

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from control4_adapter import get_all_items, item_get_bindings, item_get_variables


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _item_name_map() -> Dict[int, str]:
    items = get_all_items()
    out: Dict[int, str] = {}
    for i in items:
        if not isinstance(i, dict):
            continue
        try:
            item_id = int(i.get("id"))
        except Exception:
            continue
        name = i.get("name") or i.get("displayName") or ""
        out[item_id] = str(name)
    return out


def _select_lock_related_item_ids(room_id: int | None = None) -> List[int]:
    items = get_all_items()
    out: List[int] = []
    for i in items:
        if not isinstance(i, dict):
            continue
        if i.get("typeName") != "device":
            continue
        try:
            item_id = int(i.get("id"))
        except Exception:
            continue

        if room_id is not None:
            try:
                if int(i.get("roomId")) != int(room_id):
                    continue
            except Exception:
                continue

        control = str(i.get("control") or "").lower()
        cats = i.get("categories")
        is_lock_cat = isinstance(cats, list) and any(str(c).lower() == "locks" for c in cats)
        # Common lock-ish controls in this project:
        is_lock_control = control in {"lock", "control4_relaysingle"}
        if is_lock_cat or is_lock_control:
            out.append(item_id)
    return sorted(set(out))


def _select_room_device_item_ids(room_id: int) -> List[int]:
    items = get_all_items()
    out: List[int] = []
    for i in items:
        if not isinstance(i, dict):
            continue
        if i.get("typeName") != "device":
            continue
        try:
            if int(i.get("roomId")) != int(room_id):
                continue
        except Exception:
            continue
        try:
            out.append(int(i.get("id")))
        except Exception:
            continue
    return sorted(set(out))


def _extract_connected_ids(bindings_payload: Dict[str, Any]) -> Set[int]:
    ids: Set[int] = set()
    bindings = bindings_payload.get("bindings") if isinstance(bindings_payload, dict) else None
    if not isinstance(bindings, list):
        return ids

    for b in bindings:
        if not isinstance(b, dict):
            continue
        # Sometimes the binding object has an id field referring to the item.
        try:
            ids.add(int(b.get("id")))
        except Exception:
            pass
        conns = b.get("connections")
        if not isinstance(conns, list):
            continue
        for c in conns:
            if not isinstance(c, dict):
                continue
            try:
                ids.add(int(c.get("id")))
            except Exception:
                continue
    return ids


def build_binding_closure(start_ids: Iterable[int], depth: int) -> List[int]:
    start_ids = [int(x) for x in start_ids]
    depth = max(int(depth), 0)

    seen: Set[int] = set(start_ids)
    frontier: Set[int] = set(start_ids)

    for _ in range(depth):
        nxt: Set[int] = set()
        for item_id in sorted(frontier):
            payload = item_get_bindings(item_id)
            if not isinstance(payload, dict) or not payload.get("ok"):
                continue
            for cid in _extract_connected_ids(payload):
                if cid not in seen:
                    seen.add(cid)
                    nxt.add(cid)
        frontier = nxt
        if not frontier:
            break

    return sorted(seen)


def vars_map(item_id: int, timeout_s: float) -> Dict[str, Any]:
    payload = item_get_variables(int(item_id), timeout_s=float(timeout_s))
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("variables")
    if not isinstance(rows, list):
        return {}

    out: Dict[str, Any] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = r.get("varName")
        if name is None:
            continue
        out[str(name)] = r.get("value")
    return out


def diff_vars(prev: Dict[str, Any], curr: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    changed: Dict[str, Dict[str, Any]] = {}
    keys = set(prev.keys()) | set(curr.keys())
    for k in sorted(keys):
        pv = prev.get(k, "<missing>")
        cv = curr.get(k, "<missing>")
        if pv != cv:
            changed[k] = {"from": pv, "to": cv}
    return changed


def _open_log_file(path: str | None) -> IO[str] | None:
    if not path:
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.open("a", encoding="utf-8")


def _emit(line: str, log: IO[str] | None = None) -> None:
    print(line, flush=True)
    if log is not None:
        log.write(line + "\n")
        log.flush()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start", nargs="*", type=int, default=[], help="Starting item ids")
    p.add_argument("--all-locks", action="store_true", help="Watch all lock-related items (by control/category)")
    p.add_argument(
        "--all-room-devices",
        action="store_true",
        help="Watch all device items in --room-id (broader than --all-locks)",
    )
    p.add_argument("--room-id", type=int, default=None, help="Optional roomId filter (use with --all-locks)")
    p.add_argument("--depth", type=int, default=3, help="Binding graph expansion depth")
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--max-items", type=int, default=80, help="Safety limit")
    p.add_argument(
        "--vars-timeout",
        type=float,
        default=4.0,
        help="Timeout (seconds) for fetching variables per item (keeps watcher responsive)",
    )
    p.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to append logs to (recommended for capturing sporadic changes)",
    )
    p.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Print a heartbeat line every N seconds (0 disables)",
    )
    args = p.parse_args()

    if not args.start and not args.all_locks and not args.all_room_devices:
        raise SystemExit("Provide --start item ids or use --all-locks/--all-room-devices")

    if args.all_room_devices and args.room_id is None:
        raise SystemExit("--all-room-devices requires --room-id")

    log = _open_log_file(args.log_file)
    if log is not None:
        _emit(f"[{_now()}] Logging to: {str(Path(args.log_file).resolve())}", log)

    names = _item_name_map()

    base_ids: List[int] = []
    if args.all_locks:
        base_ids.extend(_select_lock_related_item_ids(room_id=args.room_id))
    if args.all_room_devices and args.room_id is not None:
        base_ids.extend(_select_room_device_item_ids(room_id=int(args.room_id)))
    base_ids.extend([int(x) for x in (args.start or [])])
    base_ids = sorted(set(base_ids))

    ids = build_binding_closure(base_ids, args.depth) if base_ids else []
    if len(ids) > int(args.max_items):
        ids = ids[: int(args.max_items)]

    _emit(
        f"[{_now()}] Watching {len(ids)} items for {float(args.seconds):.1f}s at {float(args.interval):.2f}s interval",
        log,
    )
    if args.all_locks:
        rid = f" roomId={int(args.room_id)}" if args.room_id is not None else ""
        _emit(f"Mode: all-locks{rid} + binding-closure depth={int(args.depth)}", log)
    elif args.all_room_devices:
        _emit(
            f"Mode: all-room-devices roomId={int(args.room_id)} + binding-closure depth={int(args.depth)}",
            log,
        )
    else:
        _emit(f"Mode: start={base_ids} + binding-closure depth={int(args.depth)}", log)
    _emit("Items:", log)
    for i in ids:
        _emit(f"  {i}: {names.get(i, '')}", log)

    try:
        prev: Dict[int, Dict[str, Any]] = {i: vars_map(i, float(args.vars_timeout)) for i in ids}

        deadline = time.time() + float(args.seconds)
        next_heartbeat = time.time() + max(float(args.heartbeat_seconds), 0.0)

        poll_count = 0
        while time.time() < deadline:
            time.sleep(max(float(args.interval), 0.1))
            poll_count += 1
            for i in ids:
                try:
                    curr = vars_map(i, float(args.vars_timeout))
                except Exception as e:
                    _emit(f"[{_now()}] ERROR vars_map item {i}: {type(e).__name__}: {e}", log)
                    continue

                ch = diff_vars(prev.get(i, {}), curr)
                if ch:
                    _emit(f"\n[{_now()}] CHANGE item {i}: {names.get(i, '')}", log)
                    _emit(json.dumps(ch, indent=2), log)
                    prev[i] = curr

            hb_every = float(args.heartbeat_seconds)
            if hb_every > 0 and time.time() >= next_heartbeat:
                remaining = max(deadline - time.time(), 0.0)
                _emit(
                    f"[{_now()}] Heartbeat: polls={poll_count} remaining={remaining:.1f}s",
                    log,
                )
                next_heartbeat = time.time() + hb_every

        _emit(f"\n[{_now()}] Done.", log)
        return 0
    finally:
        if log is not None:
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
