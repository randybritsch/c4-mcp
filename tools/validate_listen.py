"""Validate Control4 room Listen end-to-end over the MCP HTTP surface.

This exercises:
  client -> HTTP /mcp/call -> tool -> app.py -> adapter -> gateway -> Director

Safety:
- Default is read-only: lists Listen sources for a room.
- Writes require --doit.
- By default, it will only perform writes if the room is NOT currently listening (active=False)
    so it can safely restore by turning the room off (ROOM_OFF).

Usage:
  python tools/validate_listen.py --base-url http://127.0.0.1:3333 --room-id 6
  python tools/validate_listen.py --base-url http://127.0.0.1:3333 --room-id 6 --doit
  python tools/validate_listen.py --base-url http://127.0.0.1:3333 --room-id 6 --source-device-id 1772 --doit

Auth:
- If FLASK_MCP_AUTH_MODE=apikey, pass --api-key.

Exit codes:
- 0: success
- 2: requested check failed / unsafe to proceed
- 3: could not reach server / protocol errors
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional

Json = Dict[str, Any]


@dataclass(frozen=True)
class CallResult:
    ok: bool
    status: int
    payload: Json | None
    error: str | None


def _http_post_json(url: str, body: Json, timeout_s: float, headers: Dict[str, str]) -> CallResult:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)

    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read()
            status = int(getattr(resp, "status", 200))
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else None
            except Exception:
                payload = {"_raw": raw.decode("utf-8", errors="replace")}
            return CallResult(ok=200 <= status < 300, status=status, payload=payload, error=None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else None
        except Exception:
            payload = {"_raw": raw.decode("utf-8", errors="replace")}
        return CallResult(ok=False, status=int(e.code), payload=payload, error=str(e))
    except Exception as e:
        return CallResult(ok=False, status=0, payload=None, error=repr(e))


def mcp_call(base_url: str, name: str, args: Optional[Json], timeout_s: float, headers: Dict[str, str]) -> Json:
    url = base_url.rstrip("/") + "/mcp/call"
    body = {"kind": "tool", "name": name, "args": args or {}}
    r = _http_post_json(url, body=body, timeout_s=timeout_s, headers=headers)
    if not r.ok:
        raise RuntimeError(f"HTTP call failed: status={r.status} error={r.error} payload={r.payload}")
    if not isinstance(r.payload, dict):
        raise RuntimeError(f"Unexpected response payload: {r.payload!r}")
    return r.payload


def _unwrap(payload: Json) -> Json:
    if isinstance(payload, dict) and payload.get("ok") is True and isinstance(payload.get("result"), dict):
        return payload["result"]
    return payload


def _find_source_id(src: Any) -> int | None:
    if not isinstance(src, dict):
        return None
    for k in ("deviceid", "deviceId", "id"):
        if src.get(k) is None:
            continue
        try:
            return int(src.get(k))
        except Exception:
            continue
    return None


def _source_label(src: Any) -> str:
    if not isinstance(src, dict):
        return ""
    for k in ("name", "label", "display", "title"):
        v = src.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate room Listen via MCP HTTP")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:3333")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--room-id", type=int, required=True)
    parser.add_argument("--source-device-id", type=int, default=None)
    parser.add_argument("--confirm-timeout", type=float, default=10.0)
    parser.add_argument("--doit", action="store_true", help="Actually perform c4_room_listen + restore")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow switching sources even if listen is already active (NOT recommended).",
    )
    parser.add_argument(
        "--strict-restore",
        action="store_true",
        help="Fail if restore does not make listen inactive (default: warn only).",
    )

    args = parser.parse_args()

    headers: Dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = str(args.api_key)

    base_url = str(args.base_url)
    room_id = int(args.room_id)

    before_raw = mcp_call(base_url, "c4_room_listen_status", {"room_id": room_id}, timeout_s=args.timeout, headers=headers)
    before = _unwrap(before_raw)
    if not before.get("ok"):
        print(json.dumps(before_raw, indent=2)[:4000])
        print("ERROR: c4_room_listen_status failed")
        return 2

    listen = before.get("listen") if isinstance(before.get("listen"), dict) else {}
    active_before = bool(listen.get("active"))
    sources = listen.get("sources") if isinstance(listen.get("sources"), list) else []

    print(f"Room {room_id} listen.active={active_before} sources={len(sources)}")
    for s in sources[:20]:
        sid = _find_source_id(s)
        label = _source_label(s)
        if sid is None:
            continue
        print(f" - {sid}: {label}")

    if not args.doit:
        print("\nDRY RUN only (no changes).")
        if sources:
            picked = args.source_device_id or _find_source_id(sources[0])
            if picked is not None:
                print(f"Try: --source-device-id {picked} --doit")
        return 0

    if active_before and not args.force:
        print("Refusing to change audio source because listen.active is already True. Use --force to override.")
        return 2

    picked_id = int(args.source_device_id) if args.source_device_id is not None else None
    if picked_id is None:
        # choose first source with an id
        for s in sources:
            sid = _find_source_id(s)
            if sid is not None:
                picked_id = sid
                break

    if picked_id is None:
        print("No source_device_id provided and no sources with ids found; cannot proceed.")
        return 2

    print(f"\nLISTEN: selecting source_device_id={picked_id}...")
    r_listen_raw = mcp_call(
        base_url,
        "c4_room_listen",
        {"room_id": room_id, "source_device_id": picked_id, "confirm_timeout_s": float(args.confirm_timeout)},
        timeout_s=max(args.timeout, args.confirm_timeout + 25.0),
        headers=headers,
    )
    r_listen = _unwrap(r_listen_raw)
    print(json.dumps(r_listen_raw, indent=2)[:6000])

    accepted = bool(r_listen.get("accepted")) if isinstance(r_listen, dict) else False
    confirmed = bool(r_listen.get("confirmed")) if isinstance(r_listen, dict) else False
    if accepted and not confirmed:
        print("NOTE: command accepted but not confirmed within timeout (this can be normal if UI state lags).")

    after_raw = mcp_call(base_url, "c4_room_listen_status", {"room_id": room_id}, timeout_s=args.timeout, headers=headers)
    after = _unwrap(after_raw)
    active_after = bool((after.get("listen") or {}).get("active")) if isinstance(after.get("listen"), dict) else None
    print(f"After listen.active={active_after}")

    # Restore only when we started from inactive (safe): turn the room off.
    if not active_before:
        print("\nRESTORE: turning room off (ROOM_OFF) (best-effort)...")
        r_restore_raw = mcp_call(
            base_url,
            "c4_room_off",
            {"room_id": room_id, "confirm_timeout_s": float(args.confirm_timeout)},
            timeout_s=max(args.timeout, args.confirm_timeout + 25.0),
            headers=headers,
        )
        print(json.dumps(r_restore_raw, indent=2)[:4000])

        final_raw = mcp_call(base_url, "c4_room_listen_status", {"room_id": room_id}, timeout_s=args.timeout, headers=headers)
        final = _unwrap(final_raw)
        final_active = bool((final.get("listen") or {}).get("active")) if isinstance(final.get("listen"), dict) else None
        print(f"Final listen.active={final_active}")

        if final_active is not False:
            msg = "Restore did not make listen inactive (final listen.active is not False)."
            if args.strict_restore:
                print("ERROR: " + msg)
                return 2
            print("WARNING: " + msg)

    if not accepted:
        print("ERROR: Listen command was not accepted")
        return 2

    if active_after is not True:
        print("ERROR: listen did not become active after selecting a source")
        return 2

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
