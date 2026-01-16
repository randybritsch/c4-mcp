"""Validate Control4 alarm/security panel support over the MCP HTTP surface.

This is designed to be safe to run in environments WITHOUT an alarm panel.

Checks:
- List alarm panels (best-effort discovery)
- If at least one panel exists: fetch state via c4_alarm_get_state

Writes:
- None by default.
- You can opt into writes with --doit, but ONLY if you explicitly provide --device-id.

Exit codes:
- 0: success (including "no alarm present")
- 2: requested check failed / unsafe to proceed
- 3: could not reach server / protocol errors
"""

from __future__ import annotations

import argparse
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate alarm/security panel tools via MCP HTTP")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:3333")
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--timeout", type=float, default=25.0)
    parser.add_argument("--device-id", type=int, default=None, help="Alarm panel device id (required for --doit)")
    parser.add_argument("--doit", action="store_true", help="Perform a mode change (requires --device-id).")
    parser.add_argument("--mode", type=str, default="disarmed", help="Mode for --doit: disarmed, away, stay, night")
    parser.add_argument("--code", type=str, default=None, help="Optional alarm code/PIN for --doit")

    args = parser.parse_args()

    headers: Dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = str(args.api_key)

    base_url = str(args.base_url)

    panels_raw = mcp_call(base_url, "c4_alarm_list", {"limit": 200}, timeout_s=args.timeout, headers=headers)
    panels = _unwrap(panels_raw)
    if not isinstance(panels, dict) or not panels.get("ok"):
        print(json.dumps(panels_raw, indent=2)[:4000])
        print("ERROR: c4_alarm_list failed")
        return 2

    count = int(panels.get("count") or 0)
    print(f"Alarm panels found: {count}")
    for p in (panels.get("panels") or [])[:10]:
        if isinstance(p, dict):
            print(f" - {p.get('device_id')}: {p.get('name')} (room={p.get('room_name')})")

    if count <= 0:
        print("PASS: no alarm panel present (expected in some environments)")
        return 0

    device_id = int(args.device_id) if args.device_id is not None else int((panels.get("panels") or [])[0].get("device_id"))
    state_raw = mcp_call(base_url, "c4_alarm_get_state", {"device_id": str(device_id), "timeout_s": 8.0}, timeout_s=args.timeout, headers=headers)
    print("\nState:")
    print(json.dumps(state_raw, indent=2)[:4000])

    if not args.doit:
        print("\nDRY RUN only (no changes).")
        print("PASS")
        return 0

    if args.device_id is None:
        print("Refusing to write without explicit --device-id")
        return 2

    r = mcp_call(
        base_url,
        "c4_alarm_set_mode",
        {
            "device_id": str(int(args.device_id)),
            "mode": str(args.mode),
            "code": (str(args.code) if args.code is not None else None),
            "confirm_timeout_s": 12.0,
            "dry_run": False,
        },
        timeout_s=max(args.timeout, 45.0),
        headers=headers,
    )
    print("\nWrite result:")
    print(json.dumps(r, indent=2)[:6000])

    ok = bool(_unwrap(r).get("ok")) if isinstance(_unwrap(r), dict) else False
    if not ok:
        return 2

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
