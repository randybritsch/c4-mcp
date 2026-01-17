r"""Validate session memory over MCP HTTP.

This exercises the full path:
  client -> HTTP /mcp/call -> tool -> app.py -> session_memory

It verifies that a stable X-Session-Id header causes the server to remember
"last_lights" across calls.

Usage:
    .\.venv\Scripts\python.exe tools\validate_http_session_memory.py --base-url http://127.0.0.1:3333

Exit codes:
  0: pass
  2: assertion failure
  3: could not reach server / protocol errors
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
            payload = json.loads(raw.decode("utf-8")) if raw else {}
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--session-id", default="e2e-session-1")
    ap.add_argument("--timeout-s", type=float, default=25.0)
    ap.add_argument("--touch", type=int, default=3, help="How many lights to touch via c4_light_get_state.")
    args = ap.parse_args()

    headers = {"X-Session-Id": str(args.session_id)}

    try:
        devices_payload = _unwrap(
            mcp_call(
                str(args.base_url),
                name="c4_list_devices",
                args={"category": "lights"},
                timeout_s=float(args.timeout_s),
                headers=headers,
            )
        )
        devices = devices_payload.get("devices") if isinstance(devices_payload, dict) else None
        if not isinstance(devices, list) or not devices:
            raise AssertionError("Expected at least one light device")

        devices = devices[:10]

        ids: list[str] = []
        for d in devices:
            if isinstance(d, dict) and d.get("id") is not None:
                ids.append(str(d["id"]))
            if len(ids) >= int(args.touch):
                break

        print("lights_listed:", len(devices))
        print("touching:", ids)
        if not ids:
            raise AssertionError("Could not extract any light ids")

        for did in ids:
            _unwrap(
                mcp_call(
                    str(args.base_url),
                    name="c4_light_get_state",
                    args={"device_id": did},
                    timeout_s=float(args.timeout_s),
                    headers=headers,
                )
            )

        last_payload = _unwrap(mcp_call(str(args.base_url), name="c4_lights_get_last", args={}, timeout_s=float(args.timeout_s), headers=headers))
        mem_payload = _unwrap(mcp_call(str(args.base_url), name="c4_memory_get", args={}, timeout_s=float(args.timeout_s), headers=headers))

        print("session_id_echo:", mem_payload.get("session_id"))
        print("last_lights_count:", last_payload.get("count"))
        remembered = (mem_payload.get("memory") or {}).get("last_lights") if isinstance(mem_payload, dict) else None
        print("memory_last_lights_len:", len(remembered) if isinstance(remembered, list) else None)

        assert mem_payload.get("session_id") == str(args.session_id), "Expected session_id echo to match"
        assert isinstance(remembered, list) and remembered, "Expected last_lights to be remembered"

        print("OK: HTTP session memory populated")
        return 0

    except AssertionError as e:
        print("FAIL:", str(e))
        return 2
    except Exception as e:
        print("ERROR:", repr(e))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
