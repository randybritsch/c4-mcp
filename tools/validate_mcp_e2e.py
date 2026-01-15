"""End-to-end validator for Control4 MCP server.

This script exercises the public MCP HTTP surface (Flask + flask-mcp-server)
so you validate the full path:

  client -> HTTP /mcp/call -> tool -> app.py -> control4_adapter.py -> control4_gateway.py -> Director

Safety:
- By default, it performs discovery + read-only checks.
- Writes (light set/ramp, lock lock/unlock) require --do-writes.

Usage examples:
  python tools/validate_mcp_e2e.py --base-url http://127.0.0.1:3333
  python tools/validate_mcp_e2e.py --base-url http://127.0.0.1:3333 --light-id 1234 --do-writes
  python tools/validate_mcp_e2e.py --base-url http://127.0.0.1:3333 --lock-id 2214 --do-writes

Auth:
- If FLASK_MCP_AUTH_MODE=apikey, pass --api-key.

Exit codes:
- 0: all executed checks passed
- 2: a requested check failed
- 3: could not reach server / protocol errors
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def mcp_call(base_url: str, kind: str, name: str, args: Optional[Json], timeout_s: float, headers: Dict[str, str]) -> Json:
    url = base_url.rstrip("/") + "/mcp/call"
    body = {"kind": kind, "name": name, "args": args or {}}
    r = _http_post_json(url, body=body, timeout_s=timeout_s, headers=headers)
    if not r.ok:
        raise RuntimeError(f"HTTP call failed: status={r.status} error={r.error} payload={r.payload}")
    if not isinstance(r.payload, dict):
        raise RuntimeError(f"Unexpected response payload: {r.payload!r}")
    return r.payload


def mcp_list(base_url: str, timeout_s: float, headers: Dict[str, str]) -> Json:
    url = base_url.rstrip("/") + "/mcp/list"
    req = urllib.request.Request(url, method="GET")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read()
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(payload, dict):
                raise RuntimeError(f"Unexpected list payload: {payload!r}")
            return payload
    except Exception as e:
        raise RuntimeError(f"MCP list failed: {e!r}")


def _sleep_poll(seconds: float) -> None:
    time.sleep(max(0.0, float(seconds)))


def _print_json(label: str, obj: Any) -> None:
    print(f"\n== {label} ==")
    print(json.dumps(obj, indent=2, sort_keys=True)[:8000])


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _unwrap_call_payload(payload: Json) -> Json:
    """/mcp/call returns {ok, result}. Unwrap once for tool results."""
    if isinstance(payload, dict) and payload.get("ok") is True and isinstance(payload.get("result"), dict):
        return payload["result"]
    return payload


def _auto_select_thermostat_id(base_url: str, timeout_s: float, headers: Dict[str, str]) -> Optional[str]:
    devices_payload = _unwrap_call_payload(
        mcp_call(
            base_url,
            kind="tool",
            name="c4_list_devices",
            args={"category": "thermostat"},
            timeout_s=timeout_s,
            headers=headers,
        )
    )
    devices = devices_payload.get("devices") if isinstance(devices_payload, dict) else None
    if not isinstance(devices, list) or not devices:
        return None

    first = devices[0] if isinstance(devices[0], dict) else None
    tid = (first or {}).get("id") if isinstance(first, dict) else None
    return str(tid) if tid is not None else None


def _validate_thermostat(
    base_url: str,
    thermostat_id: str,
    do_writes: bool,
    target_f: float | None,
    restore: bool,
    restore_modes: bool,
    timeout_s: float,
    headers: Dict[str, str],
) -> None:
    print(f"\n-- Thermostat validation (device_id={thermostat_id}) --")
    state_payload = _unwrap_call_payload(
        mcp_call(
            base_url,
            kind="tool",
            name="c4_thermostat_get_state",
            args={"device_id": str(thermostat_id)},
            timeout_s=timeout_s,
            headers=headers,
        )
    )
    _print_json("c4_thermostat_get_state", state_payload)
    _expect(bool(state_payload.get("ok")) is True, "c4_thermostat_get_state returned ok=false")

    st = state_payload.get("state")
    _expect(isinstance(st, dict), "thermostat_get_state missing state dict")
    _expect(st.get("temperature_f") is not None, "thermostat state missing temperature_f")
    _expect(st.get("hvac_mode") is not None, "thermostat state missing hvac_mode")

    if not do_writes:
        return

    # Capture original setpoints for optional restore.
    orig_heat = st.get("heat_setpoint_f")
    orig_cool = st.get("cool_setpoint_f")
    orig_hvac_mode = st.get("hvac_mode")
    orig_fan_mode = st.get("fan_mode")
    orig_hold_mode = st.get("hold_mode")

    if target_f is not None:
        r = _unwrap_call_payload(
            mcp_call(
                base_url,
                kind="tool",
                name="c4_thermostat_set_target_f",
                args={
                    "device_id": str(thermostat_id),
                    "target_f": float(target_f),
                    "confirm_timeout_s": 8.0,
                },
                timeout_s=max(timeout_s, 25.0),
                headers=headers,
            )
        )
        _print_json("c4_thermostat_set_target_f", r)
        _expect(r.get("accepted") is True, "c4_thermostat_set_target_f not accepted")

        if restore and (orig_heat is not None or orig_cool is not None):
            print("Restoring thermostat setpoints...")
            if orig_heat is not None:
                rr = _unwrap_call_payload(
                    mcp_call(
                        base_url,
                        kind="tool",
                        name="c4_thermostat_set_heat_setpoint_f",
                        args={
                            "device_id": str(thermostat_id),
                            "setpoint_f": float(orig_heat),
                            "confirm_timeout_s": 8.0,
                        },
                        timeout_s=max(timeout_s, 25.0),
                        headers=headers,
                    )
                )
                _print_json("restore c4_thermostat_set_heat_setpoint_f", rr)
                _expect(rr.get("accepted") is True, "restore heat setpoint not accepted")

            if orig_cool is not None:
                rr = _unwrap_call_payload(
                    mcp_call(
                        base_url,
                        kind="tool",
                        name="c4_thermostat_set_cool_setpoint_f",
                        args={
                            "device_id": str(thermostat_id),
                            "setpoint_f": float(orig_cool),
                            "confirm_timeout_s": 8.0,
                        },
                        timeout_s=max(timeout_s, 25.0),
                        headers=headers,
                    )
                )
                _print_json("restore c4_thermostat_set_cool_setpoint_f", rr)
                _expect(rr.get("accepted") is True, "restore cool setpoint not accepted")

        if restore and restore_modes and (orig_hvac_mode is not None or orig_fan_mode is not None or orig_hold_mode is not None):
            print("Restoring thermostat modes...")
            if orig_hvac_mode is not None:
                rr = _unwrap_call_payload(
                    mcp_call(
                        base_url,
                        kind="tool",
                        name="c4_thermostat_set_hvac_mode",
                        args={
                            "device_id": str(thermostat_id),
                            "mode": str(orig_hvac_mode),
                            "confirm_timeout_s": 8.0,
                        },
                        timeout_s=max(timeout_s, 25.0),
                        headers=headers,
                    )
                )
                _print_json("restore c4_thermostat_set_hvac_mode", rr)
                _expect(rr.get("accepted") is True, "restore hvac_mode not accepted")

            if orig_fan_mode is not None:
                rr = _unwrap_call_payload(
                    mcp_call(
                        base_url,
                        kind="tool",
                        name="c4_thermostat_set_fan_mode",
                        args={
                            "device_id": str(thermostat_id),
                            "mode": str(orig_fan_mode),
                            "confirm_timeout_s": 8.0,
                        },
                        timeout_s=max(timeout_s, 25.0),
                        headers=headers,
                    )
                )
                _print_json("restore c4_thermostat_set_fan_mode", rr)
                _expect(rr.get("accepted") is True, "restore fan_mode not accepted")

            if orig_hold_mode is not None:
                rr = _unwrap_call_payload(
                    mcp_call(
                        base_url,
                        kind="tool",
                        name="c4_thermostat_set_hold_mode",
                        args={
                            "device_id": str(thermostat_id),
                            "mode": str(orig_hold_mode),
                            "confirm_timeout_s": 8.0,
                        },
                        timeout_s=max(timeout_s, 25.0),
                        headers=headers,
                    )
                )
                _print_json("restore c4_thermostat_set_hold_mode", rr)
                _expect(rr.get("accepted") is True, "restore hold_mode not accepted")

        return

    # Conservative writes: re-apply current values to validate command path
    cur_heat = st.get("heat_setpoint_f")
    cur_cool = st.get("cool_setpoint_f")
    cur_hvac_mode = st.get("hvac_mode")
    cur_fan_mode = st.get("fan_mode")

    if cur_hvac_mode is not None:
        r = _unwrap_call_payload(
            mcp_call(
                base_url,
                kind="tool",
                name="c4_thermostat_set_hvac_mode",
                args={"device_id": str(thermostat_id), "mode": str(cur_hvac_mode), "confirm_timeout_s": 6.0},
                timeout_s=max(timeout_s, 20.0),
                headers=headers,
            )
        )
        _print_json("c4_thermostat_set_hvac_mode", r)
        _expect(r.get("accepted") is True, "c4_thermostat_set_hvac_mode not accepted")

    if cur_fan_mode is not None:
        r = _unwrap_call_payload(
            mcp_call(
                base_url,
                kind="tool",
                name="c4_thermostat_set_fan_mode",
                args={"device_id": str(thermostat_id), "mode": str(cur_fan_mode), "confirm_timeout_s": 6.0},
                timeout_s=max(timeout_s, 20.0),
                headers=headers,
            )
        )
        _print_json("c4_thermostat_set_fan_mode", r)
        _expect(r.get("accepted") is True, "c4_thermostat_set_fan_mode not accepted")

    if cur_heat is not None:
        r = _unwrap_call_payload(
            mcp_call(
                base_url,
                kind="tool",
                name="c4_thermostat_set_heat_setpoint_f",
                args={"device_id": str(thermostat_id), "setpoint_f": float(cur_heat), "confirm_timeout_s": 6.0},
                timeout_s=max(timeout_s, 20.0),
                headers=headers,
            )
        )
        _print_json("c4_thermostat_set_heat_setpoint_f", r)
        _expect(r.get("accepted") is True, "c4_thermostat_set_heat_setpoint_f not accepted")

    if cur_cool is not None:
        r = _unwrap_call_payload(
            mcp_call(
                base_url,
                kind="tool",
                name="c4_thermostat_set_cool_setpoint_f",
                args={"device_id": str(thermostat_id), "setpoint_f": float(cur_cool), "confirm_timeout_s": 6.0},
                timeout_s=max(timeout_s, 20.0),
                headers=headers,
            )
        )
        _print_json("c4_thermostat_set_cool_setpoint_f", r)
        _expect(r.get("accepted") is True, "c4_thermostat_set_cool_setpoint_f not accepted")


def _poll_light(
    base_url: str,
    headers: Dict[str, str],
    device_id: str,
    *,
    timeout_s: float,
    poll_interval_s: float,
    poll_timeout_s: float,
    expected_level: int | None = None,
    expected_state: bool | None = None,
) -> Json:
    deadline = time.time() + float(poll_timeout_s)
    last: Json = {}
    while time.time() < deadline:
        s = _unwrap_call_payload(
            mcp_call(base_url, "tool", "c4_light_get_state", {"device_id": device_id}, float(timeout_s), headers)
        )
        lvl = _unwrap_call_payload(
            mcp_call(base_url, "tool", "c4_light_get_level", {"device_id": device_id}, float(timeout_s), headers)
        )
        last = {"state": s, "level": lvl}

        ok = True
        if expected_state is not None:
            got_state = bool(s.get("state")) if isinstance(s, dict) else None
            ok = ok and (got_state == bool(expected_state))

        if expected_level is not None:
            got_level = lvl.get("level") if isinstance(lvl, dict) else None
            ok = ok and isinstance(got_level, int) and abs(int(got_level) - int(expected_level)) <= 2

        if ok:
            break

        _sleep_poll(float(poll_interval_s))

    return last


def _assert_light_expectations(
    payload: Json,
    *,
    expected_level: int | None = None,
    expected_state: bool | None = None,
    tolerance: int = 2,
) -> None:
    if expected_state is not None:
        s = payload.get("state") if isinstance(payload, dict) else None
        got_state = s.get("state") if isinstance(s, dict) else None
        _expect(got_state in (True, False), "light_get_state did not return a boolean")
        _expect(bool(got_state) == bool(expected_state), f"light state did not match expected={expected_state}")

    if expected_level is not None:
        lvl = payload.get("level") if isinstance(payload, dict) else None
        got_level = lvl.get("level") if isinstance(lvl, dict) else None
        _expect(isinstance(got_level, int), "light_get_level did not return an integer level")
        _expect(
            abs(int(got_level) - int(expected_level)) <= int(tolerance),
            f"light level did not match expected={expected_level} (got={got_level})",
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", type=str, default="http://127.0.0.1:3333", help="Base URL for Flask server")
    p.add_argument("--api-key", type=str, default=None, help="API key (if FLASK_MCP_AUTH_MODE=apikey)")
    p.add_argument("--timeout", type=float, default=25.0, help="Per-request timeout")

    p.add_argument("--light-id", type=str, default=None, help="Light device_id to validate")
    p.add_argument(
        "--auto-light",
        action="store_true",
        help=(
            "Auto-select a dimmable light for validation (prefers ones exposing 'Brightness Percent'). "
            "Ignored if --light-id is provided."
        ),
    )
    p.add_argument("--lock-id", type=str, default=None, help="Lock device_id to validate")
    p.add_argument("--thermostat-id", type=str, default=None, help="Thermostat device_id to validate")
    p.add_argument(
        "--thermostat-target-f",
        type=float,
        default=None,
        help="If set with --do-writes, set thermostat target temp (F) via c4_thermostat_set_target_f",
    )
    p.add_argument(
        "--thermostat-restore",
        action="store_true",
        help=(
            "After thermostat writes, restore original heat/cool setpoints. "
            "Intended for smoke-testing without leaving the thermostat changed."
        ),
    )
    p.add_argument(
        "--thermostat-restore-modes",
        action="store_true",
        help=(
            "With --thermostat-restore, also attempt to restore hvac_mode, fan_mode, and hold_mode "
            "captured from the initial thermostat state."
        ),
    )
    p.add_argument(
        "--auto-thermostat",
        action="store_true",
        help="Auto-select a thermostat from c4_list_devices(thermostat). Ignored if --thermostat-id is provided.",
    )
    p.add_argument("--do-writes", action="store_true", help="Actually perform writes (set level, lock/unlock)")

    p.add_argument("--poll-interval", type=float, default=0.5)
    p.add_argument("--poll-timeout", type=float, default=10.0)

    args = p.parse_args()

    headers: Dict[str, str] = {}
    if args.api_key:
        headers["X-API-Key"] = str(args.api_key)

    try:
        items = mcp_list(args.base_url, timeout_s=float(args.timeout), headers=headers)
    except Exception as e:
        print(f"ERROR: cannot reach MCP server: {e}")
        return 3

    # Basic reachability
    ping = mcp_call(args.base_url, kind="tool", name="ping", args={}, timeout_s=float(args.timeout), headers=headers)
    _print_json("ping", ping)
    ping_r = _unwrap_call_payload(ping)
    _expect(bool(ping_r.get("ok")), "ping did not return ok")

    # Discovery
    devices_lights = mcp_call(
        args.base_url,
        kind="tool",
        name="c4_list_devices",
        args={"category": "lights"},
        timeout_s=float(args.timeout),
        headers=headers,
    )
    _print_json("c4_list_devices(lights)", devices_lights)

    selected_light_id: str | None = str(args.light_id) if args.light_id else None
    if not selected_light_id and args.auto_light:
        payload = _unwrap_call_payload(devices_lights)
        candidates = payload.get("devices") if isinstance(payload, dict) else None
        if not isinstance(candidates, list) or not candidates:
            print("WARN: --auto-light requested but no lights returned by c4_list_devices")
        else:
            # Heuristic: choose the first light that exposes Brightness Percent or PRESET_LEVEL
            # (those correlate with dimmable-level readback).
            max_probe = 25
            probed = 0
            for d in candidates:
                if not isinstance(d, dict):
                    continue
                did = d.get("id")
                if did is None:
                    continue
                probed += 1
                if probed > max_probe:
                    break

                try:
                    vars_payload = _unwrap_call_payload(
                        mcp_call(
                            args.base_url,
                            "tool",
                            "c4_item_variables",
                            {"device_id": str(did)},
                            float(args.timeout),
                            headers,
                        )
                    )
                except Exception:
                    continue

                rows = None
                if isinstance(vars_payload, dict):
                    vwrap = vars_payload.get("variables")
                    if isinstance(vwrap, dict):
                        rows = vwrap.get("variables")
                if not isinstance(rows, list):
                    continue

                names = {str(r.get("varName")) for r in rows if isinstance(r, dict) and r.get("varName") is not None}
                if "Brightness Percent" in names or "PRESET_LEVEL" in names:
                    selected_light_id = str(did)
                    print(f"Auto-selected light_id={selected_light_id} ({d.get('name')})")
                    break

            if not selected_light_id:
                # Fall back to first listed device id.
                first = candidates[0]
                if isinstance(first, dict) and first.get("id") is not None:
                    selected_light_id = str(first.get("id"))
                    print(f"Auto-selected (fallback) light_id={selected_light_id} ({first.get('name')})")

    devices_locks = mcp_call(
        args.base_url,
        kind="tool",
        name="c4_list_devices",
        args={"category": "locks"},
        timeout_s=float(args.timeout),
        headers=headers,
    )
    _print_json("c4_list_devices(locks)", devices_locks)

    devices_thermostat = mcp_call(
        args.base_url,
        kind="tool",
        name="c4_list_devices",
        args={"category": "thermostat"},
        timeout_s=float(args.timeout),
        headers=headers,
    )
    _print_json("c4_list_devices(thermostat)", devices_thermostat)

    # ---- Lights ----
    if selected_light_id:
        lid = str(selected_light_id)
        s0 = _unwrap_call_payload(
            mcp_call(args.base_url, "tool", "c4_light_get_state", {"device_id": lid}, float(args.timeout), headers)
        )
        l0 = _unwrap_call_payload(
            mcp_call(args.base_url, "tool", "c4_light_get_level", {"device_id": lid}, float(args.timeout), headers)
        )
        _print_json(f"light {lid} initial state", {"state": s0, "level": l0})

        if args.do_writes:
            off = _unwrap_call_payload(
                mcp_call(
                    args.base_url,
                    "tool",
                    "c4_item_send_command",
                    {"device_id": lid, "command": "OFF"},
                    float(args.timeout),
                    headers,
                )
            )
            _print_json(f"light {lid} OFF (via c4_item_send_command)", off)
            rb = _poll_light(
                args.base_url,
                headers,
                lid,
                timeout_s=float(args.timeout),
                poll_interval_s=float(args.poll_interval),
                poll_timeout_s=float(args.poll_timeout),
                expected_state=False,
                expected_level=0,
            )
            _print_json(f"light {lid} read-back after OFF", rb)
            _assert_light_expectations(rb, expected_state=False, expected_level=0)

            on = _unwrap_call_payload(
                mcp_call(
                    args.base_url,
                    "tool",
                    "c4_item_send_command",
                    {"device_id": lid, "command": "ON"},
                    float(args.timeout),
                    headers,
                )
            )
            _print_json(f"light {lid} ON (via c4_item_send_command)", on)
            rb = _poll_light(
                args.base_url,
                headers,
                lid,
                timeout_s=float(args.timeout),
                poll_interval_s=float(args.poll_interval),
                poll_timeout_s=float(args.poll_timeout),
                expected_state=True,
            )
            _print_json(f"light {lid} read-back after ON", rb)
            _assert_light_expectations(rb, expected_state=True)

            for target in (30, 100):
                setr = _unwrap_call_payload(
                    mcp_call(
                    args.base_url,
                    "tool",
                    "c4_light_set_level",
                    {"device_id": lid, "level": int(target)},
                    float(args.timeout),
                    headers,
                    )
                )
                _print_json(f"light {lid} set_level {target}", setr)
                _expect(bool(setr.get("ok")), "c4_light_set_level did not return ok")

                rb = _poll_light(
                    args.base_url,
                    headers,
                    lid,
                    timeout_s=float(args.timeout),
                    poll_interval_s=float(args.poll_interval),
                    poll_timeout_s=float(args.poll_timeout),
                    expected_level=int(target),
                )
                _print_json(f"light {lid} read-back after {target}", rb)
                _assert_light_expectations(rb, expected_level=int(target))

    # ---- Locks ----
    if args.lock_id:
        kid = str(args.lock_id)
        st = _unwrap_call_payload(
            mcp_call(args.base_url, "tool", "c4_lock_get_state", {"device_id": kid}, float(args.timeout), headers)
        )
        _print_json(f"lock {kid} get_state", st)
        _expect(bool(st.get("ok")), "c4_lock_get_state did not return ok")

        if args.do_writes:
            u = _unwrap_call_payload(
                mcp_call(args.base_url, "tool", "c4_lock_unlock", {"device_id": kid}, float(args.timeout), headers)
            )
            _print_json(f"lock {kid} unlock", u)
            _expect(bool(u.get("ok")), "c4_lock_unlock did not return ok")
            _expect("accepted" in u, "lock unlock missing accepted")
            _expect("confirmed" in u, "lock unlock missing confirmed")

            l = _unwrap_call_payload(
                mcp_call(args.base_url, "tool", "c4_lock_lock", {"device_id": kid}, float(args.timeout), headers)
            )
            _print_json(f"lock {kid} lock", l)
            _expect(bool(l.get("ok")), "c4_lock_lock did not return ok")
            _expect("accepted" in l, "lock lock missing accepted")
            _expect("confirmed" in l, "lock lock missing confirmed")

    # ---- Thermostats ----
    selected_thermostat_id: str | None = str(args.thermostat_id) if args.thermostat_id else None
    if not selected_thermostat_id and args.auto_thermostat:
        selected_thermostat_id = _auto_select_thermostat_id(args.base_url, timeout_s=float(args.timeout), headers=headers)
        if selected_thermostat_id:
            print(f"Auto-selected thermostat_id={selected_thermostat_id}")
        else:
            print("WARN: --auto-thermostat requested but no thermostats returned by c4_list_devices")

    if selected_thermostat_id:
        _validate_thermostat(
            args.base_url,
            selected_thermostat_id,
            do_writes=bool(args.do_writes),
            target_f=(float(args.thermostat_target_f) if args.thermostat_target_f is not None else None),
            restore=bool(args.thermostat_restore),
            restore_modes=bool(args.thermostat_restore_modes),
            timeout_s=float(args.timeout),
            headers=headers,
        )

    print("\nPASS: requested checks completed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        raise SystemExit(2)
    except KeyboardInterrupt:
        raise SystemExit(130)
