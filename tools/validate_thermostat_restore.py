"""E2E validation: thermostat set_target_f (+1F) with auto-restore.

Usage:
  python tools/validate_thermostat_restore.py

This is a safety-oriented local validation script:
- Finds a thermostatV2 device
- Reads state
- Sets target_f by +1F (mode-aware via gateway logic)
- Restores original setpoint(s)

It prints accepted/confirmed results and final state.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from control4_adapter import (
    get_all_items,
    thermostat_get_state,
    thermostat_set_cool_setpoint_f,
    thermostat_set_heat_setpoint_f,
    thermostat_set_target_f,
)


def _j(x):
    print(json.dumps(x, indent=2)[:5000])


def main() -> int:
    items = get_all_items()
    thermostats = [
        i
        for i in items
        if isinstance(i, dict)
        and i.get("typeName") == "device"
        and str(i.get("control") or "") == "thermostatV2"
    ]

    if not thermostats:
        print("No thermostatV2 devices found.")
        return 2

    t = thermostats[0]
    device_id = int(t.get("id"))
    print(f"Using thermostat {device_id}: {t.get('name')} (room={t.get('roomName') or t.get('roomId')})")

    before = thermostat_get_state(device_id)
    state = (before or {}).get("state") if isinstance(before, dict) else None
    state = state if isinstance(state, dict) else {}

    hvac_mode = str(state.get("hvac_mode") or "").strip().lower()
    heat_sp = state.get("heat_setpoint_f")
    cool_sp = state.get("cool_setpoint_f")
    deadband = state.get("heatcool_deadband_f")

    print("BEFORE:")
    _j(before)

    if hvac_mode == "off" or not hvac_mode:
        print("Thermostat HVAC mode is Off/unknown; refusing to set target.")
        return 0

    # Choose a target that is +1F relative to the relevant active setpoint.
    if hvac_mode == "heat" and heat_sp is not None:
        target_before = float(heat_sp)
        target_after = target_before + 1.0
    elif hvac_mode == "cool" and cool_sp is not None:
        target_before = float(cool_sp)
        target_after = target_before + 1.0
    else:
        # Treat any non-off/non-heat/non-cool as auto.
        if heat_sp is None:
            print("Auto mode but missing heat_setpoint_f; cannot run test.")
            return 3
        target_before = float(heat_sp)
        target_after = target_before + 1.0

    try:
        print(f"\nSETTING target_f to {target_after}F (mode={hvac_mode}, deadband={deadband})...")
        r_set = thermostat_set_target_f(device_id, target_after, confirm_timeout_s=10.0, deadband_f=None)
        _j(r_set)

        after_set = thermostat_get_state(device_id)
        print("\nAFTER SET:")
        _j(after_set)
    finally:
        print(f"\nRESTORING original setpoints (mode={hvac_mode})...")
        if hvac_mode == "auto":
            # Restore both setpoints as best-effort.
            if heat_sp is not None:
                _j(thermostat_set_heat_setpoint_f(device_id, float(heat_sp), confirm_timeout_s=10.0))
            if cool_sp is not None:
                _j(thermostat_set_cool_setpoint_f(device_id, float(cool_sp), confirm_timeout_s=10.0))
        else:
            _j(thermostat_set_target_f(device_id, target_before, confirm_timeout_s=10.0, deadband_f=None))

        after_restore = thermostat_get_state(device_id)
        print("\nAFTER RESTORE:")
        _j(after_restore)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
