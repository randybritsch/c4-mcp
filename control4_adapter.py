# control4_adapter.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from control4_gateway import Control4Gateway

Json = Dict[str, Any]

# Single shared gateway instance for the process
gateway = Control4Gateway()


# --------- Core passthroughs ---------

def list_rooms() -> List[Json]:
    # Prefer gateway method if present, else derive from items if you add get_all_items later
    if hasattr(gateway, "list_rooms"):
        return gateway.list_rooms()  # type: ignore[return-value]
    # Fallback: if your gateway doesn't implement list_rooms, return empty
    return []


def get_all_items() -> List[Json]:
    # Gateway provides item listing; adapter stays logic-free.
    return gateway.get_all_items()  # type: ignore[return-value]


def item_get_variables(device_id: int, timeout_s: float = 12.0) -> Any:
    """
    Raw variables for an item. The latest gateway code doesn't expose item_get_variables,
    but lock_get_state reads variables internally. Keep this adapter callable for MCP debug tool.
    """
    return gateway.item_get_variables(int(device_id), timeout_s=float(timeout_s))  # type: ignore[misc]


def item_get_bindings(device_id: int, timeout_s: float = 12.0) -> Any:
    return gateway.item_get_bindings(int(device_id), timeout_s=float(timeout_s))  # type: ignore[misc]


def item_get_commands(device_id: int) -> Json:
    return gateway.item_get_commands(int(device_id))


def item_execute_command(device_id: int, command_id: int) -> Json:
    return gateway.item_execute_command(int(device_id), int(command_id))


def item_send_command(device_id: int, command: str, params: Optional[Dict[str, Any]] = None) -> Json:
    return gateway.item_send_command(int(device_id), str(command or ""), params)


def debug_trace_command(
    device_id: int,
    command: str,
    params: Optional[Dict[str, Any]] = None,
    watch_var_names: Optional[List[str]] = None,
    poll_interval_s: float = 0.5,
    timeout_s: float = 30.0,
) -> Json:
    return gateway.debug_trace_command(
        int(device_id),
        str(command or ""),
        params,
        watch_var_names=watch_var_names,
        poll_interval_s=float(poll_interval_s),
        timeout_s=float(timeout_s),
    )


# --------- Locks ---------

def lock_get_state(device_id: int) -> Json:
    return gateway.lock_get_state(int(device_id))


def lock_unlock(device_id: int) -> Json:
    return gateway.lock_unlock(int(device_id))


def lock_lock(device_id: int) -> Json:
    return gateway.lock_lock(int(device_id))


# --------- Lights (stubs unless you add implementations in gateway) ---------

def light_get_state(device_id: int) -> bool:
    # If you implement these later, this adapter will pass through.
    if hasattr(gateway, "light_get_state"):
        return bool(gateway.light_get_state(int(device_id)))  # type: ignore[misc]
    return False


def light_get_level(device_id: int) -> Union[int, Any]:
    if hasattr(gateway, "light_get_level"):
        return gateway.light_get_level(int(device_id))  # type: ignore[misc]
    return 0


def light_set_level(device_id: int, level: int) -> bool:
    if hasattr(gateway, "light_set_level"):
        return bool(gateway.light_set_level(int(device_id), int(level)))  # type: ignore[misc]
    return False


def light_ramp(device_id: int, level: int, time_ms: int) -> bool:
    if hasattr(gateway, "light_ramp"):
        return bool(gateway.light_ramp(int(device_id), int(level), int(time_ms)))  # type: ignore[misc]
    return False


# --------- Thermostats ---------

def thermostat_get_state(device_id: int) -> Json:
    return gateway.thermostat_get_state(int(device_id))  # type: ignore[misc]


def thermostat_set_hvac_mode(device_id: int, mode: str, confirm_timeout_s: float = 8.0) -> Json:
    return gateway.thermostat_set_hvac_mode(int(device_id), str(mode or ""), confirm_timeout_s=float(confirm_timeout_s))  # type: ignore[misc]


def thermostat_set_fan_mode(device_id: int, mode: str, confirm_timeout_s: float = 8.0) -> Json:
    return gateway.thermostat_set_fan_mode(int(device_id), str(mode or ""), confirm_timeout_s=float(confirm_timeout_s))  # type: ignore[misc]


def thermostat_set_hold_mode(device_id: int, mode: str, confirm_timeout_s: float = 8.0) -> Json:
    return gateway.thermostat_set_hold_mode(int(device_id), str(mode or ""), confirm_timeout_s=float(confirm_timeout_s))  # type: ignore[misc]


def thermostat_set_heat_setpoint_f(device_id: int, setpoint_f: float, confirm_timeout_s: float = 8.0) -> Json:
    return gateway.thermostat_set_heat_setpoint_f(int(device_id), float(setpoint_f), confirm_timeout_s=float(confirm_timeout_s))  # type: ignore[misc]


def thermostat_set_cool_setpoint_f(device_id: int, setpoint_f: float, confirm_timeout_s: float = 8.0) -> Json:
    return gateway.thermostat_set_cool_setpoint_f(int(device_id), float(setpoint_f), confirm_timeout_s=float(confirm_timeout_s))  # type: ignore[misc]


def thermostat_set_target_f(
    device_id: int,
    target_f: float,
    confirm_timeout_s: float = 10.0,
    deadband_f: float | None = None,
) -> Json:
    return gateway.thermostat_set_target_f(
        int(device_id),
        float(target_f),
        confirm_timeout_s=float(confirm_timeout_s),
        deadband_f=(float(deadband_f) if deadband_f is not None else None),
    )  # type: ignore[misc]