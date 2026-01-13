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
    # Your latest gateway (as formatted) doesn't implement get_all_items.
    # Keep adapter compatible: return [] unless you later add gateway.get_all_items().
    if hasattr(gateway, "get_all_items"):
        return gateway.get_all_items()  # type: ignore[return-value]
    return []


def item_get_variables(device_id: int) -> Any:
    """
    Raw variables for an item. The latest gateway code doesn't expose item_get_variables,
    but lock_get_state reads variables internally. Keep this adapter callable for MCP debug tool.
    """
    # If you later add gateway.item_get_variables, this will start working automatically.
    if hasattr(gateway, "item_get_variables"):
        return gateway.item_get_variables(int(device_id))  # type: ignore[misc]
    # Provide a helpful error shape instead of throwing
    return {"ok": False, "error": "item_get_variables not implemented on Control4Gateway"}


def item_get_commands(device_id: int) -> Json:
    return gateway.item_get_commands(int(device_id))


def item_execute_command(device_id: int, command_id: int) -> Json:
    return gateway.item_execute_command(int(device_id), int(command_id))


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