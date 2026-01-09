# control4_adapter.py
from __future__ import annotations

from control4_gateway import Control4Gateway

_gateway = Control4Gateway()  # single shared instance


def list_rooms():
    return _gateway.list_rooms()


def get_all_items():
    return _gateway.get_all_items()


def get_director():
    return _gateway.get_director()


def item_get_variables(device_id: int):
    return _gateway.item_get_variables(int(device_id))


# ---- Lights ----

def light_get_state(device_id: int) -> bool:
    return _gateway.light_get_state(int(device_id))


def light_set_level(device_id: int, level: int) -> bool:
    return _gateway.light_set_level(int(device_id), int(level))


def light_ramp(device_id: int, level: int, time_ms: int) -> bool:
    return _gateway.light_ramp(int(device_id), int(level), int(time_ms))


def light_get_level(device_id: int):
    return _gateway.light_get_level(int(device_id))


# ---- Locks ----

def lock_get_state(device_id: int):
    return _gateway.lock_get_state(int(device_id))


def lock_lock(device_id: int):
    return _gateway.lock_lock(int(device_id))


def lock_unlock(device_id: int):
    return _gateway.lock_unlock(int(device_id))