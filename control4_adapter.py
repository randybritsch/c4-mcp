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


def find_rooms(search: str, limit: int = 10, include_raw: bool = False) -> Json:
    return gateway.find_rooms(str(search or ""), limit=int(limit), include_raw=bool(include_raw))  # type: ignore[misc]


def resolve_room(name: str, require_unique: bool = True, include_candidates: bool = True) -> Json:
    return gateway.resolve_room(str(name or ""), require_unique=bool(require_unique), include_candidates=bool(include_candidates))  # type: ignore[misc]


def find_devices(
    search: str | None = None,
    category: str | None = None,
    room_id: int | None = None,
    limit: int = 20,
    include_raw: bool = False,
) -> Json:
    return gateway.find_devices(
        (str(search) if search is not None else None),
        (str(category) if category is not None else None),
        (int(room_id) if room_id is not None else None),
        limit=int(limit),
        include_raw=bool(include_raw),
    )  # type: ignore[misc]


def resolve_device(
    name: str,
    category: str | None = None,
    room_id: int | None = None,
    require_unique: bool = True,
    include_candidates: bool = True,
) -> Json:
    return gateway.resolve_device(
        str(name or ""),
        category=(str(category) if category is not None else None),
        room_id=(int(room_id) if room_id is not None else None),
        require_unique=bool(require_unique),
        include_candidates=bool(include_candidates),
    )  # type: ignore[misc]


def resolve_room_and_device(
    room_name: str | None = None,
    device_name: str | None = None,
    category: str | None = None,
    require_unique: bool = True,
    include_candidates: bool = True,
) -> Json:
    return gateway.resolve_room_and_device(
        (str(room_name) if room_name is not None else None),
        (str(device_name) if device_name is not None else None),
        (str(category) if category is not None else None),
        require_unique=bool(require_unique),
        include_candidates=bool(include_candidates),
    )  # type: ignore[misc]


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


def room_send_command(room_id: int, command: str, params: Optional[Dict[str, Any]] = None) -> Json:
    return gateway.room_send_command(int(room_id), str(command or ""), params)  # type: ignore[misc]


def room_list_commands(room_id: int, search: str | None = None) -> Json:
    return gateway.room_list_commands(int(room_id), (str(search) if search is not None else None))  # type: ignore[misc]


def capabilities_report(top_n: int = 20, include_examples: bool = False, max_examples_per_bucket: int = 3) -> Json:
    return gateway.capabilities_report(int(top_n), bool(include_examples), int(max_examples_per_bucket))  # type: ignore[misc]


def uibutton_activate(device_id: int, command: str | None = None, dry_run: bool = False) -> Json:
    return gateway.uibutton_activate(int(device_id), (str(command) if command is not None else None), bool(dry_run))  # type: ignore[misc]


def contact_get_state(device_id: int, timeout_s: float = 6.0) -> Json:
    return gateway.contact_get_state(int(device_id), timeout_s=float(timeout_s))  # type: ignore[misc]


def keypad_list() -> Json:
    return gateway.keypad_list()  # type: ignore[misc]


def keypad_get_buttons(device_id: int) -> Json:
    return gateway.keypad_get_buttons(int(device_id))  # type: ignore[misc]


def keypad_button_action(device_id: int, button_id: int, action: str = "tap", tap_ms: int = 200, dry_run: bool = False) -> Json:
    return gateway.keypad_button_action(int(device_id), int(button_id), str(action or ""), int(tap_ms), bool(dry_run))  # type: ignore[misc]


def control_keypad_list() -> Json:
    return gateway.control_keypad_list()  # type: ignore[misc]


def control_keypad_send_command(device_id: int, command: str, dry_run: bool = False) -> Json:
    return gateway.control_keypad_send_command(int(device_id), str(command or ""), bool(dry_run))  # type: ignore[misc]


def fan_list() -> Json:
    return gateway.fan_list()  # type: ignore[misc]


def fan_get_state(device_id: int) -> Json:
    return gateway.fan_get_state(int(device_id))  # type: ignore[misc]


def fan_set_speed(device_id: int, speed: Any, confirm_timeout_s: float = 4.0, dry_run: bool = False) -> Json:
    return gateway.fan_set_speed(int(device_id), speed, confirm_timeout_s=float(confirm_timeout_s), dry_run=bool(dry_run))  # type: ignore[misc]


def fan_set_power(device_id: int, power: str, confirm_timeout_s: float = 4.0, dry_run: bool = False) -> Json:
    return gateway.fan_set_power(int(device_id), str(power or ""), confirm_timeout_s=float(confirm_timeout_s), dry_run=bool(dry_run))  # type: ignore[misc]


def motion_list() -> Json:
    return gateway.motion_list()  # type: ignore[misc]


def motion_get_state(device_id: int, timeout_s: float = 6.0) -> Json:
    return gateway.motion_get_state(int(device_id), timeout_s=float(timeout_s))  # type: ignore[misc]


# --------- Intercom (best-effort) ---------

def intercom_list() -> Json:
    return gateway.intercom_list()  # type: ignore[misc]


def intercom_touchscreen_set_feature(device_id: int, feature: str, enabled: bool, dry_run: bool = False) -> Json:
    return gateway.intercom_touchscreen_set_feature(int(device_id), str(feature or ""), bool(enabled), bool(dry_run))  # type: ignore[misc]


def intercom_touchscreen_screensaver(
    device_id: int,
    action: str | None = None,
    mode: str | None = None,
    start_time_s: int | None = None,
    dry_run: bool = False,
) -> Json:
    return gateway.intercom_touchscreen_screensaver(
        int(device_id),
        (str(action) if action is not None else None),
        (str(mode) if mode is not None else None),
        (int(start_time_s) if start_time_s is not None else None),
        bool(dry_run),
    )  # type: ignore[misc]


def doorstation_set_led(device_id: int, enabled: bool, dry_run: bool = False) -> Json:
    return gateway.intercom_doorstation_set_led(int(device_id), bool(enabled), bool(dry_run))  # type: ignore[misc]


def doorstation_set_external_chime(device_id: int, enabled: bool, dry_run: bool = False) -> Json:
    return gateway.intercom_doorstation_set_external_chime(int(device_id), bool(enabled), bool(dry_run))  # type: ignore[misc]


def doorstation_set_raw_setting(device_id: int, key: str, value: str, dry_run: bool = False) -> Json:
    return gateway.intercom_doorstation_set_raw_setting(int(device_id), str(key or ""), str(value or ""), bool(dry_run))  # type: ignore[misc]


# --------- Macros (Agent) ---------

def macro_list() -> Json:
    return gateway.macro_list()  # type: ignore[misc]


def macro_list_commands() -> Json:
    return gateway.macro_list_commands()  # type: ignore[misc]


def macro_execute(macro_id: int, dry_run: bool = False) -> Json:
    return gateway.macro_execute(int(macro_id), bool(dry_run))  # type: ignore[misc]


def macro_execute_by_name(name: str, dry_run: bool = False) -> Json:
    return gateway.macro_execute_by_name(str(name or ""), bool(dry_run))  # type: ignore[misc]


# --------- Scheduler (Agent) ---------

def scheduler_list(search: str | None = None) -> Json:
    return gateway.scheduler_list((str(search) if search is not None else None))  # type: ignore[misc]


def scheduler_get(event_id: int) -> Json:
    return gateway.scheduler_get(int(event_id))  # type: ignore[misc]


def scheduler_list_commands() -> Json:
    return gateway.scheduler_list_commands()  # type: ignore[misc]


def scheduler_set_enabled(event_id: int, enabled: bool, dry_run: bool = False) -> Json:
    return gateway.scheduler_set_enabled(int(event_id), bool(enabled), bool(dry_run))  # type: ignore[misc]


# --------- Announcements ---------

def announcement_list() -> Json:
    return gateway.announcement_list()  # type: ignore[misc]


def announcement_list_commands() -> Json:
    return gateway.announcement_list_commands()  # type: ignore[misc]


def announcement_execute(announcement_id: int, dry_run: bool = False) -> Json:
    return gateway.announcement_execute(int(announcement_id), bool(dry_run))  # type: ignore[misc]


def announcement_execute_by_name(name: str, dry_run: bool = False) -> Json:
    return gateway.announcement_execute_by_name(str(name or ""), bool(dry_run))  # type: ignore[misc]


def room_select_video_device(room_id: int, device_id: int, deselect: bool = False) -> Json:
    return gateway.room_select_video_device(int(room_id), int(device_id), bool(deselect))  # type: ignore[misc]


def room_select_audio_device(room_id: int, device_id: int, deselect: bool = False) -> Json:
    return gateway.room_select_audio_device(int(room_id), int(device_id), bool(deselect))  # type: ignore[misc]


def room_listen(room_id: int, device_id: int, confirm_timeout_s: float = 10.0) -> Json:
    return gateway.room_listen(int(room_id), int(device_id), confirm_timeout_s=float(confirm_timeout_s))  # type: ignore[misc]


def room_listen_status(room_id: int) -> dict[str, Any]:
    return gateway.room_listen_status(room_id=int(room_id))


def room_now_playing(room_id: int, max_sources: int = 30) -> Json:
    return gateway.room_now_playing(int(room_id), int(max_sources))  # type: ignore[misc]

def room_off(room_id: int, confirm_timeout_s: float = 10.0) -> Json:
    return gateway.room_off(int(room_id), confirm_timeout_s=float(confirm_timeout_s))  # type: ignore[misc]


def room_remote(room_id: int, button: str, press: str | None = None) -> Json:
    return gateway.room_remote(int(room_id), str(button or ""), press)  # type: ignore[misc]


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

def shade_list(limit: int = 200) -> Json:
    return gateway.shade_list(limit=int(limit))  # type: ignore[misc]

def shade_get_state(device_id: int) -> Json:
    return gateway.shade_get_state(int(device_id))  # type: ignore[misc]

def shade_open(device_id: int, confirm_timeout_s: float = 6.0, dry_run: bool = False) -> Json:
    return gateway.shade_open(int(device_id), confirm_timeout_s=float(confirm_timeout_s), dry_run=bool(dry_run))  # type: ignore[misc]

def shade_close(device_id: int, confirm_timeout_s: float = 6.0, dry_run: bool = False) -> Json:
    return gateway.shade_close(int(device_id), confirm_timeout_s=float(confirm_timeout_s), dry_run=bool(dry_run))  # type: ignore[misc]

def shade_stop(device_id: int, dry_run: bool = False) -> Json:
    return gateway.shade_stop(int(device_id), dry_run=bool(dry_run))  # type: ignore[misc]

def shade_set_position(device_id: int, position: int, confirm_timeout_s: float = 8.0, dry_run: bool = False) -> Json:
    return gateway.shade_set_position(
        int(device_id),
        int(position),
        confirm_timeout_s=float(confirm_timeout_s),
        dry_run=bool(dry_run),
    )  # type: ignore[misc]


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


# --------- Media / AV ---------

def media_get_state(device_id: int) -> Json:
    return gateway.media_get_state(int(device_id))  # type: ignore[misc]


def media_send_command(device_id: int, command: str, params: Optional[Dict[str, Any]] = None) -> Json:
    return gateway.media_send_command(int(device_id), str(command or ""), params)  # type: ignore[misc]


def media_remote(device_id: int, button: str, press: str | None = None) -> Json:
    return gateway.media_remote(int(device_id), str(button or ""), press)  # type: ignore[misc]


def media_get_now_playing(device_id: int) -> Json:
    return gateway.media_get_now_playing(int(device_id))  # type: ignore[misc]


def media_remote_sequence(device_id: int, buttons: list[str], press: str | None = None, delay_ms: int = 250) -> Json:
    return gateway.media_remote_sequence(int(device_id), list(buttons), press, int(delay_ms))  # type: ignore[misc]


def media_launch_app(device_id: int, app: str) -> Json:
    return gateway.media_launch_app(int(device_id), str(app or ""))  # type: ignore[misc]


def media_watch_launch_app(device_id: int, app: str, room_id: int | None = None, pre_home: bool = True) -> Json:
    return gateway.media_watch_launch_app(int(device_id), str(app or ""), room_id=(int(room_id) if room_id is not None else None), pre_home=bool(pre_home))  # type: ignore[misc]


def media_roku_list_apps(device_id: int, search: str | None = None) -> Json:
    return gateway.media_roku_list_apps(int(device_id), (str(search) if search is not None else None))  # type: ignore[misc]