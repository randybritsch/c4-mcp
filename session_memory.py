from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
import uuid
from typing import Any, Optional


_LAST_LIGHTS_TOKENS = {
    "__last_lights__",
    "__those_lights__",
    "those_lights",
    "those lights",
    "those",
}


_LAST_TV_TOKENS = {
    "__last_tv__",
    "__last_media__",
    "__last_av__",
    "tv",
    "the tv",
    "it",
}


def is_last_tv_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _LAST_TV_TOKENS


def is_last_lights_token(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return value.strip().lower() in _LAST_LIGHTS_TOKENS


@dataclass
class SessionMemory:
    session_id: str
    created_at_s: float = field(default_factory=lambda: time.time())
    touched_at_s: float = field(default_factory=lambda: time.time())

    last_tool: Optional[str] = None
    last_tool_args: dict[str, Any] = field(default_factory=dict)
    last_tool_at_s: Optional[float] = None

    # each: {device_id:int, name?:str, room_name?:str, room_id?:int}
    last_lights: list[dict[str, Any]] = field(default_factory=list)
    last_lights_at_s: Optional[float] = None

    # {room_id:int, room_name?:str, source_device_id?:int, source_device_name?:str, device_id?:int, device_name?:str}
    last_tv: dict[str, Any] = field(default_factory=dict)
    last_tv_at_s: Optional[float] = None

    def touch(self) -> None:
        self.touched_at_s = time.time()

    def clear(self) -> None:
        self.last_tool = None
        self.last_tool_args = {}
        self.last_tool_at_s = None
        self.last_lights = []
        self.last_lights_at_s = None
        self.last_tv = {}
        self.last_tv_at_s = None
        self.touch()

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at_s": self.created_at_s,
            "touched_at_s": self.touched_at_s,
            "last_tool": self.last_tool,
            "last_tool_at_s": self.last_tool_at_s,
            "last_lights": self.last_lights,
            "last_lights_at_s": self.last_lights_at_s,
            "last_tv": self.last_tv,
            "last_tv_at_s": self.last_tv_at_s,
        }

    def _normalize_tv(self, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}

        out: dict[str, Any] = {}

        room_id = value.get("room_id")
        try:
            room_id_i = int(room_id)
        except Exception:
            room_id_i = None
        if room_id_i is not None and room_id_i > 0:
            out["room_id"] = room_id_i

        for k in ("room_name", "source_device_name", "device_name"):
            v = value.get(k)
            if isinstance(v, str) and v.strip():
                out[k] = v.strip()

        for k in ("source_device_id", "device_id"):
            v = value.get(k)
            if v is None:
                continue
            try:
                vi = int(v)
            except Exception:
                continue
            if vi > 0:
                out[k] = vi

        return out

    def set_last_tv(self, value: dict[str, Any]) -> None:
        self.last_tv = self._normalize_tv(dict(value or {}))
        self.last_tv_at_s = time.time()
        self.touch()

    def _dedupe_lights(self, lights: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[int] = set()
        out: list[dict[str, Any]] = []
        for row in lights:
            if not isinstance(row, dict):
                continue
            did = row.get("device_id")
            try:
                did_i = int(did)
            except Exception:
                continue
            if did_i <= 0 or did_i in seen:
                continue
            seen.add(did_i)
            merged: dict[str, Any] = {"device_id": did_i}
            for k in ("name", "room_name"):
                v = row.get(k)
                if isinstance(v, str) and v.strip():
                    merged[k] = v.strip()
            if row.get("room_id") is not None:
                try:
                    merged["room_id"] = int(row.get("room_id"))
                except Exception:
                    pass
            out.append(merged)
        return out

    def set_last_lights(self, lights: list[dict[str, Any]]) -> None:
        self.last_lights = self._dedupe_lights(list(lights or []))
        self.last_lights_at_s = time.time()
        self.touch()

    def add_last_lights(self, lights: list[dict[str, Any]], window_s: float = 5.0) -> None:
        now = time.time()
        if self.last_lights_at_s is not None and (now - float(self.last_lights_at_s)) <= float(window_s):
            self.set_last_lights(list(self.last_lights or []) + list(lights or []))
        else:
            self.set_last_lights(list(lights or []))


class SessionStore:
    def __init__(self, max_sessions: int = 200, ttl_s: float = 2 * 60 * 60) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionMemory] = {}
        self._max_sessions = int(max_sessions)
        self._ttl_s = float(ttl_s)

    def create(self) -> SessionMemory:
        return self.get(str(uuid.uuid4()), create=True)

    def get(self, session_id: str, create: bool = True) -> SessionMemory:
        sid = str(session_id or "").strip()
        if not sid:
            sid = str(uuid.uuid4())
        now = time.time()
        with self._lock:
            self._prune_locked(now)
            mem = self._sessions.get(sid)
            if mem is None:
                if not create:
                    mem = SessionMemory(session_id=sid)
                else:
                    mem = SessionMemory(session_id=sid)
                self._sessions[sid] = mem
                self._evict_if_needed_locked()
            mem.touch()
            return mem

    def clear(self, session_id: str) -> None:
        with self._lock:
            mem = self._sessions.get(str(session_id or "").strip())
            if mem is not None:
                mem.clear()

    def _evict_if_needed_locked(self) -> None:
        if len(self._sessions) <= self._max_sessions:
            return
        # Evict least-recently-touched.
        items = sorted(self._sessions.values(), key=lambda m: float(m.touched_at_s))
        while len(items) > self._max_sessions:
            victim = items.pop(0)
            self._sessions.pop(victim.session_id, None)

    def _prune_locked(self, now: float) -> None:
        ttl = self._ttl_s
        expired = [sid for sid, mem in self._sessions.items() if (now - float(mem.touched_at_s)) > ttl]
        for sid in expired:
            self._sessions.pop(sid, None)


def extract_lights_from_call(tool_name: str, arguments: dict[str, Any], value: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def add(device_id: Any, name: Any = None, room_name: Any = None, room_id: Any = None) -> None:
        try:
            did = int(device_id)
        except Exception:
            return
        if did <= 0:
            return
        row: dict[str, Any] = {"device_id": did}
        if isinstance(name, str) and name.strip():
            row["name"] = name.strip()
        if isinstance(room_name, str) and room_name.strip():
            row["room_name"] = room_name.strip()
        if room_id is not None:
            try:
                row["room_id"] = int(room_id)
            except Exception:
                pass
        out.append(row)

    tn = str(tool_name or "").lower()
    args = arguments if isinstance(arguments, dict) else {}

    if "device_id" in args and any(k in tn for k in ("light", "outlet")):
        add(args.get("device_id"), name=args.get("device_name") or args.get("name"), room_name=args.get("room_name"), room_id=args.get("room_id"))

    if isinstance(value, dict):
        if "device_id" in value and any(k in tn for k in ("light", "outlet")):
            add(value.get("device_id"), name=value.get("device_name") or value.get("name"), room_name=value.get("room_name"), room_id=value.get("room_id"))

        exec_obj = value.get("execute") if isinstance(value.get("execute"), dict) else None
        if exec_obj is not None:
            devices = exec_obj.get("devices") if isinstance(exec_obj.get("devices"), list) else None
            if devices is not None and any(k in tn for k in ("room_lights_set", "lights_set")):
                for d in devices:
                    if not isinstance(d, dict):
                        continue
                    add(d.get("device_id"), name=d.get("name"), room_name=d.get("room_name"), room_id=d.get("room_id"))

        devices2 = value.get("devices") if isinstance(value.get("devices"), list) else None
        if devices2 is not None and "lights" in tn:
            for d in devices2:
                if not isinstance(d, dict):
                    continue
                add(d.get("device_id"), name=d.get("name"), room_name=d.get("room_name"), room_id=d.get("room_id"))

    return out


def extract_tv_from_call(tool_name: str, arguments: dict[str, Any], value: Any) -> dict[str, Any]:
    """Best-effort extraction of the room context for TV/media follow-ups.

    The main goal is to capture a usable room_id so follow-up tools can do things like
    'turn off the TV' without re-resolving the room.
    """

    tn = str(tool_name or "").lower()
    args = arguments if isinstance(arguments, dict) else {}

    # Only attempt for likely AV tools.
    if not any(k in tn for k in ("tv_", "media_", "room_select_video_device", "room_off", "room_listen")):
        return {}

    out: dict[str, Any] = {}

    def set_room(room_id: Any, room_name: Any = None) -> None:
        try:
            rid = int(room_id)
        except Exception:
            return
        if rid <= 0:
            return
        out["room_id"] = rid
        if isinstance(room_name, str) and room_name.strip():
            out["room_name"] = room_name.strip()

    # 1) Args commonly contain room_id.
    if "room_id" in args and args.get("room_id") is not None:
        set_room(args.get("room_id"), args.get("room_name"))

    # 2) Some tools return room_id / room_name at the top-level.
    if not out and isinstance(value, dict):
        if value.get("room_id") is not None:
            set_room(value.get("room_id"), value.get("room_name"))

    # 3) Planned payload is common in helper tools.
    if (not out) and isinstance(value, dict):
        planned = value.get("planned") if isinstance(value.get("planned"), dict) else None
        if planned is not None and planned.get("room_id") is not None:
            set_room(planned.get("room_id"), value.get("room_name"))

    # 4) Resolve payloads may include room_id.
    if (not out) and isinstance(value, dict):
        rr = value.get("resolve_room") if isinstance(value.get("resolve_room"), dict) else None
        if rr is not None and rr.get("room_id") is not None:
            set_room(rr.get("room_id"), rr.get("name") or rr.get("room_name"))

    # Optional extra detail (best-effort)
    if isinstance(value, dict):
        if value.get("source_device_name") is not None and isinstance(value.get("source_device_name"), str):
            out.setdefault("source_device_name", str(value.get("source_device_name")).strip())
        planned = value.get("planned") if isinstance(value.get("planned"), dict) else None
        if planned is not None:
            if planned.get("source_device_id") is not None:
                out.setdefault("source_device_id", planned.get("source_device_id"))
            if planned.get("device_id") is not None:
                out.setdefault("device_id", planned.get("device_id"))

    # Also capture obvious arg ids.
    if args.get("source_device_id") is not None:
        out.setdefault("source_device_id", args.get("source_device_id"))
    if args.get("device_id") is not None:
        out.setdefault("device_id", args.get("device_id"))

    # Also capture obvious arg names.
    if "source_device_name" in args and isinstance(args.get("source_device_name"), str):
        out.setdefault("source_device_name", str(args.get("source_device_name")).strip())
    if "device_name" in args and isinstance(args.get("device_name"), str):
        out.setdefault("device_name", str(args.get("device_name")).strip())

    return out


# ---- Optional MCP tool registrations (TV follow-ups) ----
#
# These tools intentionally live here so we can deploy TV follow-up behavior
# without having to rewrite/replace a large `app.py` on constrained targets.
#
# They reuse the session store created by the main `app.py` process (run as
# `python app.py` in Docker) by reaching into `sys.modules['__main__']`.


def _get_main_module() -> Any:
    import sys

    return sys.modules.get("__main__")


_FALLBACK_STORE: SessionStore | None = None


def _get_session_store() -> SessionStore:
    main = _get_main_module()
    store = getattr(main, "_SESSION_STORE", None)
    if isinstance(store, SessionStore):
        return store

    global _FALLBACK_STORE
    if _FALLBACK_STORE is None:
        _FALLBACK_STORE = SessionStore()
    return _FALLBACK_STORE


def _get_current_session_id(explicit: str | None = None) -> str:
    main = _get_main_module()
    fn = getattr(main, "_current_session_id", None)
    if callable(fn):
        try:
            return str(fn(explicit))
        except Exception:
            pass
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    return str(uuid.uuid4())


def _infer_last_room_from_memory(mem: SessionMemory) -> tuple[int | None, str | None, dict[str, Any] | None]:
    # 1) Preferred: already remembered TV room.
    if isinstance(mem.last_tv, dict) and mem.last_tv.get("room_id") is not None:
        try:
            return int(mem.last_tv.get("room_id")), mem.last_tv.get("room_name"), None
        except Exception:
            pass

    # 2) Try prior tool args (commonly from c4_tv_watch_by_name or other room-scoped tools).
    args = mem.last_tool_args if isinstance(mem.last_tool_args, dict) else {}
    room_id = args.get("room_id")
    if room_id is not None:
        try:
            return int(room_id), args.get("room_name"), None
        except Exception:
            pass

    room_name = args.get("room_name") or args.get("room")
    if isinstance(room_name, str) and room_name.strip():
        # Resolve room name to id.
        try:
            from control4_adapter import resolve_room

            rr = resolve_room(str(room_name).strip(), require_unique=True, include_candidates=True)
            if isinstance(rr, dict) and rr.get("ok") is True and rr.get("room_id") is not None:
                try:
                    return int(rr.get("room_id")), rr.get("name") or rr.get("room_name") or str(room_name).strip(), None
                except Exception:
                    return None, None, {"ok": False, "error": "bad room_id from resolver", "resolve_room": rr}
            return None, None, rr if isinstance(rr, dict) else {"ok": False, "error": "resolve_room failed"}
        except Exception as e:
            return None, None, {"ok": False, "error": repr(e)}

    return None, None, None


try:
    from flask_mcp_server import Mcp

    @Mcp.tool(
        name="c4_tv_get_last",
        description=(
            "Return the last referenced TV/media room context in this session (for follow-ups like 'turn off the TV')."
        ),
    )
    def c4_tv_get_last_tool(session_id: str | None = None) -> dict[str, Any]:
        sid = _get_current_session_id(session_id)
        mem = _get_session_store().get(sid, create=True)
        return {"ok": True, "session_id": sid, "tv": dict(mem.last_tv or {})}

    @Mcp.tool(
        name="c4_tv_off_last",
        description=(
            "Turn off the last referenced TV/media room in this session (follow-up for commands like 'turn off the TV')."
        ),
    )
    def c4_tv_off_last_tool(confirm_timeout_s: float = 10.0, session_id: str | None = None) -> dict[str, Any]:
        sid = _get_current_session_id(session_id)
        mem = _get_session_store().get(sid, create=True)

        room_id, room_name, resolve_err = _infer_last_room_from_memory(mem)
        if room_id is None:
            return {
                "ok": False,
                "error": "no remembered TV/media room in this session yet",
                "session_id": sid,
                "details": resolve_err,
            }

        # Persist for future follow-ups.
        mem.set_last_tv({"room_id": int(room_id), "room_name": room_name})

        from control4_adapter import room_off

        result = room_off(int(room_id), float(confirm_timeout_s))
        return result if isinstance(result, dict) else {"ok": True, "result": result}

    @Mcp.tool(
        name="c4_tv_remote_last",
        description=(
            "Send a room-level remote command (pause/play/mute/volume/navigation) to the last referenced TV/media room in this session."
        ),
    )
    def c4_tv_remote_last_tool(
        button: str,
        press: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        sid = _get_current_session_id(session_id)
        mem = _get_session_store().get(sid, create=True)

        room_id, room_name, resolve_err = _infer_last_room_from_memory(mem)
        if room_id is None:
            return {
                "ok": False,
                "error": "no remembered TV/media room in this session yet",
                "session_id": sid,
                "details": resolve_err,
            }

        # Persist for future follow-ups.
        mem.set_last_tv({"room_id": int(room_id), "room_name": room_name})

        from control4_adapter import room_remote

        result = room_remote(int(room_id), str(button or ""), (str(press) if press is not None else None))
        return result if isinstance(result, dict) else {"ok": True, "result": result}

except Exception:
    # If flask_mcp_server isn't available (e.g., unit tests), just skip tool registration.
    pass
