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

    def touch(self) -> None:
        self.touched_at_s = time.time()

    def clear(self) -> None:
        self.last_tool = None
        self.last_tool_args = {}
        self.last_tool_at_s = None
        self.last_lights = []
        self.last_lights_at_s = None
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
        }

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
