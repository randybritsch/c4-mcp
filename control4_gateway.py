# control4_gateway.py
from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from pyControl4.account import C4Account
from pyControl4.director import C4Director


class AsyncLoopThread:
    """Owns exactly one asyncio event loop running forever in a daemon thread."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._thread.start()
            self._started = True

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro) -> Any:
        if not self._started:
            raise RuntimeError("AsyncLoopThread not started")
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()


@dataclass(frozen=True)
class Config:
    host: str
    username: str
    password: str


class Control4Gateway:
    """
    Sync facade for Flask/MCP tools.
    Internally schedules ALL pyControl4 coroutines on a single asyncio loop thread.
    """

    def __init__(self, cfg_path: Optional[str] = None, token_ttl_s: int = 3600) -> None:
        self._cfg = self._load_config(cfg_path)
        self._token_ttl_s = token_ttl_s

        self._loop_thread = AsyncLoopThread()
        self._loop_thread.start()

        self._director_token: Optional[str] = None
        self._director_token_time: float = 0.0
        self._controller_name: Optional[str] = None

    # ---------- config / auth ----------

    def _load_config(self, cfg_path: Optional[str]) -> Config:
        path = Path(cfg_path) if cfg_path else Path(__file__).with_name("config.json")
        data = json.loads(path.read_text(encoding="utf-8"))
        return Config(host=data["host"], username=data["username"], password=data["password"])

    def _token_valid(self) -> bool:
        return bool(self._director_token) and (time.time() - self._director_token_time) < self._token_ttl_s

    async def _ensure_director_token_async(self) -> str:
        if self._token_valid():
            return self._director_token  # type: ignore[return-value]

        account = C4Account(self._cfg.username, self._cfg.password)
        await account.getAccountBearerToken()

        controller_info = await account.getAccountControllers()
        self._controller_name = controller_info["controllerCommonName"]

        token_resp = await account.getDirectorBearerToken(self._controller_name)
        self._director_token = token_resp["token"]
        self._director_token_time = time.time()
        return self._director_token

    async def _director_async(self) -> C4Director:
        token = await self._ensure_director_token_async()
        return C4Director(self._cfg.host, token)

    # ---------- discovery ----------

    def list_rooms(self) -> list[dict[str, Any]]:
        async def _run():
            director = await self._director_async()
            items = await director.getAllItemInfo()
            items_obj = json.loads(items) if isinstance(items, str) else items
            rooms = [i for i in items_obj if isinstance(i, dict) and i.get("typeName") == "room"]
            return [{"id": str(r["id"]), "name": r.get("name"), "parentId": str(r.get("parentId"))} for r in rooms]

        return self._loop_thread.run(_run())

    def get_all_items(self) -> list[dict[str, Any]]:
        async def _run():
            director = await self._director_async()
            items = await director.getAllItemInfo()
            return json.loads(items) if isinstance(items, str) else items

        return self._loop_thread.run(_run())

    def get_director(self) -> C4Director:
        async def _run():
            return await self._director_async()

        return self._loop_thread.run(_run())

    # ---------- generic variable debug ----------

    def item_get_variables(self, device_id: int):
        async def _run():
            director = await self._director_async()
            if not hasattr(director, "getItemVariables"):
                return {"_error": "Director does not support getItemVariables()"}

            try:
                vars_ = await director.getItemVariables(int(device_id))
                if isinstance(vars_, str):
                    try:
                        vars_ = json.loads(vars_)
                    except Exception:
                        return {"_error": "Variables were a string but not valid JSON", "variables": vars_}
                return vars_
            except Exception as e:
                return {
                    "_error": str(e),
                    "_error_type": type(e).__name__,
                    "_device_id": device_id
                }

        return self._loop_thread.run(_run())

    # ---------- lights ----------

    def light_get_state(self, device_id: int) -> bool:
        async def _run():
            director = await self._director_async()
            from pyControl4.light import C4Light

            light = C4Light(director, int(device_id))
            state = await light.getState()
            return bool(state)

        return self._loop_thread.run(_run())

    def light_set_level(self, device_id: int, level: int) -> bool:
        level = int(level)

        async def _run():
            director = await self._director_async()
            from pyControl4.light import C4Light

            light = C4Light(director, int(device_id))
            await light.setLevel(level)
            state = await light.getState()
            return bool(state)

        return self._loop_thread.run(_run())

    def light_ramp(self, device_id: int, level: int, time_ms: int) -> bool:
        level = int(level)
        time_ms = int(time_ms)

        async def _run():
            director = await self._director_async()
            from pyControl4.light import C4Light

            light = C4Light(director, int(device_id))
            await light.rampToLevel(level, time_ms)
            state = await light.getState()
            return bool(state)

        return self._loop_thread.run(_run())

    def light_get_level(self, device_id: int):
        """
        Returns:
          - int 0-100 when we can determine it, OR
          - dict with variables/notes when we cannot
        """

        async def _run():
            director = await self._director_async()
            from pyControl4.light import C4Light
            import asyncio as _asyncio
            import json as _json

            light = C4Light(director, int(device_id))

            # 1) Best: getLevel() exists and works
            if hasattr(light, "getLevel"):
                try:
                    lvl = await light.getLevel()
                    if isinstance(lvl, (int, float)):
                        return max(0, min(100, int(lvl)))
                except Exception:
                    pass

            # 2) Fallback: item variables
            if not hasattr(director, "getItemVariables"):
                return {"_error": "Director does not support getItemVariables()"}

            last_err = None
            for _ in range(4):
                try:
                    vars_ = await director.getItemVariables(int(device_id))

                    if isinstance(vars_, str):
                        try:
                            vars_ = _json.loads(vars_)
                        except Exception:
                            return {"_error": "Variables were a string but not valid JSON", "variables": vars_}

                    if isinstance(vars_, list):
                        for row in vars_:
                            if isinstance(row, dict) and str(row.get("varName", "")).strip().lower() == "brightness percent":
                                val = row.get("value")
                                try:
                                    return max(0, min(100, int(float(val))))
                                except Exception:
                                    pass

                        for row in vars_:
                            if isinstance(row, dict) and str(row.get("varName", "")).strip().upper() == "PRESET_LEVEL":
                                val = row.get("value")
                                try:
                                    return max(0, min(100, int(float(val))))
                                except Exception:
                                    pass

                        return {"_note": "No level variable found", "variables": vars_}

                    if vars_:
                        return {"_note": f"Unexpected variables type: {type(vars_).__name__}", "variables": vars_}

                except Exception as e:
                    last_err = e

                await _asyncio.sleep(0.15)

            return {"_error": "Failed/empty variable response", "_exception": repr(last_err)}

        return self._loop_thread.run(_run())

    # ---------- locks ----------

    def lock_get_state(self, device_id: int) -> dict[str, Any]:
        """
        Returns:
          {
            device_id: int,
            locked: True|False|None,
            state: "locked"|"unlocked"|"unknown",
            source: "variables",
            raw: ...
          }
        """
        device_id = int(device_id)

        async def _run():
            director = await self._director_async()
            if not hasattr(director, "getItemVariables"):
                return {
                    "device_id": device_id,
                    "locked": None,
                    "state": "unknown",
                    "source": "variables",
                    "raw": {"_error": "Director does not support getItemVariables()"},
                }

            try:
                vars_ = await director.getItemVariables(device_id)
            except Exception as e:
                return {
                    "device_id": device_id,
                    "locked": None,
                    "state": "unknown",
                    "source": "variables",
                    "raw": {
                        "_error": str(e),
                        "_error_type": type(e).__name__
                    },
                }

            if isinstance(vars_, str):
                try:
                    vars_ = json.loads(vars_)
                except Exception:
                    return {
                        "device_id": device_id,
                        "locked": None,
                        "state": "unknown",
                        "source": "variables",
                        "raw": {"_error": "Variables were a string but not valid JSON", "variables": vars_},
                    }

            if not isinstance(vars_, list):
                return {
                    "device_id": device_id,
                    "locked": None,
                    "state": "unknown",
                    "source": "variables",
                    "raw": {"_note": f"Unexpected variables type: {type(vars_).__name__}", "variables": vars_},
                }

            # Prefer LockStatus (lock proxy)
            for row in vars_:
                if isinstance(row, dict) and str(row.get("varName", "")).strip().lower() == "lockstatus":
                    val = str(row.get("value", "")).strip().lower()
                    if val == "locked":
                        return {"device_id": device_id, "locked": True, "state": "locked", "source": "variables", "raw": row}
                    if val == "unlocked":
                        return {"device_id": device_id, "locked": False, "state": "unlocked", "source": "variables", "raw": row}
                    return {"device_id": device_id, "locked": None, "state": "unknown", "source": "variables", "raw": row}

            # Fallback RelayState (relay-style lock): 0=locked, 1=unlocked
            for row in vars_:
                if isinstance(row, dict) and str(row.get("varName", "")).strip().lower() == "relaystate":
                    try:
                        v = int(row.get("value"))
                        if v == 0:
                            return {"device_id": device_id, "locked": True, "state": "locked", "source": "variables", "raw": row}
                        if v == 1:
                            return {"device_id": device_id, "locked": False, "state": "unlocked", "source": "variables", "raw": row}
                    except Exception:
                        pass
                    return {"device_id": device_id, "locked": None, "state": "unknown", "source": "variables", "raw": row}

            return {
                "device_id": device_id,
                "locked": None,
                "state": "unknown",
                "source": "variables",
                "raw": {"_note": "No lock state variable found", "variables": vars_},
            }

        return self._loop_thread.run(_run())

    def lock_lock(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)

        async def _run():
            director = await self._director_async()
            from pyControl4.relay import C4Relay
            import asyncio as _asyncio

            relay = C4Relay(director, device_id)

            # For locks via relay: OPEN means lock (per pyControl4 docs)
            # (C4Relay.open/close/toggle POSTs "OPEN"/"CLOSE"/"TOGGLE" to /api/v1/items/{id}/commands)
            try:
                await relay.open()
                execute_ok = True
                execute_err = None
            except Exception as e:
                execute_ok = False
                execute_err = repr(e)

            # Give the director a moment to update variables
            await _asyncio.sleep(0.35)

            after = self.lock_get_state(device_id)
            success = bool(after.get("locked") is True)

            return {
                "device_id": device_id,
                "requested": "lock",
                "success": success,
                "after": after,
                "raw": {"execute_ok": execute_ok, "execute_error": execute_err},
            }

        return self._loop_thread.run(_run())

    def lock_unlock(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)

        async def _run():
            director = await self._director_async()
            from pyControl4.relay import C4Relay
            import asyncio as _asyncio

            relay = C4Relay(director, device_id)

            # For locks via relay: CLOSE means unlock (per pyControl4 docs)
            try:
                await relay.close()
                execute_ok = True
                execute_err = None
            except Exception as e:
                execute_ok = False
                execute_err = repr(e)

            await _asyncio.sleep(0.35)

            after = self.lock_get_state(device_id)
            success = bool(after.get("locked") is False)

            return {
                "device_id": device_id,
                "requested": "unlock",
                "success": success,
                "after": after,
                "raw": {"execute_ok": execute_ok, "execute_error": execute_err},
            }

        return self._loop_thread.run(_run())