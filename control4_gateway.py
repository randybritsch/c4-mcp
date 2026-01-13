# control4_gateway.py

from __future__ import annotations

import asyncio
import inspect
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin

import aiohttp
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

    def run(self, coro, timeout_s: float | None = None) -> Any:
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout_s)


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

    def __init__(
        self,
        cfg_path: Optional[str] = None,
        token_ttl_s: int = 3600,
        auth_timeout_s: float = 10.0,
        director_timeout_s: float = 8.0,
        http_timeout_s: float = 6.0,
    ) -> None:
        self._cfg = self._load_config(cfg_path)
        self._token_ttl_s = int(token_ttl_s)
        self._auth_timeout_s = float(auth_timeout_s)
        self._director_timeout_s = float(director_timeout_s)
        self._http_timeout_s = float(http_timeout_s)

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

    async def _with_retries(self, label: str, fn, timeout_s: float, retries: int = 3):
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                return await asyncio.wait_for(fn(), timeout=timeout_s)
            except asyncio.TimeoutError:
                last_exc = RuntimeError(
                    f"Timeout: {label} (attempt {attempt}/{retries}, timeout={timeout_s}s)"
                )
            except Exception as e:
                last_exc = e
            await asyncio.sleep(0.5 * (2 ** (attempt - 1)))
        raise RuntimeError(f"{label} failed after {retries} attempts: {last_exc!r}")

    async def _ensure_director_token_async(self) -> str:
        if self._token_valid():
            return self._director_token  # type: ignore[return-value]

        account = C4Account(self._cfg.username, self._cfg.password)

        # 1) Cloud bearer token (can be slow)
        await self._with_retries(
            "getAccountBearerToken",
            lambda: account.getAccountBearerToken(),
            timeout_s=30,
            retries=3,
        )

        # 2) Controllers (cache chosen controller)
        if not self._controller_name:
            controller_info = await self._with_retries(
                "getAccountControllers",
                lambda: account.getAccountControllers(),
                timeout_s=45,
                retries=3,
            )
            # NOTE: this assumes dict response with controllerCommonName
            self._controller_name = controller_info["controllerCommonName"]

        # 3) Director bearer token
        token_resp = await self._with_retries(
            "getDirectorBearerToken",
            lambda: account.getDirectorBearerToken(self._controller_name),  # type: ignore[arg-type]
            timeout_s=45,
            retries=3,
        )

        # NOTE: this assumes dict response with "token"
        self._director_token = token_resp["token"]
        self._director_token_time = time.time()
        return self._director_token

    async def _director_async(self) -> C4Director:
        token = await self._ensure_director_token_async()
        return C4Director(self._cfg.host, token)

    # ---------- raw HTTP base helper ----------

    def _director_base_url(self) -> str:
        base = self._cfg.host.strip()
        if not base.startswith(("http://", "https://")):
            base = "http://" + base
        if not base.endswith("/"):
            base += "/"
        return base

    # ---------- low-level Director HTTP helpers ----------

    async def _director_http_get(self, path: str) -> dict[str, Any]:
        token = await self._ensure_director_token_async()
        url = urljoin(self._director_base_url(), path.lstrip("/"))
        headers = {"Authorization": f"Bearer {token}"}
        timeout = aiohttp.ClientTimeout(total=self._http_timeout_s)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(url, headers=headers, ssl=False) as r:
                    text = await r.text()
                    try:
                        data = await r.json(content_type=None)
                    except Exception:
                        data = None
                    return {"ok": r.status < 300, "status": r.status, "url": url, "json": data, "text": text}
        except Exception as e:
            return {
                "ok": False,
                "status": 0,
                "url": url,
                "json": None,
                "text": "",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def _director_http_post(self, path: str, payload: dict | None = None) -> dict[str, Any]:
        token = await self._ensure_director_token_async()
        url = urljoin(self._director_base_url(), path.lstrip("/"))
        headers = {"Authorization": f"Bearer {token}"}
        timeout = aiohttp.ClientTimeout(total=self._http_timeout_s)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.post(url, headers=headers, json=(payload or {}), ssl=False) as r:
                    text = await r.text()
                    try:
                        data = await r.json(content_type=None)
                    except Exception:
                        data = None
                    return {"ok": r.status < 300, "status": r.status, "url": url, "json": data, "text": text}
        except Exception as e:
            return {
                "ok": False,
                "status": 0,
                "url": url,
                "json": None,
                "text": "",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    # ---------- pyControl4 sendPostRequest helper (signature-safe) ----------

    async def _send_post_via_director(self, director: C4Director, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """
        Calls director.sendPostRequest with the correct arg ordering (differs across pyControl4 versions).
        Returns normalized dict containing ok/json/text/error.
        """
        try:
            fn = director.sendPostRequest
            sig = inspect.signature(fn)

            # after "self", expect 3 args in some order:
            # (path, params, body) OR (path, body, params)
            param_names = [p.name.lower() for p in sig.parameters.values()]
            if param_names and param_names[0] == "self":
                param_names = param_names[1:]

            params_obj: dict[str, Any] = {}

            if len(param_names) >= 3:
                second = param_names[1]
                third = param_names[2]
                if "param" in second and ("body" in third or "data" in third or "payload" in third):
                    res = await fn(path, params_obj, body)  # (path, params, body)
                elif ("body" in second or "data" in second or "payload" in second) and "param" in third:
                    res = await fn(path, body, params_obj)  # (path, body, params)
                else:
                    try:
                        res = await fn(path, params_obj, body)
                    except TypeError:
                        res = await fn(path, body, params_obj)
            else:
                try:
                    res = await fn(path, params_obj, body)
                except TypeError:
                    res = await fn(path, body, params_obj)

            if isinstance(res, str):
                txt = res
                try:
                    js = json.loads(txt)
                except Exception:
                    js = None
                return {"ok": True, "path": path, "text": txt, "json": js}

            return {"ok": True, "path": path, "json": res}
        except Exception as e:
            return {"ok": False, "path": path, "error": str(e), "error_type": type(e).__name__}

    # ---------- async item helpers (IMPORTANT: no deadlocks) ----------

    async def _item_get_commands_async(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)
        director = await self._director_async()

        if hasattr(director, "getItemCommands"):
            try:
                cmds = await asyncio.wait_for(director.getItemCommands(device_id), timeout=self._director_timeout_s)
                if isinstance(cmds, str):
                    try:
                        cmds = json.loads(cmds)
                    except Exception:
                        return {
                            "ok": False,
                            "device_id": device_id,
                            "source": "director.getItemCommands",
                            "raw": cmds,
                        }
                return {"ok": True, "device_id": device_id, "source": "director.getItemCommands", "commands": cmds}
            except asyncio.TimeoutError:
                return {"ok": False, "device_id": device_id, "source": "director.getItemCommands", "error": "timeout"}
            except Exception as e:
                return {
                    "ok": False,
                    "device_id": device_id,
                    "source": "director.getItemCommands",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }

        http = await self._director_http_get(f"/api/v1/items/{device_id}/commands")
        return {"device_id": device_id, "source": "http:/api/v1/items/{id}/commands", **http}

    async def _item_execute_command_async(self, device_id: int, command_id: int) -> dict[str, Any]:
        device_id = int(device_id)
        command_id = int(command_id)
        director = await self._director_async()

        attempts: list[dict[str, Any]] = []
        body = {"commandId": command_id}

        for path in (
            f"/items/{device_id}/commands",        # prefer this
            f"/api/v1/items/{device_id}/commands", # fallback
        ):
            r = await self._send_post_via_director(director, path, body)
            attempts.append({"method": f"director.sendPostRequest {path} {{commandId}}", **r})
            if r.get("ok"):
                return {"ok": True, "device_id": device_id, "command_id": command_id, "attempts": attempts}

        return {"ok": False, "device_id": device_id, "command_id": command_id, "attempts": attempts}

    # ---------- sync wrappers (OK to call from Flask/MCP) ----------

    def item_get_commands(self, device_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._item_get_commands_async(int(device_id)), timeout_s=12)

    def item_execute_command(self, device_id: int, command_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._item_execute_command_async(int(device_id), int(command_id)), timeout_s=12)

    # ---------- lock state ----------

    async def _lock_get_state_async(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)
        director = await self._director_async()

        if not hasattr(director, "getItemVariables"):
            return {
                "ok": True,
                "device_id": device_id,
                "locked": None,
                "state": "unknown",
                "source": "variables",
                "raw": {"_error": "Director does not support getItemVariables()"},
            }

        try:
            vars_ = await asyncio.wait_for(director.getItemVariables(device_id), timeout=self._director_timeout_s)
        except asyncio.TimeoutError:
            return {
                "ok": True,
                "device_id": device_id,
                "locked": None,
                "state": "unknown",
                "source": "variables",
                "raw": {"_error": "Timeout: getItemVariables"},
            }
        except Exception as e:
            return {
                "ok": True,
                "device_id": device_id,
                "locked": None,
                "state": "unknown",
                "source": "variables",
                "raw": {"_error": str(e), "_error_type": type(e).__name__},
            }

        if isinstance(vars_, str):
            try:
                vars_ = json.loads(vars_)
            except Exception:
                return {
                    "ok": True,
                    "device_id": device_id,
                    "locked": None,
                    "state": "unknown",
                    "source": "variables",
                    "raw": {"_error": "Variables were a string but not valid JSON", "variables": vars_},
                }

        if not isinstance(vars_, list):
            return {
                "ok": True,
                "device_id": device_id,
                "locked": None,
                "state": "unknown",
                "source": "variables",
                "raw": {"_note": f"Unexpected variables type: {type(vars_).__name__}", "variables": vars_},
            }

        for row in vars_:
            if not isinstance(row, dict):
                continue
            if str(row.get("varName", "")).strip().lower() != "lockstatus":
                continue

            val = str(row.get("value", "")).strip().lower()
            if val == "locked":
                return {
                    "ok": True,
                    "device_id": device_id,
                    "locked": True,
                    "state": "locked",
                    "source": "variables:LockStatus",
                    "raw": row,
                }
            if val == "unlocked":
                return {
                    "ok": True,
                    "device_id": device_id,
                    "locked": False,
                    "state": "unlocked",
                    "source": "variables:LockStatus",
                    "raw": row,
                }

            return {
                "ok": True,
                "device_id": device_id,
                "locked": None,
                "state": "unknown",
                "source": "variables:LockStatus",
                "raw": row,
            }

        return {
            "ok": True,
            "device_id": device_id,
            "locked": None,
            "state": "unknown",
            "source": "variables",
            "raw": {"_note": "No lock state variable found", "variables": vars_},
        }

    def lock_get_state(self, device_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._lock_get_state_async(int(device_id)), timeout_s=10)

    # ---------- lock actions (no sync calls inside async) ----------

    def lock_unlock(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)

        async def _run():
            before = await self._lock_get_state_async(device_id)

            cmds_resp = await self._item_get_commands_async(device_id)
            cmds = cmds_resp.get("commands")
            if not isinstance(cmds, list):
                return {
                    "ok": False,
                    "device_id": device_id,
                    "error": "Could not load commands",
                    "before": before,
                    "details": cmds_resp,
                }

            match = next(
                (c for c in cmds if isinstance(c, dict) and str(c.get("command", "")).upper() == "UNLOCK"),
                None,
            )
            if not match or "id" not in match:
                return {
                    "ok": False,
                    "device_id": device_id,
                    "error": "UNLOCK command not found",
                    "before": before,
                    "available": cmds,
                }

            cmd_id = int(match["id"])
            exec_result = await self._item_execute_command_async(device_id, cmd_id)
            if not exec_result.get("ok"):
                after = await self._lock_get_state_async(device_id)
                return {
                    "ok": False,
                    "device_id": device_id,
                    "requested": "UNLOCK",
                    "success": False,
                    "before": before,
                    "after": after,
                    "execute": exec_result,
                    "error": "Execute failed",
                }

            deadline = asyncio.get_running_loop().time() + 6.0
            last_after: dict[str, Any] | None = None

            while asyncio.get_running_loop().time() < deadline:
                after = await self._lock_get_state_async(device_id)
                last_after = after
                if after.get("locked") is False:
                    return {
                        "ok": True,
                        "device_id": device_id,
                        "requested": "UNLOCK",
                        "success": True,
                        "before": before,
                        "after": after,
                        "execute": exec_result,
                    }
                await asyncio.sleep(0.35)

            if last_after is None:
                last_after = await self._lock_get_state_async(device_id)

            return {
                "ok": True,
                "device_id": device_id,
                "requested": "UNLOCK",
                "success": False,
                "before": before,
                "after": last_after,
                "execute": exec_result,
                "note": "State did not update before deadline (may still unlock; driver might poll slowly).",
            }

        return self._loop_thread.run(_run(), timeout_s=18)

    def lock_lock(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)

        async def _run():
            cmds_resp = await self._item_get_commands_async(device_id)
            cmds = cmds_resp.get("commands")
            if not isinstance(cmds, list):
                return {
                    "ok": False,
                    "device_id": device_id,
                    "error": "Could not load commands",
                    "details": cmds_resp,
                }

            match = next(
                (c for c in cmds if isinstance(c, dict) and str(c.get("command", "")).upper() == "LOCK"),
                None,
            )
            if not match or "id" not in match:
                return {
                    "ok": False,
                    "device_id": device_id,
                    "error": "LOCK command not found",
                    "available": cmds,
                }

            cmd_id = int(match["id"])
            exec_result = await self._item_execute_command_async(device_id, cmd_id)

            for _ in range(6):
                await asyncio.sleep(0.35)
                after = await self._lock_get_state_async(device_id)
                if after.get("locked") is True:
                    return {
                        "ok": True,
                        "device_id": device_id,
                        "requested": "LOCK",
                        "success": True,
                        "after": after,
                        "execute": exec_result,
                    }

            after = await self._lock_get_state_async(device_id)
            return {
                "ok": True,
                "device_id": device_id,
                "requested": "LOCK",
                "success": False,
                "after": after,
                "execute": exec_result,
            }

        return self._loop_thread.run(_run(), timeout_s=18)