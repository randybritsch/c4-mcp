# control4_gateway.py

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
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

        # Some cloud lock drivers don't update Director variables reliably.
        # Track last *accepted* lock intent so tools can provide an estimate.
        self._last_lock_intent: dict[int, dict[str, Any]] = {}

    def _get_lock_intent_estimate(self, device_id: int, max_age_s: float = 600.0) -> dict[str, Any] | None:
        rec = self._last_lock_intent.get(int(device_id))
        if not isinstance(rec, dict):
            return None
        ts = rec.get("ts")
        intent = rec.get("intent")
        if not isinstance(ts, (int, float)) or not isinstance(intent, str):
            return None
        age = time.time() - float(ts)
        if age < 0 or age > float(max_age_s):
            return None

        intent_u = intent.upper()
        if intent_u == "UNLOCK":
            return {"locked": False, "state": "unlocked", "age_s": round(age, 1), "intent": intent_u}
        if intent_u == "LOCK":
            return {"locked": True, "state": "locked", "age_s": round(age, 1), "intent": intent_u}
        return None

    def _record_lock_intent(self, device_id: int, intent: str, execute: dict[str, Any] | None = None) -> None:
        self._last_lock_intent[int(device_id)] = {
            "ts": time.time(),
            "intent": str(intent or "").strip().upper(),
            "seq": (execute or {}).get("json", {}).get("seq") if isinstance((execute or {}).get("json"), dict) else None,
        }

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
            # Director is typically served over HTTPS; keep HTTP as a fallback
            base = "https://" + base
        if not base.endswith("/"):
            base += "/"
        return base

    def _director_base_urls(self) -> list[str]:
        host = self._cfg.host.strip()
        if host.startswith(("http://", "https://")):
            base = host
            if not base.endswith("/"):
                base += "/"
            return [base]

        https_base = f"https://{host}/"
        http_base = f"http://{host}/"
        return [https_base, http_base]

    # ---------- low-level Director HTTP helpers ----------

    async def _director_http_get(self, path: str) -> dict[str, Any]:
        token = await self._ensure_director_token_async()
        headers = {"Authorization": f"Bearer {token}"}
        timeout = aiohttp.ClientTimeout(total=self._http_timeout_s)

        last: dict[str, Any] | None = None
        for base in self._director_base_urls():
            url = urljoin(base, path.lstrip("/"))
            try:
                async with aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.get(url, headers=headers, ssl=False) as r:
                        text = await r.text()
                        try:
                            data = await r.json(content_type=None)
                        except Exception:
                            data = None
                        resp = {"ok": r.status < 300, "status": r.status, "url": url, "json": data, "text": text}
                        if resp["ok"]:
                            return resp
                        # Try next base URL if we got a 404 (common HTTP vs HTTPS mismatch)
                        last = resp
                        if r.status == 404:
                            continue
                        return resp
            except Exception as e:
                last = {
                    "ok": False,
                    "status": 0,
                    "url": url,
                    "json": None,
                    "text": "",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                continue

        return last or {"ok": False, "status": 0, "url": "", "json": None, "text": "", "error": "No base URL"}

    async def _director_http_post(self, path: str, payload: dict | None = None) -> dict[str, Any]:
        token = await self._ensure_director_token_async()
        headers = {"Authorization": f"Bearer {token}"}
        timeout = aiohttp.ClientTimeout(total=self._http_timeout_s)

        last: dict[str, Any] | None = None
        for base in self._director_base_urls():
            url = urljoin(base, path.lstrip("/"))
            try:
                async with aiohttp.ClientSession(timeout=timeout) as s:
                    async with s.post(url, headers=headers, json=(payload or {}), ssl=False) as r:
                        text = await r.text()
                        try:
                            data = await r.json(content_type=None)
                        except Exception:
                            data = None
                        resp = {"ok": r.status < 300, "status": r.status, "url": url, "json": data, "text": text}
                        if resp["ok"]:
                            return resp
                        last = resp
                        if r.status == 404:
                            continue
                        return resp
            except Exception as e:
                last = {
                    "ok": False,
                    "status": 0,
                    "url": url,
                    "json": None,
                    "text": "",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                continue

        return last or {"ok": False, "status": 0, "url": "", "json": None, "text": "", "error": "No base URL"}

    # ---------- pyControl4 sendPostRequest helper (signature-safe) ----------

    async def _send_post_via_director(
        self,
        director: C4Director,
        uri: str,
        command: str,
        params: dict[str, Any] | None = None,
        async_variable: bool = True,
    ) -> dict[str, Any]:
        """Calls director.sendPostRequest with correct signature across pyControl4 versions.

        Newer pyControl4 uses: (uri, command, params, async_variable=True)
        Older variants may use different ordering; we detect via signature.
        """
        try:
            fn = director.sendPostRequest
            sig = inspect.signature(fn)

            param_names = [p.name.lower() for p in sig.parameters.values()]
            if param_names and param_names[0] == "self":
                param_names = param_names[1:]

            params_obj: dict[str, Any] = params or {}

            # pyControl4 (current) signature: (uri, command, params, async_variable=True)
            if len(param_names) >= 3 and "command" in param_names and "param" in "".join(param_names):
                try:
                    res = await fn(uri, command, params_obj, async_variable)
                except TypeError:
                    # Some versions omit async_variable
                    res = await fn(uri, command, params_obj)
            else:
                # Legacy/fallback: best-effort ordering (uri, params, body) style.
                body = {"async": async_variable, "command": command, "tParams": params_obj}
                try:
                    res = await fn(uri, params_obj, body)
                except TypeError:
                    res = await fn(uri, body, params_obj)

            if isinstance(res, str):
                txt = res
                try:
                    js = json.loads(txt)
                except Exception:
                    js = None
                return {"ok": True, "uri": uri, "command": command, "text": txt, "json": js}

            return {"ok": True, "uri": uri, "command": command, "json": res}
        except Exception as e:
            return {
                "ok": False,
                "uri": uri,
                "command": command,
                "error": str(e),
                "error_type": type(e).__name__,
            }

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

    async def _item_execute_command_async(
        self, device_id: int, command_id: int, command_name: str | None = None
    ) -> dict[str, Any]:
        device_id = int(device_id)
        command_id = int(command_id)
        director = await self._director_async()

        attempts: list[dict[str, Any]] = []

        # Resolve command name if not provided (prefer named commands over IDs).
        resolved_name = command_name
        if not resolved_name:
            cmds_resp = await self._item_get_commands_async(device_id)
            cmds = cmds_resp.get("commands")
            if isinstance(cmds, list):
                for c in cmds:
                    if isinstance(c, dict) and int(c.get("id") or -1) == command_id:
                        resolved_name = str(c.get("command") or "").strip().upper() or None
                        break

        if not resolved_name:
            return {
                "ok": False,
                "device_id": device_id,
                "command_id": command_id,
                "command": command_name,
                "attempts": attempts,
                "error": "Could not resolve command name for command_id",
            }

        uri = f"/api/v1/items/{device_id}/commands"

        # Primary: sendPostRequest(uri, command, tParams)
        # In practice, async_variable=False yields a meaningful response (e.g., SendToDevice result/seq)
        # and has proven more reliable than async mode for device commands.
        r = await self._send_post_via_director(director, uri, resolved_name, {}, async_variable=False)
        attempts.append({"method": f"director.sendPostRequest {uri} {resolved_name}", **r})
        if r.get("ok"):
            return {
                "ok": True,
                "device_id": device_id,
                "command_id": command_id,
                "command": resolved_name,
                "attempts": attempts,
            }

        return {
            "ok": False,
            "device_id": device_id,
            "command_id": command_id,
            "command": resolved_name,
            "attempts": attempts,
        }

    async def _item_send_command_async(
        self, device_id: int, command: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        device_id = int(device_id)
        command = str(command or "").strip().upper()
        if not command:
            return {"ok": False, "device_id": device_id, "error": "command is required"}

        director = await self._director_async()
        uri = f"/api/v1/items/{device_id}/commands"
        return await self._send_post_via_director(director, uri, command, params or {}, async_variable=False)

    # ---------- room command helpers ----------

    async def _room_send_command_async(
        self, room_id: int, command: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        room_id = int(room_id)
        command = str(command or "").strip().upper()
        if not command:
            return {"ok": False, "room_id": room_id, "error": "command is required"}

        director = await self._director_async()
        uri = f"/api/v1/rooms/{room_id}/commands"
        result = await self._send_post_via_director(director, uri, command, params or {}, async_variable=False)
        if isinstance(result, dict):
            result["room_id"] = room_id
        return result

    async def _room_select_video_device_async(self, room_id: int, device_id: int, deselect: bool = False) -> dict[str, Any]:
        room_id = int(room_id)
        device_id = int(device_id)
        params = {"deviceid": device_id, "deselect": (1 if deselect else 0)}
        return await self._room_send_command_async(room_id, "SELECT_VIDEO_DEVICE", params)

    @staticmethod
    def _find_ui_watch_node(payload: Any, room_id: int) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            if payload.get("type") == "watch" and int(payload.get("room_id") or -1) == int(room_id):
                return payload
            for v in payload.values():
                found = Control4Gateway._find_ui_watch_node(v, room_id)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for it in payload:
                found = Control4Gateway._find_ui_watch_node(it, room_id)
                if found is not None:
                    return found
        return None

    async def _ui_watch_status_async(self, room_id: int) -> dict[str, Any] | None:
        room_id = int(room_id)
        http = await self._director_http_get("/api/v1/agents/ui_configuration")
        if not http.get("ok"):
            return None
        payload = http.get("json")
        node = self._find_ui_watch_node(payload, room_id)
        if not isinstance(node, dict):
            return None
        active = bool(node.get("active"))
        sources = None
        raw_sources = node.get("sources")
        if isinstance(raw_sources, dict) and isinstance(raw_sources.get("source"), list):
            sources = [s for s in raw_sources.get("source") if isinstance(s, dict)]
        return {"room_id": room_id, "active": active, "sources": sources}

    async def _resolve_room_id_for_device_async(self, device_id: int) -> int | None:
        device_id = int(device_id)
        items = await self._get_all_items_async()
        for it in items:
            if not isinstance(it, dict):
                continue
            if int(it.get("id") or -1) == device_id:
                rid = it.get("roomId")
                try:
                    return int(rid)
                except Exception:
                    return None
        return None

    async def _media_watch_launch_app_async(
        self,
        device_id: int,
        app: str,
        room_id: int | None = None,
        pre_home: bool = True,
    ) -> dict[str, Any]:
        device_id = int(device_id)
        app = str(app or "").strip()
        if not app:
            return {"ok": False, "device_id": device_id, "error": "app is required"}

        resolved_room_id = int(room_id) if room_id is not None else (await self._resolve_room_id_for_device_async(device_id))
        if resolved_room_id is None:
            return {
                "ok": False,
                "device_id": device_id,
                "room_id": room_id,
                "error": "Could not resolve room_id for device; pass room_id explicitly",
            }

        before_watch = await self._ui_watch_status_async(resolved_room_id)

        select_video = await self._room_select_video_device_async(resolved_room_id, device_id, deselect=False)
        # Give the room's Watch macro time to settle; in practice this can take a couple seconds.
        after_select_watch = None
        settle_deadline = asyncio.get_running_loop().time() + 6.0
        while asyncio.get_running_loop().time() < settle_deadline:
            await asyncio.sleep(0.6)
            after_select_watch = await self._ui_watch_status_async(resolved_room_id)
            if isinstance(after_select_watch, dict) and after_select_watch.get("active") is True:
                break

        home = None
        home_attempts: list[dict[str, Any]] | None = None
        if bool(pre_home):
            # Use media_send_command so callers can pass any Roku-related item id.
            # Send HOME twice with a short settle; Roku can ignore the first press if it's mid-transition.
            home_attempts = []
            for _ in range(2):
                home = await self._media_send_command_async(device_id, "HOME", {})
                home_attempts.append(home)
                await asyncio.sleep(0.8)

        # Retry LaunchApp if the device is still transitioning after Watch/HOME.
        launch = None
        launch_attempts: list[dict[str, Any]] = []
        for _ in range(3):
            launch = await self._media_launch_app_async(device_id, app)
            launch_attempts.append(launch)
            if isinstance(launch, dict) and launch.get("ok") is True:
                break
            await asyncio.sleep(1.2)

        after_launch_watch = await self._ui_watch_status_async(resolved_room_id)

        ok = bool(select_video.get("ok")) and bool((launch or {}).get("ok"))

        return {
            "ok": ok,
            "room_id": resolved_room_id,
            "device_id": device_id,
            "app": app,
            "watch": {
                "before": before_watch,
                "after_select_video": after_select_watch,
                "after_launch": after_launch_watch,
            },
            "select_video": select_video,
            "home": home,
            "home_attempts": home_attempts,
            "launch": launch,
            "launch_attempts": launch_attempts,
        }

    @staticmethod
    def _select_vars_for_trace(
        variables: list[dict[str, Any]],
        watch_var_names: list[str] | None,
    ) -> dict[str, Any]:
        if not watch_var_names:
            return {
                str(v.get("varName")): v.get("value")
                for v in variables
                if isinstance(v, dict) and v.get("varName") is not None
            }

        wanted = {str(n).strip().lower() for n in watch_var_names}
        out: dict[str, Any] = {}
        for row in variables:
            if not isinstance(row, dict):
                continue
            name = str(row.get("varName") or "").strip()
            if name.lower() in wanted:
                out[name] = row.get("value")
        return out

    async def _debug_trace_command_async(
        self,
        device_id: int,
        command: str,
        params: dict[str, Any] | None = None,
        watch_var_names: list[str] | None = None,
        poll_interval_s: float = 0.5,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        device_id = int(device_id)
        command = str(command or "").strip().upper()
        if not command:
            return {"ok": False, "device_id": device_id, "error": "command is required"}

        before_fetch = await self._fetch_item_variables_list_async(device_id, timeout_s=4.0)
        before_vars: list[dict[str, Any]] = (
            before_fetch.get("variables") if isinstance(before_fetch.get("variables"), list) else []
        )
        before = {
            "state": self._parse_lock_state_from_variables(device_id, before_vars),
            "vars": self._select_vars_for_trace(before_vars, watch_var_names),
        }

        execute = await self._item_send_command_async(device_id, command, params)

        start = asyncio.get_running_loop().time()
        deadline = start + float(timeout_s)
        last = before
        changes: list[dict[str, Any]] = []

        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(max(float(poll_interval_s), 0.1))
            fetched = await self._fetch_item_variables_list_async(device_id, timeout_s=4.0)
            var_list: list[dict[str, Any]] = (
                fetched.get("variables") if isinstance(fetched.get("variables"), list) else []
            )
            now = {
                "state": self._parse_lock_state_from_variables(device_id, var_list),
                "vars": self._select_vars_for_trace(var_list, watch_var_names),
            }
            if now != last:
                changes.append(
                    {"t": round(asyncio.get_running_loop().time() - start, 2), "snapshot": now}
                )
                last = now

        return {
            "ok": True,
            "device_id": device_id,
            "command": command,
            "params": params or {},
            "before": before,
            "execute": execute,
            "after": last,
            "changes": changes,
            "timeout_s": float(timeout_s),
            "poll_interval_s": float(poll_interval_s),
        }

    # ---------- sync wrappers (OK to call from Flask/MCP) ----------

    def item_get_commands(self, device_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._item_get_commands_async(int(device_id)), timeout_s=12)

    def item_execute_command(self, device_id: int, command_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._item_execute_command_async(int(device_id), int(command_id)), timeout_s=12)

    def room_send_command(self, room_id: int, command: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._loop_thread.run(
            self._room_send_command_async(int(room_id), str(command or ""), params or {}),
            timeout_s=12,
        )

    def room_select_video_device(self, room_id: int, device_id: int, deselect: bool = False) -> dict[str, Any]:
        return self._loop_thread.run(
            self._room_select_video_device_async(int(room_id), int(device_id), deselect=bool(deselect)),
            timeout_s=12,
        )

    def media_watch_launch_app(self, device_id: int, app: str, room_id: int | None = None, pre_home: bool = True) -> dict[str, Any]:
        return self._loop_thread.run(
            self._media_watch_launch_app_async(int(device_id), str(app or ""), room_id=(int(room_id) if room_id is not None else None), pre_home=bool(pre_home)),
            timeout_s=45,
        )

    def item_send_command(self, device_id: int, command: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._loop_thread.run(
            self._item_send_command_async(int(device_id), str(command or ""), params),
            timeout_s=12,
        )

    def debug_trace_command(
        self,
        device_id: int,
        command: str,
        params: dict[str, Any] | None = None,
        watch_var_names: list[str] | None = None,
        poll_interval_s: float = 0.5,
        timeout_s: float = 30.0,
    ) -> dict[str, Any]:
        return self._loop_thread.run(
            self._debug_trace_command_async(
                int(device_id),
                str(command or ""),
                params,
                watch_var_names=watch_var_names,
                poll_interval_s=float(poll_interval_s),
                timeout_s=float(timeout_s),
            ),
            timeout_s=float(timeout_s) + 20.0,
        )

    # ---------- items / rooms (used by c4_list_devices, c4_list_rooms) ----------

    @staticmethod
    def _normalize_items_payload(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return []

        if isinstance(payload, dict):
            payload = payload.get("items") or payload.get("Items") or payload.get("data") or payload

        if not isinstance(payload, list):
            return []

        items: list[dict[str, Any]] = []
        for row in payload:
            if isinstance(row, dict):
                items.append(row)
        return items

    async def _get_all_items_async(self) -> list[dict[str, Any]]:
        director = await self._director_async()

        for method_name in ("getAllItemInfo", "getAllItems", "getItems"):
            fn = getattr(director, method_name, None)
            if fn is None:
                continue
            try:
                res = await asyncio.wait_for(fn(), timeout=self._director_timeout_s)
                items = self._normalize_items_payload(res)
                if items:
                    return items
            except TypeError:
                # Some pyControl4 versions require args; ignore and fallback.
                continue
            except Exception:
                continue

        # Some pyControl4 versions support category-based listing.
        fn_by_cat = getattr(director, "getAllItemsByCategory", None)
        if fn_by_cat is not None:
            for category in ("room", "device"):
                try:
                    res = await asyncio.wait_for(fn_by_cat(category), timeout=self._director_timeout_s)
                    items = self._normalize_items_payload(res)
                    if items:
                        return items
                except Exception:
                    continue

        http = await self._director_http_get("/api/v1/items")
        if http.get("ok"):
            items = self._normalize_items_payload(http.get("json"))
            if items:
                return items
        return []

    def get_all_items(self) -> list[dict[str, Any]]:
        return self._loop_thread.run(self._get_all_items_async(), timeout_s=18)

    def list_rooms(self) -> list[dict[str, Any]]:
        items = self.get_all_items()
        rooms = [i for i in items if isinstance(i, dict) and i.get("typeName") == "room"]
        rooms.sort(key=lambda r: (str(r.get("name") or ""), str(r.get("id") or "")))
        return rooms

    # ---------- item variables (debug tool support) ----------

    async def _item_get_variables_async(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)
        director = await self._director_async()

        if hasattr(director, "getItemVariables"):
            try:
                vars_ = await asyncio.wait_for(director.getItemVariables(device_id), timeout=self._director_timeout_s)
                if isinstance(vars_, str):
                    try:
                        vars_ = json.loads(vars_)
                    except Exception:
                        return {
                            "ok": False,
                            "device_id": device_id,
                            "source": "director.getItemVariables",
                            "error": "variables string was not valid JSON",
                            "raw": vars_,
                        }
                if not isinstance(vars_, list):
                    return {
                        "ok": False,
                        "device_id": device_id,
                        "source": "director.getItemVariables",
                        "error": f"Unexpected variables type: {type(vars_).__name__}",
                        "raw": vars_,
                    }
                return {"ok": True, "device_id": device_id, "source": "director.getItemVariables", "variables": vars_}
            except asyncio.TimeoutError:
                return {"ok": False, "device_id": device_id, "source": "director.getItemVariables", "error": "timeout"}
            except Exception as e:
                return {
                    "ok": False,
                    "device_id": device_id,
                    "source": "director.getItemVariables",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }

        http = await self._director_http_get(f"/api/v1/items/{device_id}/variables")
        if http.get("ok"):
            vars_ = http.get("json")
            if isinstance(vars_, dict):
                vars_ = vars_.get("variables") or vars_.get("data") or vars_
            if isinstance(vars_, list):
                return {"ok": True, "device_id": device_id, "source": "http:/api/v1/items/{id}/variables", "variables": vars_}
        return {"ok": False, "device_id": device_id, "source": "http:/api/v1/items/{id}/variables", **http}

    def item_get_variables(self, device_id: int, timeout_s: float = 12.0) -> dict[str, Any]:
        device_id = int(device_id)
        try:
            return self._loop_thread.run(self._item_get_variables_async(device_id), timeout_s=float(timeout_s))
        except FuturesTimeoutError:
            return {
                "ok": False,
                "device_id": device_id,
                "source": "gateway.item_get_variables",
                "error": "timeout",
            }
        except Exception as e:
            return {
                "ok": False,
                "device_id": device_id,
                "source": "gateway.item_get_variables",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def item_get_bindings(self, device_id: int, timeout_s: float = 12.0) -> dict[str, Any]:
        device_id = int(device_id)
        try:
            return self._loop_thread.run(self._item_get_bindings_async(device_id), timeout_s=float(timeout_s))
        except FuturesTimeoutError:
            return {
                "ok": False,
                "device_id": device_id,
                "source": "gateway.item_get_bindings",
                "error": "timeout",
            }
        except Exception as e:
            return {
                "ok": False,
                "device_id": device_id,
                "source": "gateway.item_get_bindings",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def _item_get_bindings_async(self, device_id: int) -> dict[str, Any]:
        http = await self._director_http_get(f"/api/v1/items/{int(device_id)}/bindings")
        if http.get("ok"):
            payload = http.get("json")
            bindings = payload.get("bindings") if isinstance(payload, dict) else payload
            return {
                "ok": True,
                "device_id": int(device_id),
                "source": "http:/api/v1/items/{id}/bindings",
                "bindings": bindings,
            }

        return {
            "ok": False,
            "device_id": int(device_id),
            "source": "http:/api/v1/items/{id}/bindings",
            "error": http.get("error") or "Failed to fetch bindings",
            "http": http,
        }

    async def _fetch_item_variables_list_async(
        self, device_id: int, timeout_s: float | None = None
    ) -> dict[str, Any]:
        device_id = int(device_id)
        director = await self._director_async()

        if not hasattr(director, "getItemVariables"):
            return {
                "ok": False,
                "device_id": device_id,
                "source": "director.getItemVariables",
                "error": "Director does not support getItemVariables()",
                "variables": [],
            }

        try:
            vars_ = await asyncio.wait_for(
                director.getItemVariables(device_id), timeout=(float(timeout_s) if timeout_s is not None else self._director_timeout_s)
            )
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "device_id": device_id,
                "source": "director.getItemVariables",
                "error": "timeout",
                "variables": [],
            }
        except Exception as e:
            return {
                "ok": False,
                "device_id": device_id,
                "source": "director.getItemVariables",
                "error": str(e),
                "error_type": type(e).__name__,
                "variables": [],
            }

        if isinstance(vars_, str):
            try:
                vars_ = json.loads(vars_)
            except Exception:
                return {
                    "ok": False,
                    "device_id": device_id,
                    "source": "director.getItemVariables",
                    "error": "variables string was not valid JSON",
                    "raw": vars_,
                    "variables": [],
                }

        if not isinstance(vars_, list):
            return {
                "ok": False,
                "device_id": device_id,
                "source": "director.getItemVariables",
                "error": f"Unexpected variables type: {type(vars_).__name__}",
                "raw": vars_,
                "variables": [],
            }

        cleaned: list[dict[str, Any]] = []
        for row in vars_:
            if isinstance(row, dict):
                cleaned.append(row)
        return {"ok": True, "device_id": device_id, "source": "director.getItemVariables", "variables": cleaned}

    @staticmethod
    def _get_var_value(variables: list[dict[str, Any]], name: str) -> Any:
        target = name.strip().lower()
        for row in variables:
            var = str(row.get("varName", "")).strip().lower()
            if var == target:
                return row.get("value")
        return None

    @staticmethod
    def _coerce_bool(v: Any) -> bool | None:
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"true", "1", "yes", "on", "locked"}:
            return True
        if s in {"false", "0", "no", "off", "unlocked"}:
            return False
        return None

    def _parse_lock_state_from_variables(self, device_id: int, variables: list[dict[str, Any]]) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for row in variables:
            if not isinstance(row, dict):
                continue
            var = str(row.get("varName", "")).strip()
            if not var:
                continue
            var_l = var.lower()
            if var_l == "lockstatus":
                candidates.insert(0, row)
            elif var_l in {"lock_status", "locked", "islocked", "lock_state"}:
                candidates.append(row)
            elif "lock" in var_l and "status" in var_l:
                candidates.append(row)

        for row in candidates:
            var = str(row.get("varName", "")).strip()
            b = self._coerce_bool(row.get("value"))
            if b is True:
                return {
                    "ok": True,
                    "device_id": device_id,
                    "locked": True,
                    "state": "locked",
                    "source": f"variables:{var}",
                    "raw": row,
                }
            if b is False:
                return {
                    "ok": True,
                    "device_id": device_id,
                    "locked": False,
                    "state": "unlocked",
                    "source": f"variables:{var}",
                    "raw": row,
                }
            if str(var).strip():
                return {
                    "ok": True,
                    "device_id": device_id,
                    "locked": None,
                    "state": "unknown",
                    "source": f"variables:{var}",
                    "raw": row,
                }

        # Relay-style door lock proxy (common): RelayState 0=locked, 1=unlocked
        relay_state = self._get_var_value(variables, "RelayState")
        b = self._coerce_bool(relay_state)
        if b is not None:
            # _coerce_bool maps 0->False, 1->True; for RelayState, 0 means locked.
            locked = not b
            return {
                "ok": True,
                "device_id": device_id,
                "locked": locked,
                "state": "locked" if locked else "unlocked",
                "source": "variables:RelayState",
                "raw": {"varName": "RelayState", "value": relay_state},
            }

        return {
            "ok": True,
            "device_id": device_id,
            "locked": None,
            "state": "unknown",
            "source": "variables",
            "raw": {"_note": "No lock state variable found", "variables": variables},
        }

    # ---------- lock state ----------

    async def _lock_get_state_async(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)
        fetched = await self._fetch_item_variables_list_async(device_id)
        if not fetched.get("ok"):
            out = {
                "ok": True,
                "device_id": device_id,
                "locked": None,
                "state": "unknown",
                "source": "variables",
                "raw": {
                    "_error": fetched.get("error") or "getItemVariables failed",
                    "_error_type": fetched.get("error_type"),
                },
            }

            est = self._get_lock_intent_estimate(device_id)
            if est is not None:
                out["estimate"] = est
            return out

        variables = fetched.get("variables")
        if not isinstance(variables, list):
            variables = []
        out = self._parse_lock_state_from_variables(device_id, variables)
        est = self._get_lock_intent_estimate(device_id)
        if est is not None:
            out["estimate"] = est
        return out

    def lock_get_state(self, device_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._lock_get_state_async(int(device_id)), timeout_s=10)

    # ---------- lock actions (no sync calls inside async) ----------

    def lock_unlock(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)

        async def _run():
            before = await self._lock_get_state_async(device_id)
            before_locked = before.get("locked")
            before_vars = await self._fetch_item_variables_list_async(device_id)
            before_list: list[dict[str, Any]] = before_vars.get("variables") if isinstance(before_vars.get("variables"), list) else []
            before_activity = {
                "LastActionDescription": self._get_var_value(before_list, "LastActionDescription"),
                "LAST_UNLOCK_USER": self._get_var_value(before_list, "LAST_UNLOCK_USER"),
            }

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
                (
                    c
                    for c in cmds
                    if isinstance(c, dict)
                    and str(c.get("command", "")).upper() in {"UNLOCK", "CLOSE"}
                ),
                None,
            )
            if not match or not match.get("command"):
                return {
                    "ok": False,
                    "device_id": device_id,
                    "error": "UNLOCK/CLOSE command not found",
                    "before": before,
                    "available": cmds,
                }

            cmd_name = str(match.get("command") or "").strip().upper()
            director = await self._director_async()
            uri = f"/api/v1/items/{device_id}/commands"
            exec_result = await self._send_post_via_director(director, uri, cmd_name, {}, async_variable=False)
            accepted = bool(exec_result.get("ok"))
            if not accepted:
                after = await self._lock_get_state_async(device_id)
                return {
                    "ok": False,
                    "device_id": device_id,
                    "requested": cmd_name,
                    "accepted": False,
                    "confirmed": False,
                    "success": False,
                    "before": before,
                    "after": after,
                    "execute": exec_result,
                    "error": "Execute failed",
                }

            # Record intent immediately on accepted send.
            self._record_lock_intent(device_id, "UNLOCK", execute=exec_result)

            # UNLOCK tends to lag more than LOCK on cloud drivers.
            confirm_timeout_s = 12.0
            deadline = asyncio.get_running_loop().time() + confirm_timeout_s
            last_after: dict[str, Any] | None = None
            last_activity_after: dict[str, Any] | None = None
            activity_changed = False

            while asyncio.get_running_loop().time() < deadline:
                fetched = await self._fetch_item_variables_list_async(device_id, timeout_s=2.0)
                var_list: list[dict[str, Any]] = fetched.get("variables") if isinstance(fetched.get("variables"), list) else []
                after = self._parse_lock_state_from_variables(device_id, var_list)
                last_after = after
                last_activity_after = {
                    "LastActionDescription": self._get_var_value(var_list, "LastActionDescription"),
                    "LAST_UNLOCK_USER": self._get_var_value(var_list, "LAST_UNLOCK_USER"),
                }
                if last_activity_after != before_activity:
                    activity_changed = True
                after_locked = after.get("locked")
                if after_locked is False and (after_locked != before_locked or activity_changed):
                    return {
                        "ok": True,
                        "device_id": device_id,
                        "requested": cmd_name,
                        "accepted": True,
                        "confirmed": True,
                        "success": True,
                        "before": before,
                        "after": after,
                        "execute": exec_result,
                        "estimate": self._get_lock_intent_estimate(device_id),
                        "activity": {
                            "before": before_activity,
                            "after": last_activity_after,
                        },
                    }
                await asyncio.sleep(0.35)

            if last_after is None:
                last_after = await self._lock_get_state_async(device_id)

            return {
                "ok": True,
                "device_id": device_id,
                "requested": cmd_name,
                "accepted": True,
                "confirmed": False,
                "success": False,
                "before": before,
                "after": last_after,
                "execute": exec_result,
                "note": "State did not update before deadline (may still unlock; driver might poll slowly).",
                "confirm_timeout_s": confirm_timeout_s,
                "estimate": self._get_lock_intent_estimate(device_id),
                "activity": {
                    "changed": activity_changed,
                    "before": before_activity,
                    "after": last_activity_after,
                },
            }

        return self._loop_thread.run(_run(), timeout_s=18)

    def lock_lock(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)

        async def _run():
            before = await self._lock_get_state_async(device_id)
            before_locked = before.get("locked")
            before_vars = await self._fetch_item_variables_list_async(device_id)
            before_list: list[dict[str, Any]] = before_vars.get("variables") if isinstance(before_vars.get("variables"), list) else []
            before_activity = {
                "LastActionDescription": self._get_var_value(before_list, "LastActionDescription"),
                "LAST_UNLOCK_USER": self._get_var_value(before_list, "LAST_UNLOCK_USER"),
            }

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
                (
                    c
                    for c in cmds
                    if isinstance(c, dict)
                    and str(c.get("command", "")).upper() in {"LOCK", "OPEN"}
                ),
                None,
            )
            if not match or not match.get("command"):
                return {
                    "ok": False,
                    "device_id": device_id,
                    "error": "LOCK/OPEN command not found",
                    "before": before,
                    "available": cmds,
                }

            cmd_name = str(match.get("command") or "").strip().upper()
            # Do not short-circuit on "already locked".
            # Some cloud lock drivers report stale state; skipping would prevent re-locking.

            director = await self._director_async()
            uri = f"/api/v1/items/{device_id}/commands"
            exec_result = await self._send_post_via_director(director, uri, cmd_name, {}, async_variable=False)

            accepted = bool(exec_result.get("ok"))
            if not accepted:
                after = await self._lock_get_state_async(device_id)
                return {
                    "ok": False,
                    "device_id": device_id,
                    "requested": cmd_name,
                    "accepted": False,
                    "confirmed": False,
                    "success": False,
                    "before": before,
                    "after": after,
                    "execute": exec_result,
                    "error": "Execute failed",
                }

            # Record intent immediately on accepted send.
            self._record_lock_intent(device_id, "LOCK", execute=exec_result)

            confirm_timeout_s = 6.0
            deadline = asyncio.get_running_loop().time() + confirm_timeout_s
            last_after: dict[str, Any] | None = None
            last_activity_after: dict[str, Any] | None = None
            activity_changed = False

            while asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.35)
                fetched = await self._fetch_item_variables_list_async(device_id, timeout_s=2.0)
                var_list: list[dict[str, Any]] = fetched.get("variables") if isinstance(fetched.get("variables"), list) else []
                after = self._parse_lock_state_from_variables(device_id, var_list)
                last_after = after
                last_activity_after = {
                    "LastActionDescription": self._get_var_value(var_list, "LastActionDescription"),
                    "LAST_UNLOCK_USER": self._get_var_value(var_list, "LAST_UNLOCK_USER"),
                }
                if last_activity_after != before_activity:
                    activity_changed = True
                after_locked = after.get("locked")
                if after_locked is True and (after_locked != before_locked or activity_changed):
                    return {
                        "ok": True,
                        "device_id": device_id,
                        "requested": cmd_name,
                        "accepted": True,
                        "confirmed": True,
                        "success": True,
                        "before": before,
                        "after": after,
                        "execute": exec_result,
                        "estimate": self._get_lock_intent_estimate(device_id),
                        "activity": {
                            "before": before_activity,
                            "after": last_activity_after,
                        },
                    }

            if last_after is None:
                last_after = await self._lock_get_state_async(device_id)

            return {
                "ok": True,
                "device_id": device_id,
                "requested": cmd_name,
                "accepted": True,
                "confirmed": False,
                "success": False,
                "before": before,
                "after": last_after,
                "execute": exec_result,
                "note": "State did not update before deadline (may still lock; driver might poll slowly).",
                "confirm_timeout_s": confirm_timeout_s,
                "estimate": self._get_lock_intent_estimate(device_id),
                "activity": {
                    "changed": activity_changed,
                    "before": before_activity,
                    "after": last_activity_after,
                },
            }

        return self._loop_thread.run(_run(), timeout_s=18)

    # ---------- lights (basic) ----------

    @staticmethod
    def _coerce_int(v: Any) -> int | None:
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(round(v))
        s = str(v).strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None

    @staticmethod
    def _coerce_float(v: Any) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    @staticmethod
    def _vars_to_map(variables: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in variables:
            if not isinstance(row, dict):
                continue
            name = row.get("varName")
            if name is None:
                continue
            out[str(name)] = row.get("value")
        return out

    @staticmethod
    def _thermostat_state_from_vars(vars_by_name: dict[str, Any]) -> dict[str, Any]:
        # Prefer display-friendly values when present.
        def pick(*names: str) -> Any:
            for n in names:
                if n in vars_by_name:
                    return vars_by_name.get(n)
            return None

        return {
            "temperature_f": pick("DISPLAY_TEMPERATURE", "TEMPERATURE_F"),
            "temperature_c": pick("TEMPERATURE_C"),
            "heat_setpoint_f": pick("DISPLAY_HEATSETPOINT", "HEAT_SETPOINT_F"),
            "cool_setpoint_f": pick("DISPLAY_COOLSETPOINT", "COOL_SETPOINT_F"),
            "heatcool_deadband_f": pick("HEATCOOL_SETPOINTS_DEADBAND_F"),
            "hvac_mode": pick("HVAC_MODE", "ANA_HVACMODE", "V1 HVACMODE"),
            "hvac_state": pick("HVAC_STATE", "ANA_HVACSTATE"),
            "fan_mode": pick("FAN_MODE", "ANA_FANMODE"),
            "fan_state": pick("FAN_STATE", "ANA_FANSTATE"),
            "hold_mode": pick("HOLD_MODE", "ANA_HOLDMODE"),
            "humidity": pick("HUMIDITY"),
            "humidity_mode": pick("HUMIDITY_MODE"),
            "humidity_state": pick("HUMIDITY_STATE"),
            "outdoor_temperature_f": pick("OUTDOOR_TEMPERATURE_F"),
        }

    async def _thermostat_get_state_async(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)
        fetched = await self._fetch_item_variables_list_async(device_id, timeout_s=4.0)
        var_list: list[dict[str, Any]] = (
            fetched.get("variables") if isinstance(fetched.get("variables"), list) else []
        )
        vars_by_name = self._vars_to_map(var_list)
        state = self._thermostat_state_from_vars(vars_by_name)

        # Coerce some common numeric fields for predictable tool output.
        for k in (
            "temperature_f",
            "temperature_c",
            "heat_setpoint_f",
            "cool_setpoint_f",
            "heatcool_deadband_f",
            "humidity",
            "outdoor_temperature_f",
        ):
            if state.get(k) is not None:
                state[k] = self._coerce_float(state.get(k))

        return {
            "ok": bool(fetched.get("ok")) or True,
            "device_id": device_id,
            "source": fetched.get("source") or "director.getItemVariables",
            "state": state,
        }

    def thermostat_get_state(self, device_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._thermostat_get_state_async(int(device_id)), timeout_s=12)

    async def _thermostat_send_and_confirm_async(
        self,
        device_id: int,
        command: str,
        params: dict[str, Any],
        confirm_predicate,
        confirm_timeout_s: float = 8.0,
        poll_interval_s: float = 0.5,
    ) -> dict[str, Any]:
        device_id = int(device_id)
        before = await self._thermostat_get_state_async(device_id)

        exec_result = await self._item_send_command_async(device_id, command, params)
        if not exec_result.get("ok"):
            return {
                "ok": False,
                "device_id": device_id,
                "requested": {"command": str(command or ""), "params": params},
                "accepted": False,
                "confirmed": False,
                "before": before,
                "execute": exec_result,
            }

        deadline = asyncio.get_running_loop().time() + float(confirm_timeout_s)
        last = before

        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(max(0.2, float(poll_interval_s)))
            now = await self._thermostat_get_state_async(device_id)
            last = now
            try:
                if confirm_predicate(now):
                    return {
                        "ok": True,
                        "device_id": device_id,
                        "requested": {"command": str(command or ""), "params": params},
                        "accepted": True,
                        "confirmed": True,
                        "before": before,
                        "after": now,
                        "execute": exec_result,
                    }
            except Exception:
                # If predicate is buggy, don't crash tool.
                continue

        return {
            "ok": True,
            "device_id": device_id,
            "requested": {"command": str(command or ""), "params": params},
            "accepted": True,
            "confirmed": False,
            "before": before,
            "after": last,
            "execute": exec_result,
            "confirm_timeout_s": float(confirm_timeout_s),
        }

    def thermostat_set_hvac_mode(self, device_id: int, mode: str, confirm_timeout_s: float = 8.0) -> dict[str, Any]:
        mode = str(mode or "").strip()
        if not mode:
            return {"ok": False, "device_id": int(device_id), "error": "mode is required"}

        async def _run() -> dict[str, Any]:
            return await self._thermostat_send_and_confirm_async(
                int(device_id),
                "SET_MODE_HVAC",
                {"MODE": mode},
                lambda r: (r.get("state") or {}).get("hvac_mode") == mode,
                confirm_timeout_s=float(confirm_timeout_s),
            )

        return self._loop_thread.run(_run(), timeout_s=float(confirm_timeout_s) + 18.0)

    def thermostat_set_fan_mode(self, device_id: int, mode: str, confirm_timeout_s: float = 8.0) -> dict[str, Any]:
        mode = str(mode or "").strip()
        if not mode:
            return {"ok": False, "device_id": int(device_id), "error": "mode is required"}

        async def _run() -> dict[str, Any]:
            return await self._thermostat_send_and_confirm_async(
                int(device_id),
                "SET_MODE_FAN",
                {"MODE": mode},
                lambda r: (r.get("state") or {}).get("fan_mode") == mode,
                confirm_timeout_s=float(confirm_timeout_s),
            )

        return self._loop_thread.run(_run(), timeout_s=float(confirm_timeout_s) + 18.0)

    def thermostat_set_hold_mode(self, device_id: int, mode: str, confirm_timeout_s: float = 8.0) -> dict[str, Any]:
        mode = str(mode or "").strip()
        if not mode:
            return {"ok": False, "device_id": int(device_id), "error": "mode is required"}

        async def _run() -> dict[str, Any]:
            return await self._thermostat_send_and_confirm_async(
                int(device_id),
                "SET_MODE_HOLD",
                {"MODE": mode},
                lambda r: (r.get("state") or {}).get("hold_mode") == mode,
                confirm_timeout_s=float(confirm_timeout_s),
            )

        return self._loop_thread.run(_run(), timeout_s=float(confirm_timeout_s) + 18.0)

    def thermostat_set_heat_setpoint_f(
        self, device_id: int, setpoint_f: float, confirm_timeout_s: float = 8.0
    ) -> dict[str, Any]:
        sp = self._coerce_float(setpoint_f)
        if sp is None:
            return {"ok": False, "device_id": int(device_id), "error": "setpoint_f must be a number"}

        target = float(round(sp))

        async def _run() -> dict[str, Any]:
            return await self._thermostat_send_and_confirm_async(
                int(device_id),
                "SET_SETPOINT_HEAT",
                {"FAHRENHEIT": str(int(round(target)))},
                lambda r: self._coerce_float((r.get("state") or {}).get("heat_setpoint_f")) == target,
                confirm_timeout_s=float(confirm_timeout_s),
            )

        return self._loop_thread.run(_run(), timeout_s=float(confirm_timeout_s) + 18.0)

    def thermostat_set_cool_setpoint_f(
        self, device_id: int, setpoint_f: float, confirm_timeout_s: float = 8.0
    ) -> dict[str, Any]:
        sp = self._coerce_float(setpoint_f)
        if sp is None:
            return {"ok": False, "device_id": int(device_id), "error": "setpoint_f must be a number"}

        target = float(round(sp))

        async def _run() -> dict[str, Any]:
            return await self._thermostat_send_and_confirm_async(
                int(device_id),
                "SET_SETPOINT_COOL",
                {"FAHRENHEIT": str(int(round(target)))},
                lambda r: self._coerce_float((r.get("state") or {}).get("cool_setpoint_f")) == target,
                confirm_timeout_s=float(confirm_timeout_s),
            )

        return self._loop_thread.run(_run(), timeout_s=float(confirm_timeout_s) + 18.0)

    def thermostat_set_target_f(
        self,
        device_id: int,
        target_f: float,
        confirm_timeout_s: float = 10.0,
        deadband_f: float | None = None,
    ) -> dict[str, Any]:
        """Set a target temperature without changing HVAC mode.

        - Heat: sets heat setpoint
        - Cool: sets cool setpoint
        - Auto: sets heat setpoint to target, cool setpoint to target+deadband

        Returns accepted/confirmed semantics based on observed variable updates.
        """
        sp = self._coerce_float(target_f)
        if sp is None:
            return {"ok": False, "device_id": int(device_id), "error": "target_f must be a number"}

        target = float(round(sp))

        async def _run() -> dict[str, Any]:
            before = await self._thermostat_get_state_async(int(device_id))
            st = before.get("state") if isinstance(before, dict) else None
            hvac_mode_raw = (st or {}).get("hvac_mode") if isinstance(st, dict) else None
            mode = str(hvac_mode_raw or "").strip().lower()
            if not mode:
                return {
                    "ok": False,
                    "device_id": int(device_id),
                    "error": "Could not determine hvac_mode",
                    "before": before,
                }
            if mode == "off":
                return {
                    "ok": False,
                    "device_id": int(device_id),
                    "error": "HVAC mode is Off; refusing to set target without changing mode",
                    "before": before,
                }

            def clamp(v: float, lo: float, hi: float) -> float:
                return float(max(lo, min(hi, v)))

            # Use provided deadband, else state deadband, else default 2F.
            db = self._coerce_float(deadband_f)
            if db is None:
                db = self._coerce_float((st or {}).get("heatcool_deadband_f")) if isinstance(st, dict) else None
            if db is None:
                db = 2.0
            db = float(max(1.0, round(db)))

            steps: list[dict[str, Any]] = []

            if mode == "heat":
                heat = clamp(target, 40.0, 90.0)
                r = await self._thermostat_send_and_confirm_async(
                    int(device_id),
                    "SET_SETPOINT_HEAT",
                    {"FAHRENHEIT": str(int(round(heat)))},
                    lambda x: self._coerce_float((x.get("state") or {}).get("heat_setpoint_f")) == heat,
                    confirm_timeout_s=float(confirm_timeout_s),
                )
                steps.append(r)
            elif mode == "cool":
                cool = clamp(target, 50.0, 99.0)
                r = await self._thermostat_send_and_confirm_async(
                    int(device_id),
                    "SET_SETPOINT_COOL",
                    {"FAHRENHEIT": str(int(round(cool)))},
                    lambda x: self._coerce_float((x.get("state") or {}).get("cool_setpoint_f")) == cool,
                    confirm_timeout_s=float(confirm_timeout_s),
                )
                steps.append(r)
            else:
                # Treat any non-off/non-heat/non-cool as auto.
                heat = clamp(target, 40.0, 90.0)
                cool = clamp(target + db, 50.0, 99.0)
                if cool < heat + db:
                    cool = clamp(heat + db, 50.0, 99.0)

                r_heat = await self._thermostat_send_and_confirm_async(
                    int(device_id),
                    "SET_SETPOINT_HEAT",
                    {"FAHRENHEIT": str(int(round(heat)))},
                    lambda x: self._coerce_float((x.get("state") or {}).get("heat_setpoint_f")) == heat,
                    confirm_timeout_s=float(confirm_timeout_s),
                )
                steps.append(r_heat)

                r_cool = await self._thermostat_send_and_confirm_async(
                    int(device_id),
                    "SET_SETPOINT_COOL",
                    {"FAHRENHEIT": str(int(round(cool)))},
                    lambda x: self._coerce_float((x.get("state") or {}).get("cool_setpoint_f")) == cool,
                    confirm_timeout_s=float(confirm_timeout_s),
                )
                steps.append(r_cool)

            accepted = all(bool(s.get("accepted")) for s in steps if isinstance(s, dict)) if steps else False
            confirmed = all(bool(s.get("confirmed")) for s in steps if isinstance(s, dict)) if steps else False
            ok = all(bool(s.get("ok")) for s in steps if isinstance(s, dict)) if steps else False

            # Best-effort after snapshot from last step.
            after = steps[-1].get("after") if steps and isinstance(steps[-1], dict) else None

            return {
                "ok": bool(ok),
                "device_id": int(device_id),
                "requested": {
                    "target_f": target,
                    "hvac_mode": str(hvac_mode_raw or ""),
                    "deadband_f": db,
                },
                "accepted": bool(accepted),
                "confirmed": bool(confirmed),
                "before": before,
                "after": after,
                "steps": steps,
            }

        return self._loop_thread.run(_run(), timeout_s=float(confirm_timeout_s) * 2.0 + 25.0)

    # ---- Media / AV ----

    @staticmethod
    def _media_state_from_vars(vars_by_name: dict[str, Any]) -> dict[str, Any]:
        def pick(*names: str) -> Any:
            for n in names:
                if n in vars_by_name:
                    return vars_by_name.get(n)
            return None

        power_state = pick("POWER_STATE", "POWER")
        power_i = Control4Gateway._coerce_int(power_state)

        return {
            "power_state": power_i if power_i is not None else power_state,
            "power_on": (power_i == 1) if power_i is not None else None,
            "app_name": pick("APP_NAME"),
            "status": pick("AppleTV Status", "STATUS"),
            "transports_supported": pick("TRANSPORTS_SUPPORTED"),
            "transports_buttons": pick("TRANSPORTS_BUTTONS"),
        }

    async def _media_get_state_async(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)
        fetched = await self._fetch_item_variables_list_async(device_id, timeout_s=4.0)
        var_list: list[dict[str, Any]] = (
            fetched.get("variables") if isinstance(fetched.get("variables"), list) else []
        )
        vars_by_name = self._vars_to_map(var_list)
        state = self._media_state_from_vars(vars_by_name)
        return {
            "ok": bool(fetched.get("ok")) or True,
            "device_id": device_id,
            "source": fetched.get("source") or "director.getItemVariables",
            "state": state,
        }

    def media_get_state(self, device_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._media_get_state_async(int(device_id)), timeout_s=12)

    @staticmethod
    def _media_now_playing_from_vars(vars_by_name: dict[str, Any]) -> dict[str, Any]:
        def pick(*names: str) -> Any:
            for n in names:
                if n in vars_by_name:
                    return vars_by_name.get(n)
            return None

        # Common-ish fields across drivers (best-effort only).
        now_playing: dict[str, Any] = {
            "title": pick("TITLE", "TRACK_TITLE", "MEDIA_TITLE", "NOW_PLAYING_TITLE"),
            "artist": pick("ARTIST", "TRACK_ARTIST", "MEDIA_ARTIST", "NOW_PLAYING_ARTIST"),
            "album": pick("ALBUM", "TRACK_ALBUM", "MEDIA_ALBUM", "NOW_PLAYING_ALBUM"),
            "station": pick("STATION", "STATION_NAME", "CHANNEL", "CHANNEL_NAME"),
            "source": pick("SOURCE", "CURRENT_SOURCE", "INPUT"),
            "play_state": pick("PLAY_STATE", "TRANSPORT_STATE", "STATE"),
            "power_state": pick("POWER_STATE", "POWER"),
        }

        # Control4 Digital Media often exposes queue-related structures.
        queue_info = pick("QUEUE_INFO_V2", "QUEUE_INFO")
        queue_status = pick("QUEUE_STATUS_V2", "QUEUE_STATUS")
        if queue_info is not None:
            now_playing["queue_info"] = queue_info
        if queue_status is not None:
            now_playing["queue_status"] = queue_status

        # Strip empty/null entries to keep output clean.
        return {k: v for k, v in now_playing.items() if v not in (None, "")}

    async def _media_get_now_playing_async(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)
        fetched = await self._fetch_item_variables_list_async(device_id, timeout_s=4.0)
        var_list: list[dict[str, Any]] = (
            fetched.get("variables") if isinstance(fetched.get("variables"), list) else []
        )
        vars_by_name = self._vars_to_map(var_list)
        now_playing = self._media_now_playing_from_vars(vars_by_name)

        # Include a small list of "candidate" variables so users can quickly see what exists.
        candidates: list[dict[str, Any]] = []
        for name, value in vars_by_name.items():
            n = str(name)
            nl = n.lower()
            if any(token in nl for token in ("title", "artist", "album", "track", "station", "channel", "queue", "song", "transport", "play")):
                candidates.append({"varName": n, "value": value})

        candidates.sort(key=lambda x: str(x.get("varName") or ""))

        return {
            "ok": bool(fetched.get("ok")) or True,
            "device_id": device_id,
            "source": fetched.get("source") or "director.getItemVariables",
            "now_playing": now_playing,
            "candidates": candidates[:60],
        }

    def media_get_now_playing(self, device_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._media_get_now_playing_async(int(device_id)), timeout_s=12)

    @staticmethod
    def _normalize_remote_press(press: str | None) -> str:
        p = str(press or "Tap").strip().lower()
        if p in ("tap", "short", "press"):
            return "Tap"
        if p in ("long", "long press", "hold"):
            return "Long Press"
        if p in ("down", "key down"):
            return "Down"
        if p in ("up", "key up"):
            return "Up"
        return "Tap"

    @staticmethod
    def _media_profile_from_item(item_name: str | None, commands: list[dict[str, Any]] | None) -> str:
        n = str(item_name or "").lower()
        if "apple tv" in n:
            return "apple_tv"
        if "roku" in n:
            return "roku"

        cmds = commands if isinstance(commands, list) else []
        cmd_names = {str(c.get("command") or "").lower() for c in cmds if isinstance(c, dict)}
        if "tvhome" in cmd_names or "playpause" in cmd_names:
            return "apple_tv"
        if "launchapp" in cmd_names:
            return "roku"
        return "generic"

    @staticmethod
    def _media_remote_mapping(profile: str) -> dict[str, str]:
        # Map friendly names -> Director command name.
        if profile == "roku":
            return {
                "up": "UP",
                "down": "DOWN",
                "left": "LEFT",
                "right": "RIGHT",
                "select": "SELECT",
                "ok": "SELECT",
                "enter": "SELECT",
                "home": "HOME",
                "back": "BACK",
                "menu": "BACK",
                "info": "INFO",
                "replay": "REPLAY",
                "instant_replay": "REPLAY",
                "recall": "RECALL",
                "prev": "RECALL",
                "play": "PLAY",
                "pause": "PAUSE",
                "ff": "SCAN_FWD",
                "scan_fwd": "SCAN_FWD",
                "rew": "SCAN_REV",
                "scan_rev": "SCAN_REV",
            }

        # Default: Apple TV-style (Gen 4/4K IP driver) expects a State param.
        return {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "select": "select",
            "ok": "select",
            "menu": "menu",
            "home": "tvhome",
            "tvhome": "tvhome",
            "playpause": "playpause",
            "play_pause": "playpause",
            "play": "playpause",
            "pause": "playpause",
            "volup": "volup",
            "volume_up": "volup",
            "voldown": "voldown",
            "volume_down": "voldown",
        }

    async def _media_remote_async(self, device_id: int, button: str, press: str | None = None) -> dict[str, Any]:
        device_id = int(device_id)
        b = str(button or "").strip().lower()
        if not b:
            return {"ok": False, "device_id": device_id, "error": "button is required"}

        fetched = await self._fetch_item_variables_list_async(device_id, timeout_s=4.0)
        var_list: list[dict[str, Any]] = (
            fetched.get("variables") if isinstance(fetched.get("variables"), list) else []
        )
        item_name = None
        if var_list and isinstance(var_list[0], dict):
            item_name = var_list[0].get("name")

        cmds_payload = await self._item_get_commands_async(device_id)
        cmd_list = cmds_payload.get("commands") if isinstance(cmds_payload.get("commands"), list) else []
        profile = self._media_profile_from_item(str(item_name) if item_name is not None else None, cmd_list)
        mapping = self._media_remote_mapping(profile)

        cmd = mapping.get(b)
        if not cmd:
            return {
                "ok": False,
                "device_id": device_id,
                "profile": profile,
                "error": f"Unsupported button '{button}'. Supported: {sorted(set(mapping.keys()))}",
            }

        normalized_press = self._normalize_remote_press(press)
        params: dict[str, Any] | None
        if profile == "apple_tv":
            params = {"State": normalized_press}
        else:
            # Roku/other drivers typically ignore a press-state param; treat as a tap.
            params = None

        exec_result = await self._media_send_command_async(device_id, cmd, params)

        return {
            "ok": bool(exec_result.get("ok")),
            "device_id": device_id,
            "profile": profile,
            "requested": {"button": str(button), "press": normalized_press, "command": cmd},
            "accepted": bool(exec_result.get("ok")),
            "execute": exec_result,
        }

    def media_remote(self, device_id: int, button: str, press: str | None = None) -> dict[str, Any]:
        return self._loop_thread.run(
            self._media_remote_async(int(device_id), str(button or ""), press),
            timeout_s=12,
        )

    async def _media_remote_sequence_async(
        self,
        device_id: int,
        buttons: list[str],
        press: str | None = None,
        delay_ms: int = 250,
    ) -> dict[str, Any]:
        device_id = int(device_id)
        if not isinstance(buttons, list) or not buttons:
            return {"ok": False, "device_id": device_id, "error": "buttons must be a non-empty list"}

        results: list[dict[str, Any]] = []
        for idx, btn in enumerate(buttons):
            r = await self._media_remote_async(device_id, str(btn or ""), press)
            results.append(r)
            if not bool(r.get("ok")):
                return {
                    "ok": False,
                    "device_id": device_id,
                    "error": "remote sequence step failed",
                    "failed_index": idx,
                    "failed_step": r,
                    "results": results,
                }
            await asyncio.sleep(max(0.0, float(delay_ms)) / 1000.0)

        return {"ok": True, "device_id": device_id, "count": len(results), "results": results}

    def media_remote_sequence(self, device_id: int, buttons: list[str], press: str | None = None, delay_ms: int = 250) -> dict[str, Any]:
        return self._loop_thread.run(
            self._media_remote_sequence_async(int(device_id), list(buttons), press, int(delay_ms)),
            timeout_s=20,
        )

    async def _media_launch_app_async(self, device_id: int, app: str) -> dict[str, Any]:
        device_id = int(device_id)
        a = str(app or "").strip()
        if not a:
            return {"ok": False, "device_id": device_id, "error": "app is required"}

        # For Roku, the same physical device can be represented by multiple items (protocol root,
        # media_service, media_player, avswitch). Users may call any of these item IDs, but we want
        # LaunchApp to be reliable regardless.
        async def _get_item_by_id(items: list[dict[str, Any]], item_id: int) -> dict[str, Any] | None:
            for row in items:
                if isinstance(row, dict) and int(row.get("id") or -1) == int(item_id):
                    return row
            return None

        async def _resolve_roku_targets(input_id: int) -> dict[str, Any]:
            # Use the fast HTTP items listing for routing to avoid the slower multi-method
            # pyControl4 enumeration path (which can accumulate several timeouts).
            http = await self._director_http_get("/api/v1/items")
            if http.get("ok"):
                items = self._normalize_items_payload(http.get("json"))
            else:
                items = await self._get_all_items_async()
            rec = await _get_item_by_id(items, int(input_id))
            if not isinstance(rec, dict):
                return {
                    "input_device_id": int(input_id),
                    "protocol_id": None,
                    "primary_device_id": int(input_id),
                    "targets": [int(input_id)],
                }

            protocol_filename = str(rec.get("protocolFilename") or "")
            protocol_name = str(rec.get("protocolName") or "")

            protocol_id = rec.get("protocolId")
            try:
                protocol_id_i = int(protocol_id)
            except Exception:
                protocol_id_i = int(input_id)

            room_id = rec.get("roomId")
            try:
                room_id_i = int(room_id)
            except Exception:
                room_id_i = -1

            group = [
                r
                for r in items
                if isinstance(r, dict) and int(r.get("protocolId") or -1) == protocol_id_i
            ]

            # Prefer the media_player proxy for control.
            media_players = [r for r in group if str(r.get("proxy") or "").lower() == "media_player"]
            if room_id_i != -1:
                media_players.sort(key=lambda r: (int(r.get("roomId") or -1) != room_id_i, int(r.get("proxyOrder") or 9999), int(r.get("id") or 0)))
            else:
                media_players.sort(key=lambda r: (int(r.get("proxyOrder") or 9999), int(r.get("id") or 0)))

            primary_id = int(media_players[0].get("id")) if media_players else int(input_id)

            # Build a stable target list; keep unique order.
            targets: list[int] = []
            for candidate in [
                primary_id,
                protocol_id_i,
                # service + switcher can also accept LaunchApp in some driver setups
                *[int(r.get("id")) for r in group if str(r.get("proxy") or "").lower() == "media_service"],
                *[int(r.get("id")) for r in group if str(r.get("proxy") or "").lower() == "avswitch"],
                int(input_id),
            ]:
                if candidate not in targets:
                    targets.append(candidate)

            return {
                "input_device_id": int(input_id),
                "protocol_id": protocol_id_i,
                "primary_device_id": primary_id,
                "room_id": room_id_i if room_id_i != -1 else None,
                "protocolFilename": protocol_filename,
                "protocolName": protocol_name,
                "targets": targets,
            }

        async def _roku_current_app(protocol_id: int) -> dict[str, Any] | None:
            # Prefer HTTP for polling; pyControl4 variable reads can be slow and stack timeouts.
            http = await self._director_http_get(f"/api/v1/items/{int(protocol_id)}/variables")
            if not http.get("ok"):
                return None
            payload = http.get("json")
            var_list = payload.get("variables") if isinstance(payload, dict) else payload
            if not isinstance(var_list, list):
                return None
            if not var_list:
                return None
            current_app = self._get_var_value(var_list, "CURRENT_APP")
            current_app_id = self._get_var_value(var_list, "CURRENT_APP_ID")
            try:
                current_app_id_i = int(current_app_id)
            except Exception:
                current_app_id_i = None
            return {
                "CURRENT_APP": current_app,
                "CURRENT_APP_ID": current_app_id,
                "CURRENT_APP_ID_INT": current_app_id_i,
            }

        # Resolve Roku routing early so we can avoid slow profile inference when it's clearly Roku.
        routing_hint = await _resolve_roku_targets(device_id)
        is_roku_hint = "roku" in str(routing_hint.get("protocolFilename") or "").lower()

        # Try to infer profile so we can apply Roku-specific behavior.
        if is_roku_hint:
            profile = "roku"
        else:
            fetched = await self._fetch_item_variables_list_async(device_id, timeout_s=4.0)
            var_list: list[dict[str, Any]] = fetched.get("variables") if isinstance(fetched.get("variables"), list) else []
            item_name = None
            if var_list and isinstance(var_list[0], dict):
                item_name = var_list[0].get("name")

            cmds_payload = await self._item_get_commands_async(device_id)
            cmd_list = cmds_payload.get("commands") if isinstance(cmds_payload.get("commands"), list) else []
            profile = self._media_profile_from_item(str(item_name) if item_name is not None else None, cmd_list)

        resolved: dict[str, Any] | None = None
        app_param: Any = a

        # If the app looks numeric (e.g., Roku channel id), pass as an int.
        if a.isdigit():
            app_param = int(a)
            resolved = {"mode": "direct_id", "app": a, "roku_app_id": int(a)}

        # Otherwise, for Roku, try to resolve app name -> UM_ROKU id from universal mini-app variables.
        if resolved is None and profile == "roku":
            apps_resp = await self._media_roku_list_apps_async(device_id, search=a)
            apps = apps_resp.get("apps") if isinstance(apps_resp, dict) else None
            if isinstance(apps, list) and apps:
                target_l = a.lower()
                # Prefer exact match, then contains match.
                exact = [row for row in apps if isinstance(row, dict) and str(row.get("app_name") or "").lower() == target_l]
                cand = exact[0] if exact else apps[0]

                roku_app_id = cand.get("roku_app_id")
                if isinstance(roku_app_id, int):
                    app_param = roku_app_id
                    resolved = {
                        "mode": "resolved_name",
                        "app": a,
                        "resolved_app_name": cand.get("app_name"),
                        "roku_app_id": roku_app_id,
                        "source_item_id": cand.get("item_id"),
                    }

        # Roku drivers commonly support LaunchApp with param name 'App'.
        expected_roku_app_id: int | None = None
        if isinstance(app_param, int):
            expected_roku_app_id = int(app_param)

        attempts: list[dict[str, Any]] = []

        # For Roku, broadcast LaunchApp across the protocol group until the Roku reports the expected app.
        if profile == "roku":
            routing = routing_hint
            protocol_id = routing.get("protocol_id")
            if isinstance(protocol_id, int):
                before = await _roku_current_app(protocol_id)
            else:
                before = None

            director = await self._director_async()

            for target_id in routing.get("targets") if isinstance(routing.get("targets"), list) else [device_id]:
                try:
                    target_i = int(target_id)
                except Exception:
                    continue
                uri = f"/api/v1/items/{target_i}/commands"
                exec_result = await self._send_post_via_director(director, uri, "LaunchApp", {"App": app_param}, async_variable=False)
                attempts.append({"target_device_id": target_i, "execute": exec_result})
                if isinstance(protocol_id, int) and expected_roku_app_id is not None:
                    await asyncio.sleep(0.6)
                    snap = await _roku_current_app(protocol_id)
                    attempts[-1]["roku_after"] = snap
                    if isinstance(snap, dict) and snap.get("CURRENT_APP_ID_INT") == expected_roku_app_id:
                        break

            after = await (_roku_current_app(protocol_id) if isinstance(protocol_id, int) else asyncio.sleep(0) or None)

            ok = False
            if isinstance(after, dict) and expected_roku_app_id is not None:
                ok = after.get("CURRENT_APP_ID_INT") == expected_roku_app_id
            else:
                ok = any(bool(a.get("execute", {}).get("ok")) for a in attempts)

            out: dict[str, Any] = {
                "ok": bool(ok),
                "device_id": device_id,
                "profile": profile,
                "requested": {"app": a, "command": "LaunchApp"},
                "accepted": bool(ok),
                "routing": routing,
                "roku": {"before": before, "after": after, "expected_app_id": expected_roku_app_id},
                "attempts": attempts,
            }
        else:
            exec_result = await self._media_send_command_async(device_id, "LaunchApp", {"App": app_param})
            out = {
                "ok": bool(exec_result.get("ok")),
                "device_id": device_id,
                "profile": profile,
                "requested": {"app": a, "command": "LaunchApp"},
                "accepted": bool(exec_result.get("ok")),
                "execute": exec_result,
            }
        if resolved is not None:
            out["resolved"] = resolved
        return out

    def media_launch_app(self, device_id: int, app: str) -> dict[str, Any]:
        # Roku LaunchApp may broadcast across multiple proxy items and poll for CURRENT_APP_ID.
        # Give it a bit more time than a single command.
        return self._loop_thread.run(self._media_launch_app_async(int(device_id), str(app or "")), timeout_s=25)

    async def _get_all_items_variable_values_async(self, var_names: list[str]) -> dict[str, Any]:
        director = await self._director_async()
        names = [str(n or "").strip() for n in (var_names or []) if str(n or "").strip()]
        if not names:
            return {"ok": False, "error": "var_names must be non-empty", "variables": []}

        if hasattr(director, "getAllItemVariableValue"):
            try:
                res = await asyncio.wait_for(director.getAllItemVariableValue(names), timeout=self._director_timeout_s)
                if isinstance(res, str):
                    try:
                        res = json.loads(res)
                    except Exception:
                        return {"ok": False, "source": "director.getAllItemVariableValue", "raw": res, "variables": []}
                if isinstance(res, list):
                    cleaned = [row for row in res if isinstance(row, dict)]
                    return {"ok": True, "source": "director.getAllItemVariableValue", "variables": cleaned}
                return {
                    "ok": False,
                    "source": "director.getAllItemVariableValue",
                    "error": f"Unexpected response type: {type(res).__name__}",
                    "raw": res,
                    "variables": [],
                }
            except asyncio.TimeoutError:
                return {"ok": False, "source": "director.getAllItemVariableValue", "error": "timeout", "variables": []}
            except Exception as e:
                return {
                    "ok": False,
                    "source": "director.getAllItemVariableValue",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "variables": [],
                }

        q = ",".join(names)
        http = await self._director_http_get(f"/api/v1/items/variables?varnames={q}")
        if http.get("ok"):
            payload = http.get("json")
            if isinstance(payload, list):
                cleaned = [row for row in payload if isinstance(row, dict)]
                return {"ok": True, "source": "http:/api/v1/items/variables", "variables": cleaned}
        return {"ok": False, "source": "http:/api/v1/items/variables", **http, "variables": []}

    async def _media_roku_list_apps_async(self, device_id: int, search: str | None = None) -> dict[str, Any]:
        device_id = int(device_id)
        fetched = await self._fetch_item_variables_list_async(device_id, timeout_s=4.0)
        var_list: list[dict[str, Any]] = fetched.get("variables") if isinstance(fetched.get("variables"), list) else []

        room_name = None
        if var_list and isinstance(var_list[0], dict):
            room_name = var_list[0].get("roomName")
        room_name_s = str(room_name or "").strip()
        if not room_name_s:
            return {"ok": False, "device_id": device_id, "error": "Could not determine roomName for device", "source": fetched.get("source")}

        all_vars = await self._get_all_items_variable_values_async(["APP_NAME", "UM_ROKU"])
        rows = all_vars.get("variables") if isinstance(all_vars.get("variables"), list) else []

        grouped: dict[int, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("roomName") or "").strip() != room_name_s:
                continue
            item_id = row.get("id")
            if item_id is None:
                continue
            try:
                iid = int(item_id)
            except Exception:
                continue

            rec = grouped.setdefault(iid, {"item_id": iid, "roomName": room_name_s, "name": row.get("name")})
            var_name = str(row.get("varName") or "").strip().upper()
            if var_name == "APP_NAME":
                rec["app_name"] = row.get("value")
            elif var_name == "UM_ROKU":
                v = row.get("value")
                try:
                    rec["roku_app_id"] = int(v)
                except Exception:
                    rec["roku_app_id"] = v

        query = str(search or "").strip().lower()
        apps: list[dict[str, Any]] = []
        for rec in grouped.values():
            if "roku_app_id" not in rec:
                continue
            app_name = str(rec.get("app_name") or "").strip()
            if not app_name:
                continue
            if query and query not in app_name.lower():
                continue
            if not isinstance(rec.get("roku_app_id"), int):
                continue
            apps.append({"item_id": rec.get("item_id"), "app_name": app_name, "roku_app_id": rec.get("roku_app_id")})

        apps.sort(key=lambda a: str(a.get("app_name") or "").lower())
        return {
            "ok": True,
            "device_id": device_id,
            "roomName": room_name_s,
            "search": str(search or "") if search is not None else None,
            "count": len(apps),
            "apps": apps,
            "source": all_vars.get("source") or "director.getAllItemVariableValue",
        }

    def media_roku_list_apps(self, device_id: int, search: str | None = None) -> dict[str, Any]:
        return self._loop_thread.run(self._media_roku_list_apps_async(int(device_id), search), timeout_s=18)

    async def _media_send_command_async(
        self,
        device_id: int,
        command: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        device_id = int(device_id)
        cmd = str(command or "").strip()
        if not cmd:
            return {"ok": False, "device_id": device_id, "error": "command is required"}

        # Special-case Roku LaunchApp: users may pass any of the Roku-related item IDs.
        cmd_u = cmd.upper()
        if cmd_u in {"LAUNCHAPP", "LAUNCH_APP"}:
            p = params or {}
            app_val = None
            if isinstance(p, dict):
                app_val = p.get("App") if "App" in p else p.get("APP")

            # If we don't have an App param, fall back to raw send.
            if app_val is not None:
                # Determine whether this looks like a Roku protocol group.
                http = await self._director_http_get("/api/v1/items")
                if http.get("ok"):
                    items = self._normalize_items_payload(http.get("json"))
                else:
                    items = await self._get_all_items_async()

                rec = next((r for r in items if isinstance(r, dict) and int(r.get("id") or -1) == device_id), None)
                proto_fn = str((rec or {}).get("protocolFilename") or "").lower()
                if "roku" in proto_fn:
                    # Resolve all items in this Roku protocol group and broadcast LaunchApp.
                    protocol_id = None
                    try:
                        protocol_id = int((rec or {}).get("protocolId") or device_id)
                    except Exception:
                        protocol_id = None

                    group = [
                        r
                        for r in items
                        if isinstance(r, dict) and protocol_id is not None and int(r.get("protocolId") or -1) == int(protocol_id)
                    ]

                    targets: list[int] = []
                    # Prefer media_player first.
                    for r in group:
                        if str(r.get("proxy") or "").lower() == "media_player":
                            targets.append(int(r.get("id")))
                    # Then protocol root + other proxies.
                    if protocol_id is not None:
                        targets.append(int(protocol_id))
                    for r in group:
                        pid = int(r.get("id"))
                        if pid not in targets:
                            targets.append(pid)
                    if device_id not in targets:
                        targets.append(int(device_id))

                    # De-dupe while preserving order.
                    uniq: list[int] = []
                    for t in targets:
                        if t not in uniq:
                            uniq.append(t)
                    targets = uniq

                    director = await self._director_async()
                    attempts: list[dict[str, Any]] = []
                    any_ok = False
                    for t in targets:
                        uri = f"/api/v1/items/{int(t)}/commands"
                        r = await self._send_post_via_director(director, uri, "LaunchApp", {"App": app_val}, async_variable=False)
                        attempts.append({"target_device_id": int(t), "execute": r})
                        any_ok = any_ok or bool(r.get("ok"))
                    return {
                        "ok": bool(any_ok),
                        "device_id": int(device_id),
                        "command": "LaunchApp",
                        "routing": {
                            "protocol_id": protocol_id,
                            "targets": targets,
                        },
                        "attempts": attempts,
                    }

        director = await self._director_async()
        uri = f"/api/v1/items/{device_id}/commands"
        return await self._send_post_via_director(director, uri, cmd, params or {}, async_variable=False)

    def media_send_command(self, device_id: int, command: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._loop_thread.run(
            self._media_send_command_async(int(device_id), str(command or ""), params),
            timeout_s=12,
        )

    async def _light_get_state_async(self, device_id: int) -> bool:
        fetched = await self._fetch_item_variables_list_async(int(device_id))
        variables: list[dict[str, Any]] = fetched.get("variables") if isinstance(fetched.get("variables"), list) else []

        v = self._get_var_value(variables, "LIGHT_STATE")
        b = self._coerce_bool(v)
        if b is not None:
            return bool(b)

        # Fallback: if we have a brightness level, infer on/off
        lvl = self._get_var_value(variables, "Brightness Percent")
        li = self._coerce_int(lvl)
        if li is not None:
            return li > 0

        return False

    def light_get_state(self, device_id: int) -> bool:
        return bool(self._loop_thread.run(self._light_get_state_async(int(device_id)), timeout_s=10))

    async def _light_get_level_async(self, device_id: int) -> int:
        fetched = await self._fetch_item_variables_list_async(int(device_id))
        variables: list[dict[str, Any]] = fetched.get("variables") if isinstance(fetched.get("variables"), list) else []

        for name in ("Brightness Percent", "PRESET_LEVEL"):
            v = self._get_var_value(variables, name)
            li = self._coerce_int(v)
            if li is not None:
                return max(0, min(100, int(li)))
        return 0

    def light_get_level(self, device_id: int) -> int:
        return int(self._loop_thread.run(self._light_get_level_async(int(device_id)), timeout_s=10))

    async def _light_set_level_async(self, device_id: int, level: int) -> bool:
        level = max(0, min(100, int(level)))
        res = await self._item_send_command_async(int(device_id), "SET_LEVEL", {"LEVEL": level})
        return bool(res.get("ok"))

    def light_set_level(self, device_id: int, level: int) -> bool:
        return bool(self._loop_thread.run(self._light_set_level_async(int(device_id), int(level)), timeout_s=12))

    async def _light_ramp_async(self, device_id: int, level: int, time_ms: int) -> bool:
        level = max(0, min(100, int(level)))
        time_ms = max(0, int(time_ms))
        res = await self._item_send_command_async(int(device_id), "RAMP_TO_LEVEL", {"LEVEL": level, "TIME": time_ms})
        return bool(res.get("ok"))

    def light_ramp(self, device_id: int, level: int, time_ms: int) -> bool:
        return bool(
            self._loop_thread.run(self._light_ramp_async(int(device_id), int(level), int(time_ms)), timeout_s=12)
        )