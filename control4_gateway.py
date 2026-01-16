# control4_gateway.py

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
from difflib import SequenceMatcher
import inspect
import json
import os
import re
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
        env_host = (os.environ.get("C4_HOST") or os.environ.get("CONTROL4_HOST") or "").strip()
        env_user = (os.environ.get("C4_USERNAME") or os.environ.get("CONTROL4_USERNAME") or "").strip()
        env_pass = (os.environ.get("C4_PASSWORD") or os.environ.get("CONTROL4_PASSWORD") or "").strip()

        if env_host or env_user or env_pass:
            missing = [
                k
                for k, v in {
                    "C4_HOST": env_host,
                    "C4_USERNAME": env_user,
                    "C4_PASSWORD": env_pass,
                }.items()
                if not v
            ]
            if missing:
                raise RuntimeError(
                    "Incomplete Control4 config from environment. Missing: "
                    + ", ".join(missing)
                    + ". Provide all three env vars (or none)."
                )
            return Config(host=env_host, username=env_user, password=env_pass)

        env_cfg = (os.environ.get("C4_CONFIG_PATH") or os.environ.get("CONTROL4_CONFIG_PATH") or "").strip()
        path = Path(cfg_path or env_cfg) if (cfg_path or env_cfg) else Path(__file__).with_name("config.json")
        if not path.exists():
            raise RuntimeError(
                "Missing Control4 config. Set env vars C4_HOST/C4_USERNAME/C4_PASSWORD, "
                "or create config.json (or set C4_CONFIG_PATH)."
            )

        data = json.loads(path.read_text(encoding="utf-8"))
        try:
            host = str(data["host"]).strip()
            username = str(data["username"]).strip()
            password = str(data["password"]).strip()
        except Exception as e:
            raise RuntimeError(f"Invalid config file {str(path)!r}: expected keys host/username/password") from e

        if not host or not username or not password:
            raise RuntimeError(f"Invalid config file {str(path)!r}: host/username/password must be non-empty")

        return Config(host=host, username=username, password=password)

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

    async def _director_http_post_https_only(self, path: str, payload: dict | None = None, timeout_s: float | None = None) -> dict[str, Any]:
        """POST to Director using HTTPS base only.

        Some endpoints behave poorly when we fall back to HTTP. Scheduler writes in particular can be slow,
        so we allow an explicit timeout override.
        """

        token = await self._ensure_director_token_async()
        headers = {"Authorization": f"Bearer {token}"}
        timeout = aiohttp.ClientTimeout(total=(float(timeout_s) if timeout_s is not None else self._http_timeout_s))

        base = self._director_base_urls()[0]
        url = urljoin(base, path.lstrip("/"))
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

    async def _director_http_request_https_only(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Perform an authenticated HTTPS-only JSON request.

        We intentionally avoid HTTP fallback here because certain Director endpoints (notably scheduler writes)
        behave inconsistently when accessed over HTTP.
        """

        verb = str(method or "").strip().upper()
        if verb not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            return {"ok": False, "status": 0, "url": "", "json": None, "text": "", "error": f"Unsupported method: {verb}"}

        token = await self._ensure_director_token_async()
        headers = {"Authorization": f"Bearer {token}"}
        timeout = aiohttp.ClientTimeout(total=(float(timeout_s) if timeout_s is not None else self._http_timeout_s))

        base = self._director_base_urls()[0]
        url = urljoin(base, path.lstrip("/"))

        try:
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.request(verb, url, headers=headers, json=(payload or {}), ssl=False) as r:
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

    async def _item_send_command_preserve_async(
        self, device_id: int, command: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send an item command without uppercasing.

        Some Control4 endpoints expose human-readable command names (e.g. door station commands)
        where preserving case/spaces is safer.
        """

        device_id = int(device_id)
        command_s = str(command or "").strip()
        if not command_s:
            return {"ok": False, "device_id": device_id, "error": "command is required"}

        director = await self._director_async()
        uri = f"/api/v1/items/{device_id}/commands"
        return await self._send_post_via_director(director, uri, command_s, params or {}, async_variable=False)

    async def _agent_send_command_async(
        self, agent: str, command: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        agent_s = str(agent or "").strip().strip("/")
        command_s = str(command or "").strip()
        if not agent_s:
            return {"ok": False, "error": "agent is required"}
        if not command_s:
            return {"ok": False, "agent": agent_s, "error": "command is required"}

        director = await self._director_async()
        uri = f"/api/v1/agents/{agent_s}/commands"
        primary = await self._send_post_via_director(director, uri, command_s, params or {}, async_variable=False)
        if primary.get("ok"):
            return primary

        # Fallback to direct HTTP if pyControl4 chokes on the response body.
        body = {"async": False, "command": command_s, "tParams": (params or {})}
        http = await self._director_http_post(uri, body)
        return {"ok": bool(http.get("ok")), "uri": uri, "command": command_s, "http": http}

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

    async def _room_list_commands_async(self, room_id: int, search: str | None = None) -> dict[str, Any]:
        room_id = int(room_id)
        needle = str(search or "").strip().lower()

        http = await self._director_http_get(f"/api/v1/rooms/{room_id}/commands")
        if not http.get("ok"):
            return {
                "ok": False,
                "room_id": room_id,
                "search": (str(search) if search is not None else None),
                "error": "failed to fetch room commands",
                "http": http,
                "commands": [],
            }

        payload = http.get("json")
        cmds = payload if isinstance(payload, list) else (payload.get("commands") if isinstance(payload, dict) else None)
        if not isinstance(cmds, list):
            cmds = []

        normalized = [c for c in cmds if isinstance(c, dict)]
        if needle:
            def _hay(row: dict[str, Any]) -> str:
                return " ".join([
                    str(row.get("command") or ""),
                    str(row.get("label") or ""),
                    str(row.get("display") or ""),
                ]).lower()

            normalized = [c for c in normalized if needle in _hay(c)]

        return {
            "ok": True,
            "room_id": room_id,
            "search": (str(search) if search is not None else None),
            "count": len(normalized),
            "commands": normalized,
            "source": "director.http.get",
        }

    async def _room_select_video_device_async(self, room_id: int, device_id: int, deselect: bool = False) -> dict[str, Any]:
        room_id = int(room_id)
        device_id = int(device_id)
        params = {"deviceid": device_id, "deselect": (1 if deselect else 0)}
        return await self._room_send_command_async(room_id, "SELECT_VIDEO_DEVICE", params)

    async def _room_select_audio_device_async(self, room_id: int, device_id: int, deselect: bool = False) -> dict[str, Any]:
        room_id = int(room_id)
        device_id = int(device_id)
        params = {"deviceid": device_id, "deselect": (1 if deselect else 0)}
        return await self._room_send_command_async(room_id, "SELECT_AUDIO_DEVICE", params)

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

    @staticmethod
    def _find_ui_listen_node(payload: Any, room_id: int) -> dict[str, Any] | None:
        if isinstance(payload, dict):
            if payload.get("type") == "listen" and int(payload.get("room_id") or -1) == int(room_id):
                return payload
            for v in payload.values():
                found = Control4Gateway._find_ui_listen_node(v, room_id)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for it in payload:
                found = Control4Gateway._find_ui_listen_node(it, room_id)
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

    async def _ui_listen_status_async(self, room_id: int) -> dict[str, Any] | None:
        room_id = int(room_id)
        http = await self._director_http_get("/api/v1/agents/ui_configuration")
        if not http.get("ok"):
            return None
        payload = http.get("json")
        node = self._find_ui_listen_node(payload, room_id)
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

        async def _resolve_watch_source_device_id(input_device_id: int, resolved_room_id: int) -> int:
            """Resolve the best device_id to pass to SELECT_VIDEO_DEVICE.

            For Roku, callers may pass any of the protocol group item ids (media_service/media_player/avswitch).
            Watch selection typically needs the AVSwitch (app switcher) or media_player item, not the media_service.
            """
            input_device_id = int(input_device_id)
            resolved_room_id = int(resolved_room_id)

            http = await self._director_http_get("/api/v1/items")
            items = self._normalize_items_payload(http.get("json")) if http.get("ok") else await self._get_all_items_async()
            rec = next(
                (
                    r
                    for r in items
                    if isinstance(r, dict) and int(r.get("id") or -1) == input_device_id
                ),
                None,
            )
            if not isinstance(rec, dict):
                return input_device_id

            protocol_filename = str(rec.get("protocolFilename") or "")
            protocol_id = rec.get("protocolId")
            try:
                protocol_id_i = int(protocol_id)
            except Exception:
                protocol_id_i = input_device_id

            is_roku = "roku" in protocol_filename.lower() or "roku" in str(rec.get("filename") or "").lower()
            if not is_roku:
                return input_device_id

            group = [
                r
                for r in items
                if isinstance(r, dict) and int(r.get("protocolId") or -1) == protocol_id_i
            ]

            def in_room(row: dict[str, Any]) -> bool:
                try:
                    return int(row.get("roomId") or -1) == resolved_room_id
                except Exception:
                    return False

            def id_of(rows: list[dict[str, Any]]) -> int | None:
                for row in rows:
                    try:
                        return int(row.get("id") or 0)
                    except Exception:
                        continue
                return None

            # Prefer AVSwitch (Roku App Switcher) in this room.
            avswitch = [r for r in group if str(r.get("proxy") or "").lower() == "avswitch" and in_room(r)]
            picked = id_of(avswitch)
            if picked is not None:
                return picked

            # Next prefer media_player in this room.
            media_player = [r for r in group if str(r.get("proxy") or "").lower() == "media_player" and in_room(r)]
            picked = id_of(media_player)
            if picked is not None:
                return picked

            # Fall back to protocol root if present; otherwise the input id.
            return int(protocol_id_i) if protocol_id_i else input_device_id

        before_watch = await self._ui_watch_status_async(resolved_room_id)

        watch_source_device_id = await _resolve_watch_source_device_id(device_id, resolved_room_id)

        select_video = await self._room_select_video_device_async(resolved_room_id, watch_source_device_id, deselect=False)
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
            "watch_source_device_id": watch_source_device_id,
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

    def room_list_commands(self, room_id: int, search: str | None = None) -> dict[str, Any]:
        return self._loop_thread.run(
            self._room_list_commands_async(int(room_id), (str(search) if search is not None else None)),
            timeout_s=15,
        )

    def room_select_video_device(self, room_id: int, device_id: int, deselect: bool = False) -> dict[str, Any]:
        return self._loop_thread.run(
            self._room_select_video_device_async(int(room_id), int(device_id), deselect=bool(deselect)),
            timeout_s=12,
        )

    def room_select_audio_device(self, room_id: int, device_id: int, deselect: bool = False) -> dict[str, Any]:
        return self._loop_thread.run(
            self._room_select_audio_device_async(int(room_id), int(device_id), deselect=bool(deselect)),
            timeout_s=12,
        )

    def room_listen_status(self, room_id: int) -> dict[str, Any]:
        status = self._loop_thread.run(self._ui_listen_status_async(int(room_id)), timeout_s=12)
        if not isinstance(status, dict):
            return {"ok": False, "room_id": int(room_id), "error": "Listen UI status not available"}
        return {"ok": True, "room_id": int(room_id), "listen": status}

    async def _room_listen_async(self, room_id: int, device_id: int, confirm_timeout_s: float = 10.0) -> dict[str, Any]:
        room_id = int(room_id)
        device_id = int(device_id)

        before_listen = await self._ui_listen_status_async(room_id)
        execute = await self._room_select_audio_device_async(room_id, device_id, deselect=False)
        accepted = bool(execute.get("ok"))
        if not accepted:
            return {
                "ok": False,
                "room_id": room_id,
                "device_id": device_id,
                "accepted": False,
                "confirmed": False,
                "listen": {"before": before_listen, "after": before_listen},
                "execute": execute,
            }

        # Best-effort: confirm the room is actively listening.
        deadline = asyncio.get_running_loop().time() + float(confirm_timeout_s)
        last_listen = before_listen
        confirmed = False

        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.6)
            last_listen = await self._ui_listen_status_async(room_id)
            if isinstance(last_listen, dict) and last_listen.get("active") is True:
                confirmed = True
                break

        return {
            "ok": True,
            "room_id": room_id,
            "device_id": device_id,
            "accepted": True,
            "confirmed": bool(confirmed),
            "listen": {"before": before_listen, "after": last_listen},
            "execute": execute,
            "confirm_timeout_s": float(confirm_timeout_s),
        }

    def room_listen(self, room_id: int, device_id: int, confirm_timeout_s: float = 10.0) -> dict[str, Any]:
        return self._loop_thread.run(
            self._room_listen_async(int(room_id), int(device_id), confirm_timeout_s=float(confirm_timeout_s)),
            timeout_s=float(confirm_timeout_s) + 18.0,
        )

    async def _room_off_async(self, room_id: int, confirm_timeout_s: float = 10.0) -> dict[str, Any]:
        room_id = int(room_id)
        before_watch = await self._ui_watch_status_async(room_id)

        execute = await self._room_send_command_async(room_id, "ROOM_OFF", {})
        accepted = bool(execute.get("ok"))
        if not accepted:
            return {
                "ok": False,
                "room_id": room_id,
                "accepted": False,
                "confirmed": False,
                "watch": {"before": before_watch, "after": before_watch},
                "execute": execute,
            }

        # Best-effort: confirm the room is no longer actively watching.
        deadline = asyncio.get_running_loop().time() + float(confirm_timeout_s)
        last_watch = before_watch
        confirmed = False

        while asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.6)
            last_watch = await self._ui_watch_status_async(room_id)
            if isinstance(last_watch, dict) and last_watch.get("active") is False:
                confirmed = True
                break

        return {
            "ok": True,
            "room_id": room_id,
            "accepted": True,
            "confirmed": bool(confirmed),
            "watch": {"before": before_watch, "after": last_watch},
            "execute": execute,
            "confirm_timeout_s": float(confirm_timeout_s),
        }

    def room_off(self, room_id: int, confirm_timeout_s: float = 10.0) -> dict[str, Any]:
        # Room off can take a moment to reflect in UI configuration; give it some extra headroom.
        return self._loop_thread.run(
            self._room_off_async(int(room_id), confirm_timeout_s=float(confirm_timeout_s)),
            timeout_s=float(confirm_timeout_s) + 18.0,
        )

    @staticmethod
    def _room_remote_mapping() -> dict[str, str]:
        # Friendly names -> room command.
        # Prefer room commands because they work regardless of the underlying TV/receiver driver.
        return {
            "up": "UP",
            "down": "DOWN",
            "left": "LEFT",
            "right": "RIGHT",
            "select": "ENTER",
            "ok": "ENTER",
            "enter": "ENTER",
            "back": "BACK",
            "menu": "MENU",
            "info": "INFO",
            "exit": "EXIT",
            "guide": "GUIDE",
            "play": "PLAY",
            "pause": "PAUSE",
            "ff": "SCAN_FWD",
            "scan_fwd": "SCAN_FWD",
            "rew": "SCAN_REV",
            "scan_rev": "SCAN_REV",
            "recall": "RECALL",
            "prev": "RECALL",
            "page_up": "PAGE_UP",
            "page_down": "PAGE_DOWN",
            "volup": "PULSE_VOL_UP",
            "volume_up": "PULSE_VOL_UP",
            "voldown": "PULSE_VOL_DOWN",
            "volume_down": "PULSE_VOL_DOWN",
            "mute": "MUTE_TOGGLE",
            "mute_toggle": "MUTE_TOGGLE",
            "ch_up": "PULSE_CH_UP",
            "channel_up": "PULSE_CH_UP",
            "ch_down": "PULSE_CH_DOWN",
            "channel_down": "PULSE_CH_DOWN",
            "power_off": "ROOM_OFF",
            "off": "ROOM_OFF",
            "room_off": "ROOM_OFF",
        }

    async def _room_remote_async(self, room_id: int, button: str, press: str | None = None) -> dict[str, Any]:
        room_id = int(room_id)
        b = str(button or "").strip().lower()
        if not b:
            return {"ok": False, "room_id": room_id, "error": "button is required"}

        normalized_press = self._normalize_remote_press(press)
        mapping = self._room_remote_mapping()
        base_cmd = mapping.get(b)
        if not base_cmd:
            return {
                "ok": False,
                "room_id": room_id,
                "error": f"Unsupported button '{button}'. Supported: {sorted(set(mapping.keys()))}",
            }

        # Support press semantics for commands that have start/stop variants.
        cmd = base_cmd
        if base_cmd in {"PULSE_VOL_UP", "PULSE_VOL_DOWN", "PAGE_UP", "PAGE_DOWN", "PULSE_CH_UP", "PULSE_CH_DOWN"}:
            variants = {
                "PULSE_VOL_UP": ("PULSE_VOL_UP", "START_VOL_UP", "STOP_VOL_UP"),
                "PULSE_VOL_DOWN": ("PULSE_VOL_DOWN", "START_VOL_DOWN", "STOP_VOL_DOWN"),
                "PAGE_UP": ("PAGE_UP", "START_PAGE_UP", "STOP_PAGE_UP"),
                "PAGE_DOWN": ("PAGE_DOWN", "START_PAGE_DOWN", "STOP_PAGE_DOWN"),
                "PULSE_CH_UP": ("PULSE_CH_UP", "START_CH_UP", "STOP_CH_UP"),
                "PULSE_CH_DOWN": ("PULSE_CH_DOWN", "START_CH_DOWN", "STOP_CH_DOWN"),
            }
            pulse, start, stop = variants[base_cmd]
            if normalized_press == "Down":
                cmd = start
            elif normalized_press == "Up":
                cmd = stop
            else:
                cmd = pulse

        exec_result = await self._room_send_command_async(room_id, cmd, {})
        return {
            "ok": bool(exec_result.get("ok")),
            "room_id": room_id,
            "requested": {"button": str(button), "press": normalized_press, "command": cmd},
            "accepted": bool(exec_result.get("ok")),
            "execute": exec_result,
        }

    def room_remote(self, room_id: int, button: str, press: str | None = None) -> dict[str, Any]:
        return self._loop_thread.run(
            self._room_remote_async(int(room_id), str(button or ""), press),
            timeout_s=12,
        )

    def media_watch_launch_app(self, device_id: int, app: str, room_id: int | None = None, pre_home: bool = True) -> dict[str, Any]:
        return self._loop_thread.run(
            self._media_watch_launch_app_async(int(device_id), str(app or ""), room_id=(int(room_id) if room_id is not None else None), pre_home=bool(pre_home)),
            timeout_s=45,
        )

    async def _capabilities_report_async(self, top_n: int = 20, include_examples: bool = False, max_examples_per_bucket: int = 3) -> dict[str, Any]:
        try:
            top_n_i = max(1, min(int(top_n), 200))
        except Exception:
            top_n_i = 20

        include_examples_b = bool(include_examples)
        try:
            max_examples_i = max(0, min(int(max_examples_per_bucket), 10))
        except Exception:
            max_examples_i = 3

        http = await self._director_http_get("/api/v1/items")
        items = self._normalize_items_payload(http.get("json")) if http.get("ok") else await self._get_all_items_async()

        def _bucket(v: Any) -> str:
            s = str(v or "").strip()
            return s if s else "(none)"

        controls: dict[str, int] = {}
        proxies: dict[str, int] = {}
        protocol_files: dict[str, int] = {}
        rooms: dict[str, int] = {}

        examples: dict[str, list[dict[str, Any]]] = {}

        for r in items:
            if not isinstance(r, dict):
                continue
            control = _bucket(r.get("control"))
            proxy = _bucket(r.get("proxy"))
            proto = _bucket(r.get("protocolFilename"))
            room = _bucket(r.get("roomName"))

            controls[control] = controls.get(control, 0) + 1
            proxies[proxy] = proxies.get(proxy, 0) + 1
            protocol_files[proto] = protocol_files.get(proto, 0) + 1
            rooms[room] = rooms.get(room, 0) + 1

            if include_examples_b and max_examples_i > 0:
                key = f"control:{control}"
                lst = examples.setdefault(key, [])
                if len(lst) < max_examples_i:
                    lst.append({
                        "id": r.get("id"),
                        "name": r.get("name"),
                        "roomId": r.get("roomId"),
                        "roomName": r.get("roomName"),
                        "proxy": r.get("proxy"),
                        "protocolFilename": r.get("protocolFilename"),
                    })

        def _top(d: dict[str, int]) -> list[dict[str, Any]]:
            return [
                {"key": k, "count": c}
                for k, c in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n_i]
            ]

        out: dict[str, Any] = {
            "ok": True,
            "items_total": len([r for r in items if isinstance(r, dict)]),
            "top_n": top_n_i,
            "controls_top": _top(controls),
            "proxies_top": _top(proxies),
            "protocol_filenames_top": _top(protocol_files),
            "rooms_top": _top(rooms),
            "source": "director.http.get" if http.get("ok") else "director.getAllItems",
        }
        if include_examples_b and examples:
            out["examples"] = examples
        return out

    def capabilities_report(self, top_n: int = 20, include_examples: bool = False, max_examples_per_bucket: int = 3) -> dict[str, Any]:
        return self._loop_thread.run(
            self._capabilities_report_async(int(top_n), bool(include_examples), int(max_examples_per_bucket)),
            timeout_s=25,
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

    # ---------- name-based discovery / resolution ----------

    @staticmethod
    def _norm_search_text(text: str) -> str:
        s = str(text or "").strip().lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    @classmethod
    def _score_match(cls, query: str, candidate: str) -> int:
        qn = cls._norm_search_text(query)
        cn = cls._norm_search_text(candidate)
        if not qn or not cn:
            return 0
        if qn == cn:
            return 100
        if qn in cn:
            # Prefer tighter substring matches.
            return max(60, 90 - min(30, abs(len(cn) - len(qn))))

        q_tokens = set(qn.split(" "))
        c_tokens = set(cn.split(" "))
        overlap = len(q_tokens & c_tokens)
        union = max(1, len(q_tokens | c_tokens))
        token_score = int(round((overlap / union) * 75))

        ratio = SequenceMatcher(a=qn, b=cn).ratio()
        seq_score = int(round(ratio * 70))

        return max(token_score, seq_score)

    def find_rooms(self, search: str, limit: int = 10, include_raw: bool = False) -> dict[str, Any]:
        search = str(search or "").strip()
        limit = max(1, min(50, int(limit)))
        include_raw = bool(include_raw)

        if not search:
            return {"ok": False, "error": "search is required", "matches": []}

        rooms = self.list_rooms()
        matches: list[dict[str, Any]] = []
        for r in rooms:
            if not isinstance(r, dict):
                continue
            name = str(r.get("name") or "")
            score = self._score_match(search, name)
            if score <= 0:
                continue
            out: dict[str, Any] = {
                "room_id": str(r.get("id")),
                "name": name,
                "score": int(score),
            }
            if include_raw:
                out["raw"] = r
            matches.append(out)

        matches.sort(key=lambda m: (-int(m.get("score") or 0), str(m.get("name") or ""), str(m.get("room_id") or "")))
        return {"ok": True, "search": search, "matches": matches[:limit]}

    def resolve_room(self, name: str, require_unique: bool = True, include_candidates: bool = True) -> dict[str, Any]:
        name = str(name or "").strip()
        if not name:
            return {"ok": False, "error": "name is required"}

        found = self.find_rooms(name, limit=8, include_raw=False)
        if not found.get("ok"):
            return found

        matches = list(found.get("matches") or [])
        if not matches:
            return {"ok": False, "error": "not_found", "details": f"No rooms matched '{name}'", "candidates": []}

        best = matches[0]
        best_score = int(best.get("score") or 0)
        second_score = int(matches[1].get("score") or 0) if len(matches) > 1 else 0

        # Accept if it's an exact match, or a clearly better fuzzy match.
        accept = (best_score >= 100) or (best_score >= 80 and (best_score - second_score) >= 10)
        if not accept and bool(require_unique):
            return {
                "ok": False,
                "error": "ambiguous",
                "details": f"Multiple rooms could match '{name}'.",
                "candidates": matches if include_candidates else [],
            }

        out: dict[str, Any] = {"ok": True, "room_id": str(best.get("room_id")), "name": str(best.get("name"))}
        if include_candidates:
            out["candidates"] = matches
        return out

    def find_devices(
        self,
        search: str | None = None,
        category: str | None = None,
        room_id: int | None = None,
        limit: int = 20,
        include_raw: bool = False,
    ) -> dict[str, Any]:
        search = (str(search).strip() if search is not None else "")
        category = (str(category).lower().strip() if category is not None else "")
        limit = max(1, min(100, int(limit)))
        include_raw = bool(include_raw)
        room_id_i = int(room_id) if room_id is not None else None

        category_controls = {
            "lights": {"light_v2", "control4_lights_gen3", "outlet_light", "outlet_module_v2"},
            # Locks may appear either as a lock proxy (control=lock) or as a relay-style door lock proxy.
            "locks": {"lock", "control4_relaysingle"},
            "thermostat": {"thermostatV2"},
            "media": {
                "media_player",
                "media_service",
                "receiver",
                "tv",
                "dvd",
                "tuner",
                "satellite",
                "avswitch",
                "av_gen",
                "control4_digitalaudio",
            },
        }

        if category and category not in category_controls:
            return {"ok": False, "error": f"Unknown category '{category}'. Use one of: {sorted(category_controls.keys())}"}

        items = self.get_all_items()
        rooms_by_id = {
            str(i.get("id")): i.get("name")
            for i in items
            if isinstance(i, dict) and i.get("typeName") == "room" and i.get("id") is not None
        }

        allowed_controls = category_controls.get(category) if category else None
        is_lock_category = category == "locks"

        matches: list[dict[str, Any]] = []
        for i in items:
            if not isinstance(i, dict) or i.get("typeName") != "device":
                continue

            device_room_id = i.get("roomId") or i.get("parentId")
            if room_id_i is not None and device_room_id is not None and int(device_room_id) != int(room_id_i):
                continue

            control = str(i.get("control") or "").lower()
            if allowed_controls is not None:
                if control not in allowed_controls and not is_lock_category:
                    continue
                if is_lock_category:
                    # In lock category, accept either explicit lock control or relay single.
                    if control not in allowed_controls and str(i.get("proxy") or "").lower() != "lock":
                        continue

            name_i = str(i.get("name") or "")
            score = self._score_match(search, name_i) if search else 1
            if score <= 0:
                continue

            resolved_room_name = i.get("roomName") or (rooms_by_id.get(str(device_room_id)) if device_room_id is not None else None)
            out: dict[str, Any] = {
                "device_id": str(i.get("id")),
                "name": name_i,
                "room_id": (str(device_room_id) if device_room_id is not None else None),
                "room_name": resolved_room_name,
                "control": (i.get("control") or None),
                "proxy": (i.get("proxy") or None),
                "score": int(score),
            }
            if include_raw:
                out["raw"] = i
            matches.append(out)

        matches.sort(key=lambda m: (-int(m.get("score") or 0), str(m.get("name") or ""), str(m.get("device_id") or "")))
        return {
            "ok": True,
            "search": (search if search else None),
            "category": (category if category else None),
            "room_id": (str(room_id_i) if room_id_i is not None else None),
            "matches": matches[:limit],
        }

    def resolve_device(
        self,
        name: str,
        category: str | None = None,
        room_id: int | None = None,
        require_unique: bool = True,
        include_candidates: bool = True,
    ) -> dict[str, Any]:
        name = str(name or "").strip()
        if not name:
            return {"ok": False, "error": "name is required"}

        found = self.find_devices(search=name, category=category, room_id=room_id, limit=8, include_raw=False)
        if not found.get("ok"):
            return found

        matches = list(found.get("matches") or [])
        if not matches:
            return {
                "ok": False,
                "error": "not_found",
                "details": f"No devices matched '{name}'",
                "candidates": [],
            }

        best = matches[0]
        best_score = int(best.get("score") or 0)
        second_score = int(matches[1].get("score") or 0) if len(matches) > 1 else 0

        accept = (best_score >= 100) or (best_score >= 80 and (best_score - second_score) >= 10)
        if not accept and bool(require_unique):
            return {
                "ok": False,
                "error": "ambiguous",
                "details": f"Multiple devices could match '{name}'.",
                "candidates": matches if include_candidates else [],
            }

        out: dict[str, Any] = {
            "ok": True,
            "device_id": str(best.get("device_id")),
            "name": str(best.get("name")),
            "room_id": best.get("room_id"),
            "room_name": best.get("room_name"),
        }
        if include_candidates:
            out["candidates"] = matches
        return out

    def resolve_room_and_device(
        self,
        room_name: str | None = None,
        device_name: str | None = None,
        category: str | None = None,
        require_unique: bool = True,
        include_candidates: bool = True,
    ) -> dict[str, Any]:
        room_name = (str(room_name).strip() if room_name is not None else "")
        device_name = (str(device_name).strip() if device_name is not None else "")
        category = (str(category).strip() if category is not None else None)

        if not room_name and not device_name:
            return {"ok": False, "error": "room_name or device_name is required"}

        room_res: dict[str, Any] | None = None
        room_id: int | None = None
        if room_name:
            room_res = self.resolve_room(room_name, require_unique=bool(require_unique), include_candidates=bool(include_candidates))
            if not room_res.get("ok"):
                return {"ok": False, "error": "room_resolve_failed", "room": room_res}
            try:
                room_id = int(room_res.get("room_id"))
            except Exception:
                room_id = None

        dev_res: dict[str, Any] | None = None
        if device_name:
            dev_res = self.resolve_device(
                device_name,
                category=category,
                room_id=room_id,
                require_unique=bool(require_unique),
                include_candidates=bool(include_candidates),
            )
            if not dev_res.get("ok"):
                return {"ok": False, "error": "device_resolve_failed", "room": room_res, "device": dev_res}

        out: dict[str, Any] = {"ok": True}
        if room_res is not None:
            out["room"] = room_res
        if dev_res is not None:
            out["device"] = dev_res
        return out

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

    # ---------- UI Buttons / Scenes (best-effort) ----------

    @staticmethod
    def _pick_best_command(cmds: list[dict[str, Any]], preferred: list[str]) -> str | None:
        if not cmds:
            return None

        available: list[str] = []
        for row in cmds:
            if not isinstance(row, dict):
                continue
            c = str(row.get("command") or "").strip()
            if c:
                available.append(c)

        if not available:
            return None

        pref = [str(p).strip() for p in preferred if str(p).strip()]
        lower_map = {a.lower(): a for a in available}
        for p in pref:
            hit = lower_map.get(p.lower())
            if hit:
                return hit
        return available[0]

    async def _uibutton_activate_async(
        self,
        device_id: int,
        command: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        device_id = int(device_id)
        cmd_override = str(command or "").strip()

        cmds_payload = await self._item_get_commands_async(device_id)
        cmds = cmds_payload.get("commands") if isinstance(cmds_payload.get("commands"), list) else []
        cmds_norm = [c for c in cmds if isinstance(c, dict)]

        preferred = [
            "Select",
            "PRESS",
            "ACTIVATE",
            "TRIGGER",
            "ON",
            "GetState",
        ]

        resolved = cmd_override or self._pick_best_command(cmds_norm, preferred)
        if not resolved:
            return {
                "ok": False,
                "device_id": device_id,
                "error": "No commands available for this UI button",
                "commands": cmds_norm,
            }

        if bool(dry_run):
            return {
                "ok": True,
                "device_id": device_id,
                "dry_run": True,
                "resolved_command": resolved,
                "available_commands": [
                    str(c.get("command"))
                    for c in cmds_norm
                    if isinstance(c, dict) and c.get("command") is not None
                ],
            }

        exec_result = await self._item_send_command_async(device_id, resolved, {})
        return {
            "ok": bool(exec_result.get("ok")),
            "device_id": device_id,
            "command": resolved,
            "accepted": bool(exec_result.get("ok")),
            "execute": exec_result,
        }

    def uibutton_activate(self, device_id: int, command: str | None = None, dry_run: bool = False) -> dict[str, Any]:
        return self._loop_thread.run(
            self._uibutton_activate_async(int(device_id), (str(command) if command is not None else None), bool(dry_run)),
            timeout_s=12,
        )

    # ---------- Sensors (best-effort parsing) ----------

    @staticmethod
    def _vars_to_map(variables: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in variables:
            if not isinstance(row, dict):
                continue
            k = str(row.get("varName") or "").strip()
            if not k:
                continue
            out[k.upper()] = row.get("value")
        return out

    @staticmethod
    def _coerce_bool(v: Any) -> bool | None:
        if v in (True, False):
            return bool(v)
        try:
            if isinstance(v, str) and v.strip().isdigit():
                v = int(v.strip())
            if isinstance(v, (int, float)):
                return bool(int(v))
        except Exception:
            return None
        return None

    async def _contact_get_state_async(self, device_id: int, timeout_s: float = 6.0) -> dict[str, Any]:
        device_id = int(device_id)
        fetched = await self._fetch_item_variables_list_async(device_id, timeout_s=float(timeout_s))
        var_list: list[dict[str, Any]] = fetched.get("variables") if isinstance(fetched.get("variables"), list) else []

        # Build a case-insensitive map for sensors.
        # NOTE: control4_gateway.py also has a thermostat-specific _vars_to_map later in the file
        # that preserves original varName casing. To avoid subtle conflicts, sensor parsing uses
        # its own uppercase-normalized view.
        vmap_u: dict[str, Any] = {}
        for row in var_list:
            if not isinstance(row, dict):
                continue
            k = str(row.get("varName") or "").strip()
            if not k:
                continue
            vmap_u[k.upper()] = row.get("value")

        battery_level = vmap_u.get("BATTERY_LEVEL")
        battery_voltage = vmap_u.get("BATTERY_VOLTAGE")
        temperature = vmap_u.get("TEMPERATURE")
        light_level = vmap_u.get("LIGHT_LEVEL")
        night_mode = vmap_u.get("NIGHT_MODE")

        # Best-effort state extraction across common drivers.
        # NOTE: vmap keys are normalized to UPPERCASE via _vars_to_map.
        candidates = [
            ("contact", "CONTACT"),
            ("contact", "CONTACT_STATE"),
            ("contact", "CONTACTSTATE"),
            ("contact", "OPEN"),
            ("contact", "CLOSED"),
            ("contact", "IS_OPEN"),
            ("motion", "MOTION"),
            ("motion", "MOTION_STATE"),
            ("motion", "OCCUPANCY"),
            ("generic", "STATE"),
        ]
        state = None
        state_kind = None
        state_var = None

        for kind, key in candidates:
            if key not in vmap_u:
                continue
            state_var = key
            state_kind = kind
            val = vmap_u.get(key)
            b = self._coerce_bool(val)
            state = b if b is not None else val
            break

        return {
            "ok": bool(fetched.get("ok")),
            "device_id": device_id,
            "variables": var_list,
            "battery_level": battery_level,
            "battery_voltage": battery_voltage,
            "temperature": temperature,
            "light_level": light_level,
            "night_mode": night_mode,
            "state": state,
            "state_kind": state_kind,
            "state_var": state_var,
            "source": fetched.get("source"),
        }

    # ---------- Keypads (best-effort) ----------

    def keypad_list(self) -> dict[str, Any]:
        items = self.get_all_items()
        keypads: list[dict[str, Any]] = []
        for i in items:
            if not isinstance(i, dict) or i.get("typeName") != "device":
                continue
            if str(i.get("proxy") or "").lower() != "keypad_proxy":
                continue
            keypads.append(
                {
                    "device_id": int(i.get("id")),
                    "name": i.get("name"),
                    "room_id": i.get("roomId") or i.get("parentId"),
                    "room_name": i.get("roomName"),
                    "control": i.get("control"),
                    "proxy": i.get("proxy"),
                    "protocolFilename": i.get("protocolFilename"),
                }
            )
        keypads.sort(key=lambda d: (str(d.get("room_name") or ""), str(d.get("name") or "")))
        return {"ok": True, "count": len(keypads), "keypads": keypads}

    async def _keypad_get_buttons_async(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)
        cmds_resp = await self._item_get_commands_async(device_id)
        if not cmds_resp.get("ok"):
            return {
                "ok": False,
                "device_id": device_id,
                "error": "Could not load commands",
                "details": cmds_resp,
            }

        buttons_by_id: dict[int, str] = {}
        for cmd in (cmds_resp.get("commands") or []):
            if not isinstance(cmd, dict):
                continue
            for p in (cmd.get("params") or []):
                if not isinstance(p, dict):
                    continue
                if str(p.get("name") or "").strip().upper() != "BUTTON_ID":
                    continue
                values = p.get("values")
                if not isinstance(values, list):
                    continue
                for row in values:
                    if not isinstance(row, dict):
                        continue
                    bid = row.get("id")
                    if isinstance(bid, bool) or not isinstance(bid, (int, float, str)):
                        continue
                    try:
                        bid_i = int(bid)
                    except Exception:
                        continue
                    name = row.get("name")
                    if name is None:
                        name = row.get("display")
                    if name is None:
                        name = str(bid_i)
                    buttons_by_id[bid_i] = str(name)

        buttons = [{"button_id": k, "name": buttons_by_id[k]} for k in sorted(buttons_by_id.keys())]
        return {
            "ok": True,
            "device_id": device_id,
            "buttons": buttons,
            "button_count": len(buttons),
            "source": cmds_resp.get("source"),
        }

    def keypad_get_buttons(self, device_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._keypad_get_buttons_async(int(device_id)), timeout_s=12)

    async def _keypad_button_action_async(
        self,
        device_id: int,
        button_id: int,
        action: str,
        tap_ms: int,
        dry_run: bool,
    ) -> dict[str, Any]:
        device_id = int(device_id)
        button_id = int(button_id)
        action_n = str(action or "").strip().lower()
        tap_ms = int(tap_ms)
        if tap_ms < 0:
            tap_ms = 0

        def _action_to_int(a: str) -> int | None:
            if a in {"press", "down", "pressed"}:
                return 1
            if a in {"release", "up", "released"}:
                return 0
            return None

        executes: list[dict[str, Any]] = []
        if action_n == "tap":
            if bool(dry_run):
                return {
                    "ok": True,
                    "device_id": device_id,
                    "button_id": button_id,
                    "action": "tap",
                    "tap_ms": tap_ms,
                    "dry_run": True,
                    "planned": [
                        {"action": "press", "command": "KEYPAD_BUTTON_ACTION", "params": {"BUTTON_ID": button_id, "ACTION": 1}},
                        {"action": "release", "command": "KEYPAD_BUTTON_ACTION", "params": {"BUTTON_ID": button_id, "ACTION": 0}},
                    ],
                }
            press = await self._item_send_command_async(device_id, "KEYPAD_BUTTON_ACTION", {"BUTTON_ID": button_id, "ACTION": 1})
            executes.append({"action": "press", "execute": press})
            await asyncio.sleep(max(0.05, tap_ms / 1000.0) if tap_ms else 0.05)
            release = await self._item_send_command_async(device_id, "KEYPAD_BUTTON_ACTION", {"BUTTON_ID": button_id, "ACTION": 0})
            executes.append({"action": "release", "execute": release})
            ok = all(bool(x.get("execute", {}).get("ok")) for x in executes)
            return {
                "ok": ok,
                "device_id": device_id,
                "button_id": button_id,
                "action": "tap",
                "tap_ms": tap_ms,
                "accepted": ok,
                "executes": executes,
            }

        a_int = _action_to_int(action_n)
        if a_int is None:
            return {
                "ok": False,
                "device_id": device_id,
                "button_id": button_id,
                "error": "action must be one of: tap, press, release",
            }

        if bool(dry_run):
            return {
                "ok": True,
                "device_id": device_id,
                "button_id": button_id,
                "action": action_n,
                "dry_run": True,
                "planned": {"command": "KEYPAD_BUTTON_ACTION", "params": {"BUTTON_ID": button_id, "ACTION": a_int}},
            }

        res = await self._item_send_command_async(device_id, "KEYPAD_BUTTON_ACTION", {"BUTTON_ID": button_id, "ACTION": a_int})
        return {
            "ok": bool(res.get("ok")),
            "device_id": device_id,
            "button_id": button_id,
            "action": action_n,
            "accepted": bool(res.get("ok")),
            "execute": res,
        }

    def keypad_button_action(
        self,
        device_id: int,
        button_id: int,
        action: str = "tap",
        tap_ms: int = 200,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self._loop_thread.run(
            self._keypad_button_action_async(
                int(device_id),
                int(button_id),
                str(action or ""),
                int(tap_ms),
                bool(dry_run),
            ),
            timeout_s=12,
        )

    # ---------- Room Control Keypads (programmed triggers) ----------

    def control_keypad_list(self) -> dict[str, Any]:
        items = self.get_all_items()
        devs: list[dict[str, Any]] = []
        for i in items:
            if not isinstance(i, dict) or i.get("typeName") != "device":
                continue
            if str(i.get("proxy") or "").lower() != "room_control_keypad":
                continue
            devs.append(
                {
                    "device_id": int(i.get("id")),
                    "name": i.get("name"),
                    "room_id": i.get("roomId") or i.get("parentId"),
                    "room_name": i.get("roomName"),
                    "control": i.get("control"),
                    "proxy": i.get("proxy"),
                    "protocolFilename": i.get("protocolFilename"),
                }
            )
        devs.sort(key=lambda d: (str(d.get("room_name") or ""), str(d.get("name") or "")))
        return {"ok": True, "count": len(devs), "control_keypads": devs}

    def control_keypad_send_command(self, device_id: int, command: str, dry_run: bool = False) -> dict[str, Any]:
        if bool(dry_run):
            return {
                "ok": True,
                "device_id": int(device_id),
                "command": str(command or ""),
                "dry_run": True,
            }
        return self.item_send_command(int(device_id), str(command or ""), None)

    # ---------- Fans ----------

    def fan_list(self) -> dict[str, Any]:
        items = self.get_all_items()
        fans: list[dict[str, Any]] = []
        for i in items:
            if not isinstance(i, dict) or i.get("typeName") != "device":
                continue
            if str(i.get("proxy") or "").lower() != "fan":
                continue
            fans.append(
                {
                    "device_id": int(i.get("id")),
                    "name": i.get("name"),
                    "room_id": i.get("roomId") or i.get("parentId"),
                    "room_name": i.get("roomName"),
                    "control": i.get("control"),
                    "proxy": i.get("proxy"),
                    "protocolFilename": i.get("protocolFilename"),
                }
            )
        fans.sort(key=lambda d: (str(d.get("room_name") or ""), str(d.get("name") or "")))
        return {"ok": True, "count": len(fans), "fans": fans}

    async def _fan_get_state_async(self, device_id: int) -> dict[str, Any]:
        device_id = int(device_id)
        fetched = await self._fetch_item_variables_list_async(device_id)
        var_list: list[dict[str, Any]] = fetched.get("variables") if isinstance(fetched.get("variables"), list) else []
        vmap = self._vars_to_map(var_list)

        is_on = self._coerce_bool(vmap.get("IS_ON"))
        current_speed = vmap.get("CURRENT_SPEED")
        fan_speed = vmap.get("FAN_SPEED")

        def _as_int(v: Any) -> int | None:
            if isinstance(v, bool) or v is None:
                return None
            try:
                return int(str(v).strip())
            except Exception:
                return None

        speed = _as_int(current_speed)
        if speed is None:
            speed = _as_int(fan_speed)

        return {
            "ok": bool(fetched.get("ok")),
            "device_id": device_id,
            "is_on": is_on,
            "speed": speed,
            "variables": var_list,
            "source": fetched.get("source"),
        }

    def fan_get_state(self, device_id: int) -> dict[str, Any]:
        return self._loop_thread.run(self._fan_get_state_async(int(device_id)), timeout_s=12)

    @staticmethod
    def _fan_speed_to_int(speed: Any) -> int | None:
        if speed is None:
            return None
        if isinstance(speed, bool):
            return None
        if isinstance(speed, (int, float)):
            return int(speed)
        s = str(speed).strip().lower()
        if s in {"0", "off", "stop"}:
            return 0
        if s in {"1", "low"}:
            return 1
        if s in {"2", "med", "medium"}:
            return 2
        if s in {"3", "medhigh", "mediumhigh", "medium high"}:
            return 3
        if s in {"4", "high"}:
            return 4
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return None
        return None

    def fan_set_speed(self, device_id: int, speed: Any, confirm_timeout_s: float = 4.0, dry_run: bool = False) -> dict[str, Any]:
        device_id = int(device_id)
        speed_i = self._fan_speed_to_int(speed)
        if speed_i is None or speed_i < 0 or speed_i > 4:
            return {"ok": False, "device_id": device_id, "error": "speed must be 0-4 or one of: off, low, medium, medium high, high"}

        async def _run():
            before = await self._fan_get_state_async(device_id)
            if bool(dry_run):
                return {
                    "ok": True,
                    "device_id": device_id,
                    "speed": speed_i,
                    "dry_run": True,
                    "planned": {"command": "SET_SPEED", "params": {"SPEED": speed_i}},
                    "before": before,
                }
            exec_result = await self._item_send_command_async(device_id, "SET_SPEED", {"SPEED": speed_i})
            accepted = bool(exec_result.get("ok"))

            confirmed = False
            after = None
            deadline = time.time() + float(confirm_timeout_s)
            while time.time() < deadline:
                after = await self._fan_get_state_async(device_id)
                if after.get("speed") == speed_i:
                    confirmed = True
                    break
                await asyncio.sleep(0.25)

            return {
                "ok": accepted,
                "device_id": device_id,
                "speed": speed_i,
                "accepted": accepted,
                "confirmed": confirmed,
                "before": before,
                "after": after,
                "execute": exec_result,
            }

        return self._loop_thread.run(_run(), timeout_s=float(confirm_timeout_s) + 12.0)

    def fan_set_power(self, device_id: int, power: str, confirm_timeout_s: float = 4.0, dry_run: bool = False) -> dict[str, Any]:
        device_id = int(device_id)
        p = str(power or "").strip().lower()
        if p not in {"on", "off", "toggle"}:
            return {"ok": False, "device_id": device_id, "error": "power must be one of: on, off, toggle"}
        cmd = {"on": "ON", "off": "OFF", "toggle": "TOGGLE"}[p]

        async def _run():
            before = await self._fan_get_state_async(device_id)
            if bool(dry_run):
                return {
                    "ok": True,
                    "device_id": device_id,
                    "power": p,
                    "dry_run": True,
                    "planned": {"command": cmd, "params": {}},
                    "before": before,
                }
            exec_result = await self._item_send_command_async(device_id, cmd, {})
            accepted = bool(exec_result.get("ok"))

            confirmed = False
            after = None
            deadline = time.time() + float(confirm_timeout_s)
            while time.time() < deadline:
                after = await self._fan_get_state_async(device_id)
                if p == "on" and after.get("is_on") is True:
                    confirmed = True
                    break
                if p == "off" and after.get("is_on") is False:
                    confirmed = True
                    break
                if p == "toggle" and after.get("is_on") in (True, False) and after.get("is_on") != before.get("is_on"):
                    confirmed = True
                    break
                await asyncio.sleep(0.25)

            return {
                "ok": accepted,
                "device_id": device_id,
                "power": p,
                "accepted": accepted,
                "confirmed": confirmed,
                "before": before,
                "after": after,
                "execute": exec_result,
            }

        return self._loop_thread.run(_run(), timeout_s=float(confirm_timeout_s) + 12.0)

    # ---------- Motion sensors (best-effort; mostly ContactState-based) ----------

    def motion_list(self) -> dict[str, Any]:
        items = self.get_all_items()
        out: list[dict[str, Any]] = []
        wanted = {"contactsingle_motionsensor", "cardaccess_wirelesspir", "control4_wirelesspir"}
        for i in items:
            if not isinstance(i, dict) or i.get("typeName") != "device":
                continue
            proxy = str(i.get("proxy") or "").lower()
            if proxy not in wanted:
                continue
            out.append(
                {
                    "device_id": int(i.get("id")),
                    "name": i.get("name"),
                    "room_id": i.get("roomId") or i.get("parentId"),
                    "room_name": i.get("roomName"),
                    "control": i.get("control"),
                    "proxy": i.get("proxy"),
                    "protocolFilename": i.get("protocolFilename"),
                }
            )
        out.sort(key=lambda d: (str(d.get("room_name") or ""), str(d.get("name") or "")))
        return {"ok": True, "count": len(out), "motions": out}

    def motion_get_state(self, device_id: int, timeout_s: float = 6.0) -> dict[str, Any]:
        base = self.contact_get_state(int(device_id), timeout_s=float(timeout_s))
        state = base.get("state")
        kind = base.get("state_kind")
        motion = None
        if kind in {"motion", "contact"} and state in (True, False):
            motion = bool(state)
        base["motion_detected"] = motion
        return base

    # ---------- Intercom (best-effort) ----------

    def intercom_list(self) -> dict[str, Any]:
        items = self.get_all_items()
        out: list[dict[str, Any]] = []
        for i in items:
            if not isinstance(i, dict) or i.get("typeName") != "device":
                continue
            if "intercom" not in str(i.get("proxy") or "").lower():
                continue
            out.append(
                {
                    "device_id": int(i.get("id")),
                    "name": i.get("name"),
                    "room_id": i.get("roomId") or i.get("parentId"),
                    "room_name": i.get("roomName"),
                    "control": i.get("control"),
                    "proxy": i.get("proxy"),
                    "protocolFilename": i.get("protocolFilename"),
                }
            )
        out.sort(key=lambda d: (str(d.get("room_name") or ""), str(d.get("name") or "")))
        return {"ok": True, "count": len(out), "intercoms": out}

    def intercom_touchscreen_set_feature(
        self,
        device_id: int,
        feature: str,
        enabled: bool,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        device_id = int(device_id)
        feat = str(feature or "").strip().lower()
        if feat not in {"autobrightness", "proximity", "alexa"}:
            return {"ok": False, "device_id": device_id, "error": "feature must be one of: autobrightness, proximity, alexa"}

        prefix = {
            "autobrightness": "SET_AUTOBRIGHTNESS_ENABLED",
            "proximity": "SET_PROXIMITY_SENSOR_ENABLED",
            "alexa": "SET_ALEXA_ENABLED",
        }[feat]
        cmd = f"{prefix}:{'True' if bool(enabled) else 'False'}"

        async def _run():
            if bool(dry_run):
                return {"ok": True, "device_id": device_id, "feature": feat, "enabled": bool(enabled), "dry_run": True, "planned": {"command": cmd, "params": {}}}
            r = await self._item_send_command_async(device_id, cmd, {})
            return {"ok": bool(r.get("ok")), "device_id": device_id, "feature": feat, "enabled": bool(enabled), "execute": r}

        return self._loop_thread.run(_run(), timeout_s=12)

    def intercom_touchscreen_screensaver(
        self,
        device_id: int,
        action: str | None = None,
        mode: str | None = None,
        start_time_s: int | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        device_id = int(device_id)
        act = str(action or "").strip().lower() if action is not None else ""
        if act and act not in {"enter", "exit"}:
            return {"ok": False, "device_id": device_id, "error": "action must be enter/exit (or omitted)"}

        planned: list[dict[str, Any]] = []
        if mode is not None:
            mode_s = str(mode or "").strip()
            if not mode_s:
                return {"ok": False, "device_id": device_id, "error": "mode was provided but empty"}
            planned.append({"command": "SET_SCREENSAVER_MODE", "params": {"MODE": mode_s}})
        if start_time_s is not None:
            try:
                t = int(start_time_s)
            except Exception:
                return {"ok": False, "device_id": device_id, "error": "start_time_s must be an integer"}
            planned.append({"command": "SET_SCREENSAVER_START_TIME", "params": {"TIME": str(t)}})
        if act:
            planned.append({"command": "SCREENSAVER_ENTER" if act == "enter" else "SCREENSAVER_EXIT", "params": {}})
        if not planned:
            return {"ok": False, "device_id": device_id, "error": "No screensaver operation specified"}

        async def _run():
            if bool(dry_run):
                return {"ok": True, "device_id": device_id, "dry_run": True, "planned": planned}

            attempts: list[dict[str, Any]] = []
            ok = True
            for step in planned:
                r = await self._item_send_command_async(device_id, str(step.get("command") or ""), step.get("params") or {})
                attempts.append({"step": step, "execute": r})
                if not r.get("ok"):
                    ok = False
                    break
            return {"ok": ok, "device_id": device_id, "attempts": attempts}

        return self._loop_thread.run(_run(), timeout_s=18)

    def intercom_doorstation_set_led(self, device_id: int, enabled: bool, dry_run: bool = False) -> dict[str, Any]:
        device_id = int(device_id)
        cmd = "Enable LED Indicator" if bool(enabled) else "Disable LED Indicator"

        async def _run():
            if bool(dry_run):
                return {"ok": True, "device_id": device_id, "led_enabled": bool(enabled), "dry_run": True, "planned": {"command": cmd, "params": {}}}
            r = await self._item_send_command_preserve_async(device_id, cmd, {})
            return {"ok": bool(r.get("ok")), "device_id": device_id, "led_enabled": bool(enabled), "execute": r}

        return self._loop_thread.run(_run(), timeout_s=12)

    def intercom_doorstation_set_external_chime(self, device_id: int, enabled: bool, dry_run: bool = False) -> dict[str, Any]:
        device_id = int(device_id)
        cmd = "Enable External Chime" if bool(enabled) else "Disable External Chime"

        async def _run():
            if bool(dry_run):
                return {"ok": True, "device_id": device_id, "external_chime_enabled": bool(enabled), "dry_run": True, "planned": {"command": cmd, "params": {}}}
            r = await self._item_send_command_preserve_async(device_id, cmd, {})
            return {"ok": bool(r.get("ok")), "device_id": device_id, "external_chime_enabled": bool(enabled), "execute": r}

        return self._loop_thread.run(_run(), timeout_s=12)

    def intercom_doorstation_set_raw_setting(self, device_id: int, key: str, value: str, dry_run: bool = False) -> dict[str, Any]:
        device_id = int(device_id)
        key_s = str(key or "").strip()
        value_s = str(value or "").strip()
        if not key_s:
            return {"ok": False, "device_id": device_id, "error": "key is required"}

        cmd = "Set Raw Settings"
        params = {"Key": key_s, "Value": value_s}

        async def _run():
            if bool(dry_run):
                return {"ok": True, "device_id": device_id, "dry_run": True, "planned": {"command": cmd, "params": params}}
            r = await self._item_send_command_preserve_async(device_id, cmd, params)
            return {"ok": bool(r.get("ok")), "device_id": device_id, "execute": r}

        return self._loop_thread.run(_run(), timeout_s=12)

    # ---------- Macros agent ----------

    def macro_list(self) -> dict[str, Any]:
        async def _run():
            http = await self._director_http_get("/api/v1/agents/macros")
            payload = http.get("json")
            macros = payload if isinstance(payload, list) else []
            macros_norm = [m for m in macros if isinstance(m, dict)]
            macros_norm.sort(key=lambda m: (str(m.get("name") or ""), int(m.get("id") or 0)))
            return {"ok": bool(http.get("ok")), "count": len(macros_norm), "macros": macros_norm, "http": http}

        return self._loop_thread.run(_run(), timeout_s=12)

    def macro_list_commands(self) -> dict[str, Any]:
        async def _run():
            http = await self._director_http_get("/api/v1/agents/macros/commands")
            payload = http.get("json")
            cmds = payload if isinstance(payload, list) else (payload.get("commands") if isinstance(payload, dict) else [])
            cmds_norm = [c for c in (cmds or []) if isinstance(c, dict)]
            return {"ok": bool(http.get("ok")), "count": len(cmds_norm), "commands": cmds_norm, "http": http}

        return self._loop_thread.run(_run(), timeout_s=12)

    def macro_execute(self, macro_id: int, dry_run: bool = False) -> dict[str, Any]:
        mid = int(macro_id)

        async def _run():
            planned = {"agent": "macros", "command": "EXECUTE_MACRO", "params": {"id": mid}}
            if bool(dry_run):
                return {"ok": True, "macro_id": mid, "dry_run": True, "planned": planned}
            r = await self._agent_send_command_async("macros", "EXECUTE_MACRO", {"id": mid})
            return {"ok": bool(r.get("ok")), "macro_id": mid, "execute": r}

        return self._loop_thread.run(_run(), timeout_s=18)

    def macro_execute_by_name(self, name: str, dry_run: bool = False) -> dict[str, Any]:
        """Execute a macro by its configured name.

        Safety behavior:
        - Match is case-insensitive exact equality after trimming.
        - If 0 matches or >1 matches, do not execute.
        """

        needle = str(name or "").strip()
        if not needle:
            return {"ok": False, "error": "name is required"}

        async def _run():
            http = await self._director_http_get("/api/v1/agents/macros")
            payload = http.get("json")
            macros = payload if isinstance(payload, list) else []
            macros_norm = [m for m in macros if isinstance(m, dict)]

            needle_l = needle.lower()
            exact = [m for m in macros_norm if str(m.get("name") or "").strip().lower() == needle_l]

            if len(exact) == 0:
                contains = [
                    {"id": m.get("id"), "name": m.get("name")}
                    for m in macros_norm
                    if needle_l in str(m.get("name") or "").strip().lower()
                ]
                contains = contains[:10]
                return {
                    "ok": False,
                    "name": needle,
                    "error": "No macro matched name (exact match required)",
                    "suggestions": contains,
                    "http": http,
                }

            if len(exact) > 1:
                return {
                    "ok": False,
                    "name": needle,
                    "error": "Macro name is ambiguous (multiple exact matches)",
                    "matches": [{"id": m.get("id"), "name": m.get("name")} for m in exact],
                    "http": http,
                }

            mid = int(exact[0].get("id") or 0)
            if mid <= 0:
                return {"ok": False, "name": needle, "error": "Matched macro had invalid id", "match": exact[0]}

            planned = {"agent": "macros", "command": "EXECUTE_MACRO", "params": {"id": mid}}
            if bool(dry_run):
                return {"ok": True, "name": needle, "macro_id": mid, "dry_run": True, "planned": planned}

            r = await self._agent_send_command_async("macros", "EXECUTE_MACRO", {"id": mid})
            return {"ok": bool(r.get("ok")), "name": needle, "macro_id": mid, "execute": r}

        return self._loop_thread.run(_run(), timeout_s=18)

    # ---------- Scheduler agent ----------

    def scheduler_list(self, search: str | None = None) -> dict[str, Any]:
        needle = str(search or "").strip().lower()

        async def _run():
            http = await self._director_http_get("/api/v1/agents/scheduler")
            payload = http.get("json")
            events = payload if isinstance(payload, list) else []
            events_norm = [e for e in events if isinstance(e, dict)]

            if needle:
                def _hay(row: dict[str, Any]) -> str:
                    return " ".join(
                        [
                            str(row.get("display") or ""),
                            str(row.get("category") or ""),
                            str(row.get("eventId") or ""),
                        ]
                    ).lower()

                events_norm = [e for e in events_norm if needle in _hay(e)]

            events_norm.sort(key=lambda e: (str(e.get("display") or ""), int(e.get("eventId") or 0)))
            return {
                "ok": bool(http.get("ok")),
                "count": len(events_norm),
                "search": (str(search) if search is not None else None),
                "events": events_norm,
                "http": http,
            }

        return self._loop_thread.run(_run(), timeout_s=12)

    def scheduler_get(self, event_id: int) -> dict[str, Any]:
        eid = int(event_id)
        if eid <= 0:
            return {"ok": False, "error": "event_id must be a positive integer"}

        async def _run():
            http = await self._director_http_get(f"/api/v1/agents/scheduler/{eid}")
            payload = http.get("json")
            event = payload if isinstance(payload, dict) else None
            return {"ok": bool(http.get("ok")), "event_id": eid, "event": event, "http": http}

        return self._loop_thread.run(_run(), timeout_s=18)

    def scheduler_list_commands(self) -> dict[str, Any]:
        async def _run():
            http = await self._director_http_get("/api/v1/agents/scheduler/commands")
            payload = http.get("json")
            cmds = payload if isinstance(payload, list) else (payload.get("commands") if isinstance(payload, dict) else [])
            cmds_norm = [c for c in (cmds or []) if isinstance(c, dict)]
            return {"ok": bool(http.get("ok")), "count": len(cmds_norm), "commands": cmds_norm, "http": http}

        return self._loop_thread.run(_run(), timeout_s=12)

    def scheduler_set_enabled(self, event_id: int, enabled: bool, dry_run: bool = False) -> dict[str, Any]:
        eid = int(event_id)
        if eid <= 0:
            return {"ok": False, "error": "event_id must be a positive integer"}

        desired = bool(enabled)

        async def _run():
            before_http = await self._director_http_get(f"/api/v1/agents/scheduler/{eid}")
            before = before_http.get("json") if isinstance(before_http.get("json"), dict) else None

            # Some Director builds appear to be finicky about scheduler writes: certain payload shapes return
            # server-side errors like "Timeout Modifying Scheduled Event". We try a small matrix of
            # method/path/payload options and always confirm by rereading.

            enabled_int = int(desired)

            # Minimal payloads (what we'd expect to work)
            payload_min = {"eventid": eid, "enabled": enabled_int}

            # Full event payload (best-effort) - some builds accept this even when minimal updates fail.
            payload_full: dict[str, Any]
            if isinstance(before, dict):
                payload_full = dict(before)
                payload_full["enabled"] = bool(desired)
            else:
                payload_full = dict(payload_min)

            planned_attempts = [
                # Preferred: try id-specific endpoints first.
                {"method": "PUT", "path": f"/api/v1/agents/scheduler/{eid}", "payload": payload_min},
                {"method": "PUT", "path": f"/api/v1/agents/scheduler/events/{eid}", "payload": payload_min},
                {"method": "POST", "path": f"/api/v1/agents/scheduler/{eid}", "payload": payload_min},
                {"method": "POST", "path": f"/api/v1/agents/scheduler/events/{eid}", "payload": payload_min},
                {"method": "POST", "path": "/api/v1/agents/scheduler/events", "payload": payload_min},
                # Fallback: try full payload variants.
                {"method": "POST", "path": f"/api/v1/agents/scheduler/{eid}", "payload": payload_full},
                {"method": "POST", "path": "/api/v1/agents/scheduler/events", "payload": payload_full},
            ]

            if bool(dry_run):
                return {
                    "ok": True,
                    "event_id": eid,
                    "enabled": desired,
                    "dry_run": True,
                    "planned": planned_attempts,
                    "before": before,
                    "before_http": before_http,
                }

            attempts: list[dict[str, Any]] = []
            accepted = False
            last: dict[str, Any] | None = None
            after_http: dict[str, Any] | None = None
            after: dict[str, Any] | None = None
            confirmed = False

            for plan in planned_attempts:
                method = str(plan.get("method") or "POST")
                path = str(plan.get("path") or "")
                payload = plan.get("payload") if isinstance(plan.get("payload"), dict) else {}

                # Scheduler writes can be slow; allow a slightly longer client timeout. Note this does not
                # fix server-side timeouts returned as 400 errors.
                r = await self._director_http_request_https_only(method, path, payload, timeout_s=20.0)
                attempts.append({"method": method, "path": path, "payload": payload, "http": r})
                last = r

                if r.get("ok"):
                    accepted = True

                    # Re-read after each successful attempt; some endpoints return 200 but do nothing.
                    after_http = await self._director_http_get(f"/api/v1/agents/scheduler/{eid}")
                    after = after_http.get("json") if isinstance(after_http.get("json"), dict) else None
                    after_enabled = after.get("enabled") if isinstance(after, dict) else None
                    confirmed = (after_enabled is True and desired is True) or (after_enabled is False and desired is False)
                    if confirmed:
                        break

            if after_http is None:
                after_http = await self._director_http_get(f"/api/v1/agents/scheduler/{eid}")
                after = after_http.get("json") if isinstance(after_http.get("json"), dict) else None

            return {
                "ok": bool(accepted),
                "event_id": eid,
                "enabled": desired,
                "accepted": bool(accepted),
                "confirmed": bool(confirmed),
                "warning": (
                    "Write attempt returned success but enabled state did not change (check attempts + confirmed)"
                    if bool(accepted) and not bool(confirmed)
                    else None
                ),
                "before": before,
                "after": after,
                "attempts": attempts,
                "last": last,
            }

        return self._loop_thread.run(_run(), timeout_s=35)

    # ---------- Announcements agent ----------

    def announcement_list(self) -> dict[str, Any]:
        async def _run():
            http = await self._director_http_get("/api/v1/agents/announcements")
            payload = http.get("json")
            ann = payload if isinstance(payload, list) else []
            ann_norm = [a for a in ann if isinstance(a, dict)]
            ann_norm.sort(key=lambda a: (str(a.get("name") or ""), int(a.get("id") or 0)))
            return {"ok": bool(http.get("ok")), "count": len(ann_norm), "announcements": ann_norm, "http": http}

        return self._loop_thread.run(_run(), timeout_s=12)

    def announcement_list_commands(self) -> dict[str, Any]:
        async def _run():
            http = await self._director_http_get("/api/v1/agents/announcements/commands")
            payload = http.get("json")
            cmds = payload if isinstance(payload, list) else (payload.get("commands") if isinstance(payload, dict) else [])
            cmds_norm = [c for c in (cmds or []) if isinstance(c, dict)]
            return {"ok": bool(http.get("ok")), "count": len(cmds_norm), "commands": cmds_norm, "http": http}

        return self._loop_thread.run(_run(), timeout_s=12)

    def announcement_execute(self, announcement_id: int, dry_run: bool = False) -> dict[str, Any]:
        aid = int(announcement_id)

        async def _run():
            planned = {"agent": "announcements", "command": "execute_announcement", "params": {"id": aid}}
            if bool(dry_run):
                return {"ok": True, "announcement_id": aid, "dry_run": True, "planned": planned}
            r = await self._agent_send_command_async("announcements", "execute_announcement", {"id": aid})
            return {"ok": bool(r.get("ok")), "announcement_id": aid, "execute": r}

        return self._loop_thread.run(_run(), timeout_s=18)

    def announcement_execute_by_name(self, name: str, dry_run: bool = False) -> dict[str, Any]:
        """Execute an announcement by its configured name.

        Safety behavior:
        - Match is case-insensitive exact equality after trimming.
        - If 0 matches or >1 matches, do not execute.
        """

        needle = str(name or "").strip()
        if not needle:
            return {"ok": False, "error": "name is required"}

        async def _run():
            http = await self._director_http_get("/api/v1/agents/announcements")
            payload = http.get("json")
            ann = payload if isinstance(payload, list) else []
            ann_norm = [a for a in ann if isinstance(a, dict)]

            needle_l = needle.lower()
            exact = [a for a in ann_norm if str(a.get("name") or "").strip().lower() == needle_l]

            if len(exact) == 0:
                # Provide suggestions but do not attempt fuzzy execution.
                contains = [
                    {"id": a.get("id"), "name": a.get("name")}
                    for a in ann_norm
                    if needle_l in str(a.get("name") or "").strip().lower()
                ]
                contains = contains[:10]
                return {
                    "ok": False,
                    "name": needle,
                    "error": "No announcement matched name (exact match required)",
                    "suggestions": contains,
                    "http": http,
                }

            if len(exact) > 1:
                return {
                    "ok": False,
                    "name": needle,
                    "error": "Announcement name is ambiguous (multiple exact matches)",
                    "matches": [{"id": a.get("id"), "name": a.get("name")} for a in exact],
                    "http": http,
                }

            aid = int(exact[0].get("id") or 0)
            if aid <= 0:
                return {"ok": False, "name": needle, "error": "Matched announcement had invalid id", "match": exact[0]}

            planned = {"agent": "announcements", "command": "execute_announcement", "params": {"id": aid}}
            if bool(dry_run):
                return {
                    "ok": True,
                    "name": needle,
                    "announcement_id": aid,
                    "dry_run": True,
                    "planned": planned,
                }

            r = await self._agent_send_command_async("announcements", "execute_announcement", {"id": aid})
            return {
                "ok": bool(r.get("ok")),
                "name": needle,
                "announcement_id": aid,
                "execute": r,
            }

        return self._loop_thread.run(_run(), timeout_s=18)

    def contact_get_state(self, device_id: int, timeout_s: float = 6.0) -> dict[str, Any]:
        return self._loop_thread.run(
            self._contact_get_state_async(int(device_id), timeout_s=float(timeout_s)),
            timeout_s=float(timeout_s) + 10.0,
        )

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
            "title": pick(
                "TITLE",
                "Title",
                "TRACK_TITLE",
                "Track Title",
                "SONG_TITLE",
                "Song Title",
                "MEDIA_TITLE",
                "Media Title",
                "NOW_PLAYING_TITLE",
                "Now Playing Title",
            ),
            "artist": pick(
                "ARTIST",
                "Artist",
                "TRACK_ARTIST",
                "Track Artist",
                "ARTIST_NAME",
                "Artist Name",
                "MEDIA_ARTIST",
                "Media Artist",
                "NOW_PLAYING_ARTIST",
                "Now Playing Artist",
            ),
            "album": pick(
                "ALBUM",
                "Album",
                "TRACK_ALBUM",
                "Track Album",
                "ALBUM_NAME",
                "Album Name",
                "MEDIA_ALBUM",
                "Media Album",
                "NOW_PLAYING_ALBUM",
                "Now Playing Album",
            ),
            "station": pick(
                "STATION",
                "Station",
                "STATION_NAME",
                "Station Name",
                "CHANNEL",
                "CHANNEL_NAME",
                "Channel",
                "Channel Name",
            ),
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

    async def _room_now_playing_async(self, room_id: int, max_sources: int = 30) -> dict[str, Any]:
        room_id = int(room_id)
        try:
            limit = max(1, min(int(max_sources), 80))
        except Exception:
            limit = 30

        listen = await self._ui_listen_status_async(room_id)
        if not isinstance(listen, dict):
            return {"ok": False, "room_id": room_id, "error": "Listen UI status not available"}

        sources = listen.get("sources") if isinstance(listen.get("sources"), list) else []
        active = bool(listen.get("active"))

        # If the room isn't actively listening, it's unlikely we'll find audio now-playing.
        if not active:
            return {"ok": True, "room_id": room_id, "active": False, "probe": [], "best": None}

        # Prefer DIGITAL_AUDIO_CLIENT first (often holds queue/transport state).
        def _score(src: dict[str, Any]) -> int:
            t = str(src.get("type") or "").upper()
            if t == "DIGITAL_AUDIO_CLIENT":
                return 0
            if t in {"AUDIO_SELECTION", "DIGITAL_AUDIO_SERVER"}:
                return 1
            return 2

        ordered = [s for s in sources if isinstance(s, dict) and s.get("id") is not None]
        ordered.sort(key=_score)

        probed: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None

        for src in ordered[:limit]:
            try:
                device_id = int(src.get("id"))
            except Exception:
                continue

            np = await self._media_get_now_playing_async(device_id)
            nowp = np.get("now_playing") if isinstance(np, dict) else None
            cands = np.get("candidates") if isinstance(np, dict) else None

            has_nowp = isinstance(nowp, dict) and len(nowp) > 0
            has_cands = isinstance(cands, list) and len(cands) > 0
            probed.append(
                {
                    "device_id": device_id,
                    "name": src.get("name"),
                    "type": src.get("type"),
                    "has_now_playing": bool(has_nowp),
                    "candidates_count": len(cands) if isinstance(cands, list) else 0,
                }
            )

            if has_nowp or has_cands:
                best = {
                    "device_id": device_id,
                    "name": src.get("name"),
                    "type": src.get("type"),
                    "now_playing": nowp if isinstance(nowp, dict) else {},
                    "candidates": cands if isinstance(cands, list) else [],
                    "source": np.get("source") if isinstance(np, dict) else None,
                }
                # Prefer a result with normalized now_playing keys.
                if has_nowp:
                    break

        return {
            "ok": True,
            "room_id": room_id,
            "active": True,
            "best": best,
            "probe": probed,
            "max_sources": limit,
        }

    def room_now_playing(self, room_id: int, max_sources: int = 30) -> dict[str, Any]:
        # Give the async probe a bit of headroom (each item vars fetch has its own bounded timeout).
        return self._loop_thread.run(
            self._room_now_playing_async(int(room_id), int(max_sources)),
            timeout_s=25,
        )

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

            # Prefer a stable variable source for CURRENT_APP(_ID) polling.
            # Some proxies (notably legacy/placeholder media_service entries) may not reliably update
            # these variables even when LaunchApp succeeds.
            state_candidates = [r for r in group if str(r.get("proxy") or "").lower() == "media_service" and "roku" in str(r.get("protocolFilename") or "").lower()]
            state_candidates.sort(key=lambda r: (int(r.get("proxyOrder") or 9999), int(r.get("id") or 0)))
            state_device_id = int(state_candidates[0].get("id")) if state_candidates else int(primary_id)

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
                "state_device_id": state_device_id,
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
            poll_id = routing.get("state_device_id")
            if not isinstance(poll_id, int):
                poll_id = routing.get("protocol_id")
            if isinstance(poll_id, int):
                before = await _roku_current_app(poll_id)
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
                if isinstance(poll_id, int) and expected_roku_app_id is not None:
                    await asyncio.sleep(0.6)
                    snap = await _roku_current_app(poll_id)
                    attempts[-1]["roku_after"] = snap
                    if isinstance(snap, dict) and snap.get("CURRENT_APP_ID_INT") == expected_roku_app_id:
                        break

            after = await (_roku_current_app(poll_id) if isinstance(poll_id, int) else asyncio.sleep(0) or None)

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