r"""One-shot end-to-end runner for c4-mcp.

This script:
- (optionally) starts the Flask MCP HTTP server (read-only / guardrails mode)
- waits for /mcp/list to respond
- runs the HTTP validator suite
- runs STDIO validator(s)
- writes logs under ./logs/

Usage (PowerShell):
  .\.venv\Scripts\python.exe tools\run_e2e.py
  .\.venv\Scripts\python.exe tools\run_e2e.py --no-server --base-url http://127.0.0.1:3333

Exit codes:
  0 = all steps passed
  2 = at least one step failed
  3 = runner error (startup / configuration)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class StepResult:
    name: str
    ok: bool
    returncode: int
    out_path: Path
    err_path: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _venv_python(repo_root: Path) -> str:
    cand = repo_root / ".venv" / "Scripts" / "python.exe"
    if cand.exists():
        return str(cand)
    return sys.executable


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _http_get_json(url: str, timeout_s: float) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        raw = resp.read()
        if not raw:
            return {}
        import json

        obj = json.loads(raw.decode("utf-8"))
        return obj if isinstance(obj, dict) else {"_raw": obj}


def _wait_for_server(base_url: str, timeout_s: float, poll_s: float = 0.25) -> None:
    deadline = time.time() + float(timeout_s)
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            _http_get_json(base_url.rstrip("/") + "/mcp/list", timeout_s=2.0)
            return
        except Exception as e:
            last_err = repr(e)
            time.sleep(float(poll_s))
    raise TimeoutError(f"Timed out waiting for server at {base_url} (last_err={last_err})")


def _start_server(
    *,
    repo_root: Path,
    python_exe: str,
    logs_dir: Path,
    env: Dict[str, str],
) -> subprocess.Popen:
    out_path = logs_dir / "e2e_http_server_out.txt"
    err_path = logs_dir / "e2e_http_server_err.txt"

    out_f = open(out_path, "w", encoding="utf-8")
    err_f = open(err_path, "w", encoding="utf-8")

    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    proc = subprocess.Popen(
        [python_exe, str(repo_root / "app.py")],
        cwd=str(repo_root),
        stdout=out_f,
        stderr=err_f,
        env=env,
        text=True,
        creationflags=creationflags,
    )

    (logs_dir / "e2e_http_server.pid").write_text(str(proc.pid), encoding="utf-8")
    return proc


def _stop_server(proc: subprocess.Popen, timeout_s: float = 6.0) -> None:
    try:
        if proc.poll() is not None:
            return
        proc.terminate()
        proc.wait(timeout=float(timeout_s))
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _run_step(
    *,
    name: str,
    cmd: List[str],
    cwd: Path,
    logs_dir: Path,
    env: Dict[str, str],
) -> StepResult:
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_")
    out_path = logs_dir / f"e2e_{safe_name}.out.txt"
    err_path = logs_dir / f"e2e_{safe_name}.err.txt"

    with open(out_path, "w", encoding="utf-8") as out_f, open(err_path, "w", encoding="utf-8") as err_f:
        p = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=out_f, stderr=err_f, text=True)
        return StepResult(name=name, ok=(p.returncode == 0), returncode=int(p.returncode), out_path=out_path, err_path=err_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="One-shot E2E runner for c4-mcp")
    ap.add_argument("--base-url", default="http://127.0.0.1:3333", help="MCP HTTP base URL")
    ap.add_argument("--no-server", action="store_true", help="Do not start/stop app.py; just run validators")
    ap.add_argument("--skip-http", action="store_true", help="Skip HTTP validator suite")
    ap.add_argument("--skip-stdio", action="store_true", help="Skip STDIO validator suite")
    ap.add_argument("--server-wait-s", type=float, default=25.0, help="Time to wait for HTTP server readiness")
    ap.add_argument("--timeout-s", type=float, default=25.0, help="Timeout passed to validators where supported")
    ap.add_argument("--session-id", default="e2e-session-1", help="X-Session-Id used by session-memory validator")
    ap.add_argument("--api-key", default=None, help="X-API-Key passed to validators (if auth enabled)")
    ap.add_argument("--logs-dir", default=str(_repo_root() / "logs"), help="Directory to write logs")
    ap.add_argument("--keep-server", action="store_true", help="If starting server, leave it running")

    args = ap.parse_args()

    repo_root = _repo_root()
    logs_dir = Path(args.logs_dir)
    _ensure_dir(logs_dir)

    python_exe = _venv_python(repo_root)

    # Use a stable run id to make it easy to correlate logs in CI or repeated runs.
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    (logs_dir / "e2e_last_run.txt").write_text(run_id + "\n", encoding="utf-8")

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(repo_root))

    # Default to safe-by-default: guardrails on, writes off.
    env.setdefault("C4_WRITE_GUARDRAILS", "true")
    env.setdefault("C4_WRITES_ENABLED", "false")

    base_url = str(args.base_url)

    server_proc: Optional[subprocess.Popen] = None
    try:
        if not args.no_server:
            server_proc = _start_server(repo_root=repo_root, python_exe=python_exe, logs_dir=logs_dir, env=env)
            _wait_for_server(base_url, timeout_s=float(args.server_wait_s))

        results: List[StepResult] = []

        headers_args: List[str] = []
        if args.api_key:
            headers_args = ["--api-key", str(args.api_key)]

        if not args.skip_http:
            results.append(
                _run_step(
                    name="http_validate_mcp_e2e",
                    cmd=[python_exe, str(repo_root / "tools" / "validate_mcp_e2e.py"), "--base-url", base_url, "--timeout", str(args.timeout_s)]
                    + headers_args,
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    env=env,
                )
            )

            results.append(
                _run_step(
                    name="http_validate_listen",
                    cmd=[python_exe, str(repo_root / "tools" / "validate_listen.py"), "--base-url", base_url, "--timeout", str(args.timeout_s)]
                    + headers_args,
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    env=env,
                )
            )

            results.append(
                _run_step(
                    name="http_validate_alarm",
                    cmd=[python_exe, str(repo_root / "tools" / "validate_alarm.py"), "--base-url", base_url, "--timeout", str(args.timeout_s)]
                    + headers_args,
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    env=env,
                )
            )

            results.append(
                _run_step(
                    name="http_validate_scenes",
                    cmd=[python_exe, str(repo_root / "tools" / "validate_scenes.py"), "--base-url", base_url, "--timeout-s", str(args.timeout_s)],
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    env=env,
                )
            )

            results.append(
                _run_step(
                    name="http_validate_shades",
                    cmd=[python_exe, str(repo_root / "tools" / "validate_shades.py"), "--base-url", base_url, "--timeout-s", str(args.timeout_s)],
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    env=env,
                )
            )

            results.append(
                _run_step(
                    name="http_validate_session_memory",
                    cmd=[
                        python_exe,
                        str(repo_root / "tools" / "validate_http_session_memory.py"),
                        "--base-url",
                        base_url,
                        "--timeout-s",
                        str(args.timeout_s),
                        "--session-id",
                        str(args.session_id),
                    ],
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    env=env,
                )
            )

        if not args.skip_stdio:
            results.append(
                _run_step(
                    name="stdio_validate_mcp_stdio",
                    cmd=[python_exe, str(repo_root / "tools" / "validate_mcp_stdio.py")],
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    env=env,
                )
            )

            results.append(
                _run_step(
                    name="stdio_validate_claude_stdio",
                    cmd=[python_exe, str(repo_root / "tools" / "validate_claude_stdio.py")],
                    cwd=repo_root,
                    logs_dir=logs_dir,
                    env=env,
                )
            )

        failed = [r for r in results if not r.ok]

        summary_path = logs_dir / "e2e_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"run_id={run_id}\n")
            f.write(f"base_url={base_url}\n")
            f.write(f"python={python_exe}\n")
            for r in results:
                f.write(f"{r.name}: rc={r.returncode} ok={r.ok} out={r.out_path.name} err={r.err_path.name}\n")

        # Human-friendly console summary.
        ok_count = len([r for r in results if r.ok])
        print(f"E2E steps passed: {ok_count}/{len(results)}")
        print(f"Logs: {logs_dir}")
        print(f"Summary: {summary_path}")

        if failed:
            print("FAILED steps:")
            for r in failed:
                print(f"- {r.name} (rc={r.returncode}) -> {r.err_path.name}")
            return 2

        print("PASS")
        return 0

    except TimeoutError as e:
        print(f"ERROR: {e}")
        return 3
    except Exception as e:
        print(f"ERROR: {e!r}")
        return 3
    finally:
        if server_proc is not None and not args.keep_server:
            _stop_server(server_proc)


if __name__ == "__main__":
    raise SystemExit(main())
