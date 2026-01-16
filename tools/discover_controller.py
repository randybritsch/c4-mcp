"""Discover the primary Control4 controller IP and (optionally) write it to config.json.

Why this exists:
- The MCP server needs a Director host in config.json (or env vars) to start.
- In a fresh setup, users may not know the controller IP.

Approach (best-effort):
- Enumerate local IPv4 subnets (Windows: parse `ipconfig` output), or accept explicit `--subnet`.
- Probe each candidate host for a Control4 Director API path.
- If a likely Director endpoint is found, optionally persist it to config.json.

Safety:
- Bounded per-host timeouts.
- Concurrency-limited.
- Only scans the local subnets detected (or user-specified).

Usage:
    .\\.venv\\Scripts\\python.exe tools\\discover_controller.py
    .\\.venv\\Scripts\\python.exe tools\\discover_controller.py --write
    .\\.venv\\Scripts\\python.exe tools\\discover_controller.py --subnet 192.168.1.0/24 --write

Exit codes:
- 0: found (and optionally wrote)
- 2: not found
- 3: invalid args / config write failure
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network, ip_address, ip_network
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    import aiohttp
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing dependency aiohttp. Run with the project venv: .\\.venv\\Scripts\\python.exe tools\\discover_controller.py"
    ) from e


DIRECTOR_PROBE_PATHS = (
    "/api/v1/agents/ui_configuration",
    "/api/v1/items",
)


@dataclass(frozen=True)
class Candidate:
    ip: str
    scheme: str
    path: str
    status: int


def _is_probably_private_lan_ip(ip: str) -> bool:
    try:
        a = ip_address(ip)
        return a.version == 4 and (a.is_private or a.is_link_local) and not a.is_loopback
    except Exception:
        return False


def _parse_windows_ipconfig_subnets() -> List[IPv4Network]:
    """Parse local IPv4 + subnet mask from `ipconfig` output.

    This is best-effort but works well on typical Windows systems.
    """

    try:
        raw = subprocess.check_output(["ipconfig"], text=True, encoding="utf-8", errors="replace")
    except Exception:
        return []

    # Matches:
    #   IPv4 Address. . . . . . . . . . . : 192.168.1.123
    #   Subnet Mask . . . . . . . . . . . : 255.255.255.0
    ip_re = re.compile(r"IPv4 Address[^:]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})")
    mask_re = re.compile(r"Subnet Mask[^:]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})")

    ips = ip_re.findall(raw)
    masks = mask_re.findall(raw)

    # ipconfig is structured; simplest robust approach is pair by order.
    subnets: List[IPv4Network] = []
    for ip_s, mask_s in zip(ips, masks):
        try:
            if ip_s.startswith("169.254.") or ip_s.startswith("127."):
                continue
            net = IPv4Network((IPv4Address(ip_s), mask_s), strict=False)
            subnets.append(net)
        except Exception:
            continue

    # De-dupe
    uniq: List[IPv4Network] = []
    seen = set()
    for n in subnets:
        key = str(n)
        if key not in seen:
            seen.add(key)
            uniq.append(n)
    return uniq


def _default_subnets() -> List[IPv4Network]:
    if sys.platform.startswith("win"):
        nets = _parse_windows_ipconfig_subnets()
        if nets:
            return nets

    # Generic fallbacks for non-Windows or if parsing fails.
    fallbacks = [
        "192.168.0.0/24",
        "192.168.1.0/24",
        "10.0.0.0/24",
    ]
    out: List[IPv4Network] = []
    for s in fallbacks:
        try:
            out.append(ip_network(s))
        except Exception:
            pass
    return out


def _iter_host_ips(network: IPv4Network, max_hosts: int) -> Iterable[str]:
    # Skip network and broadcast automatically; ipaddress.hosts() does that.
    count = 0
    for ip in network.hosts():
        if count >= max_hosts:
            break
        count += 1
        yield str(ip)


async def _probe_one(session: aiohttp.ClientSession, ip: str, scheme: str, timeout_s: float) -> Optional[Candidate]:
    timeout = aiohttp.ClientTimeout(total=float(timeout_s))
    headers = {"User-Agent": "c4-mcp-discovery/1.0"}

    for path in DIRECTOR_PROBE_PATHS:
        url = f"{scheme}://{ip}{path}"
        try:
            async with session.get(url, headers=headers, timeout=timeout, ssl=False) as r:
                status = int(r.status)
                # Control4 Director endpoints typically require auth; 401/403 is a strong signal.
                if status in (401, 403):
                    return Candidate(ip=ip, scheme=scheme, path=path, status=status)

                # Some controllers may respond with 200 + JSON even without auth on some endpoints;
                # treat any 2xx as a possible signal (but lower confidence).
                if 200 <= status < 300:
                    return Candidate(ip=ip, scheme=scheme, path=path, status=status)

        except asyncio.CancelledError:
            raise
        except Exception:
            continue

    return None


async def _discover_async(
    subnets: List[IPv4Network],
    timeout_s: float,
    concurrency: int,
    max_hosts_per_subnet: int,
    prefer_https: bool,
) -> List[Candidate]:
    sem = asyncio.Semaphore(max(1, int(concurrency)))
    found: List[Candidate] = []

    schemes = ["https", "http"] if prefer_https else ["http", "https"]

    async with aiohttp.ClientSession() as session:

        async def run_ip(ip: str) -> None:
            async with sem:
                for scheme in schemes:
                    cand = await _probe_one(session, ip, scheme, timeout_s=timeout_s)
                    if cand is not None:
                        found.append(cand)
                        return

        tasks: List[asyncio.Task] = []
        for net in subnets:
            for ip in _iter_host_ips(net, max_hosts=max_hosts_per_subnet):
                # Quick skip for obviously non-LAN-ish addresses.
                if not _is_probably_private_lan_ip(ip):
                    continue
                tasks.append(asyncio.create_task(run_ip(ip)))

        if not tasks:
            return []

        # We want the first few hits, not necessarily exhaustive.
        # Wait for completion but allow cancellation once we have a confident candidate.
        try:
            while tasks:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = list(pending)
                if found:
                    # If we have at least one strong signal (401/403), stop early.
                    if any(c.status in (401, 403) for c in found):
                        for t in pending:
                            t.cancel()
                        break
        finally:
            # Best-effort drain
            for t in tasks:
                if not t.done():
                    t.cancel()

        return found


def _choose_best(cands: List[Candidate]) -> Optional[Candidate]:
    if not cands:
        return None
    # Prefer auth-required Director endpoints.
    strong = [c for c in cands if c.status in (401, 403)]
    if strong:
        return strong[0]
    return cands[0]


def _load_or_create_config(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to parse config file: {path}") from e

    # Create a new config from env vars if possible.
    user = (os.environ.get("C4_USERNAME") or os.environ.get("CONTROL4_USERNAME") or "").strip()
    pw = (os.environ.get("C4_PASSWORD") or os.environ.get("CONTROL4_PASSWORD") or "").strip()

    if not user or not pw:
        raise RuntimeError(
            "config.json does not exist and username/password are not set in env vars. "
            "Set C4_USERNAME and C4_PASSWORD (or CONTROL4_*) to create a config automatically."
        )

    return {"host": "", "username": user, "password": pw}


def _write_config(path: Path, cfg: dict) -> None:
    path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Discover Control4 controller IP and optionally write config.json")
    p.add_argument("--subnet", action="append", default=[], help="CIDR subnet(s) to scan, e.g. 192.168.1.0/24")
    p.add_argument("--timeout-s", type=float, default=0.6, help="Per-host HTTP timeout (seconds)")
    p.add_argument("--concurrency", type=int, default=96, help="Max concurrent probes")
    p.add_argument("--max-hosts", type=int, default=256, help="Max hosts to scan per subnet")
    p.add_argument("--prefer-https", action="store_true", default=True)
    p.add_argument("--prefer-http", dest="prefer_https", action="store_false")
    p.add_argument("--write", action="store_true", help="Write discovered host to config.json")
    p.add_argument(
        "--config-path",
        default=str(Path(__file__).resolve().parents[1] / "config.json"),
        help="Path to config.json (default: repo root config.json)",
    )

    args = p.parse_args(argv)

    subnets: List[IPv4Network] = []
    if args.subnet:
        for s in args.subnet:
            try:
                subnets.append(ip_network(str(s), strict=False))
            except Exception:
                print(f"Invalid subnet: {s!r}", file=sys.stderr)
                return 3
    else:
        subnets = _default_subnets()

    if not subnets:
        print("No subnets detected/provided.", file=sys.stderr)
        return 3

    print("Scanning subnets:")
    for n in subnets:
        print(f"- {n}")

    cands = asyncio.run(
        _discover_async(
            subnets=subnets,
            timeout_s=float(args.timeout_s),
            concurrency=int(args.concurrency),
            max_hosts_per_subnet=int(args.max_hosts),
            prefer_https=bool(args.prefer_https),
        )
    )

    best = _choose_best(cands)
    if best is None:
        print("No Control4 Director candidates found.")
        return 2

    print("\nFound candidate:")
    print(f"- host: {best.ip}")
    print(f"- scheme: {best.scheme}")
    print(f"- probe_path: {best.path}")
    print(f"- status: {best.status}")

    if args.write:
        cfg_path = Path(args.config_path)
        cfg = _load_or_create_config(cfg_path)
        cfg["host"] = best.ip
        _write_config(cfg_path, cfg)
        print(f"\nWrote host to: {cfg_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
