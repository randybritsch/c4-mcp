# c4-mcp

<!-- mcp-name: io.github.randybritsch/c4-mcp -->

Turn your Control4 system into a **Model Context Protocol (MCP)** toolset, so any MCP-capable client (Claude Desktop, custom agents, scripts) can **query rooms/devices** and **safely run automations**.

[![CI](https://github.com/randybritsch/c4-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/randybritsch/c4-mcp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/github/license/randybritsch/c4-mcp)](LICENSE)

## Why this is cool

- **Works with real MCP clients**: HTTP transport for dev/scripts + **STDIO JSON-RPC** for clients like **Claude Desktop**.
- **One integration point for many clients**: use the same toolset from Claude Desktop, scripts, or your own agents without rewriting Control4 logic.
- **Structured tool schemas = fewer mistakes**: explicit inputs/outputs (room/device IDs, levels, setpoints, etc.) reduce ambiguity vs. prompt-only automations.
- **Safe-by-default controls**: optional write guardrails, read-only mode, and allow/deny lists for state-changing tools.
- **Session memory for follow-ups**: enables natural multi-step flows like “turn on the basement lights… now dim those lights”.
- **Smarter “lights” semantics**: room-based lighting ops avoid accidentally targeting fans/heaters/outlets.
- **One-command validation**: an end-to-end runner exercises HTTP + STDIO so you can ship changes with confidence.
- **Tunable performance**: inventory caching + env-configurable timeouts for slower Control4 projects.
- **Credentials stay local**: keep `config.json` on your machine (gitignored) and choose local-only STDIO or LAN HTTP based on your risk tolerance.

## What you can do

- Discover rooms/devices by name, category, and room (plus resolvers for “best-effort” name-based calls).
- Activate scenes, control shades, query variables/commands, and (optionally) change state (lights/locks/thermostat/media).
- Use it as a local “home automation brain” for chat + agents without hard-coding your project’s device IDs.

## Example prompts (copy/paste)

These work well in MCP clients like Claude Desktop (the client will call tools under the hood):

```text
List all rooms.

Show me the lights in the Basement.

Turn on the basement lights.

Now dim those lights to 30%.

Activate the "Movie Time" scene in the Living Room.

Which doors are currently unlocked?
```

## Advanced prompts (hard in the stock Control4 app UI)

These are examples of the kind of *cross-device*, *conditional*, and *multi-step* requests that are awkward (or not possible) to do purely in the standard Control4 app UI without building custom automation logic elsewhere.

Note: prompts that *change state* (lights/locks/thermostat/media) require `C4_WRITES_ENABLED=true`. Read-only prompts (inventory/status/reporting) work fine with the safe default `C4_WRITES_ENABLED=false`.

```text
Run a “Good Night” sweep: turn off all lights except Hallway (10%), lock all exterior doors, set Downstairs thermostat to 68°F, then report what succeeded/failed.

If any door is unlocked, lock it — but do NOT lock the Garage door.

Find anything in the Basement that is currently on (lights, outlets), list it, then turn off everything except the dehumidifier outlet.

I’m leaving: turn off all AV devices, activate the “Away” scene, and confirm the house is secured (all locks locked).

The Basement lights are on — tell me which specific loads are on and turn off only the ones that are above 50%.

Compare the Living Room lights vs. Kitchen lights: which room is brighter right now? Then set them to match.

Do a safety check: list any unlocked doors, any lights left on in the Basement, and the current thermostat setpoints for each zone.
```

Tip: If you run with `C4_WRITE_GUARDRAILS=true` and `C4_WRITES_ENABLED=false`, you’ll get a safe read-only experience until you explicitly enable writes.

## Ambiguity & disambiguation (recommended)

Name-based tools can legitimately return **multiple matches** (e.g., “Basement” might match several rooms). In that case, `c4-mcp` returns a structured failure with an **ambiguous** marker and a candidate list.

Recommended client pattern:

1) Call the name-based tool with `include_candidates=true` (or accept the default if the tool always includes them).
2) If the response indicates ambiguity, show the candidates to the user and let them pick.
3) Re-call the tool with `require_unique=true` and a more specific scope (e.g., `room_id` / `room_name`, or exact `device_name`).

This is how higher-level apps can support natural commands like “turn on the basement lights” while still being deterministic and safe.

## Direct HTTP examples (no MCP client required)

If you’re not using an MCP client yet, you can still call the server directly.

List tools:

- `GET http://127.0.0.1:3333/mcp/list`

Synology/Compose note:

- Inside Docker/Compose, `c4-mcp` commonly listens on `:3333`.
- On the NAS/LAN, it’s often published as host port `:3334` → container `:3333`.
	- Example: `GET http://<NAS_IP>:3334/mcp/list`

Call a tool (example: list rooms):

PowerShell:

```powershell
$base = 'http://127.0.0.1:3333'  # or: http://<NAS_IP>:3334
Invoke-RestMethod -Method Post -Uri ($base + '/mcp/call') -ContentType 'application/json' -Body (
	@{ kind = 'tool'; name = 'c4_list_rooms'; args = @{} } | ConvertTo-Json -Depth 10
)
```

curl:

```bash
curl -s http://127.0.0.1:3333/mcp/call \
	-H "Content-Type: application/json" \
	-d '{"kind":"tool","name":"c4_list_rooms","args":{}}'
```

### PowerShell tips (Windows)

**1) `/mcp/list` returns a tool *map* (not an array).**

In PowerShell, `tools` is a `PSCustomObject` where each property name is a tool name.

```powershell
$r = Invoke-RestMethod -Method Get -Uri 'http://127.0.0.1:3333/mcp/list' -TimeoutSec 10
$toolNames = $r.tools.PSObject.Properties.Name | Sort-Object
"tools_count=$($r.tools.PSObject.Properties.Count)"
$toolNames | Select-Object -First 25
```

**2) Quick start/stop on Windows (detached, logs captured).**

This avoids confusion around multiple terminals / Ctrl+C and makes it easy to inspect server logs.

```powershell
# Safe-by-default: guardrails on, writes off
$env:C4_WRITE_GUARDRAILS='true'
$env:C4_WRITES_ENABLED='false'
$env:PYTHONUTF8='1'

New-Item -ItemType Directory -Force -Path logs | Out-Null
$p = Start-Process -FilePath .\.venv\Scripts\python.exe -ArgumentList @('app.py') -PassThru -WindowStyle Hidden `
  -RedirectStandardOutput 'logs\http_server_out.txt' -RedirectStandardError 'logs\http_server_err.txt'
$p.Id | Set-Content -Encoding ascii 'logs\http_server.pid'
"started_pid=$($p.Id)"

# Sanity check
Test-NetConnection 127.0.0.1 -Port 3333 | Select-Object TcpTestSucceeded
```

Stop it later:

```powershell
Stop-Process -Id (Get-Content .\logs\http_server.pid)
```

If `/mcp/list` hangs or errors, check `logs/http_server_err.txt`.

## Hosting on a NAS (Synology) — LAN only

The HTTP server is designed to run locally. To run it on a NAS and reach it from other machines on your LAN:

- Bind to all interfaces with `C4_BIND_HOST=0.0.0.0` (default is localhost-only).
- Keep it LAN-only using Synology firewall rules (recommended) or a VPN (for remote access later).

### Docker (recommended on Synology)

This repo includes a `Dockerfile` and `docker-compose.yml`.

On Synology (Container Manager), run a compose project that:

- Publishes port `3333` to your LAN.
- Mounts your real `config.json` (keep credentials off git).
- Keeps writes off by default: `C4_WRITES_ENABLED=false`.

Before you start the compose project, create your local config file:

- Copy `config.example.json` → `config.json` and fill in values (this repo ignores `config.json`).

`docker-compose.yml` already sets `C4_BIND_HOST=0.0.0.0`.

LAN-only note: do **not** expose port 3333 to the internet. Use Synology Firewall to allow only your LAN subnet (e.g. `192.168.0.0/16`) to reach TCP 3333.

#### Troubleshooting Synology builds

If Container Manager fails with an error like:

`unable to prepare context: unable to evaluate symlinks in Dockerfile path: lstat /volume1/...`

That means Docker cannot find or access the folder you selected as the **build context** (the folder that should contain your `Dockerfile` and source code).

Fix:

- Put the repo files on the NAS under a real shared-folder path, e.g. `/volume1/docker/c4-mcp/`.
- Ensure that folder contains at least: `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `app.py`, and the Python modules.
- In Container Manager, create the Compose project using that exact folder as the project path (don’t point it at just `/volume1/docker/` unless the files are actually there).
- If your shared folder isn’t on `volume1`, use the correct volume (e.g. `/volume2/...`).

### Host/port env vars

- `C4_BIND_HOST` (default `127.0.0.1`)
- `C4_PORT` (default `3333`)

Example (LAN): `C4_BIND_HOST=0.0.0.0` and `C4_PORT=3333`

## Security / publishing note (read this)

This project talks to your Control4 system using credentials (and often a local controller IP).

- Never commit real credentials. Keep `config.json` local-only (it is ignored by `.gitignore`).
- If you accidentally committed credentials at any point, rotate them immediately and rewrite git history before making the repo public.

### Public GitHub checklist (do this before you publish)

- Ensure `config.json` is not in git history. At minimum it should not be tracked in your current tree.
	- Quick check: `git ls-files config.json` should return nothing.
	- If it was ever committed: rotate your Control4 password and rewrite history (e.g., `git filter-repo`), then force-push.
- Prefer `C4_CONFIG_PATH` (pointing to a file outside the repo) for the safest setup.

## Registry metadata

This section is meant to be **copy/paste-friendly** for MCP registries and "server list" directories.

- **Name**: `c4-mcp`
- **Category**: Home Automation / Control4
- **Repo**: https://github.com/randybritsch/c4-mcp
- **License**: MIT
- **Transports**:
  - **STDIO (JSON-RPC)**: `claude_stdio_server.py` (for Claude Desktop and other stdio-based MCP clients)
	- **HTTP**: `app.py` (defaults to `http://127.0.0.1:3333`; override with `C4_BIND_HOST`/`C4_PORT`; endpoints: `/mcp/list`, `/mcp/call`)
- **Configuration / secrets**:
  - Recommended: set `C4_CONFIG_PATH` to a local `config.json` that contains `host`, `username`, `password` (keep this file gitignored)
  - Optional: set `C4_HOST` (non-secret) to override `host` from `config.json`
  - Optional: set `C4_USERNAME`/`C4_PASSWORD` (secret) via OS env vars (must be provided together)
- **Safety defaults (recommended)**:
  - For read-only-by-default runs: `C4_WRITE_GUARDRAILS=true` + `C4_WRITES_ENABLED=false`
  - Optional filters: `C4_WRITE_ALLOWLIST` / `C4_WRITE_DENYLIST`
	- Scheduler Agent writes are additionally gated: `c4_scheduler_set_enabled` requires `C4_SCHEDULER_WRITES_ENABLED=true`

## Python version note (important)

This project depends on `flask-mcp-server`, which in turn depends on `pydantic`/`pydantic-core`.
At the time of writing, **Python 3.14 will not work out-of-the-box on Windows** because `pydantic-core` does not ship wheels for it yet.

Use **Python 3.12** (recommended) or another version with `pydantic-core` wheels available.

## Install from PyPI (recommended for most users)

If you just want to use the server (not hack on the repo), you can install it from PyPI:

- `python -m pip install c4-mcp`

Then run either transport:

- STDIO (for Claude Desktop / stdio MCP clients): `c4-mcp`
- HTTP (for scripts / curl / local dev): `c4-mcp-http`

You still need to provide Control4 config via `C4_CONFIG_PATH` (recommended) or `C4_HOST`/`C4_USERNAME`/`C4_PASSWORD`.

## Easy install (almost one command)

If you have **Node.js + npm** installed, you can bootstrap the Python venv + dependencies with one command:

- Install Node.js (includes npm): https://nodejs.org/

- `npm run setup`

Then:

- Start HTTP server: `npm run start`
- Start STDIO server (Claude-style): `npm run start:stdio`
- Run end-to-end checks: `npm run e2e`

This is just a convenience wrapper around the existing Python setup steps (it creates `.venv` and installs `requirements.txt`).

## Setup

This project is intended to work with any Control4 system. Nothing in the server is hard-coded to a specific home.

1) Install dependencies (in a venv)

This repo uses `requirements.txt` as the source of truth.

Windows (PowerShell):

- `python -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`
- `python -m pip install -r requirements.txt`

macOS / Linux (bash/zsh):

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `python -m pip install -r requirements.txt`

2) Provide Control4 connection config via either:

- Environment variables:
	- `C4_HOST` (Director/Controller IP or hostname; scheme optional)
	- `C4_USERNAME` (Control4 account email)
	- `C4_PASSWORD` (Control4 account password)

or

- A local config file (not committed): copy `config.example.json` to `config.json` and fill in values.

## VS Code setup (recommended)

1) Install VS Code extensions

- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)

2) Open the repo folder in VS Code

- File → Open Folder… → select this repo.

3) Create/select a virtual environment

Option A (VS Code UI):

- `Ctrl+Shift+P` → **Python: Create Environment** → choose `venv` → select your Python 3.12 interpreter.

Option B (terminal):

- `python -m venv .venv`
- Activate it (see Setup section above)
- `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`

4) Install dependencies

- `python -m pip install -r requirements.txt`

5) Provide Control4 config while developing

- **Config file**: copy `config.example.json` → `config.json` (kept local-only; ignored by git)

or

- **Environment variables**: set `C4_HOST`, `C4_USERNAME`, `C4_PASSWORD`

Tip (VS Code-friendly): create a local `.env` file in the repo root (ignored by git) and use it from a debug config.

6) Run the server

- VS Code Terminal: `python app.py`

7) Optional: Debug with F5

Create a local `.vscode/launch.json` (this repo ignores `.vscode/` by default) like:

```json
{
	"version": "0.2.0",
	"configurations": [
		{
			"name": "c4-mcp (HTTP server)",
			"type": "python",
			"request": "launch",
			"program": "${workspaceFolder}/app.py",
			"console": "integratedTerminal",
			"justMyCode": true,
			"envFile": "${workspaceFolder}/.env"
		},
		{
			"name": "c4-mcp (STDIO server)",
			"type": "python",
			"request": "launch",
			"program": "${workspaceFolder}/claude_stdio_server.py",
			"console": "integratedTerminal",
			"justMyCode": true,
			"envFile": "${workspaceFolder}/.env"
		}
	]
}
```

### First-time setup (recommended)

If you're setting this up for the first time and don't know your controller IP yet, this is the quickest flow:

1) Set credentials for discovery + server login:

Windows (PowerShell):

- `$env:C4_USERNAME = "you@example.com"`
- `$env:C4_PASSWORD = "your-password"`

macOS / Linux (bash/zsh):

- `export C4_USERNAME="you@example.com"`
- `export C4_PASSWORD="your-password"`

2) Auto-discover the controller IP and write it to `config.json`:

- `python tools\discover_controller.py --write`

3) Start the MCP server:

- `python app.py`

4) Verify it's up:

- `GET http://127.0.0.1:3333/mcp/list`

### Optional: auto-discover controller IP (Windows-friendly)

If you don't know the controller IP yet, you can run a best-effort LAN discovery scan and optionally write the discovered IP into `config.json`:

- Dry run (no writes): `python tools\discover_controller.py`
- Write `host` to `config.json`: `python tools\discover_controller.py --write`

Notes:
- Scans only the local subnets detected from `ipconfig` (or use `--subnet 192.168.1.0/24`).
- Uses bounded per-host timeouts + concurrency limits.
- If `config.json` does not exist, you can set `C4_USERNAME` and `C4_PASSWORD` so the script can create it.

## Run

- Start server: `python app.py`
- Verify MCP is up: `GET http://127.0.0.1:3333/mcp/list`

## End-to-end validation (one command)

This starts the HTTP server in **read-only guardrails mode**, runs the HTTP validator suite, runs both STDIO validators, and then stops the server.

Windows (PowerShell):

- `.\.venv\Scripts\python.exe tools\run_e2e.py`

macOS / Linux (bash/zsh):

- `./.venv/bin/python tools/run_e2e.py`

If you already have the server running and only want to run validators:

Windows (PowerShell):

- `.\.venv\Scripts\python.exe tools\run_e2e.py --no-server --base-url http://127.0.0.1:3333`

macOS / Linux (bash/zsh):

- `./.venv/bin/python tools/run_e2e.py --no-server --base-url http://127.0.0.1:3333`

## Claude Desktop (MCP stdio) setup (Windows)

Claude Desktop launches MCP servers over **STDIO** (it starts a subprocess and speaks JSON-RPC over stdin/stdout).

Claude Desktop uses the official MCP method surface (`initialize`, `tools/list`, `tools/call`).
This repo includes a small shim, `claude_stdio_server.py`, that adapts Claude's method surface to the existing Flask MCP tool registry.

1) Create your venv using Python 3.12 (recommended; 3.13 also works):

- `py -3.12 -m venv .venv`
- `.\.venv\Scripts\Activate.ps1`
- `python -m pip install -r requirements.txt`

2) Edit Claude Desktop config:

- File: `%APPDATA%\Claude\claude_desktop_config.json`

Security note: avoid pasting your Control4 password directly into the Claude Desktop config file.
Prefer one of these safer options:

- Put credentials in a local `config.json` in this repo (gitignored), and only keep non-secret settings in Claude's config.
- Or set `C4_USERNAME` / `C4_PASSWORD` as OS environment variables, and omit them from Claude's config.

Recommended Claude-side config: set **only** `C4_CONFIG_PATH` (pointing at your local `config.json`) and keep all secrets out of Claude Desktop.
If you set `C4_USERNAME`/`C4_PASSWORD` via env vars, they must be provided together.

Add an MCP server entry like this (edit paths + optional env vars):

```json
{
	"mcpServers": {
		"c4-mcp": {
			"command": "C:\\Users\\YOUR_USER\\c4-mcp\\.venv\\Scripts\\python.exe",
			"args": ["-u", "C:\\Users\\YOUR_USER\\c4-mcp\\claude_stdio_server.py"],
			"cwd": "C:\\Users\\YOUR_USER\\c4-mcp",
			"env": {
				"PYTHONUTF8": "1",
				"PYTHONIOENCODING": "utf-8",
				"C4_STDIO_TOOL_MODE": "compact",
				"C4_CONFIG_PATH": "C:\\Users\\YOUR_USER\\c4-mcp\\config.json",
				"C4_WRITE_GUARDRAILS": "true",
				"C4_WRITES_ENABLED": "false",
				"C4_DIRECTOR_TIMEOUT_S": "30",
				"C4_GET_ALL_ITEMS_TIMEOUT_S": "75"
			}
		}
	}
}
```

Notes:
- `-u` is recommended on Windows so STDIO JSON-RPC responses are not buffered.
- `C4_STDIO_TOOL_MODE=compact` keeps `tools/list` small and avoids Claude Desktop choking on a huge tool catalog. Set it to `all` to expose everything.

If Claude fails to start the server and you see an error like:

`python.exe: can't open file '...\\AnthropicClaude\\...\\mcp_cli.py': [Errno 2] No such file or directory`

it means Claude is trying to resolve a **relative path** from its own install directory.
Fix by using **absolute paths** for scripts in `args` and keeping `cwd` set to the repo root.

Tip: set `C4_STDIO_DEBUG=true` in the Claude config `env` to log each JSON-RPC request/response to the Claude MCP log.

If you use `config.json` for credentials, copy `config.example.json` to `config.json` and set `host`, `username`, and `password` there.

If Claude starts the server but tool calls fail and you see an error like:

`RuntimeError: Invalid config file '...\\config.json': username/password must be non-empty (or provide C4_USERNAME/C4_PASSWORD env vars)`

then your `config.json` has blank credentials. Fix by filling in `username`/`password` in `config.json` or setting `C4_USERNAME` and `C4_PASSWORD` in the Claude config `env` (they must be provided together).

Optional (non-secret) env vars you can add to Claude's config if needed:

- `C4_CONFIG_PATH`: point to a `config.json` stored elsewhere

To change the Director host, edit `config.json` (recommended) or point `C4_CONFIG_PATH` at a different config file.

3) Restart Claude Desktop.

If everything is wired up correctly, Claude should show the `c4-mcp` tools as available.

Notes:
- All non-protocol logging must go to **stderr**; this repo's logging defaults to stderr.
- If you want to enable write tools, flip `C4_WRITES_ENABLED` to `true` (guardrails still apply).

### Troubleshooting timeouts

If listing rooms/devices times out on first run, increase these (Claude config `env` or your shell env):

- `C4_DIRECTOR_TIMEOUT_S` (per Director request timeout)
- `C4_GET_ALL_ITEMS_TIMEOUT_S` (overall inventory fetch timeout)

### Optional: write guardrails (recommended for safety)

By default, tools that change state (locks, lights, thermostat, media remote, etc.) are **blocked** unless you explicitly enable writes.
If you want an extra safety layer (recommended), set:

- `C4_WRITE_GUARDRAILS=true` (turns on enforcement)
- `C4_WRITES_ENABLED=false` (keep write tools blocked; flip to `true` when you actually want writes)

Optional filters (comma-separated tool names):

- `C4_WRITE_ALLOWLIST=c4_light_set_level,c4_light_ramp` (only allow these write tools)
- `C4_WRITE_DENYLIST=c4_lock_unlock,c4_lock_lock` (block these write tools)

### Performance knobs

Some tools rely on inventory scans (`get_all_items`) for name-based resolution and listing. You can speed these up with a small in-process cache:

- `C4_ITEMS_CACHE_TTL_S=5` (default) caches the inventory for 5 seconds
- `C4_ITEMS_CACHE_TTL_S=0` disables the cache

## Discover IDs

Device and room ids are specific to your Control4 project. Use:

- `c4_list_rooms`
- `c4_find_rooms` / `c4_resolve_room` (search by name)
- `c4_list_devices` (by category)
- `c4_list_devices` category: `shades` (best-effort discovery)
- `c4_list_devices` category: `scenes` (best-effort; based on UI Buttons)
- `c4_find_devices` / `c4_resolve_device` (search by name; optional category/room filters)
- `c4_resolve` (resolve room + device together; device resolution can be scoped to the resolved room)

### Shades / blinds (best-effort)

If your project has shades/blinds, try:

- `c4_shade_list`
- `c4_shade_get_state` (returns `position` 0-100 when available)
- `c4_shade_open` / `c4_shade_close` / `c4_shade_stop`
- `c4_shade_set_position`

Troubleshooting:
- Use `c4_item_commands(device_id)` to see the exact command names your shade driver exposes.
- Use `c4_item_variables(device_id)` to inspect which variable contains position/level.

Then pass the discovered ids into tools like `c4_media_watch_launch_app` or the scripts in `tools/`.

If you prefer name-based calls (no ids), use `c4_media_watch_launch_app_by_name`.

### Lighting scenes (best-effort)

Control4 "scenes" vary by project. In many projects they show up as UI Button devices.

Try:

- `c4_scene_list` (alias of `c4_uibutton_list`)
- `c4_scene_activate(device_id)` (alias of `c4_uibutton_activate`)
- `c4_scene_activate_by_name(scene_name, room_name=...)` (best-effort resolver + activate)

Troubleshooting:

- Use `c4_item_commands(device_id)` to see which command(s) a given scene/button supports.
- Read-only validator: `.\.venv\Scripts\python.exe tools\validate_scenes.py --show-commands`
