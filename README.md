# c4-mcp

Control4 MCP Server (Flask) exposing Control4 automation as MCP tools.

## Setup

This project is intended to work with any Control4 system. Nothing in the server is hard-coded to a specific home.

1) Install dependencies (in a venv)

2) Provide Control4 connection config via either:

- Environment variables:
	- `C4_HOST` (Director/Controller IP or hostname; scheme optional)
	- `C4_USERNAME` (Control4 account email)
	- `C4_PASSWORD` (Control4 account password)

or

- A local config file (not committed): copy `config.example.json` to `config.json` and fill in values.

### First-time setup (recommended)

If youre setting this up for the first time and dont know your controller IP yet, this is the quickest flow (PowerShell):

1) Set credentials for discovery + server login:

- `$env:C4_USERNAME = "you@example.com"`
- `$env:C4_PASSWORD = "your-password"`

2) Auto-discover the controller IP and write it to `config.json`:

- `\.venv\Scripts\python.exe tools\discover_controller.py --write`

3) Start the MCP server:

- `\.venv\Scripts\python.exe app.py`

4) Verify its up:

- `GET http://127.0.0.1:3333/mcp/list`

### Optional: auto-discover controller IP (Windows-friendly)

If you don't know the controller IP yet, you can run a best-effort LAN discovery scan and optionally write the discovered IP into `config.json`:

- Dry run (no writes): `\.venv\\Scripts\\python.exe tools\\discover_controller.py`
- Write `host` to `config.json`: `\.venv\\Scripts\\python.exe tools\\discover_controller.py --write`

Notes:
- Scans only the local subnets detected from `ipconfig` (or use `--subnet 192.168.1.0/24`).
- Uses bounded per-host timeouts + concurrency limits.
- If `config.json` does not exist, you can set `C4_USERNAME` and `C4_PASSWORD` so the script can create it.

## Run

- Start server: `\.venv\\Scripts\\python.exe app.py`
- Verify MCP is up: `GET http://127.0.0.1:3333/mcp/list`

### Optional: write guardrails (recommended for safety)

By default, tools that change state (locks, lights, thermostat, media remote, etc.) are allowed.
If you want a **read-only** server unless explicitly enabled, set:

- `C4_WRITE_GUARDRAILS=true` (turns on enforcement)
- `C4_WRITES_ENABLED=true` (allows write tools)

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
- Read-only validator: `\.venv\Scripts\python.exe tools\validate_scenes.py --show-commands`
