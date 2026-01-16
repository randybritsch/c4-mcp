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

## Discover IDs

Device and room ids are specific to your Control4 project. Use:

- `c4_list_rooms`
- `c4_list_devices` (by category)

Then pass the discovered ids into tools like `c4_media_watch_launch_app` or the scripts in `tools/`.
