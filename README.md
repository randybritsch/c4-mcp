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

## Run

- Start server: `python app.py`
- Verify MCP is up: `GET http://127.0.0.1:3333/mcp/list`

## Discover IDs

Device and room ids are specific to your Control4 project. Use:

- `c4_list_rooms`
- `c4_list_devices` (by category)

Then pass the discovered ids into tools like `c4_media_watch_launch_app` or the scripts in `tools/`.
