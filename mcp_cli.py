"""Thin wrapper to run flask-mcp-server's Typer CLI via python.

Why this exists:
- The installed console script (flask-mcp.exe) runs with sys.path rooted at the
  venv Scripts directory, so importing repo-local modules (like app.py) can fail.
- Claude Desktop prefers launching MCP servers as subprocesses; pointing it at
  python.exe + this wrapper keeps the repo root on sys.path.

Examples:
  ./.venv/Scripts/python.exe mcp_cli.py list --module app
  ./.venv/Scripts/python.exe mcp_cli.py serve-stdio --module app
"""

from __future__ import annotations

from flask_mcp_server.cli import app


if __name__ == "__main__":
    app()
