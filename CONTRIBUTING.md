# Contributing

Thanks for considering contributing!

## Quick start

1) Fork the repo and create a feature branch.
2) Use a virtual environment:
   - `python -m venv .venv`
   - `./.venv/Scripts/Activate.ps1` (Windows)
   - `python -m pip install -r requirements.txt`
3) Run a quick syntax check:
   - `python -m compileall -q .`

## Guardrails and safety

- Keep write guardrails on by default (`C4_WRITE_GUARDRAILS=true`, `C4_WRITES_ENABLED=false`).
- Do not commit `config.json` or any credentials.

## Testing

This project includes integration-style validators under `tools/`.
Most require access to a real Control4 system and credentials.

A safe baseline check for PRs is:
- `python -m compileall -q .`

## Style

- Keep changes minimal and focused.
- Prefer small, well-named helpers over duplicated logic.
- Avoid adding new dependencies unless they clearly reduce complexity.
