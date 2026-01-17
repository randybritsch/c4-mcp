# Security Policy

## Reporting a vulnerability

If you believe you’ve found a security vulnerability, please do **not** open a public GitHub issue.

Instead, email the maintainer with:
- A clear description of the issue and potential impact
- Reproduction steps (ideally minimal)
- Any suggested fixes or mitigations

## Secrets & credentials

This project requires Control4 credentials and typically a controller/Director host.

- Never commit credentials to git.
- Keep `config.json` local-only (it is ignored via `.gitignore`).
- Prefer environment variables for secrets (`C4_USERNAME`, `C4_PASSWORD`).
- If credentials were ever committed, rotate them immediately and rewrite git history before making the repo public.

## Write safety

By default, you should run with guardrails enabled:
- `C4_WRITE_GUARDRAILS=true`
- `C4_WRITES_ENABLED=false` (read-only)

Only enable writes when you’re ready to test automation:
- `C4_WRITES_ENABLED=true`

You can further restrict write operations with allow/deny lists.
