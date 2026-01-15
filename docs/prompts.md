Below are concise, battle‑tested prompts you can drop straight into a chat. They cover:

*   **project\overview\.md** (single-source-of-truth)
*   **Modular /docs** (architecture, modules, API, roadmap, glossary)
*   **Bootstrap Summary** (for instant context reload in new chats)
*   **Conventions & Guardrails** (so the AI codes the way you want)
*   **Diff‑based updates** (so your overview stays current, safely)
*   **Context Pack** (compressed + structured)
*   **New Chat Boot Prompt** (so any fresh chat is ready in seconds)
*   **Bonus prompts** for file trees, PR summaries, risks, and migration notes

Use as-is; replace bracketed placeholders like `[PROJECT NAME]`.

***

## 0) Quick Start — What to do first

1.  Create `/project_overview.md` in your repo (empty is fine for now).
2.  Create `/docs/` folder (empty).
3.  Open a new chat and paste **Prompt A**, then **Prompt B**, then **Prompt C**.

***

## Prompt A — Create `project_overview.md` (initial build)

> **Goal:** Create a single, accurate source-of-truth.

```text
You are my technical editor. We’re initializing a new “project brain” file for a codebase.

Project name: [PROJECT NAME]  
Primary goal: [GOAL IN 1–2 SENTENCES]  
Tech stack: [LANGUAGES/FRAMEWORKS/TOOLS]  
Deployment: [CLOUD/ON-PREM/CONTAINERS/CI-CD]  
Constraints: [PERF, SECURITY, COMPLIANCE, COST, DATA RESIDENCY]  
Users & personas: [WHO USES IT, HOW]  

Create a complete `project_overview.md` with:
1) One-paragraph executive summary  
2) Architecture (logical + runtime): components, data flow, external deps  
3) Directory structure (proposed initial tree) with brief purpose per folder/file  
4) Module inventory: name, purpose, inputs/outputs, boundaries, upstream/downstream  
5) Data & schemas (key tables, events, contracts, versioning strategy)  
6) API surface (internal + external): endpoints, payload shapes, auth  
7) Decision log (ADR-style bullets with rationale)  
8) Non-functional requirements: performance, security, scalability, reliability, observability  
9) Testing strategy (unit/integration/e2e) and sample coverage goals  
10) Operational runbook: environments, secrets, config, alerts, SLIs/SLOs  
11) Coding conventions (style, folder naming, error handling, logging)  
12) Current risks/unknowns and assumptions  
13) Short-term roadmap (0–2 weeks), mid-term (2–8 weeks)  
14) Glossary of important terms and abbreviations

Output valid Markdown, concise but complete. Prefer lists and tables where helpful. Use headings and anchors.
```

***

## Prompt B — Initialize `/docs` structure (modular, scalable)

> **Goal:** Smaller docs for deep dives, linked from the overview.

```text
Generate a `/docs` set with the following files and content. Write each file’s full Markdown content inline, headed by a clear filename header like: 
`=== docs/architecture.md ===`

Files:
- `docs/architecture.md`: context diagram, component diagram, data flow diagram (described textually), deployment diagram, tradeoff notes
- `docs/modules/README.md`: how to add/edit module specs
- `docs/modules/[MODULE].md` for each proposed module: purpose, inputs/outputs, interfaces, data, failure modes, test points, telemetry, KPIs
- `docs/api/endpoints.md`: all internal/external endpoints with example requests/responses, error codes, auth flows
- `docs/data/contracts.md`: schemas (tables, events), compatibility/versioning policy
- `docs/ops/runbook.md`: envs, deploy steps, rollback, observability, dashboards, alerts, on-call playbook
- `docs/roadmap.md`: milestones, epics, target dates (approx), risks, mitigations
- `docs/glossary.md`: terms and definitions

Anchor links back to `project_overview.md`. Keep each file pragmatic and implementation-ready.
```

***

## Prompt C — Create the “Bootstrap Summary” (for new chats)

> **Goal:** A compact, re‑importable brief you can paste to reload context.

```text
Create a compact “PROJECT BOOTSTRAP SUMMARY” focused on fast context reloads in new chats. Include:

1) One-line purpose  
2) Architecture overview (3–6 bullets)  
3) Key modules and roles (bullet list)  
4) Data & contracts (top 3–5 only)  
5) APIs (key endpoints only)  
6) Coding conventions (only the rules the AI must always follow)  
7) Current priorities (Top 5)  
8) Open risks/unknowns (Top 5)  
9) Links/paths to the full docs (`project_overview.md`, `/docs/...`)

Keep it under ~400–700 words, highly scannable, with bolded section headers.
```

***

## Prompt D — Conventions & Guardrails (so AI builds consistently)

> **Goal:** Make the AI follow your standards without repeating them constantly.

```text
From our project overview, extract a concise “Conventions & Guardrails” block I can paste into future coding prompts. Include:

- Language/framework versions and style guides
- Directory/file naming conventions
- Error handling, logging, and observability (levels, structure, IDs)
- Testing requirements and structure
- Security and privacy must-haves
- Performance targets and memory/latency constraints
- Commit message format / PR template bullets
- Documentation expectations for new modules/APIs

Write as a short, enforceable checklist. Max ~200–300 words. Make it copy-pastable.
```

***

## Prompt E — Diff‑Based Update of `project_overview.md` (use after sessions)

> **Goal:** Keep the “project brain” file accurate without overwriting.

```text
We made changes in this session (code/architecture/decisions). Generate:
1) A concise bullet list of updates  
2) A unified diff (old → new) for `docs\project_overview.md`  
3) Update the `docs\project_overview.md`

Rules:
- Only change sections that truly need updates
- Preserve existing structure and anchors
- Be explicit and minimal in the diff
```

***

## Prompt F — Generate a “Context Pack” (compressed + structured)

> **Goal:** A single paste that primes a new chat with the *right* amount of detail.

```text
Create a “Context Pack” for re-importing this project into a new chat. Include:

- Mini executive summary (≤ 120 words)
- Critical architecture bullets (≤ 6)
- Current working set (the 3–7 files/modules we’re actively changing) with short purpose
- Interfaces/contracts that must not break
- Today’s objectives and acceptance criteria
- Guardrails block (from conventions)
- Links/paths for deeper docs
- A “Next Prompt to Paste” (suggested instruction to kick off the next task)

Keep it ≤ 900 words and extremely actionable.
Update the 'docs/context_pack.md' with the new Context Pack
```

***

## Prompt G — New Chat Boot Prompt (paste this to start any new chat)

> **Goal:** Instantly load context from your external files.

```text
You are joining an ongoing software project. Load context strictly from the following pasted summaries and references.

[Paste the latest BOOTSTRAP SUMMARY]
[Paste the latest CONTEXT PACK]

Your tasks:
1) Acknowledge understanding of architecture and constraints
2) Ask only the 1–2 highest leverage clarifying questions
3) Begin executing on “Today’s objectives” using the Guardrails
4) Before making changes that could break contracts, propose a minimal plan
5) Proceed one step at a time, testing after each step

Do NOT re-architect unless asked. Be concise and code-first.
```

***

## Prompt H — File Tree & Intent (great early on or after refactors)

> **Goal:** Create/verify the directory structure before coding.

```text
Propose a `tree`-style directory structure for the codebase consistent with the docs. For each folder/file, add a one-line purpose. Then output:

1) The `tree` with comments  
2) A list of files to implement first (order matters), with rationale  
3) For each first-implementation file: key functions/classes, signatures, and TODOs
```

***

## Prompt I — PR Summary & Release Notes (keep stakeholders aligned)

```text
Based on today’s changes, produce:
- A PR description with “What/Why/How/Validation/Risks/Rollback”
- Release notes (user-facing) with breaking-change callouts
- Changelog entry following Keep a Changelog format (Unreleased → Added/Changed/Fixed/Removed/Deprecated/Security)
```

***

## Prompt J — Risk Register & Mitigations (decision-quality clarity)

```text
Create/refresh a risk register:
- Risk ID, description, likelihood (L/M/H), impact (L/M/H), owner, mitigation, trigger, contingency
- Focus on architecture fragility, data contract drift, security exposures, performance regressions, operational gaps
Output as a compact Markdown table and 3–5 key watch items.
```

***

## Prompt K — Migration / Breaking Changes Plan (when evolving)

```text
We need to make a breaking change: [describe]. Produce a migration plan including:
- Semver impact and version plan
- Strangler/feature-flag rollout plan
- Back-compat shims and deprecation timeline
- Data migration steps and verification
- Telemetry to confirm safe rollout and rollback criteria
```

***

## Prompt L — Bug Intake & Triage (keeps the flow productive)

```text
Transform these raw bug notes into a triage board:
- Normalize titles, add repro steps, expected vs actual, environment
- Tag severity/priority and owner
- Suggest root-cause hypothesis and next diagnostic steps
Output as a Markdown table plus top 3 quick wins.
```

***

# Tips for Smooth Usage

*   **Pin your Bootstrap Summary** in your repo and in your notes app—keep it \~weekly fresh.
*   After any substantial session, run **Prompt E** to update `project_overview.md`.
*   Keep **docs modular** and small; it makes context loads faster and more reliable.
*   Use the **Context Pack** for day-to-day continuity and the **Boot Prompt** for restarting chats.
*   For really large codebases, paste only the **active modules** into the Context Pack to avoid overload.

***

If you want, tell me your **typical stack and project types** (e.g., Python data pipelines, TypeScript/React web apps, microservices on AWS, embedded, etc.), and I’ll tailor all templates (directory trees, testing patterns, CI/CD bits, logging format, and security guardrails) to match your world.
