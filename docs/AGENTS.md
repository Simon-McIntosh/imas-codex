This file governs the `docs/` subtree — the feature-plan documentation lifecycle under
`plans/`.



## Feature Plan Documentation

Plans live in `plans/features/`. Lifecycle: `features/<name>.md` (active) → `features/pending/<name>.md` (partially implemented, gaps documented) → **DELETE** (fully implemented — the code is the documentation). Gap docs (`gaps-*.md`) consolidate remaining work across related pending plans.

**Every plan must have a "Documentation Updates" section** listing which targets need updates: `AGENTS.md` (new CLI/MCP/config/workflows), `README.md` (user-facing), `plans/README.md` (status), `.claude/skills/*.md`, `.claude/agents/*.md`, `docs/` (mature architecture), prompt templates, schema reference (auto via `uv run build-models`).

**Self-consistency rule:** a feature is not done until code is committed + tested, every applicable doc target is updated, `plans/README.md` reflects the new status, and the plan file is deleted or moved to `pending/`.
