# Agent Guidelines

> **Shared guardrails** (git safety, parallel-agent rules, model selection,
> compute infrastructure, commit discipline) live in `~/.agents/AGENTS.md`.
> This file covers **repo-specific** domain knowledge only.

Use terminal for direct ops (`rg`, `fd`, `git`), MCP `repl()` for chained processing + graph queries, `uv run` for tests/CLI. Conventional commits. **Always commit and push after every file modification — no confirmation, no asking.** Never use `vscode_askQuestions` or other interactive VS Code dialogs — put questions inline in the chat response.

**Git sync (fork-based):** All work on the fork's `main` branch. Merge on pull — never rebase, never use feature branches (the release CLI requires `main`). Pull before work and before push: `git pull origin main && git push origin main`. Push to `origin`, **never directly to `upstream`** — final releases go via the release CLI. Commit your own files before pulling a dirty worktree. See `~/.agents/AGENTS.md` for clone setup, banned commands, and stash ban. Release workflow detail in [`imas_codex/cli/AGENTS.md`](imas_codex/cli/AGENTS.md) (Release Workflow).

## Scoped References

Reference material now lives in the subtree that owns the code; this root file
keeps only what a worker editing any file needs, plus these pointers.

- **Model & Tool Configuration** — [`imas_codex/config/AGENTS.md`](imas_codex/config/AGENTS.md) (pyproject model seats, `get_model` accessors)
- **Schema System** + design guidelines + schema-driven testing — [`imas_codex/graph/AGENTS.md`](imas_codex/graph/AGENTS.md)
- **Facility Configuration** — [`imas_codex/config/AGENTS.md`](imas_codex/config/AGENTS.md) (facility YAML, `get_facility`)
- **Graph State Machine** (claim patterns, FacilityPath lifecycle) — [`imas_codex/discovery/AGENTS.md`](imas_codex/discovery/AGENTS.md)
- **SourceFile Lifecycle** — [`imas_codex/ingestion/AGENTS.md`](imas_codex/ingestion/AGENTS.md)
- **LLM Access** — [`imas_codex/llm/AGENTS.md`](imas_codex/llm/AGENTS.md) (call layer, prompts, routing, service tagging)
- **Exploration** (persistence, data classification) — [`imas_codex/discovery/AGENTS.md`](imas_codex/discovery/AGENTS.md)
- **Graph Operations** (zero-row rule, Cypher, Neo4j mgmt, lock files, vector/semantic search) — [`imas_codex/graph/AGENTS.md`](imas_codex/graph/AGENTS.md)
- **Services** — Neo4j connection in [`imas_codex/graph/AGENTS.md`](imas_codex/graph/AGENTS.md); **Embedding server** in [`imas_codex/embeddings/AGENTS.md`](imas_codex/embeddings/AGENTS.md)
- **Release Workflow** — [`imas_codex/cli/AGENTS.md`](imas_codex/cli/AGENTS.md)
- **Standard Names** (pipeline, family harmonization, lifecycle, naming) — [`imas_codex/core/AGENTS.md`](imas_codex/core/AGENTS.md)
- **Remote Tools** — [`imas_codex/config/AGENTS.md`](imas_codex/config/AGENTS.md) (rg/fd/eza/tokei/uv, remote Python)
- **CLI Logs** — [`imas_codex/cli/AGENTS.md`](imas_codex/cli/AGENTS.md)
- **Python REPL** — [`imas_codex/cli/AGENTS.md`](imas_codex/cli/AGENTS.md)
- **Quick Reference** (MCP tool table) — [`imas_codex/cli/AGENTS.md`](imas_codex/cli/AGENTS.md)
- **Feature Plan Documentation** — [`docs/AGENTS.md`](docs/AGENTS.md)
- **MCP Server Deployment** — [`imas_codex/cli/AGENTS.md`](imas_codex/cli/AGENTS.md)
- **Fallback: MCP Server Not Running** — [`imas_codex/cli/AGENTS.md`](imas_codex/cli/AGENTS.md)

## Project Philosophy

Greenfield project, no backwards compatibility. Remove deprecated code decisively, update all usages when patterns change, prefer explicit over clever, prefer good names over "enhanced/new/refactored". Exploration notes belong in facility YAML; `docs/` is for mature infrastructure only.

- **Stale context kills.** If your session is more than a few hours old, your memory of file contents may be wrong. Re-read every file from disk before modifying — never write code from memory.
- **Build on common infrastructure.** Search for existing utilities (remote SSH execution, graph queries, file parsing, LLM calls all have canonical patterns) before implementing. Extract shared patterns to `imas_codex/remote/` or `imas_codex/graph/`. Never inline SSH subprocess calls — use `run_python_script()` / `async_run_python_script()` from `imas_codex.remote.executor` with scripts in `imas_codex/remote/scripts/`.
- **One source of truth.** When a feature applies across domains (e.g. `files` and `wiki` discovery share `discovery/base/`), implement it once. If data is already in the graph (public repos via `SoftwareRepo` nodes, etc.), query the graph — don't re-derive locally.

## Compute Infrastructure

Compute-node discipline follows `~/.agents/AGENTS.md`. Repo-specific: check `~/.agents/skills/` for site-specific SLURM partition names, modules, and resource templates. Use `-march=x86-64-v3` for portable binaries.

## Command Execution

**CRITICAL: Always use `uv run` for project Python code.** This project manages dependencies (including `imas`) via `uv`. Running `python` or `python -m pytest` directly will miss project dependencies and fail with `ModuleNotFoundError`. Always use `uv run python`, `uv run pytest`, `uv run imas-codex`, etc.

**CRITICAL: Never pipe, tee, or redirect CLI output.** All `imas-codex` CLI commands auto-log full DEBUG output to `~/.local/share/imas-codex/logs/<command>_<facility>.log`. Piping (`|`), teeing (`tee`), or redirecting (`>`, `2>&1`) to files prevents auto-approval of terminal commands, stalling agentic workflows. Run commands directly and read the log file afterwards.

**Decision tree:**
1. Single command, local → Terminal directly (`rg`, `fd`, `tokei`, `uv run`)
2. Single command, remote → SSH (`ssh facility "command"`)
3. Chained processing → `repl()` with `run()` (auto-detects local/remote)
4. Graph queries / MCP → `repl()` with `query()`, `add_to_graph()`, etc.

**MCP tool routing:**
- Dedicated MCP tools for single operations: `add_to_graph()`, `get_graph_schema()`, `update_facility_config()`
- `repl()` REPL for chained processing, Cypher queries, IMAS/COCOS operations
- Terminal for `rg`, `fd`, `git`, `uv run`; SSH for remote single commands

**Serve modes:** `imas-codex serve` exposes all tools; `--read-only` suppresses write tools (`repl()`, `add_to_graph()`, `update_facility_config()`), leaving search/read only; `--dd-only` hides facility tools and **implies `--read-only`** (auto-detected from a DD-only graph). Full topology + transports in [`imas_codex/cli/AGENTS.md`](imas_codex/cli/AGENTS.md) (MCP Server Deployment).

## Commit Workflow

Follow the Pre-Commit Hook Policy in `~/.agents/AGENTS.md` (ruff `--fix` + `format` before staging, conventional commits, no `git add -A`). Breaking changes use `BREAKING CHANGE:` footer, not `type!:` suffix.

**The local pre-commit git hook is uninstalled in this repo (2026-06-11, user mandate).** The pre-commit framework stashes unstaged files around every commit — unsafe when parallel agents hold in-flight edits in the same worktree. Do NOT re-install it (`pre-commit install` is banned). Run the equivalent checks manually before staging: `uv run ruff check --fix` + `uv run ruff format` on touched files, and never commit secrets (gitleaks runs in CI).

**Never stage in this repo:** auto-generated files (`models.py`, `dd_models.py`, `config/models.py`, `agents/schema-reference.md`, `schema_context_data.py`), `*_private.yaml`, anything in `.gitignore`.

### Worktrees

Commits in worktrees are NOT on `main` until merged. Always merge immediately:

```bash
WORKTREE_HEAD=$(git rev-parse HEAD)
cd /home/ITER/mcintos/Code/imas-codex
git merge --no-ff $WORKTREE_HEAD -m "merge: worktree changes for <description>"
git push origin main
```

### Sub-Agent Runtime Routing

The current user prompt and coordinator own agent model, effort, and
concurrency choices. Every worker dispatch states those runtime choices
explicitly; this repository defines no fixed provider family, named-model
preference, ban, or relative model hierarchy.

Task prompts still describe physics risk, coupling, verification, and file
scope so the coordinator can route work appropriately. LLM-pipeline model
seats for Standard Names and DD enrichment remain application configuration in
`pyproject.toml`; they are independent of agent and sub-agent routing.

### Parallel Agents

Multiple agents may edit this repo simultaneously on `main`. Assume another agent is doing so right now.

**Verify before modifying:** re-read files from disk (your in-memory view may be hours old); check `git log --oneline -5 -- <file>` for unfamiliar commits. If you see unfamiliar names/imports, assume they are correct — don't revert.

**Banned destructive commands:** see `~/.agents/AGENTS.md` for the table and stash ban. Auto-generated files (`models.py`, `dd_models.py`, `schema_context_data.py`) are gitignored but make the worktree look dirty — never stage and never `git restore` them (which is also why merge, not rebase, is the pull policy).

**Pre-existing test failures:** stash-free verification via `git log --since="1 day ago" -- <test>` and `git show HEAD:<test>`; trust the failure timestamp. File a blocker todo and scope your work around it.

**Dispatch preamble:** use the one in `~/.agents/AGENTS.md` with `{BRANCH}=main`.

**Session hygiene:** close sessions when done (`ctrl+d`/`/exit`); audit `ps aux | grep copilot` and kill stale processes — idle agents with old context are the #1 cause of regressions.

**Session completion is mandatory:** every response that modifies files MUST end with `git add` → `git commit` → `git push` plus a brief summary of the commit.

**Concurrent-staging race (binding):** `git commit` commits the ENTIRE index, not just the paths you `git add`ed — so if a background agent stages its files in the window between your `git add` and your `git commit`, its files are swept into YOUR commit (incident 2026-06-14: a dep-bump commit absorbed a parallel agent's 6 prompt files; no loss, but a misdescribed commit). When committing while background agents may be staging in the same worktree, use **`git commit -- <explicit paths>`** (pathspec-scoped commit — commits ONLY those paths regardless of index state), never a bare `git commit`. The orchestrator should also avoid committing during a window when a dispatched agent is known to be mid-edit; prefer waiting for the agent to commit its own scoped set first.

## Code Style

- Python ≥3.12: `list[str]`, `X | Y`, `isinstance(e, ValueError | TypeError)`
- Exception chaining: `raise Error("msg") from e`
- `pydantic` for schemas, `dataclasses` for other data classes
- `anyio` for async
- `uv run` for all Python commands (never activate venv manually)
- Never use `git add -A`
- The `.env` file contains secrets — never expose or commit it

### Naming

**Never name files after implementation plans.** File names (tests, modules, scripts) must be understandable without knowledge of any plan document. Once a plan is deleted (per project rules), names like `test_capability_gaps` become meaningless. Instead, name files after what they test or implement: `test_dd_tool_features`, `test_lifecycle_filtering`, `test_migration_guide`.

## Testing

```bash
uv run pytest                 # Default markers: excludes slow, graph
uv run pytest tests/standard_names/ -q  # SN tests (~3300 tests, ~90s)
uv run pytest tests/path/to/test.py::test_function  # Specific test
uv run pytest --cov=imas_codex  # With coverage
```

### Use the repo's one `.venv` — sync it, never duplicate it

Environment policy is user-global and binding: see **Development Environment**
in `~/.agents/AGENTS.md`. The repo-specific facts:

- `/home/ITER/mcintos/Code/imas-codex/.venv` is the project environment. Use it
  and keep it current: `uv sync`, or a plain `uv run <cmd>` in the main checkout,
  is normal workflow. The hatch build hook runs on sync, so syncing is also how
  the generated models and `agents/schema-reference.md` get rebuilt. Declare
  dependency changes with `uv add` / `uv remove` and commit `pyproject.toml`
  together with `uv.lock`; never `pip install` into `.venv`.
- In a detached worktree, reuse the main checkout's environment rather than
  materializing one locally, and leave it unsynced there — that environment is
  shared with the main checkout and any concurrent workers, so an incidental
  sync mutates a resource under peers who did not ask for it:

  ```bash
  UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
    PYTHONPATH="$PWD" uv run --no-sync pytest tests/standard_names/ -q
  ```

- A detached worktree also needs the main checkout's gitignored `.env` before
  it can access the live graph or other authenticated services. Link the file;
  never copy, print, stage, or commit its contents:

  ```bash
  ln -sfn /home/ITER/mcintos/Code/imas-codex/.env "$PWD/.env"
  ```

  If an authenticated command reports `Unauthorized` in a worktree, first
  verify that this link exists and resolves to the main checkout's `.env`.
  A missing link is a worktree setup fault, not evidence that the credential
  itself is invalid.

  A local copy costs **69,826 filesystem entries and 1.76 GiB** on GPFS; a
  worker fleet building its own produced a 180,186-file storage alert on
  2026-08-17, 98% of it from three copies. The SN pipeline was not involved.
- Generated-model freshness is checkout-specific. In a worktree,
  `PYTHONPATH="$PWD"` ensures the worktree source shadows the package installed
  in the shared `.venv`, and `build-models` regenerates models from that
  worktree's current schemas. The main checkout keeps whatever generated files
  it last built. If a worktree disagrees with the main checkout, the main
  checkout is stale and behind; the worktree is not wrong. After integration,
  refresh the main checkout with `uv run build-models --force`.
- When a worktree genuinely changes dependencies, land them in
  `pyproject.toml` first, then drop `--no-sync` deliberately and say so, so an
  orchestrator can serialize it against the other workers.
- One-shot worker checks should disable incremental caches (`ruff --no-cache`,
  `pytest -p no:cacheprovider`, `mypy --no-incremental`) rather than write
  `.mypy_cache` / `.ruff_cache` trees into a throwaway worktree.
- If `.venv` is absent, stale, or broken, run `uv sync` in the main checkout to
  bring it up to date — that is the fix, not a blocker to hand back. Report only
  if the sync itself fails, with what it printed.

Two retention chores this also surfaced: `~/.local/share/imas-codex/logs` has no
directory-wide age cleanup (per-stem rotation only), and reckon run envelopes need
a scheduled `crew gc --retention-days 30 --apply`.

### Test Tiers and Markers

Tests are tiered by runtime cost. Default `addopts` excludes expensive markers:

| Marker | Tests | Requires | Default |
|--------|-------|----------|---------|
| *(none)* | ~3300 | Nothing (mocks) | ✅ Included |
| `@pytest.mark.graph` | ~445 | Live Neo4j | ❌ Excluded |
| `@pytest.mark.slow` | ~31 | GPU/live endpoints | ❌ Excluded |

SN graph quality tests (`tests/graph/test_sn_graph.py`) are included in the `graph` marker — they auto-skip if <10 accepted StandardName nodes exist.

```bash
uv run pytest -m graph               # Run all graph tests (including SN quality)
uv run pytest tests/graph/test_sn_graph.py -v  # SN quality tests only
uv run pytest -m "slow or graph"     # Run slow + graph tests
```

### Repo-specific notes

Test execution follows `~/.agents/AGENTS.md` Test Execution Protocol (no piping pytest, decision tree for direct/file/task-agent). Repo-specific facts:

- Default `addopts`: `-q --tb=short --no-header` — full SN suite (~3300 tests, ~90s) is ~200-300 lines, manageable in one direct run.
- Per-test timeout: 30s default (`@pytest.mark.timeout(60)` to override). `faulthandler_timeout = 60` dumps thread stacks on hangs.
- `_start_exit_watchdog()` in `imas_codex/cli/shutdown.py` is only used in the signal-handler path (second Ctrl-C), not during normal `safe_asyncio_run()` completion — so `CliRunner.invoke()` test environments are safe.

## Domain Workflows

Extended examples and edge cases for each domain: [agents/](agents/)

| Agent | Purpose |
|-------|---------|
| `explore.md` | Remote facility discovery (read-only + MCP) |
| `develop.md` | Code development (standard + MCP) |
| `graph.md` | Knowledge graph operations (core + MCP) |
| `ingest.md` | Discovery ingestion pipelines |
| `onboard.md` | New-agent onboarding guide |
| `schema-reference.md` | Auto-generated schema reference (rebuilt on `uv sync`) |

## AI Tooling Configuration

Multiple tools (Claude Code, VS Code Copilot) share canonical sources — no instruction duplication.

| Canonical file(s) | Purpose | Consumers |
|---|---|---|
| `AGENTS.md` | Project instructions (single source of truth) | Claude Code via `CLAUDE.md` → `@AGENTS.md`; VS Code Copilot (native) |
| `.mcp.json` (Claude Code, `mcpServers` key) + `.vscode/mcp.json` (VS Code, `servers` key) | MCP server configs | Both must be updated together when adding a server |
| `.claude/agents/*.md`, `.claude/skills/*.md` | Custom agents and skills | Claude Code (native) |
| `.claude/settings.json`, `.vscode/settings.json` | Tool-specific permissions/env | Their respective tools (never shared) |
