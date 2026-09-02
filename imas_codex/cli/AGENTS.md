This file governs the `imas_codex/cli/` subtree — the `imas-codex` CLI commands, the
software release workflow, the MCP serve interface, and the REPL/search tooling it serves.



### Release Workflow

**Standard Names catalog releases are a separate workflow.** The sole
operational recipe—including the live-help flag contract, additive baseline,
reviewer/preview surfaces, fail-closed approval routes, contested resolution,
receipts, and undo—is the scoped
[`standard_names/AGENTS.md`](imas_codex/standard_names/AGENTS.md#release-recipe).
Do not duplicate that runbook here or substitute the software-release commands
below for the catalog flow.

The release CLI is state-machine driven from the latest git tag. **Stable** = `vX.Y.Z`, **RC** = `vX.Y.Z-rcN`. RCs target `origin` (fork); finals target `upstream` (iterorganization) — override with `--remote`. RC releases tolerate dirty worktrees; `--final` requires clean.

```bash
uv run imas-codex release status                     # current state + permitted bumps

# From stable: bump major/minor/patch → vN.Y.Z-rc1 (origin); add --final for direct upstream
uv run imas-codex release --bump minor -m '<msg>'
# From RC mode: re-run iterates rc → rc2/rc3...; --final cuts the stable to upstream
uv run imas-codex release -m '<msg>'
uv run imas-codex release --final -m '<msg>'
# Options: --remote, --skip-git, --dry-run, --version
```

**The CLI does it all:** computes the next version, validates no private fields in the graph, tags DDVersion, pushes graph variants to GHCR (dd-only + full for RC; + per-facility for final), pushes the git tag → triggers CI. CI runs `graph-quality`, `smoke-test`, `build-and-push` to ACR. Azure Web App continuously deploys from ACR (5–15 min lag).

**Workflow:** RC on fork → verify on Azure test (`https://app-imas-mcp-server-test-frc.azurewebsites.net/health`) → PR fork/main → upstream/main → `release --final` from upstream. **Never** push the same tag to both remotes (causes ACR race conditions). RC tags on fork are disposable.

## CLI Logs

All discovery and DD CLI commands write DEBUG-level rotating logs to disk. The rich progress display suppresses most log output to keep the TUI clean, but full details are always available in the log files.

**Log directory:** `~/.local/share/imas-codex/logs/`

**Log naming:** `{command}_{facility}.log` (e.g. `paths_tcv.log`, `wiki_jet.log`, `imas_dd.log`). Logs rotate at 10 MB with 3 backups.

```bash
tail -f ~/.local/share/imas-codex/logs/paths_tcv.log  # Follow live
rg "ERROR|WARNING" ~/.local/share/imas-codex/logs/     # Find errors
```

(The no-pipe rule from [Command Execution](../AGENTS.md#command-execution) applies here too — logs are already on disk; never redirect CLI output.)

## Python REPL

`repl()` is a persistent MCP REPL for custom queries not covered by the search tools. **Prefer `search_signals`/`search_docs`/`search_code`/`search_dd_paths` first** — they handle embeddings, multi-index fan-out, enrichment, and formatting in one call.

Use `repl()` for: signal→IMAS mapping, facility overviews, flexible `graph_search()`, raw Cypher, or chaining domain functions. Chain operations in a single call — each call has overhead. Before raw Cypher, call `schema_for(task='wiki')` to get node labels/properties/relationships/enums from the LinkML schemas (`get_schema()` for the full object; `repl_help()` for the API reference). **Never guess property names.** Format structured results with `as_table(pick(results, 'col1', 'col2'))`.

## Quick Reference

**Primary MCP tools** — use these first, they return formatted reports:

| Task | MCP Tool |
|------|----------|
| Signal lookup | `search_signals("plasma current", facility="tcv")` |
| Documentation | `search_docs("fishbone instabilities", facility="jet")` |
| Code examples | `search_code("equilibrium reconstruction", facility="tcv")` |
| IMAS DD paths | `search_dd_paths("electron temperature", facility="tcv")` — results include semantic cluster labels and "See Also" cross-IDS siblings for top hits |
| Full content | `fetch_content("jet:Fishbone_proposal_2018.ppt")` — use IDs/URLs from search results |

**repl() REPL** — for custom queries not covered by the search tools:

| Task | Command |
|------|---------|
| Wiki keyword | `repl("print(find_wiki(text_contains='fishbone'))")` |
| Page chunks | `repl("print(wiki_page_chunks('equilibrium', facility='tcv'))")` |
| Signal→IMAS map | `repl("print(map_signals_to_imas(facility='tcv', physics_domain='magnetics'))")` |
| Graph search | `repl("print(graph_search('WikiChunk', where={'text__contains': 'IMAS'}))")` |
| Format table | `repl("print(as_table(find_signals('ip', facility='tcv')))")` |
| Facility info | `repl("print(get_facility('tcv'))")` |
| Raw Cypher | `repl("print(query('MATCH (n) RETURN n.id LIMIT 5'))")` |
| Add to graph | `add_to_graph('SourceFile', [...])` |
| Remote command | `ssh facility "rg pattern /path"` |

Chain multiple operations in a single `repl()` call to minimize round-trips.

## MCP Server Deployment

Add `--transport streamable-http` (containers/HTTP) or `--transport stdio`
(VS Code, Claude Desktop) to any mode below.

| Deployment | Command | Tools available |
|------------|---------|-----------------|
| Development | `imas-codex serve` | All (REPL, search, write, infrastructure) |
| Public / read-only | `imas-codex serve --read-only` | Search and read only |
| DD-only container | `imas-codex serve --dd-only` | DD search and read only (implies read-only) |

## Fallback: MCP Server Not Running

```bash
uv run imas-codex graph status          # Graph operations
uv run imas-codex graph shell           # Interactive Cypher
uv run imas-codex llm status            # LLM proxy status (lightweight, no API calls)
uv run imas-codex llm status --deep     # Full model health check (makes real LLM API calls — billable)
uv run pytest                           # Testing
```

Automated health checks (e.g., Azure `/health/readiness`) make no LLM API calls and incur no token cost. Only `imas-codex llm status --deep` exercises the model endpoint and is billable.
