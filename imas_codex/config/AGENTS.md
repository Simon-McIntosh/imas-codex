This file governs the `imas_codex/config/` subtree — the model and tool configuration in
`pyproject.toml` read through `imas_codex.settings`, the per-facility YAML configs, and the
remote-tools table.



## Model & Tool Configuration

All model and tool settings live in `pyproject.toml` under `[tool.imas-codex]`. No backward-compatible aliases — use the canonical accessors from `imas_codex.settings`.

**Sections** (each with a `model` parameter where relevant):

| Section | Purpose | Accessor |
|---------|---------|----------|
| `[graph]` | Neo4j connection, active symlink identity, server location | `get_graph_profile()`, `get_graph_name()`, `get_graph_location()`, `get_graph_uri()` |
| `[hosts]` | Host/location metadata, login nodes, local hosts | `is_local_host()` |
| `[logs]` | CLI log directory and rotation | — |
| `[embedding]` | Embedding model, dimension, location, scheduler | `get_model("embedding")`, `get_embedding_location()` |
| `[language]` | Structured output (scoring, discovery, labeling), batch-size | `get_model("language")` |
| `[vision]` | Image/document tasks | `get_model("vision")` |
| `[reasoning]` | Complex structured output (IMAS mapping, multi-step reasoning) | `get_model("reasoning")` |
| `[discovery]` | Discovery threshold for high-value processing | `get_discovery_threshold()` |
| `[data-dictionary]` | DD version, include-ggd, include-error-fields | `get_dd_version()` |
| `[dd-enrichment]` | DD enrichment worker concurrency and batching | `get_model("dd-enrichment")` |
| `[llm]` | LLM proxy URL, timeouts, retry policy | — |
| `[sn]` | Standard names paths (staging-dir, isnc-dir), retry knobs | — |
| `[sn-compose]` | SN name composition model, batch sizes, max-concurrency | `get_model("sn-compose")` |
| `[sn-docs]` | SN documentation generation model | `get_model("sn-docs")` |
| `[sn-refine]` | SN refine_name + refine_docs tier | `get_model("sn-refine")` |
| `[sn-escalation]` | Final refine attempt at chain cap (name + docs); vendor-diverse from refine/compose | `get_model("sn-escalation")` |
| `[sn-parent-enrich]` | Derived-parent description synthesis (enrich_parents pool) | `get_model("sn-parent-enrich")` |
| `[sn-classifier]` | Physics-domain classifier (DD paths → domain; SN names inherit it) | `get_model("sn-classifier")` |
| `[sn-prose-adjudicator]` | Banned-prose grep-flag adjudication at the campaign convergence gate | `get_model("sn-prose-adjudicator")` |
| `[sn-fanout]` | Structured fan-out (Proposer/Executor/Synthesizer) | — |
| `[sn-review]` | Shared RD-quorum settings (disagreement threshold, max cycles, active profile) | `get_sn_review_disagreement_threshold()`, `get_sn_review_max_cycles()`, `get_sn_review_active_profile()` |
| `[sn-review.names]` / `[sn-review.docs]` | Reviewer model chain per axis (1–3 models) | `get_sn_review_names_models()`, `get_sn_review_docs_models()` |
| `[sn-review.names.profiles.*]` | Named profiles (default, opus-only, quality-cost-balanced) | — |
| `[sn-benchmark]` | SN benchmark compose-models, candidate-models (seat evaluation slate), reviewer-model (judge) | `get_sn_benchmark_compose_models()`, `get_sn_benchmark_candidate_models()`, `get_sn_benchmark_reviewer_model()` |

**Model access:** `get_model(section)` is the single entry point for all model lookups. Pass the pyproject.toml section name directly: `"language"`, `"vision"`, `"reasoning"`, or `"embedding"`. Priority: section env var → pyproject.toml config → default.

**Canonical model names live in `pyproject.toml` — NEVER hardcode model IDs in source.** Both the pipeline seat models and the benchmark model lists are managed there and read through the `imas_codex.settings` accessors above:

- **Pipeline seats:** each seat's production model is its own `[sn-*]` section — `[sn-compose]`, `[sn-docs]`, `[sn-refine]`, `[sn-escalation]`, `[sn-parent-enrich]`, `[sn-classifier]`, `[sn-prose-adjudicator]` (`get_model("sn-<seat>")`), and the reviewer chains in `[sn-review.names]` / `[sn-review.docs]` (+ named profiles under `[sn-review.names.profiles.*]`), read via `get_sn_review_names_models()` / `get_sn_review_docs_models()`. **Every model the SN pipeline drives must live under an `sn-*` seat, never a generic section (`[language]`, `[reasoning]`, …), so its config and model choice are attributable to the SN project** — when you find an SN-path call borrowing a generic seat, give it a dedicated `sn-*` section. **Never introduce a new model choice as a module-level constant in feature code** (e.g. `defaults.py`, a worker, an adapter): add an `sn-*` seat, register it in `settings.MODEL_SECTIONS` / `_MODEL_DEFAULTS` / `_MODEL_ENV_VARS`, and read it with `get_model()`. The only code-level model strings permitted are the centralized fallbacks in `settings._MODEL_DEFAULTS` (and the equivalent Pydantic-settings field defaults, e.g. `[sn-fanout]`).
- **No results in config/source comments:** a seat comment states WHAT the seat is and WHY it is vendor/tier-shaped in mechanism terms — never the benchmark scores that motivated the choice. Bench numbers (AUC, defect-resolution, collateral, cost/call) live in the run's result files (`~/.local/share/imas-codex/benchmarks/*.json`) and the results doc, not baked into `pyproject.toml` or docstrings where they rot. Reference the results doc if provenance is needed.
- **Benchmarks:** the compose bake-off slate is `[sn-benchmark].compose-models`; the cross-seat evaluation slate (`sn bench --role <seat>`) is `[sn-benchmark].candidate-models`. A `--role` run measures the candidate slate against that seat's **live production model as the incumbent** — derived from the seat's own `[sn-*]` config at runtime, not written into the CLI. To change what a seat bench tests, edit `candidate-models`; to change what a seat runs in production, edit that seat's section.
- **Keeping families current:** review the model families periodically and keep at least the latest ~2 iterations of each family in use (e.g. `gpt-5.5` **and** `gpt-5.6`, `claude-sonnet-4.6` → `claude-sonnet-5`, `opus-4.7` → `opus-4.8`). A model can be excluded on **provider capacity** rather than version — e.g. `qwen3.7-max` / `minimax-m3` are omitted from every seat because their OpenRouter providers upstream-throttle (429) under the review pools' concurrency; vet a new reviewer with `sn bench --role concurrency` before seating it.
- **Effort is a benchmark axis:** `sn bench --role <seat> --efforts minimal,low,medium,high` sweeps reasoning-effort (one row per model×effort) — for judgment seats (review/refine) lower effort can beat high because models overthink, so measure it rather than defaulting to `high`.

**Graph access:** The active `~/.local/share/imas-codex/neo4j` symlink selects **what data** Neo4j serves; change it with `imas-codex graph switch NAME`. The independent `IMAS_CODEX_GRAPH_LOCATION` override selects **where Neo4j runs** and therefore its host and Bolt/HTTP port slot (iter 7687/7474, tcv 7688/7475, jt-60sa 7689/7476). `IMAS_CODEX_GRAPH` does not select data. `NEO4J_URI`/`NEO4J_USERNAME`/`NEO4J_PASSWORD` are connection escape hatches; `resolve_neo4j()` (`imas_codex.graph.profiles`) combines the active symlink identity with the resolved location. Graph subcommands expose their own options; there is no universal `--graph/-g` selector. Full detail: `docs/architecture/graph-profiles.md`.

**Location-aware connections:** `is_local_host(host)` picks direct vs tunnel at connect time; for edge cases configure `login_nodes`/`local_hosts` in the facility's private YAML (`imas-codex config local-hosts`).

## Facility Configuration

Per-facility YAML configs define discovery roots, wiki sites, data sources, and infrastructure details. Schema enforced via LinkML (`imas_codex/schemas/facility_config.yaml`).

**Files:**
- `imas_codex/config/facilities/<facility>.yaml` - Public config (git-tracked)
- `imas_codex/config/facilities/<facility>_private.yaml` - Private config (gitignored)

**CRITICAL: All facility-specific configuration MUST live in YAML files.** Never hardcode facility names, tree names, version numbers, setup commands, system descriptions, or any other facility-specific values in Python code. Scripts and CLI commands must be fully generic — they load all configuration from the facility YAML at runtime via `get_facility(facility)`.

**What goes in public facility YAML** (`<facility>.yaml`):
- `discovery_roots` — paths to scan for code/data
- `data_systems.tdi.*` — TDI function directories, reference shots, exclude lists
- `data_systems.mdsplus.*` — tree names, subtrees, node usages, setup commands
- `data_systems.mdsplus.static_trees` — static tree versions, first_shot, descriptions, systems
- `data_access_patterns` — primary method, naming conventions, key tools
- `wiki_sites` — wiki URLs for scraping

**What goes in private facility YAML** (`<facility>_private.yaml`, gitignored):
- Hostnames, IPs, NFS mount points
- OS versions, kernel info
- Login node names, local host overrides
- User-specific paths, tool locations

**How to load config:** `get_facility(facility)` from `imas_codex.discovery.base.facility` loads both public + private YAML and returns a dict.

**When adding a new discovery pipeline or data source**, add the required config fields to the facility YAML schema (`imas_codex/schemas/facility_config.yaml`) and load them via `get_facility()`. The Python code should work unchanged across all facilities — only the YAML differs.

**Editing configs:** Always use MCP tools rather than direct file editing:

```python
# Update public facility config (wiki sites, discovery roots, data systems)
update_facility_config('tcv', {'discovery_roots': ['/new/path']})

# For infrastructure notes, use the repl tool directly
repl("update_infrastructure('tcv', {'exploration_notes': ['Found equilibrium codes at /home/codes/liuqe']})")
```

**Validation:** `validate_facility_config('tcv')` returns a list of error strings. The config schema is also exposed via the `get_graph_schema()` MCP tool.

## Remote Tools

Prefer these Rust-based CLI tools over standard Unix commands. Defined in `imas_codex/config/remote_tools.yaml`.

| Tool | Purpose | Use Instead Of |
|------|---------|----------------|
| `rg` | Pattern search | `grep -r` |
| `fd` | File finder | `find` |
| `eza` | Directory listing with tree view | `ls -la`, `tree` |
| `tokei` | LOC by language | `wc -l`, `cloc` |
| `uv` | Python package manager | `pip`, `virtualenv` |

Install on any facility: `uv run imas-codex tools install <facility>`

**Critical:** `fd` requires a path argument on large filesystems to avoid hanging: `fd -e py /path`

**Critical:** `rg` also requires an explicit path in scripted/non-tty contexts: with no path and no match it falls back to reading stdin and waits forever (a June-9 session shell hung 46 h on exactly this). Always `rg pattern <path>` in agent commands.

**Remote Python — two-interpreter architecture:**

- `run_python_script()` / `async_run_python_script()` — venv `python3` (3.12+) via `_REMOTE_PATH_PREFIX`. Modern syntax OK (`X | Y`, `match`).
- `SSHWorkerPool` / `pooled_run_python_script()` — hardcoded `/usr/bin/python3` (3.9+, stdlib-only) to avoid 60–100s NFS venv startup. **No 3.10+ syntax** in pool scripts. Each script declares its Python version in a docstring header. Ruff skips type-hint modernization for `imas_codex/remote/scripts/*` (see per-file ignores).

**Remote zombie prevention:** every executor function wraps the SSH command with server-side `timeout <local_timeout + 5s>` so the remote process self-terminates when the local SSH client is killed. Never construct raw SSH calls — always use the executor functions.
