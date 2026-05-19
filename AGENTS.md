# Agent Guidelines

> **Shared guardrails** (git safety, parallel-agent rules, model selection,
> compute infrastructure, commit discipline) live in `~/.agents/AGENTS.md`.
> This file covers **repo-specific** domain knowledge only.

Use terminal for direct ops (`rg`, `fd`, `git`), MCP `repl()` for chained processing + graph queries, `uv run` for tests/CLI. Conventional commits. **Always commit and push after every file modification — no confirmation, no asking.** Never use `vscode_askQuestions` or other interactive VS Code dialogs — put questions inline in the chat response.

**Git sync (fork-based):** All work on the fork's `main` branch. Merge on pull — never rebase, never use feature branches (the release CLI requires `main`). Pull before work and before push: `git pull origin main && git push origin main`. Push to `origin`, **never directly to `upstream`** — final releases go via the release CLI. Commit your own files before pulling a dirty worktree. See `~/.agents/AGENTS.md` for clone setup, banned commands, and stash ban. Release workflow detail in [Release Workflow](#release-workflow).

## Project Philosophy

Greenfield project, no backwards compatibility. Remove deprecated code decisively, update all usages when patterns change, prefer explicit over clever, prefer good names over "enhanced/new/refactored". Exploration notes belong in facility YAML; `docs/` is for mature infrastructure only.

- **Stale context kills.** If your session is more than a few hours old, your memory of file contents may be wrong. Re-read every file from disk before modifying — never write code from memory.
- **Build on common infrastructure.** Search for existing utilities (remote SSH execution, graph queries, file parsing, LLM calls all have canonical patterns) before implementing. Extract shared patterns to `imas_codex/remote/` or `imas_codex/graph/`. Never inline SSH subprocess calls — use `run_python_script()` / `async_run_python_script()` from `imas_codex.remote.executor` with scripts in `imas_codex/remote/scripts/`.
- **One source of truth.** When a feature applies across domains (e.g. `files` and `wiki` discovery share `discovery/base/`), implement it once. If data is already in the graph (public repos via `SoftwareRepo` nodes, etc.), query the graph — don't re-derive locally.

## Model & Tool Configuration

All model and tool settings live in `pyproject.toml` under `[tool.imas-codex]`. No backward-compatible aliases — use the canonical accessors from `imas_codex.settings`.

**Sections** (each with a `model` parameter where relevant):

| Section | Purpose | Accessor |
|---------|---------|----------|
| `[graph]` | Neo4j connection, graph name/location | `get_graph_uri()`, `get_graph_username()`, `get_graph_password()`, `resolve_graph()` |
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
| `[sn-enrich]` | SN per-item context enrichment | `get_model("sn-enrich")` |
| `[sn-fanout]` | Structured fan-out (Proposer/Executor/Synthesizer) | — |
| `[sn-review]` | Shared RD-quorum settings (disagreement threshold, max cycles, active profile) | `get_sn_review_disagreement_threshold()`, `get_sn_review_max_cycles()`, `get_sn_review_active_profile()` |
| `[sn-review.names]` / `[sn-review.docs]` | Reviewer model chain per axis (1–3 models) | `get_sn_review_names_models()`, `get_sn_review_docs_models()` |
| `[sn-review.names.profiles.*]` | Named profiles (default, opus-only, quality-cost-balanced) | — |
| `[sn-benchmark]` | SN benchmark compose-models list and reviewer-model | `get_sn_benchmark_compose_models()`, `get_sn_benchmark_reviewer_model()` |

**Model access:** `get_model(section)` is the single entry point for all model lookups. Pass the pyproject.toml section name directly: `"language"`, `"vision"`, `"reasoning"`, or `"embedding"`. Priority: section env var → pyproject.toml config → default.

**Graph access:** Graph profiles separate **name** (what data) from **location** (where Neo4j runs). The default graph `"codex"` contains all facilities + IMAS DD and runs at location `"iter"`. `IMAS_CODEX_GRAPH` env var selects the graph name. `IMAS_CODEX_GRAPH_LOCATION` overrides where it runs. Each location maps to a unique bolt+HTTP port pair by convention:

| Location | Bolt | HTTP |
|----------|------|------|
| iter | 7687 | 7474 |
| tcv | 7688 | 7475 |
| jt-60sa | 7689 | 7476 |

Env var overrides (`NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`) still apply as escape hatches over any profile. Use `resolve_graph(name)` from `imas_codex.graph.profiles` for direct profile resolution. All CLI `graph` commands accept `--graph/-g` to target a specific graph.

**Location-aware connections:** `is_local_host(host)` determines direct vs tunnel access at connection time. For edge cases, configure `login_nodes` and `local_hosts` in the facility's private YAML. Check with: `imas-codex config local-hosts`.

## Schema System

All graph node types, relationships, and properties are defined in LinkML schemas — the single source of truth.

**Schema files:**
- `imas_codex/schemas/facility.yaml` - Facility graph: SourceFile, SignalNode, CodeChunk, FacilityPath, FacilitySignal, etc.
- `imas_codex/schemas/imas_dd.yaml` - DD graph: IMASNode, DDVersion, Unit, IMASCoordinateSpec, Cluster, NodeCategory
- `imas_codex/schemas/standard_name.yaml` - Standard names: StandardName, StandardNameSource, Review, DocsRevision, VocabGap, LLMCost, SNRun
- `imas_codex/schemas/grammar_graph.yaml` - ISN grammar: GrammarSegment, GrammarToken, GrammarTemplate, ISNGrammarVersion
- `imas_codex/schemas/facility_config.yaml` - Per-facility YAML config schema
- `imas_codex/schemas/task_groups.yaml` - Worker task grouping
- `imas_codex/schemas/common.yaml` - Shared enums and mixins

**Build pipeline:**
- Models auto-generated during `uv sync` via hatch build hook
- Regenerate manually: `uv run build-models --force`
- Output: `imas_codex/graph/models.py`, `imas_codex/graph/dd_models.py`, `imas_codex/config/models.py`, `agents/schema-reference.md`, `imas_codex/graph/schema_context_data.py`

**CRITICAL: Never commit auto-generated files.** These are gitignored and rebuilt on `uv sync`. If `git status` shows a generated model file as untracked or modified, do NOT stage it. Generated files:
- `imas_codex/graph/models.py`
- `imas_codex/graph/dd_models.py`
- `imas_codex/config/models.py`
- `agents/schema-reference.md`
- `imas_codex/graph/schema_context_data.py`

**PhysicsDomain enum**: Imported from the `imas-standard-names` PyPI package and re-exported from `imas_codex.core.physics_domain`. The canonical vocabulary is maintained in the imas-standard-names project. Contains 32 physics domain values. `imas_codex/core/physics_domain.py` is a hand-written one-line re-export — it IS committed and should NOT be treated as auto-generated.

**NodeCategory enum** (`imas_dd.yaml`): DD node classification — 9 values: `quantity`, `geometry`, `coordinate`, `metadata`, `error`, `structural`, `identifier`, `fit_artifact`, `representation`. Classifier lives in `imas_codex/core/node_classifier.py` (two-pass: Pass 1 attribute-only, Pass 2 graph-relational). Category sets for pipeline participation in `imas_codex/core/node_categories.py`.

**IMASNodeStatus lifecycle** (`imas_dd.yaml`): Tracks DD build pipeline progress — `built → enriched → refined → embedded → classified`. Seven workers: EXTRACT → BUILD → ENRICH → REFINE → EMBED → CLASSIFY → CLUSTER. CLASSIFY runs after EMBED, before CLUSTER, using `get_model("language")` (service tag: `"data-dictionary"`). Three-tier domain assignment: Tier 1 = LLM classification for physics paths; Tier 2 = inheritance for error paths (via `HAS_ERROR`) and metadata (via `HAS_PARENT`); Tier 3 = none for infrastructure metadata (`ids_properties/*`, `code/*`). Paths initially assigned `"general"` are retried with expanded cluster context. `--reset-to embedded` clears domain classifications and re-classifies only (~$2.60, the cheapest domain fix).

Always import enums and classes from generated models. Never hardcode status values:

```python
from imas_codex.graph.models import SourceFile, SourceFileStatus, SignalNode

sf = SourceFile(
    id="tcv:/home/codes/liuqe.py",
    facility_id="tcv",
    path="/home/codes/liuqe.py",
    status=SourceFileStatus.discovered,  # Use enum, not string
)
add_to_graph("SourceFile", [sf.model_dump()])
```

**Extending schemas:** Edit LinkML YAML → `uv run build-models --force` → import from `imas_codex.graph.models`. Prefer additive changes, but renames and removals are fine when they improve consistency — the schema must stay clean. When renaming or removing: update all code references, migrate graph data, and rebuild models in a single commit.

**Full schema reference:** [agents/schema-reference.md](agents/schema-reference.md) — auto-generated list of all node labels, properties, vector indexes, relationships, and enums. Rebuilt on `uv sync`.

### Schema Design Guidelines

Conventions when adding classes, properties, or relationships. The build pipeline, `create_nodes()`, and query builder depend on predictable schema structure.

**Dual property + relationship.** Every slot with a class range produces **both** a node property (fast `WHERE` filtering) AND a Neo4j relationship (graph traversal). `create_nodes()` in `client.py` does `SET n += item` then `MERGE (n)-[:REL]->(t:Target {id: item.slot})`. Never remove one side of the dual model.

**Relationship type names.** If the slot has a `relationship_type` annotation, that's used; otherwise the slot name is uppercased (`has_chunk` → `HAS_CHUNK`). Add an explicit annotation when the auto-derived name is unclear. **All `facility_id` slots MUST have `range: Facility` + `annotations: { relationship_type: AT_FACILITY }`** — no exceptions. Prefer verb-based names (`SOURCE_PATH`, `BELONGS_TO_DIAGNOSTIC`).

**Class template:**

```yaml
MyNewNode:
  description: >-
    What this node represents. Include example Cypher queries.
  class_uri: facility:MyNewNode
  attributes:
    id: { identifier: true, required: true, description: "Composite key (e.g., 'tcv:unique_part')" }
    facility_id:
      required: true
      range: Facility
      annotations: { relationship_type: AT_FACILITY }
    status: { range: MyNewNodeStatus, required: true }
    description: { description: "Human-readable, drives semantic search" }
    embedding: { multivalued: true, range: float }
    embedded_at: { range: datetime }
```

**Rules:**
- Use `identifier: true` on exactly one slot per class (always `id`).
- Composite IDs use colon separator: `facility_id:unique_part`. Must be globally unique.
- Nodes with `embedding` + `description` auto-get a vector index `{snake_case_label}_desc_embedding`. Override with `vector_index_name` annotation if needed.
- Status enums live in the same schema file. **Durable states only** — never `scanning`/`processing`. Worker coordination via `claimed_at` timestamps.
- `is_private: true` excludes a slot from the graph (config-only).
- Never hardcode enum values in Python — import from generated models.
- Never skip the `description` field — it drives semantic search.
- Don't use `multivalued: true` on relationship slots unless genuinely many-to-many.

### Schema-Driven Testing

Tests in `tests/graph/` are **parametrized from the schema** — they validate graph data against LinkML declarations. Key modules: `test_schema_compliance.py` (node labels/properties/enums), `test_referential_integrity.py` (relationship types with correct `relationship_type` annotation), `test_data_quality.py` (embedding coverage).

**On test failure, fix the root cause, not the schema.** Three cases: (a) building a new capability → declare in LinkML first, then write code; (b) code writing non-compliant data → fix the code or the bad data; (c) stale data from a prior schema version → migrate/remove the data. Never add schema declarations just to make tests green.

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

## Graph State Machine

Status enums represent **durable states only**. No transient states like `scanning`, `scoring`, or `ingesting`.

**Worker coordination:** Claim via `claimed_at = datetime()` (status unchanged), complete by updating status and clearing `claimed_at = null`. Orphan recovery is automatic via timeout check in claim queries.

### Claim Patterns — Deadlock Avoidance

All claim functions **must** use three anti-deadlock patterns. Reference implementations: `discovery/wiki/graph_ops.py`, `discovery/code/graph_ops.py`. Shared infrastructure: `discovery/base/claims.py`.

1. **`@retry_on_deadlock()`** — decorator from `claims.py`. Retries on `TransientError` with exponential backoff + jitter. Apply to every function that writes `claimed_at`.
2. **`ORDER BY rand()`** — randomize lock acquisition order. Deterministic ordering (`ORDER BY v.version ASC`, `ORDER BY score DESC`) causes lock convoys where concurrent workers deadlock on the same rows.
3. **`claim_token` two-step verify** — SET a UUID token in step 1, then read back by token in step 2. Prevents double-claiming race conditions.

```python
from imas_codex.discovery.base.claims import retry_on_deadlock

@retry_on_deadlock()
def claim_items(facility: str, limit: int = 10) -> list[dict]:
    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query("""
            MATCH (n:MyNode {facility_id: $facility})
            WHERE n.status = 'discovered' AND n.claimed_at IS NULL
            WITH n ORDER BY rand() LIMIT $limit
            SET n.claimed_at = datetime(), n.claim_token = $token
        """, facility=facility, limit=limit, token=token)
        return list(gc.query("""
            MATCH (n:MyNode {claim_token: $token})
            RETURN n.id AS id, n.path AS path
        """, token=token))
```

**Never** use deterministic `ORDER BY` in claim queries. **Never** write a manual retry loop for deadlocks — use `@retry_on_deadlock()`. See `imas_codex/discovery/README.md` for detailed rationale.

### FacilityPath Lifecycle

```
discovered → explored | skipped | stale
```

| Score | Use Case |
|-------|----------|
| 0.9+ | IMAS integration, IDS read/write |
| 0.7+ | MDSplus access, equilibrium codes |
| 0.5+ | General analysis codes |
| <0.3 | Config files, documentation |

### SourceFile Lifecycle

```
discovered → ingested | failed | stale
```

Ingestion is interrupt-safe — rerun to continue. Already-ingested files are skipped.

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

**Read-only mode:** `imas-codex serve --read-only` suppresses all write tools (`repl()` REPL, `add_to_graph()`, `update_facility_config()`) and exposes only the search/read tools (`search_signals`, `search_docs`, `search_code`, `search_dd_paths`, `fetch_content`, `get_graph_schema`, etc.). Use for any context where graph mutation is not desired.

**DD-only mode:** `imas-codex serve --dd-only` hides facility-specific tools and **implies `--read-only`**. Use for container deployments with a DD-only graph. Auto-detected from graph content if omitted.

```bash
# Full mode (default) — all tools including REPL and write operations
imas-codex serve

# Read-only mode — search and read tools only, no REPL or graph writes
imas-codex serve --read-only

# DD-only mode — DD tools only, implies read-only (typical container deployment)
imas-codex serve --dd-only --transport streamable-http
```

## LLM Access

All LLM interaction flows through two canonical modules. Never call `litellm.completion()` directly — the shared functions handle prompt caching flags, cost tracking, retries with exponential backoff, and structured output parsing.

### Calling LLMs

Use `call_llm_structured()` / `acall_llm_structured()` from `imas_codex.discovery.base.llm`:

```python
from imas_codex.discovery.base.llm import call_llm_structured

result, cost, tokens = call_llm_structured(
    model=get_model("language"),
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ],
    response_model=MyPydanticModel,
)
```

These functions automatically: apply `inject_cache_control()` to system messages, retry on API/parse errors with backoff, accumulate cost across retries, and parse structured output via Pydantic `response_format`.

### Rendering Prompts

Use `render_prompt()` from `imas_codex.llm.prompt_loader` — never construct paths to prompt files manually:

```python
from imas_codex.llm.prompt_loader import render_prompt

system_prompt = render_prompt("paths/scorer", {"facility": "tcv", "batch": batch_data})
```

For path access (e.g., in tests), import `PROMPTS_DIR` from the same module — never hardcode path segments like `"llm" / "prompts"`.

### Rules

- Model identifiers require the `openrouter/` prefix to preserve `cache_control` blocks
- Use `get_model(section)` from `imas_codex.settings` for model selection — never hardcode model names
- All models live in `pyproject.toml` under `[tool.imas-codex.<section>].model`. **Never hardcode `cc:*`, raw provider IDs, or any other model string in pipeline code or worker prompts.** If you think you need a new model variant, add a section to `pyproject.toml` and reference it via `get_model("section")`.
- Pydantic schema injection via `get_pydantic_schema_json()` — never hardcode JSON examples in prompts
- Each prompt declares `schema_needs` in frontmatter to load only required schema context

### Routing: Direct vs Proxy

`call_llm_structured()` chooses between two paths automatically. Understanding both is critical because they have different cost-tracking and caching behavior.

| Path | Trigger | Cost tracking | Prompt caching | Use for |
|------|---------|---------------|----------------|---------|
| **Direct (bypass)** | `supports_cache(model)` AND `OPENROUTER_API_KEY_IMAS_CODEX` set | ✅ `response_cost` populated | ✅ `cache_control` preserved | All cache-capable models on the codex billing account |
| **Proxy** | Otherwise (no IMAS_CODEX key, or non-caching model) | ❌ `response_cost = 0` | ❌ `cache_control` stripped | Air-gapped clusters, models without caching, dev environments |

**Validated empirically (2026-04-27)** with `openrouter/anthropic/claude-haiku-4.5` on the direct path:
- COLD call: cost = $0.006335, cache_write = 4801 tokens
- WARM call: cost = $0.000814 (87% cheaper), cache_read = 4801 tokens

The bypass logic lives in `imas_codex/discovery/base/llm.py` lines 794-799. Keep `OPENROUTER_API_KEY_IMAS_CODEX` set and use `openrouter/anthropic/<model>` (or unprefixed `anthropic/<model>` — `ensure_model_prefix()` normalizes) and you get cost + cache for free.

**Anti-pattern: do not invent `cc:*` model strings.** A `cc:opus` / `cc:sonnet` / `cc:haiku` proxy alias exists in `litellm_config.yaml` for billing isolation to a separate OpenRouter account, but it routes via the **proxy path** which silently breaks both `response_cost` and `cache_control`. If you need spend isolation, add a per-service env var (`OPENROUTER_API_KEY_<SERVICE>`) and let the direct path handle routing — it preserves cost + cache.

### Service Tagging

All LLM calls are tagged with a `service` parameter for spend visibility. The tag flows to:
- **X-Title** header → OpenRouter dashboard (shows as `imas-codex:<service>`)
- **Langfuse metadata** → trace analytics
- **Per-service API keys** → optional spend isolation via separate OpenRouter keys

**Service taxonomy:**

| Service Tag | Description | Call Sites |
|-------------|-------------|------------|
| `facility-discovery` | Facility path/wiki/signal/code discovery | `discovery/paths/*`, `discovery/wiki/*`, `discovery/signals/*`, `discovery/base/image.py`, `discovery/code/*`, `discovery/static/*` |
| `standard-names` | Standard name generation, review, enrichment | `standard_names/workers.py`, `benchmark.py`, `review/pipeline.py` |
| `data-dictionary` | DD enrichment, domain classification, and cluster labeling | `graph/dd_enrichment.py`, `dd_ids_enrichment.py`, `dd_identifier_enrichment.py`, `dd_workers.py`, `clusters/labeler.py` |
| `imas-mapping` | IMAS signal-to-path mapping | `ids/mapping.py`, `ids/metadata.py` |
| `untagged` | Default — surfaces missed call sites | Any call without explicit `service=` |

**Usage:**
```python
result, cost, tokens = call_llm_structured(
    model=model, messages=messages, response_model=MyModel,
    service="facility-discovery",  # Required — AST test enforces this
)
```

**Per-service API keys** (optional): Set `OPENROUTER_API_KEY_<SERVICE_UPPER>` to use a separate OpenRouter key for a service pipeline. Hyphens become underscores. Falls back to `OPENROUTER_API_KEY_IMAS_CODEX`.

```bash
# Example: isolate discovery spend to its own key
export OPENROUTER_API_KEY_FACILITY_DISCOVERY=sk-or-v1-...
```

### Prompt Structure and Caching

Static-first ordering maximises OpenRouter prompt-cache hit rates. **System prompt** (schema, enums, rules, output format) is static and shared — `inject_cache_control()` sets a `cache_control: {"type": "ephemeral"}` breakpoint at its end. **User prompt** is per-call dynamic. In Jinja templates, place `{% include %}` schema/rules blocks BEFORE dynamic variables to maximise the cacheable prefix.

## Exploration

Before disk-intensive operations, **check facility excludes** to avoid repeating known timeouts:

```python
info = get_facility('tcv')
excludes = info.get('excludes', {})
print(excludes.get('large_dirs', []))
print(excludes.get('depth_limits', {}))
print(info.get('exploration_notes', [])[-3:])
```

When a command times out, **persist the constraint immediately** via `update_infrastructure()` in the repl. Never repeat a timeout.

### Persistence

| Discovery Type | Destination |
|----------------|-------------|
| Source files, paths, codes, trees | `add_to_graph()` (public graph) |
| Public facility config, wiki sites, discovery roots | `update_facility_config()` |
| Infrastructure notes (hostnames, tool versions) | `repl("update_infrastructure('tcv', {...})")` |

### Data Classification

- **Graph (public):** MDSplus tree names, analysis code names/versions/paths, TDI functions, diagnostic names
- **Infrastructure (private):** Hostnames, IPs, NFS mounts, OS versions, tool availability, user directories

## Graph Operations

**Schema verification:** Before writing Cypher queries, verify property names against `agents/schema-reference.md` (auto-generated) or call `get_graph_schema()`. Common pitfall: WikiChunk/CodeChunk text content is stored in the `text` property.

### Cypher Compatibility — Neo4j 2026

We run **Neo4j 2026.01.x** with `db.query.default_language: CYPHER_5`. The only breaking syntax change that affects this codebase: `x NOT IN [list]` is removed — write `NOT (x IN [list])` instead. `CASE WHEN` is fully supported — use it freely. For "keep old value if new is empty," prefer `SET s.f = coalesce(nullIf(new, ''), old)` over `CASE WHEN`. Test new Cypher against the live graph before committing.

### Neo4j Management

```bash
uv run imas-codex graph start                 # Start Neo4j (auto-detects mode)
uv run imas-codex graph stop                  # Stop Neo4j
uv run imas-codex graph status                # Check graph status
uv run imas-codex graph profiles              # List all profiles and ports
uv run imas-codex graph shell                 # Interactive Cypher (active profile)
uv run imas-codex graph export               # Export graph to archive (add -f <facility> for filtered)
uv run imas-codex graph load graph.tar.gz    # Load graph archive
uv run imas-codex graph pull                 # Pull latest from GHCR (add --facility for per-facility)
uv run imas-codex graph push --dev           # Push to GHCR (add --facility for per-facility)
uv run imas-codex graph fetch                # Download archive (no load)
uv run imas-codex graph clear                # Clear all graph data
uv run imas-codex graph init NAME            # Initialize a new graph instance
uv run imas-codex graph switch NAME          # Activate a different graph
uv run imas-codex graph list                 # List local graph instances
uv run imas-codex graph secure               # Rotate Neo4j password
uv run imas-codex graph tags                 # List GHCR versions
uv run imas-codex graph prune --dev          # Remove all dev GHCR tags
uv run imas-codex tunnel start iter          # Start SSH tunnel to remote host
uv run imas-codex tunnel status              # Show active tunnels
uv run imas-codex config private push        # Push private YAML to Gist
uv run imas-codex config secrets push iter   # Push .env to remote host
```

Never use `DETACH DELETE` on production data without user confirmation. For re-embedding: update nodes in place, don't delete and recreate.

### Graph Migrations

Run migrations as inline Cypher via `imas-codex graph shell` or the MCP `repl()` (`query()`). Never create `scripts/migrate_*.py` or `repair_*.py`. For >10K-node migrations, batch with `LIMIT` to avoid transaction timeouts; verify counts before and after.

### LLMCost Node Properties

`LLMCost` nodes track per-call LLM spend. **All `LLMOperation`-mixin fields are prefixed with `llm_`** — never use bare `cost`, `model`, or `service`.

| Property | Type | Description |
|----------|------|-------------|
| `llm_cost` | float | Dollar cost of the call (USD) |
| `llm_model` | string | Model identifier |
| `llm_service` | string | Service tag (e.g. `standard-names`) |
| `llm_tokens_in` / `llm_tokens_out` | int | Input/output tokens |
| `llm_tokens_cached_read` / `llm_tokens_cached_write` | int | Prompt-cache hit/write tokens |
| `llm_at` | datetime | When the call was made |
| `run_id` (required), `phase` (required), `cycle`, `pool`, `batch_id` | string/int | Pipeline grouping |
| `sn_ids` | string[] | SNs touched by this call |
| `event_type`, `overspend` | string/bool | Charge classification |
| `for_run` | string | `SNRun` ID this cost belongs to |

**Canonical cost queries:**

```cypher
-- Total LLM spend
MATCH (c:LLMCost) RETURN round(sum(c.llm_cost)*100)/100 AS total_usd

-- Per-pool / per-model breakdown
MATCH (c:LLMCost)
RETURN c.pool AS pool, count(c) AS calls, round(sum(c.llm_cost)*100)/100 AS usd
ORDER BY usd DESC

-- Spend for a specific run
MATCH (c:LLMCost {for_run: $run_id}) RETURN sum(c.llm_cost) AS total

-- SNRun budget tracking
MATCH (r:SNRun) RETURN r.cost_spent AS spent, r.cost_limit AS budget, r.stop_reason
ORDER BY r.started_at DESC LIMIT 1
```

`SNRun.cost_spent` / `cost_limit` / `cost_total` are aggregates; `LLMCost.llm_cost` is the per-call source of truth. Embedding costs are always zero — only OpenRouter LLM calls incur cost.

### Neo4j Lock Files — CRITICAL

Neo4j uses several lock file types. Mishandling them **causes data loss**.

| Lock File | Location | Purpose | Safe to Delete? |
|-----------|----------|---------|----------------|
| `store_lock` | `data/databases/` | Coordinates single-writer access | Yes — after confirming Neo4j is stopped |
| `database_lock` | `data/databases/*/` | Per-database writer lock | Yes — after confirming Neo4j is stopped |
| `write.lock` | `data/databases/*/schema/index/*/` | Lucene index segment lock | **NEVER** — deletion corrupts vector indexes |

**Rules:**
1. **Never use `find -name "*.lock"` to clean locks** — this matches Lucene `write.lock` files inside vector index directories.
2. Only remove `store_lock` and `database_lock` explicitly by path, and only after confirming Neo4j has fully stopped.
3. On GPFS/NFS, stale POSIX locks can survive process death. The safe workaround is inode replacement (`cp file file.unlock && mv -f file.unlock file`), not deletion.
4. If Lucene `write.lock` is deleted while Neo4j is running, it triggers `AlreadyClosedException`, checkpoint failure, and potential database reinitialization on next start.

**Never use the Docker entrypoint** (`/startup/docker-entrypoint.sh`) to start Neo4j in Apptainer. It calls `neo4j-admin dbms set-initial-password` and runs `rm -rf conf/*` on every start, which can reinitialize an existing database after a crash. Always use `neo4j console` directly with a host-side `conf/` bind mount.

### Vector Indexes

Vector indexes use Neo4j 2026.01's native `SEARCH` clause for in-index pre-filtering.
All indexes include quantization for ~4× memory savings:

```python
from imas_codex.settings import get_embedding_dimension

dim = get_embedding_dimension()
gc.query(f"""
    CREATE VECTOR INDEX my_index IF NOT EXISTS
    FOR (n:MyNodeType) ON n.embedding
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {dim},
            `vector.similarity_function`: 'cosine',
            `vector.quantization.enabled`: true
        }}
    }}
""")
```

`ensure_vector_indexes()` auto-detects dimension mismatches and drops/recreates stale indexes.

**Existing indexes:** See [agents/schema-reference.md](agents/schema-reference.md) for the full list of vector indexes derived from the LinkML schema.

### Semantic Search & Graph RAG

Use `semantic_search(text, index, k)` in the python REPL:

```python
# Document content (wiki, code)
semantic_search("COCOS sign conventions", index="wiki_chunk_embedding", k=5)

# Descriptive metadata (signals, paths - search by physics meaning)
semantic_search("plasma current measurement", index="facility_signal_desc_embedding", k=10)
```

Combine vector similarity with link traversal using the Cypher 25 SEARCH clause:

```python
results = query("""
    CYPHER 25
    MATCH (signal:FacilitySignal)
    SEARCH signal IN (
      VECTOR INDEX facility_signal_desc_embedding
      FOR $embedding
      LIMIT 5
    ) SCORE AS score
    WHERE signal.facility_id = $facility
    WITH signal, score
    MATCH (signal)-[:DATA_ACCESS]->(da:DataAccess)
    OPTIONAL MATCH (signal)-[:HAS_DATA_SOURCE_NODE]->(dn:SignalNode)
        <-[:SOURCE_PATH]-(m:IMASMapping)-[:TARGET_PATH]->(imas:IMASNode)
    RETURN signal.id, signal.description, da.data_template,
           collect(imas.id) AS imas_paths, score
    ORDER BY score DESC
""", embedding=embed("electron density profile"), facility="tcv")
```

Use `build_vector_search()` from `imas_codex.graph.vector_search` to generate
SEARCH clauses programmatically. All WHERE conditions are post-filters (in-index
pre-filtering requires properties registered as additional vector index properties).

**Key relationships for traversal:**

| From | Relationship | To |
|------|--------------|-----|
| FacilitySignal | DATA_ACCESS | DataAccess |
| FacilitySignal | HAS_DATA_SOURCE_NODE | SignalNode |
| IMASMapping | SOURCE_PATH | SignalNode |
| IMASMapping | TARGET_PATH | IMASNode |
| WikiChunk | HAS_CHUNK← | WikiPage |
| FacilityPath | AT_FACILITY | Facility |

**Token cost:** Always project specific properties in Cypher (`RETURN n.id, n.name`), never return full nodes. Use Cypher aggregations instead of Python post-processing.

### Batch Operations

Use `UNWIND` for batch graph writes:

```python
query('''
    UNWIND $items AS item
    MERGE (n:Tool {id: item.id})
    SET n += item
    WITH n
    MATCH (f:Facility {id: 'tcv'})
    MERGE (n)-[:AT_FACILITY]->(f)
''', items=tools)
```

### Release Workflow

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

## Standard Names

> Architecture docs: `docs/architecture/standard-names.md` (pipeline detail),
> `docs/architecture/standard-names-decisions.md` (design rationale).

### Pipeline

**Six-pool concurrent run loop:**

| Pool | Label | Stage gate | Key operation |
|------|-------|------------|---------------|
| 1 | `GENERATE_NAME` | `StandardNameSource.status=pending` | LLM generates name; new SN persisted at `name_stage='drafted'`. Unit injected from DD — never from LLM. |
| 2 | `REVIEW_NAME` | `name_stage='drafted'` | RD-quorum scores name; atomic transition → `accepted` (rsn≥min) / `reviewed` / `exhausted`. Also admits `origin='derived'` parents (Phase 4+): scores include a `specificity` dimension; `specificity < threshold` → `proposed_verdict='reject'`. |
| 3 | `REFINE_NAME` | `name_stage='reviewed' AND rsn<min AND chain_length<cap` | Creates NEW SN node; predecessor flipped to `superseded`; source edges migrated; `REFINED_FROM` edge added |
| 4 | `GENERATE_DOCS` | `name_stage='accepted' AND docs_stage='pending'` | LLM generates documentation; `docs_stage → 'drafted'`. Cross-pipeline gate: fires only after name is accepted |
| 5 | `REVIEW_DOCS` | `docs_stage='drafted'` | RD-quorum scores docs; atomic transition → `accepted` / `reviewed` / `exhausted` |
| 6 | `REFINE_DOCS` | `docs_stage='reviewed' AND rds<min AND docs_chain_length<cap` | Rewrites docs in-place; prior content snapshotted on a `DocsRevision` node via `DOCS_REVISION_OF`; `docs_stage → 'drafted'` |

**Derived parents** (`origin='derived'`) flow through the same six-pool loop with two differences: (a) placeholder nodes are created at write time by `_write_standard_name_edges` only if they pass the deterministic admission gate in `imas_codex/standard_names/parents.py` (two-clause: IR specificity OR vector-like topology); (b) REVIEW_NAME includes a `specificity` dimension that fires only on derived parents. Failed-on-name parents are deleted with their `HAS_PARENT` edges (Phase 6). Description-quality failures route to `REFINE_DOCS`, not delete. No separate `sn parents` CLI — all parent handling is folded into `sn run`. The `apply_derived_parent_migration` idempotent self-healing pass runs on every `sn run` startup to migrate legacy `origin='deterministic'` entries to the new lifecycle.

**Acceptance always overrides cap:** even at `chain_length == cap − 1`, a passing score wins (no forced exhaustion on a good result). **Escalation:** on the final refine attempt (`chain_length == cap − 1`), the pool switches to `DEFAULT_ESCALATION_MODEL` (default `openrouter/anthropic/claude-opus-4.6`). **Backlog throttle:** when refine_name backlog > 0.5 × generate_name backlog, `BudgetManager.pool_admit` dampens generate_name effective weight by 0.5×.

**EXTRACT → COMPOSE** sub-pipeline runs inside GENERATE_NAME (pool path) or via `pool_adapter.run_explicit_paths()` (single-pass path): DD paths queried (filtered by `SN_SOURCE_CATEGORIES`), classified, clustered, batched; ISN 3-layer validation (Pydantic → semantic → description) + grammar round-trip; cross-batch dedup; conflict-detecting Neo4j writes.

**Key modules:**

| Module | Purpose |
|--------|---------|
| `imas_codex/standard_names/pools.py` | Pool specifications: `POOL_WEIGHTS`, `POOL_NAMES`, `_build_pool_specs`, backlog throttle |
| `imas_codex/standard_names/loop.py` | Six-pool loop orchestrator (`run_sn_pools()`); pulls weights from `pools.py` |
| `imas_codex/standard_names/workers.py` | Claim/process/persist functions for all six pools |
| `imas_codex/standard_names/enrichment.py` | Primary cluster selection (IDS > domain > global scope), grouping cluster selection (global > domain > IDS), global grouping by (cluster × unit) |
| `imas_codex/standard_names/consolidation.py` | Cross-batch dedup, conflict checks, coverage gap accounting |
| `imas_codex/standard_names/graph_ops.py` | Neo4j read/write with unit conflict detection; StandardNameSource CRUD; `persist_refined_name`, `persist_refined_docs` |
| `imas_codex/standard_names/pool_adapter.py` | Routes `--focus` through pool compose; handles explicit path seeding |
| `imas_codex/standard_names/defaults.py` | Central constants: `DEFAULT_MIN_SCORE=0.80`, `DEFAULT_ORPHAN_SWEEP_TIMEOUT_S=600`, `DEFAULT_REFINE_ROTATIONS`, `DEFAULT_ESCALATION_MODEL`, backlog caps |
| `imas_codex/standard_names/models.py` | Pydantic response models (`StandardNameComposeBatch`, `StandardNameAttachment`) |
| `imas_codex/standard_names/source_paths.py` | Central encode/parse/split/merge utilities for StandardName source paths |
| `imas_codex/standard_names/context.py` | Grammar context builder (vocabulary, examples, tokamak ranges) |
| `imas_codex/standard_names/search.py` | Vector search for similar existing StandardName nodes (collision avoidance) |
| `imas_codex/standard_names/orphan_sweep.py` | Periodic sweep that reverts stale `refining` claims after timeout |
| `imas_codex/standard_names/parents.py` | Derived-parent admission gate (`is_admissible_parent_name`, `recompute_parent_kind`); two-clause deterministic check; callable without a live graph (stub `gc` for Clause B) |

SN classification responsibilities are owned by DD `node_category` and pre-filtered via `SN_SOURCE_CATEGORIES` in `imas_codex/core/node_categories.py`.

**Unit safety:** Units flow exclusively from the DD `HAS_UNIT` relationship → EXTRACT → prompt
(marked read-only) → injected into candidate dict by worker → `HAS_UNIT` relationship in graph.
The LLM never provides the unit field.

### CLI Commands

| Command | Purpose | Key Options |
|---------|---------|-------------|
| `sn run` | Run the 6-pool standard name pipeline (GENERATE_NAME → REVIEW_NAME → REFINE_NAME → GENERATE_DOCS → REVIEW_DOCS → REFINE_DOCS). Auto-seeds all eligible domains by default; `--domain` restricts. `--focus <path>` routes specific DD paths through the full 6-pool loop scoped by a UUID; positional args, quoted space-separated, or repeated flags all work. `--flush` drains existing work without composing. `--only <phase>` runs a single phase. `--override-edits <name>` lets the pipeline overwrite catalog-edited fields. Key flags: `--min-score` (default `0.80`), `--rotation-cap` (default 3), `--escalation-model` (default `openrouter/anthropic/claude-opus-4.6`). | `--source {dd,signals}`, `--domain` (multi), `--facility`, `--focus` (multi), `--limit`, `--max-sources`, `-c/--cost-limit`, `--dry-run`, `--force`, `--reset-to`, `--reset-only`, `--from-model`, `--since`, `--before`, `--below-score`, `--tier`, `--retry-quarantined`, `--retry-skipped`, `--retry-vocab-gap`, `--min-score`, `--rotation-cap`, `--escalation-model`, `--review-name-backlog-cap`, `--review-docs-backlog-cap`, `--skip-review`, `--only`, `--override-edits`, `--flush` |
| `sn review` | Score existing valid standard names via RD-quorum reviewer pipeline | `--ids`, `--physics-domain`, `--status`, `--unreviewed`, `--force`, `--models`, `--batch-size`, `--neighborhood`, `--target`, `--reviewer-profile` |
| `sn export` | Export validated StandardName nodes to YAML staging dir. Applies quality gates (reviewer_score_name ≥ 0.65 + description sub-score). Staging dir defaults to `~/.cache/imas-codex/staging`. | `--staging`, `--min-score`, `--include-unreviewed`, `--min-description-score`, `--gate-only`, `--gate-scope {all,a,b,c,d}`, `--domain`, `--force`, `--skip-gate`, `--override-edits` |
| `sn preview` | Auto-export + local MkDocs preview via ISN's CatalogRenderer. Use `--no-export` to serve an existing staging dir. SSH tunnel: `ssh -L 8000:localhost:8000 <host>`. | `--export/--no-export`, `--staging`, `--port`, `--host` |
| `sn release` | Release standard names to the ISNC catalog. RC → origin (fork); final → upstream. Use `sn release status` for current state. | `-m/--message`, `--bump {major,minor,patch}`, `--final`, `--remote`, `--isnc`, `--staging`, `--skip-export`, `--dry-run` |
| `sn import` | Import reviewed YAML from ISNC back into graph. Diff-based origin tracking flips edited names to `origin=catalog_edit`. | `--isnc`, `--accept-unit-override`, `--accept-cocos-override`, `--dry-run` |
| `sn status` | Show StandardName and StandardNameSource pipeline statistics | — |
| `sn coverage` | Report DD/signal coverage by domain, cluster, and IDS | `--domain`, `--ids`, `--format` |
| `sn gaps` | List grammar vocabulary gaps from composition | `--segment`, `--export {table,yaml}` |
| `sn clear` | Unconditional full-subsystem wipe with auto grammar re-seed | `--dry-run`, `--force`, `--no-reseed` |
| `sn prune` | Scoped delete of StandardName nodes; relationship-first safety | `--status`, `--all`, `--source`, `--ids`, `--include-accepted`, `--include-sources`, `--dry-run` |
| `sn sync-grammar` | Seed/refresh ISN grammar vocabulary in the graph | `--dry-run` |
| `sn bench` | Benchmark LLM models on standard name generation quality | `--models`, `--max-candidates`, `--runs`, `--temperature`, `--output`, `--reviewer-model`, `--reviewer-models` |

**`sn run` scope routing:** By default, `sn run` runs the 6-pool completion loop across all eligible work — all pools run concurrently. `--only <phase>` runs a single phase in isolation (e.g. `--only reconcile` to mark stale sources). `--focus <path>` routes specific DD paths through the full 6-pool production pipeline, scoped by a UUID `scope_run_id` — use for iterative prompt development and quality investigation on individual paths without a full rotation. Focus paths can be provided three ways: trailing positional arguments (`sn run path1 path2`), quoted space-separated (`--focus "path1 path2"`), or repeated flags (`--focus path1 --focus path2`).

### Benchmark

`sn bench` uses the same prompt pipeline as `sn run` (system/user message split via
`build_compose_context()`). Model lists default from `[tool.imas-codex.sn-benchmark]` in
pyproject.toml. Output table includes a **Cache %** column showing the prompt-cache
hit rate per model (provider-side via OpenRouter — not something we implement). Scoring is
**6-dimensional**: grammar, semantic, documentation, convention, completeness, and compliance
(each 0-20 integer, aggregate normalized to 0-1 via `sum / 120.0`), evaluated by a reviewer LLM.
Scoring criteria are defined in `imas_codex/llm/config/sn_review_criteria.yaml`.

**Qualified models** (benchmark evidence from equilibrium + core_profiles + magnetics):

| Role | Model | Avg Quality | Notes |
|------|-------|-------------|-------|
| **Compose (recommended)** | `openai/gpt-5.5` | 78.2 | Best overall, strong grammar + docs |
| **Compose (alt)** | `anthropic/claude-sonnet-4.6` | 76.5 | 32% Outstanding, best grammar + documentation |
| **Compose (budget)** | `google/gemini-3.1-pro-preview` | 74.6 | Near-top quality, good consistency |
| **Compose (light)** | `anthropic/claude-haiku-4.5` | 61.2 | Adequate for bulk generation |

**Reviewer models** (7-model benchmark, GPT-5.5 compose, 4 items, 2026-05-15):

| Role | Model | Names Avg | Docs Avg | Cost/run | Notes |
|------|-------|-----------|----------|----------|-------|
| **Cycle 0 primary** | `deepseek/deepseek-v4-flash` | 0.756 | 0.992 | $0.002 | Cheapest, negative rank-corr (−0.5) maximises escalator |
| **Cycle 1 secondary** | `openai/gpt-5.4` | 0.672 | 0.881 | $0.033 | Fastest (31s), cross-vendor |
| **Cycle 2 escalator** | `google/gemini-3.1-pro-preview` | 0.884 | 1.000 | $0.469 | Highest quality, authoritative |
| Eliminated | `anthropic/claude-sonnet-4.6` | 0.581 | 0.875 | $0.079 | Lowest commercial names score |
| Eliminated | `deepseek/deepseek-v4-pro` | 0.519 | 0.931 | $0.040 | Lowest names, very slow (259s) |
| Eliminated | `moonshot/kimi-k2.6` | 0.694 | 0.966 | $0.066 | Adequate but no advantage |
| Failed | `alibaba/qwen3.6-plus` | 0.703 | 0.000 | — | 100% docs review structured output failure |

Projected cost at 10K sources: $659 (vs prior Sonnet→GPT→Opus: $2,578, **−83%**).

**GPT-5.x compatibility:** GPT-5.x models require `strict: false` JSON schema wrapping
(handled automatically in `llm.py`) and cannot use `temperature=0.0` (handled in benchmark).

### RD-Quorum Review

`sn review` uses a **Rational-Disagreement quorum** for high-confidence axis scores. Per batch: **Cycle 0** (primary, blind) scores with `models[0]`; **Cycle 1** (secondary, blind) re-scores with `models[1]` — blindness enforced (no `prior_reviews` block). For each item, any per-dimension diff > `disagreement-threshold` (default 0.15) marks it disputed. **Cycle 2** (escalator, context-aware) runs only if there are disputed items AND `len(models) == 3`; uses both prior critiques and is authoritative for those items. Reviews persist immediately after each cycle (crash-safety). Partial-failure ladder: both missing → retry → `retry_item` (quarantine); one missing → `single_review`.

**`resolution_method` values on Review nodes:** `quorum_consensus` (cycles 0+1 agreed; final = mean) / `authoritative_escalation` (cycle 2 wins for disputed items) / `max_cycles_reached` (disagreement, no escalator) / `retry_item` / `single_review`. `update_review_aggregates` picks the most-recent winning group and mirrors final scores onto SN axis slots.

**Configuration:** `[tool.imas-codex.sn-review]` (`disagreement-threshold`, `max-cycles`, `active-profile`) + `[tool.imas-codex.sn-review.{names,docs}]` (`models = [cycle0, cycle1, cycle2]`). Profiles in `[tool.imas-codex.sn-review.names.profiles.*]`. Accessors: `get_sn_review_{names,docs}_models()`, `get_sn_review_max_cycles()`, `get_sn_review_disagreement_threshold()`, `get_sn_review_active_profile()`.

```toml
[tool.imas-codex.sn-review.names]
models = [
  "openrouter/deepseek/deepseek-v4-flash",           # cycle 0 primary (blind, cheapest)
  "openrouter/openai/gpt-5.4",                        # cycle 1 secondary (blind, fast)
  "openrouter/google/gemini-3.1-pro-preview",         # cycle 2 escalator (highest quality)
]
```

**Budget:** `review_pipeline` reserves `batch_cost × num_models × 1.3` upfront via `BudgetLease.reserve()`, then charges per cycle — prevents secondary-cost leaks on mid-cycle crashes.

### Structured fan-out

`imas_codex/standard_names/fanout/` runs bounded Proposer → Executor → Synthesizer fan-out for `refine_name`. The Proposer emits a closed-catalog `FanoutPlan` (Pydantic discriminated union on `fn_id`, all bounds enforced at parse time); a pure-Python executor runs it in parallel via `asyncio.to_thread`; the call-site's LLM call ingests the rendered evidence block. No agentic loop, no runtime function generation.

- **Default off** (`enabled=False` in `[tool.imas-codex.sn-fanout]`): `run_fanout()` is a true no-op.
- **One `GraphClient` per refine cycle** — worker passes its `gc` in; runners never instantiate their own.
- **Cost ownership:** Proposer call is charged to the caller's `BudgetLease` as a sub-event with `batch_id=fanout_run_id`; callers stamp the same id on the Synthesizer charge so the `Fanout` ↔ `LLMCost` graph join works.
- **`Fanout` node** is telemetry-only, runtime-written (like `LLMCost`); exempt from LinkML schema management.

### StandardName Lifecycle

Four lifecycle axes on each `StandardName`:

| Axis | States | Set by | Notes |
|---|---|---|---|
| `name_stage` / `docs_stage` | `pending → drafted → reviewed → {accepted \| refining → drafted \| exhausted \| superseded}` | Pool workers | Cross-pipeline gate: `GENERATE_DOCS` fires only when `name_stage='accepted'`. `refining` reverts to `reviewed` after `DEFAULT_ORPHAN_SWEEP_TIMEOUT_S` (600 s). `chain_length` / `docs_chain_length` track refinement depth (root = 0). `superseded` = predecessor in `REFINED_FROM` chain; source edges migrate to the latest. |
| `pipeline_status` | `drafted → published → accepted` | `sn run` → `sn export` → `sn import` | Catalog round-trip state. |
| `status` | `draft → published → deprecated` | Catalog import | ISN vocabulary lifecycle; pipeline defaults to `draft`. Deprecated names link via `superseded_by` ↔ `deprecates`. |
| `validation_status` | `pending → valid \| quarantined` | Compose worker | Gates `sn review`, consolidation, and `sn export`. Critical failures (grammar round-trip, Pydantic, ambiguity) quarantine; semantic warnings persist in `validation_issues` but stay `valid`. **Review never demotes** — low-scoring names stay `valid` and route to refine pools. |

**`origin`:** `pipeline` (LLM-generated), `catalog_edit` (human-edited via catalog PR), or `derived` (deterministic structural parent created by the admission gate in `parents.py` — see Derived Parents below). `filter_protected()` skips `PROTECTED_FIELDS` on `catalog_edit` names unless overridden via `sn run --override-edits <name>`. PR provenance fields: `catalog_pr_number`, `catalog_pr_url`, `catalog_commit_sha`; round-trip timestamps: `exported_at`, `imported_at`.

**Derived parent lifecycle.** Structural parent SNs inferred from the ISN grammar peel enter with `origin='derived'`, `name_stage='pending'`. They are only created when the two-clause admission gate passes: Clause A — the parsed IR has at least one qualifier, operator, projection, locus, or mechanism (filters bare-base scalars like `pressure`, `density`); Clause B — the name already has ≥2 distinct-axis `projection` children in the graph (admits true vectors like `magnetic_field`). Names that fail both clauses are never created; their children remain parentless and group via shared grammar fields (`physical_base`, `qualifier`) in Cypher queries. Once created, `origin='derived'` names flow through the standard `REVIEW_NAME → GENERATE_DOCS → REVIEW_DOCS` path. Phase 4+ adds a `specificity` scoring dimension (gated on `origin='derived'`) — a failing specificity score routes the name to delete (Phase 6), not refine. Advisory slots `proposed_verdict` (admit \| reject \| uncertain) and `specificity_score` (float) track the verdict before deletion is enabled.

### StandardNameSource Lifecycle

`StandardNameSource` nodes track individual DD path / facility signal extraction through the pipeline. Written by the extract worker, updated by the compose worker.

```
extracted → composed | attached | vocab_gap | failed | stale
```

- **extracted**: Path queued for composition (written by extract worker)
- **composed**: LLM generated a new standard name for this source
- **attached**: Source auto-attached to an existing standard name (no LLM call needed)
- **vocab_gap**: Grammar vocabulary gap prevented naming this source
- **failed**: Composition failed (LLM error, validation rejection)
- **stale**: Source no longer exists in DD/signals graph (set by reconcile phase of `sn run`)

**ID format**: `dd:{full_dd_path}` or `signals:{facility}:{signal_id}` — the `dd:` prefix is the canonical URI scheme for DD sources (e.g. `dd:equilibrium/time_slice/profiles_1d/psi`).

**Reconciliation**: The reconcile phase (`sn run --only reconcile`) detects StandardNameSource nodes whose backing DD path or facility signal no longer exists in the graph and marks them `stale`.

### Reset and Clear Semantics

- **`sn run --reset-to {drafted|extracted}`** — Re-processes existing nodes (`drafted`) or clears matching SN nodes for a full re-run (`extracted`). Clears transient fields (embedding, model, generated_at) and removes `HAS_STANDARD_NAME`/`HAS_UNIT` edges. Scoped to `--source`; narrow further with `--since`, `--before`, `--below-score`, `--tier`, `--retry-quarantined`.
- **`sn run --reset-only`** — Performs the `--reset-to` cleanup then exits. Requires `--reset-to`.
- **`sn run --from-model <substring>`** — Provenance-based filter; re-generates names produced by a specific model.
- **`sn clear`** — Full-subsystem wipe (SN + Review + StandardNameSource + VocabGap + SNRun + grammar tree) with auto re-seed from ISN. No scoping — use `sn prune` for targeted deletes.
- **`sn prune`** — Relationship-first safe delete. Requires `--status` or `--all`; `--include-sources` to also drop StandardNameSource nodes.
- **`sn sync-grammar`** — Idempotent grammar re-sync; auto-run by `sn clear`, manual after ISN version bumps.

**Chain history is permanent.** `--reset-to` leaves `REFINED_FROM` chains and `DocsRevision` snapshots in place. **Safety guard:** `--reset-to` and `sn prune` require `--include-accepted` to touch `pipeline_status=accepted` names (catalog-authoritative). `sn clear` has no such guard.

**Loop semantics.** Pools run concurrently weighted by `POOL_WEIGHTS` (see [Pipeline](#pipeline)). Stops on zero eligible work, `--cost-limit` exhausted, or per-pool admission threshold. `--min-score F` routes below-threshold names/docs to refine pools; `--rotation-cap N` (default 3) caps chain depth; `--escalation-model` fires on the final rotation attempt. `--cost-limit` is a **single shared budget pool** across all six pools — per-category spend reported on the `SNRun` node. On `Ctrl-C`, writes an audit `SNRun` (`cost`, pool counters, `min_score`, `rotation_cap`, `stop_reason`); `sn status` surfaces the most recent run.

**Cost is graph-backed** via `LLMCost` nodes written async by `BudgetManager`. `SNRun.status`: `started → completed | interrupted | failed | degraded`. The only charge API is `lease.charge_event(cost, event)` (soft, never raises). Start `BudgetManager` with `await shared_mgr.start()`; finalize each run with `drain_pending()` + `get_total_spent()` in a `finally` block.

### Write Semantics

- **`write_standard_names()` (build path):** `coalesce(b.field, sn.field)` for all fields — passing `None` preserves existing data. Also persists `validation_issues` and `validation_layer_summary`.
- **`_write_catalog_entries()` (import path):** Catalog fields SET directly (catalog is authoritative). Graph-only fields (embedding, model, generated_at) preserved via coalesce.
- **Review write path:** Each RD-quorum cycle persists its `Review` nodes immediately. After all cycles, `update_review_aggregates` mirrors final axis scores onto SN slots (`reviewer_score_name`/`reviewer_score_docs`).

Both build and import paths call shared `_write_standard_name_edges(gc, names)` (in `graph_ops.py`) as a tail pass after node MERGE.

### StandardName Graph Edges

All structural edges for `StandardName` nodes are emitted by the shared
`_write_standard_name_edges(gc, names)` helper (tail pass — after primary MERGE).
Forward-reference targets are MERGEd as bare placeholder nodes.

| Edge | From | To | Source field / Derivation |
|------|------|-----|--------------------------|
| `HAS_ARGUMENT` | derived `StandardName` | parent `StandardName` | ISN parser: outermost unary prefix/postfix or projection layer; `{operator, operator_kind, [role, separator, axis, shape]}` |
| `HAS_PARENT` | child `StandardName` | parent `StandardName` | ISN grammar peel: child belongs to a family headed by the parent. Direction child→parent, matching the `(IMASNode)-[:HAS_PARENT]->(IMASNode)` convention from `imas_dd.yaml`. Edge props: `{operator, operator_kind, [role, separator, axis, shape]}`. Only emitted when the parent passes the admission gate (`_filter_admissible_parents` in `graph_ops.py`). Renamed from `COMPONENT_OF` in Phase 7. |
| `MAGNITUDE_OF` | magnitude `StandardName` | vector `StandardName` | Passive — emitted by `_emit_magnitude_of_edges` only when a DD/signal-sourced SN composes to `magnitude_of_<X>` AND `<X>` is an admitted `kind='vector'` parent. Never created speculatively. Algebraic-sibling relationship (not hierarchical like `HAS_PARENT`). |
| `HAS_ERROR` | inner `StandardName` | uncertainty sibling | ISN parser: `upper_uncertainty` / `lower_uncertainty` / `uncertainty_index` prefix; direction inverted; `{error_type ∈ upper\|lower\|index}` |
| `HAS_PREDECESSOR` | `StandardName` | predecessor | `predecessor` field (pipeline) or `deprecates` field (catalog import) |
| `HAS_SUCCESSOR` | `StandardName` | successor | `successor` field (pipeline) or `superseded_by` field (catalog import) |
| `IN_CLUSTER` | `StandardName` | `IMASSemanticCluster` | `primary_cluster_id` field |
| `HAS_PHYSICS_DOMAIN` | `StandardName` | `PhysicsDomain` | `physics_domain` field (slug) → singleton seeded at graph init |
| `HAS_UNIT` | `StandardName` | `Unit` | `unit` field — written by both paths (existing) |
| `HAS_COCOS` | `StandardName` | `COCOS` | `cocos` integer — written by pipeline path (existing) |
| `REFINED_FROM` | new `StandardName` | predecessor `StandardName` | Set by `persist_refined_name`; marks the chain. Source edges (`PRODUCED_NAME`, `HAS_STANDARD_NAME`) migrate to the new SN in the same transaction |
| `DOCS_REVISION_OF` | `StandardName` | `DocsRevision` | Set by `persist_refined_docs`; snapshots prior docs+reviewer state before in-place rewrite |

`HAS_ARGUMENT`/`HAS_ERROR` derivation lives in `imas_codex/standard_names/derivation.py` — exports `derive_edges(name) -> list[DerivedEdge]` (pure logic). Each name is peeled one layer only; the inner name runs its own derivation when written. Unparseable names silently produce no derived edges.

### PR-driven round-trip

The graph is authoritative for pipeline state; the catalog is authoritative for human-reviewed editorial fields. The round-trip flow:

```
sn export → sn preview → sn release -m "msg" → GitHub Pages / PR review → PR merged → sn import
```

1. **`sn export`** — Reads validated StandardName nodes from graph, applies quality gates (reviewer_score_name ≥ 0.65 + description sub-score), writes YAML to `<staging>/standard_names/<domain>/<name>.yml`. Uses default staging dir (`~/.cache/imas-codex/staging`) unless `--staging` is specified.
2. **`sn preview`** — Auto-exports from graph and launches a local MkDocs dev server using ISN's CatalogRenderer API. Use `--no-export` to serve an existing staging dir. Press Ctrl-C to stop. SSH tunnel: `ssh -L 8000:localhost:8000 <host>`.
3. **`sn release -m "msg"`** — Auto-exports from graph, copies staging YAML into ISNC git checkout, commits, tags with next semver version, and pushes to remote. RC releases go to origin (fork) by default; final releases (`--final`) go to upstream. The tag push triggers ISNC CI which deploys to GitHub Pages. For custom export filtering, run `sn export` first, then use `--skip-export`.
4. **User reviews the PR on GitHub** — edits description, documentation, tags, kind, links, status, etc.
5. **PR merged to ISNC main.**
6. **`sn import`** — Reads YAML from ISNC (auto-discovered from `isnc-dir` config, or pass `--isnc` explicitly). Diffs against graph. If any `PROTECTED_FIELDS` were edited, flips `origin=catalog_edit` on that name. Subsequent pipeline runs preserve these edits.

**Protection model:** `PROTECTED_FIELDS` = {description, documentation, kind, tags, links, status, deprecates, superseded_by, validity_domain, constraints}. Pipeline writers call `filter_protected()` before graph writes — names with `origin=catalog_edit` have these fields stripped from pipeline updates unless `override=True` or the name is in `override_names`.

**Idempotent re-run:** `ImportWatermark` (singleton) records the last imported `catalog_commit_sha`; `ImportLock` (singleton) prevents concurrent imports. Together they ensure `sn import` can be re-run safely.

**Override escape hatch:** `sn run --override-edits <name>` (repeatable) lets the pipeline overwrite catalog-edited fields for specific names — use when intentional re-generation should replace human edits.

### MCP Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `search_standard_names` | Semantic + keyword search over StandardName descriptions | `query`, `kind`, `physics_domain`, `pipeline_status`, `cocos_type`, `k`, `physical_base`, `subject`, `component`, `coordinate`, `position`, `process`, `region`, `geometric_base`, `device` |
| `fetch_standard_names` | Fetch full entries by name ID | `names` (space/comma separated) |
| `list_standard_names` | List with optional filters | `physics_domain`, `kind`, `pipeline_status`, `cocos_type` |
| `list_grammar_vocabulary` | Distinct tokens + usage counts for a grammar segment | `segment` (one of: `physical_base`, `subject`, `component`, `coordinate`, `position`, `process`, `geometry`, `object`, `geometric_base`, `device`, `region`, `qualifier`) |

Grammar-segment filters (`physical_base`, `subject`, …) on `search_standard_names` match exactly against the parsed `sn.<segment>` property. Use `list_grammar_vocabulary` to discover valid tokens before filtering. The `physics_domain` filter (canonical scalar + `source_domains` list) is pushed into Cypher in all three search branches (segment-filter, vector, keyword), so it does not lose results below `LIMIT $k`.

**All grammar segments are closed.** All segments including `physical_base`
(100 tokens) have a closed vocabulary defined in ISN's `SEGMENT_TOKEN_MAP`.
VocabGap reports on pseudo segments like `grammar_ambiguity` are filtered
out at write time (`imas_codex.standard_names.segments.filter_closed_segment_gaps`).
Reviewers audit the `physical_base` slot via the decomposition rule
(see `sn_review_criteria.yaml` I4.6).

### Schema

StandardName and StandardNameSource nodes defined in `imas_codex/schemas/standard_name.yaml`. Key relationships:

- `(IMASNode)-[:HAS_STANDARD_NAME]->(StandardName)`
- `(FacilitySignal)-[:HAS_STANDARD_NAME]->(StandardName)`
- `(StandardName)-[:HAS_UNIT]->(Unit)`
- `(StandardName)-[:HAS_COCOS]->(COCOS)`
- `(StandardNameSource)-[:FROM_DD_PATH]->(IMASNode)` — DD-sourced extraction tracking
- `(StandardNameSource)-[:FROM_SIGNAL]->(FacilitySignal)` — signal-sourced extraction tracking
- `(StandardNameSource)-[:PRODUCED_NAME]->(StandardName)` — links source to result
- `(StandardName)-[:REFINED_FROM]->(StandardName)` — predecessor in name refinement chain
- `(StandardName)-[:DOCS_REVISION_OF]->(DocsRevision)` — prior docs snapshot before in-place rewrite
- `(StandardName)-[:HAS_PARENT]->(StandardName)` — child belongs to a structural parent family; direction matches `(IMASNode)-[:HAS_PARENT]->(IMASNode)` convention; only created when parent passes admission gate
- `(StandardName)-[:MAGNITUDE_OF]->(StandardName)` — scalar magnitude SN derived from a vector parent; passive (source-driven, never speculative)

**COCOS / physics_domain / units (DD-authoritative, injected post-LLM):** `unit` flows from the DD `HAS_UNIT` relationship; `cocos` (int → COCOS singleton) and `cocos_transformation_type` (e.g. `psi_like`, `ip_like`) record convention behaviour; `dd_version` records the DD snapshot. `physics_domain` comes directly from `IMASNode.physics_domain` (falls back to `"general"` if absent). The LLM never fills any of these.

**Axis-split review storage:** name and docs reviews persist to independent column families so a docs pass cannot clobber name-only data and vice-versa. Each axis has paired slots: `reviewer_score_{axis}`, `reviewer_scores_{axis}` (per-dim JSON), `reviewer_comments_{axis}`, `reviewer_comments_per_dim_{axis}`, `reviewer_model_{axis}`. Same-axis re-review requires `--force`; guard is `_axis_overwrite_blocked` in `review/pipeline.py`.

**Score-canonical policy:** the rubric-driven numeric `score` (0–1) is the sole accept/refine signal. No separate `verdict` field exists — the reviewer LLM produces scores plus optional `revised_name` / `suggested_name`.

**RD-quorum fields on `Review`:** `review_axis`, `cycle_index` (0/1/2), `review_group_id` (UUID), `resolution_role` (primary/secondary/escalator), `resolution_method` (see [RD-Quorum Review](#rd-quorum-review)). `id` format: `{sn_id}:{axis}:{review_group_id}:{cycle_index}`.

**Provenance / catalog / pool fields:** `vocab_gap_detail`, `validation_issues`, `validation_layer_summary` (JSON); catalog round-trip: `pipeline_status`, `status`, `origin`, `deprecates`, `superseded_by`, `catalog_pr_number`, `catalog_pr_url`, `catalog_commit_sha`, `exported_at`, `imported_at`; pool state: `name_stage`, `docs_stage`, `chain_length`, `docs_chain_length`, `refine_name_escalated_at`, `refine_docs_escalated_at`.

**VocabGap nodes** record missing grammar tokens from composition. Linked via `HAS_SN_VOCAB_GAP` from `IMASNode`/`FacilitySignal`; carry segment, needed token, and reason. Use `sn gaps` for a table view or `sn gaps --export yaml` to produce vocabulary issues for ISN.

### Architecture Boundary

ISN owns grammar, vocabulary, and validation. Codex owns the pipeline, evaluation, and graph persistence. Full details in `docs/architecture/boundary.md`.

**Import boundary** (ISN ≥0.8.0rc7): `get_grammar_context()` (single entry point, 19 keys), `create_standard_name_entry()` (Pydantic, 18 validators), `run_semantic_checks()` (9 checks), `validate_description()`, `parse_standard_name()` / `compose_standard_name()` (round-trip). Never import from ISN private modules. Never hardcode grammar rules — pull from `get_grammar_context()`. Review criteria live in codex (`sn_review_criteria.yaml`).

### Grammar Vocabulary

Closed-vocabulary system: **physical bases** (~78, irreducible dimensional quantities, CI-gated), **qualifiers** (~92, prefix modifiers stripped recursively by the parser), **processes** (~90, suffix-only via `_due_to_{token}`).

**Composition order:** `{subject}_{qualifier1}_{qualifier2}_{physical_base}_{due_to_process}`

**Update rules:** (1) Never add compounds to `physical_bases` — use qualifiers. (2) Qualifier order is insertion-order, preserved through round-trip. (3) Subjects win over qualifiers (parser stage 3 before stage 5). (4) Process tokens as prefixes are qualifiers, not processes. When rotations surface gaps, follow [Vocabulary Rotation](#vocabulary-rotation-isn-fork-rc-workflow). Never hardcode vocabulary tokens in Python.

### Vocabulary Rotation: ISN Fork RC Workflow

When rotations surface `VocabGap` nodes blocked by the current ISN vocabulary, add tokens on the fork (`~/Code/imas-standard-names` → `Simon-McIntosh/IMAS-Standard-Names`) and cut an RC release. Upstream is `iterorganization/IMAS-Standard-Names`; the dep in imas-codex pins to a git tag on the fork, so RC tags on origin are sufficient.

**Vocab-addition rules** (apply BEFORE editing YAML):

- Every legitimate physics quantity deserves a token regardless of frequency — `ejima_coefficient` is as valid as `temperature`. Classify each VocabGap first: TRUE_GAP (add), COMPOSE_ERROR (fix the prompt instead), REJECT (not a valid grammar concept).
- Single-word base preferred; physical-quantity semantics only; no overlap with existing tokens; not a unit or geometry primitive.
- **No compound tokens that subsume `physical_base` words.** `trapped_particle` as a subject greedily consumes "particle" and breaks grouping with `particle_density`. Use `trapped` alone.
- **Prefer atomic qualifiers.** Orbit class (`trapped`, `co_passing`) and species (`fast_particle`, `electron`) are independent axes — never combine into one token.
- **Grouping invariant:** all orbit/species variants of the same physical quantity MUST parse to the same `physical_base`. If they don't, the token is wrong.
- **Round-trip every existing name** containing the candidate string: `assert compose_standard_name(parse_standard_name(name)) == name`.

**Release procedure:**

```bash
cd ~/Code/imas-standard-names
# Edit vocabulary YAML, run tests, commit to main:
uv run pytest && git push origin main

# Cut an RC via the state-machine CLI (NEVER tag manually):
uv run standard-names release status            # inspect state
uv run standard-names release -m "feat: ..."    # increment RC (e.g. v0.8.0rc7 → v0.8.0rc8)
# Or --bump minor/major to start a new series; --final to finalize to upstream.

# In imas-codex: bump dep (appears twice in pyproject.toml), sync, test, push
cd ~/Code/imas-codex
sed -i 's|@v0\.8\.0rc[0-9]\+|@v0.8.0rc<NN>|g' pyproject.toml
uv sync && uv run pytest tests/standard_names/ -x -q
git commit -am "deps: bump imas-standard-names to v0.8.0rc<NN>" && git push origin main
```

**Multi-RC chains are normal during bootstrap.** Re-rotate vocab-bound domains after each bump to measure lift.

### Prompt Context Injection

Four context channels are injected per-item into compose, review, and enrich prompts:

1. **Hybrid DD search neighbours** — concept-similar DD paths found via vector similarity + keyword search (`_hybrid_search_neighbours` in `workers.py`, backed by `hybrid_dd_search` in `dd_search.py`). Injected as `hybrid_neighbours` per item.
2. **Related DD paths** — cross-IDS structural siblings via explicit graph relationships (cluster membership, shared coordinates, matching units, identifier schemas, COCOS transformation type). Injected as `related_neighbours` per item via `_related_path_neighbours`.
3. **Error companions** — uncertainty/error fields (`_error_upper`, `_error_lower`, `_error_index`) associated with each DD path. Injected as `error_fields` per item.
4. **Identifier enum values** — when a DD path references an identifier schema, the allowed enumeration values (name, index, description) are injected as `identifier_values` per item.

**Compose retry with expanded context:** On grammar/validation failure, the compose worker retries up to `retry_attempts` times (default 1), re-enriching items with expanded hybrid search (`search_k=retry_k_expansion`, default 12) before resubmission. Configurable via `[tool.imas-codex.sn]` or `IMAS_CODEX_SN_RETRY_*` env vars.

**Path defaults** (`[tool.imas-codex.sn]` in pyproject.toml):
- `staging-dir` — local staging directory used by `sn export`, `sn preview`, and `sn release` (default: `~/.cache/imas-codex/staging`).
- `isnc-dir` — path to a local ISNC git checkout used by `sn release` and `sn import` (default: `""`, i.e. must be provided via flag if not set).

**Scored-example injection:** Compose and review prompts include dynamically selected exemplar StandardName nodes at target score thresholds `(1.0, 0.8, 0.65, 0.4)`. Examples are graph-backed and selected by the example loader (W3-K3K4); `benchmark_calibration.yaml` and the static calibration code path have been removed. Context keys: `compose_scored_examples`, `review_scored_examples`.

### Prompt Infrastructure

Compose and review prompts use shared fragments via `{% include %}`:
- `{% include "sn/_grammar_reference.md" %}` — grammar vocabulary and segment order (used in `compose_system.md`)
- `llm/prompts/shared/sn/_scoring_rubric.md` — 6-dimension scoring rubric (shared reference)
- `llm/config/sn_review_criteria.yaml` — scoring dimensions and tiers (loaded via `load_prompt_config()`)
- ISN context keys (`quick_start`, `common_patterns`, `critical_distinctions`) rendered in compose prompt

**Axis review prompts:** `review_names.md` (4-dim rubric: grammar/semantic/convention/completeness)
and `review_docs.md` (4-dim rubric: description_quality/documentation_quality/completeness/physics_accuracy).
Both share a `{% if prior_reviews %}...{% endif %}` block that is only rendered for the cycle-2
escalator (context-aware). Cycles 0 and 1 never receive this block (blindness enforced).

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

**Remote Python — two-interpreter architecture:**

- `run_python_script()` / `async_run_python_script()` — venv `python3` (3.12+) via `_REMOTE_PATH_PREFIX`. Modern syntax OK (`X | Y`, `match`).
- `SSHWorkerPool` / `pooled_run_python_script()` — hardcoded `/usr/bin/python3` (3.9+, stdlib-only) to avoid 60–100s NFS venv startup. **No 3.10+ syntax** in pool scripts. Each script declares its Python version in a docstring header. Ruff skips type-hint modernization for `imas_codex/remote/scripts/*` (see per-file ignores).

**Remote zombie prevention:** every executor function wraps the SSH command with server-side `timeout <local_timeout + 5s>` so the remote process self-terminates when the local SSH client is killed. Never construct raw SSH calls — always use the executor functions.

## Commit Workflow

Follow the Pre-Commit Hook Policy in `~/.agents/AGENTS.md` (ruff `--fix` + `format` before staging, conventional commits, no `git add -A`). Breaking changes use `BREAKING CHANGE:` footer, not `type!:` suffix.

**Never stage in this repo:** auto-generated files (`models.py`, `dd_models.py`, `config/models.py`, `agents/schema-reference.md`, `schema_context_data.py`), `*_private.yaml`, anything in `.gitignore`.

### Worktrees

Commits in worktrees are NOT on `main` until merged. Always merge immediately:

```bash
WORKTREE_HEAD=$(git rev-parse HEAD)
cd /home/ITER/mcintos/Code/imas-codex
git merge --no-ff $WORKTREE_HEAD -m "merge: worktree changes for <description>"
git push origin main
```

### Sub-Agent Model Selection

Model selection follows `~/.agents/AGENTS.md` (3-tier: Opus 4.6 top, Sonnet 4.6 floor, Haiku 4.6 menial-only). Repo-specific additions:

- **EFIT++ campaigns, solver/numerics/physics analysis, schema redesigns, cross-cutting refactors** → always `claude-opus-4.6`.
- **Rubber-duck reviews** of complex designs → `claude-opus-4.6` (weak critic produces weak critiques).
- **LLM-pipeline model choices** (standard names, DD enrichment, etc.) are governed by `pyproject.toml` `[tool.imas-codex.*]` sections — independent of sub-agent model policy.

### Parallel Agents

Multiple agents may edit this repo simultaneously on `main`. Assume another agent is doing so right now.

**Verify before modifying:** re-read files from disk (your in-memory view may be hours old); check `git log --oneline -5 -- <file>` for unfamiliar commits. If you see unfamiliar names/imports, assume they are correct — don't revert.

**Banned destructive commands:** see `~/.agents/AGENTS.md` for the table and stash ban. Auto-generated files (`models.py`, `dd_models.py`, `schema_context_data.py`) are gitignored but make the worktree look dirty — never stage and never `git restore` them (which is also why merge, not rebase, is the pull policy).

**Pre-existing test failures:** stash-free verification via `git log --since="1 day ago" -- <test>` and `git show HEAD:<test>`; trust the failure timestamp. File a blocker todo and scope your work around it.

**Dispatch preamble:** use the one in `~/.agents/AGENTS.md` with `{BRANCH}=main`.

**Session hygiene:** close sessions when done (`ctrl+d`/`/exit`); audit `ps aux | grep copilot` and kill stale processes — idle agents with old context are the #1 cause of regressions.

**Session completion is mandatory:** every response that modifies files MUST end with `git add` → `git commit` → `git push` plus a brief summary of the commit.

## Feature Plan Documentation

Plans live in `plans/features/`. Lifecycle: `features/<name>.md` (active) → `features/pending/<name>.md` (partially implemented, gaps documented) → **DELETE** (fully implemented — the code is the documentation). Gap docs (`gaps-*.md`) consolidate remaining work across related pending plans.

**Every plan must have a "Documentation Updates" section** listing which targets need updates: `AGENTS.md` (new CLI/MCP/config/workflows), `README.md` (user-facing), `plans/README.md` (status), `.claude/skills/*.md`, `.claude/agents/*.md`, `docs/` (mature architecture), prompt templates, schema reference (auto via `uv run build-models`).

**Self-consistency rule:** a feature is not done until code is committed + tested, every applicable doc target is updated, `plans/README.md` reflects the new status, and the plan file is deleted or moved to `pending/`.

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

## CLI Logs

All discovery and DD CLI commands write DEBUG-level rotating logs to disk. The rich progress display suppresses most log output to keep the TUI clean, but full details are always available in the log files.

**Log directory:** `~/.local/share/imas-codex/logs/`

**Log naming:** `{command}_{facility}.log` (e.g. `paths_tcv.log`, `wiki_jet.log`, `imas_dd.log`). Logs rotate at 10 MB with 3 backups.

```bash
tail -f ~/.local/share/imas-codex/logs/paths_tcv.log  # Follow live
rg "ERROR|WARNING" ~/.local/share/imas-codex/logs/     # Find errors
```

(The no-pipe rule from [Command Execution](#command-execution) applies here too — logs are already on disk; never redirect CLI output.)

## Testing

```bash
uv sync --extra test          # Required in worktrees
uv run pytest                 # Default markers: excludes slow, graph
uv run pytest tests/standard_names/ -q  # SN tests (~3300 tests, ~90s)
uv run pytest tests/path/to/test.py::test_function  # Specific test
uv run pytest --cov=imas_codex  # With coverage
```

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

## Services

**Neo4j graph and the embedding server are always running** as SLURM jobs on all dev machines (ITER, WSL). Assume both are available. If a service is down, restart it — don't work around it. Always connect via the Python client methods (`GraphClient`, `Encoder`) — never raw HTTP/bolt; they handle SLURM node discovery, tunnel setup, auth from `.env`, and retries.

**SLURM-only rule.** Both services MUST run as SLURM jobs — never bypass with `nohup`, `ssh … &`, `screen`, `tmux`, or anything else. SLURM provides cgroup isolation, clean lifecycle (`scancel`), accounting, and drain cleanup. Rogue processes cause "Duplicate jobid" errors that drain nodes for all users. If SLURM won't schedule, get the node resumed (`scontrol update NodeName=<node> State=RESUME`) — don't work around it.

### Embedding server

Config: `[tool.imas-codex.embedding]`. `get_embedding_location()` returns the facility or `"local"`. Port = `18765 + offset` in the shared `locations` list.

```bash
imas-codex embed start [-g 2]    # Start (optionally with N GPUs)
imas-codex embed status          # Health + SLURM job + node state
imas-codex embed restart -g 8    # Restart with 8 GPUs (~18s cycle)
imas-codex embed stop            # Stop SLURM job + cleanup rogue processes
imas-codex embed logs            # View SLURM logs
imas-codex embed service install # Install systemd service (login node only)
```

Troubleshooting: `embed status` shows node state. Common: node draining → ask admin to RESUME; rogue process → `embed stop` kills it; package issue → check `embed logs` and `uv sync` on node; timeouts → check tunnel (`lsof -i :18765`).

### Neo4j connection

On ITER login/compute nodes, `GraphClient()` (no args) discovers the SLURM compute node and connects directly — never hardcode `bolt://localhost:7687`:

```python
from imas_codex.graph.client import GraphClient
gc = GraphClient()    # handles SLURM, tunnels, env overrides
```

From WSL/remote, start a tunnel first: `imas-codex tunnel start iter` then `tunnel status`. The profile system auto-tunnels for remote hosts. Override with `export IMAS_CODEX_TUNNEL_BOLT_ITER=17687` if needed.

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

## MCP Server Deployment

```bash
# Default: full mode with all tools (REPL, search, write)
uv run imas-codex serve

# DD-only mode: DD tools only, implies read-only (container deployments)
uv run imas-codex serve --dd-only --transport streamable-http

# Read-only: search and read tools only
uv run imas-codex serve --read-only --transport streamable-http

# STDIO transport for MCP clients (VS Code, Claude Desktop)
uv run imas-codex serve --transport stdio
```

**Deployment topology:**

| Deployment | Command | Tools Available |
|------------|---------|-----------------|
| Development | `imas-codex serve` | All (REPL, search, write, infrastructure) |
| DD-only container | `imas-codex serve --dd-only` | DD search and read only |
| Public / read-only | `imas-codex serve --read-only` | Search and read only |
| MCP STDIO client | `imas-codex serve --transport stdio` | All (inside the calling process) |

## Fallback: MCP Server Not Running

```bash
uv run imas-codex graph status          # Graph operations
uv run imas-codex graph shell           # Interactive Cypher
uv run imas-codex llm status            # LLM proxy status (lightweight, no API calls)
uv run imas-codex llm status --deep     # Full model health check (makes real LLM API calls — billable)
uv run pytest                           # Testing
```

Automated health checks (e.g., Azure `/health/readiness`) make no LLM API calls and incur no token cost. Only `imas-codex llm status --deep` exercises the model endpoint and is billable.
