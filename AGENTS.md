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
| `[sn-fanout]` | Structured fan-out (Proposer/Executor/Synthesizer) | — |
| `[sn-review]` | Shared RD-quorum settings (disagreement threshold, max cycles, active profile) | `get_sn_review_disagreement_threshold()`, `get_sn_review_max_cycles()`, `get_sn_review_active_profile()` |
| `[sn-review.names]` / `[sn-review.docs]` | Reviewer model chain per axis (1–3 models) | `get_sn_review_names_models()`, `get_sn_review_docs_models()` |
| `[sn-review.names.profiles.*]` | Named profiles (default, opus-only, quality-cost-balanced) | — |
| `[sn-benchmark]` | SN benchmark compose-models list and reviewer-model | `get_sn_benchmark_compose_models()`, `get_sn_benchmark_reviewer_model()` |

**Model access:** `get_model(section)` is the single entry point for all model lookups. Pass the pyproject.toml section name directly: `"language"`, `"vision"`, `"reasoning"`, or `"embedding"`. Priority: section env var → pyproject.toml config → default.

**Graph access:** Profiles separate **name** (what data) from **location** (where Neo4j runs). Default graph `"codex"` (all facilities + IMAS DD) runs at location `"iter"`. Select via `IMAS_CODEX_GRAPH` / `IMAS_CODEX_GRAPH_LOCATION`; each location maps to a unique bolt+HTTP port pair (iter 7687/7474, tcv 7688/7475, jt-60sa 7689/7476). `NEO4J_URI`/`NEO4J_USERNAME`/`NEO4J_PASSWORD` override any profile; `resolve_graph(name)` (`imas_codex.graph.profiles`) resolves directly; all CLI `graph` commands take `--graph/-g`. Full detail: `docs/architecture/graph-profiles.md`.

**Location-aware connections:** `is_local_host(host)` picks direct vs tunnel at connect time; for edge cases configure `login_nodes`/`local_hosts` in the facility's private YAML (`imas-codex config local-hosts`).

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
- **CRITICAL — gitignored, auto-generated, never commit (never stage even if `git status` shows them modified/untracked):** `imas_codex/graph/models.py`, `imas_codex/graph/dd_models.py`, `imas_codex/config/models.py`, `agents/schema-reference.md`, `imas_codex/graph/schema_context_data.py`

**PhysicsDomain enum**: Imported from the `imas-standard-names` PyPI package and re-exported from `imas_codex.core.physics_domain`. The canonical vocabulary is maintained in the imas-standard-names project. Contains 32 physics domain values. `imas_codex/core/physics_domain.py` is a hand-written one-line re-export — it IS committed and should NOT be treated as auto-generated.

**NodeCategory enum** (`imas_dd.yaml`): DD node classification — 9 values: `quantity`, `geometry`, `coordinate`, `metadata`, `error`, `structural`, `identifier`, `fit_artifact`, `representation`. Classifier lives in `imas_codex/core/node_classifier.py` (two-pass: Pass 1 attribute-only, Pass 2 graph-relational). Category sets for pipeline participation in `imas_codex/core/node_categories.py`.

**IMASNodeStatus lifecycle** (`imas_dd.yaml`): DD build pipeline `built → enriched → refined → embedded → classified` across seven workers EXTRACT → BUILD → ENRICH → REFINE → EMBED → CLASSIFY → CLUSTER. CLASSIFY (after EMBED) uses `get_model("language")` for three-tier domain assignment — LLM for physics paths; inheritance (`HAS_ERROR`/`HAS_PARENT`) for error/metadata; none for infra metadata (`ids_properties/*`, `code/*`). `"general"` paths retry with expanded cluster context; `--reset-to embedded` re-classifies only (~$2.60, cheapest domain fix).

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

**Serve modes:** `imas-codex serve` exposes all tools; `--read-only` suppresses write tools (`repl()`, `add_to_graph()`, `update_facility_config()`), leaving search/read only; `--dd-only` hides facility tools and **implies `--read-only`** (auto-detected from a DD-only graph). Full topology + transports in [MCP Server Deployment](#mcp-server-deployment).

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

Keep `OPENROUTER_API_KEY_IMAS_CODEX` set and use `openrouter/anthropic/<model>` (or unprefixed — `ensure_model_prefix()` normalizes) to get cost tracking + caching for free (empirically ~87% cheaper on a warm cache). Bypass logic in `imas_codex/discovery/base/llm.py`.

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

`imas-codex graph <cmd>` (`--help` for full list): lifecycle `start`/`stop`/`status`/`shell`/`clear`; archives `export [-f <facility>]`/`load`/`fetch`; GHCR `pull`/`push --dev`/`tags`/`prune --dev` (all take `--facility`); instances `init`/`switch`/`list`; `profiles`; `secure` (rotate password). Also `imas-codex tunnel start <host>`/`status` and `config private push` / `config secrets push <host>`.

Never use `DETACH DELETE` on production data without user confirmation. For re-embedding: update nodes in place, don't delete and recreate.

### Graph Migrations

Run migrations as inline Cypher via `imas-codex graph shell` or the MCP `repl()` (`query()`). Never create `scripts/migrate_*.py` or `repair_*.py`. For >10K-node migrations, batch with `LIMIT` to avoid transaction timeouts; verify counts before and after.

### LLMCost Node Properties

`LLMCost` nodes track per-call LLM spend. **All `LLMOperation`-mixin fields are prefixed with `llm_`** — never use bare `cost`, `model`, or `service`. Full property list is in `agents/schema-reference.md`; key fields: `llm_cost`, `llm_model`, `llm_service`, `llm_tokens_{in,out,cached_read,cached_write}`, grouping (`run_id`, `phase`, `pool`, `batch_id`, `for_run`), and `sn_ids`.

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

Nodes with `embedding` + `description` auto-get a quantized cosine vector index (~4× memory savings). `ensure_vector_indexes()` creates them, auto-detects dimension mismatches, and drops/recreates stale indexes — never hand-write `CREATE VECTOR INDEX`. Query with Neo4j 2026.01's native `SEARCH` clause (in-index pre-filtering). Index names and the full list are in `agents/schema-reference.md`.

### Semantic Search & Graph RAG

Use `semantic_search(text, index, k)` in the python REPL:

```python
# Document content (wiki, code)
semantic_search("COCOS sign conventions", index="wiki_chunk_embedding", k=5)

# Descriptive metadata (signals, paths - search by physics meaning)
semantic_search("plasma current measurement", index="facility_signal_desc_embedding", k=10)
```

Combine vector similarity with link traversal via the Cypher 25 `SEARCH` clause
(`MATCH (s:FacilitySignal) SEARCH s IN (VECTOR INDEX <name> FOR $embedding LIMIT
k) SCORE AS score WHERE … WITH s, score MATCH (s)-[:DATA_ACCESS]->…`). Use
`build_vector_search()` from `imas_codex.graph.vector_search` to generate SEARCH
clauses programmatically rather than hand-writing them. All WHERE conditions are
post-filters (in-index pre-filtering requires properties registered as
additional vector index properties).

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

> **Full reference:** [`docs/architecture/standard-names.md`](docs/architecture/standard-names.md)
> (pipeline, RD-quorum, fanout, derived parents, prompt architecture, graph
> edges, write semantics, CLI flag detail, benchmark results) and
> [`standard-names-decisions.md`](docs/architecture/standard-names-decisions.md)
> (rationale). This section is **orientation + tripwires only** — flags are in
> `--help`, schema in `agents/schema-reference.md`, live stats in `sn status`.

### Pipeline (six-pool `sn run` loop)

| Pool | Stage gate | Operation |
|------|------------|-----------|
| `GENERATE_NAME` | `StandardNameSource.status=pending` | LLM generates name; new SN at `name_stage='drafted'`. **Unit from DD, never LLM.** Runs the EXTRACT→COMPOSE→VALIDATE→CONSOLIDATE→PERSIST sub-pipeline. |
| `REVIEW_NAME` | `name_stage='drafted'` | RD-quorum scores → `accepted`/`reviewed`/`exhausted`. Derived parents add a `specificity` dim. |
| `REFINE_NAME` | `name_stage='reviewed' AND rsn<min AND chain_length<cap` | New SN node; predecessor `superseded`; `REFINED_FROM` edge; source edges migrate. |
| `GENERATE_DOCS` | `name_stage='accepted' AND docs_stage='pending'` | LLM docs → `docs_stage='drafted'`. Cross-gate: fires only after name accepted. |
| `REVIEW_DOCS` | `docs_stage='drafted'` | RD-quorum scores → `accepted`/`reviewed`/`exhausted`. |
| `REFINE_DOCS` | `docs_stage='reviewed' AND rds<min AND docs_chain_length<cap` | Rewrites docs in-place; prior snapshot on `DocsRevision` via `DOCS_REVISION_OF`. |

Pools run concurrently weighted by `POOL_WEIGHTS`. **Acceptance overrides cap**
(a passing score wins even at the final rotation). **Escalation:** the final
refine attempt switches to `--escalation-model` (default
`openrouter/anthropic/claude-opus-4.6`). **Backlog throttle:** refine_name
backlog > 0.5 × generate_name backlog dampens generate weight 0.5×.
`--cost-limit` is a single shared budget pool; `Ctrl-C` writes an audit `SNRun`.
Scope routing: `--only <phase>` (single phase, e.g. `--only reconcile`),
`--focus <path>` (specific paths through the full loop, UUID-scoped).

**Budget & run-completion discipline.** Mid-pipeline names (drafted,
unreviewed, docs pending) are NOT stranded or lost — they are normal durable
graph state that any subsequent `sn run` claims and continues. Design each
run so its cohort completes within its `-c` cap: size the cap to carry every
seeded name through review + docs (≈$0.10–0.15/name name-axis,
≈$0.30/name docs-axis at 2026-06 rates), especially for `--focus` rotations,
which should never need a follow-up. `--flush` is a GATE, not a recovery
tool: it blocks new work from entering (skips seeding + generate_name) so
the existing backlog drains — use it to converge the graph before an audit
or release cut, not as routine post-run cleanup. Never kill a running
campaign to "save money" — the cap already bounds spend; interrupt only when
the configuration underneath the run has been invalidated, and prefer letting
the cap expire.

**Tripwires** (the rest is reference — see the doc):

- **Unit safety:** units flow DD `HAS_UNIT` → EXTRACT → prompt (read-only) →
  worker injects → graph. The LLM never provides `unit`, `cocos`, or
  `physics_domain` — all DD-authoritative, injected post-LLM.
- **Score-canonical:** the numeric `score` (0–1) is the *sole* accept/refine
  signal. There is **no `verdict` field**; the reviewer emits scores +
  optional `revised_name`/`suggested_name`.
- **Chain history is permanent.** `--reset-to` leaves `REFINED_FROM` chains and
  `DocsRevision` snapshots in place.
- **Data-safety guard:** `sn run --reset-to` and `sn prune` require
  `--include-accepted` to touch `pipeline_status=accepted` (catalog-
  authoritative) names. `sn clear` has no guard — it wipes everything.
- **Review never demotes:** a low-scoring `valid` name stays `valid` and routes
  to a refine pool; it is not quarantined.
- **Import boundary (ISN ≥0.8.0rc7):** import only the public surface
  (`get_grammar_context()`, `create_standard_name_entry()`,
  `run_semantic_checks()`, `validate_description()`, `parse_standard_name()` /
  `compose_standard_name()`). Never import ISN private modules; never hardcode
  grammar rules or vocabulary tokens — pull from `get_grammar_context()`.
  Review criteria live in codex (`sn_review_criteria.yaml`). Boundary detail:
  `docs/architecture/boundary.md`.
- **Closed segments:** *all* grammar segments — including `physical_base` — are
  closed (ISN `SEGMENT_TOKEN_MAP`). A composer "missing token" report against
  `physical_base` is not a real gap; pseudo segments (`grammar_ambiguity`) are
  filtered at write time. When a true gap blocks naming, follow the vocab
  rotation workflow in the architecture doc (add tokens on the ISN fork, cut an
  RC, bump the dep — appears twice in `pyproject.toml`).

### CLI commands

`sn run` (six-pool loop), `review`, `export`, `preview`, `release`, `import`,
`status`, `coverage`, `gaps`, `clear`, `prune`, `sync-grammar`, `bench`. Run
`uv run imas-codex sn <cmd> --help` for flags; semantics and the full flag
matrix are in the architecture doc.

### Lifecycle axes

Four independent axes on each `StandardName` (full state tables in the doc):

| Axis | States | Driver |
|------|--------|--------|
| `name_stage` / `docs_stage` | `pending → drafted → reviewed → {accepted \| refining → drafted \| exhausted \| superseded}` | pool workers (`refining` reverts after 600 s orphan sweep) |
| `pipeline_status` | `drafted → published → accepted` | `sn run` → `export` → `import` (catalog round-trip) |
| `status` | `draft → active → {deprecated \| superseded}` | catalog import (ISN vocabulary lifecycle) |
| `validation_status` | `pending → valid \| quarantined` | compose worker (gates review/consolidation/export) |

`origin`: `pipeline` | `catalog_edit` (human-edited; `filter_protected()` skips
`PROTECTED_FIELDS` unless `--override-edits`) | `derived` (structural parent
from the `parents.py` admission gate). `StandardNameSource`:
`extracted → composed | attached | vocab_gap | failed | stale`; ID scheme
`dd:{path}` or `signals:{facility}:{id}`.

### Key modules

`pools.py` (pool specs + throttle) · `loop.py` (`run_sn_pools()`) · `workers.py`
(claim/process/persist) · `pool_adapter.py` (`--focus` seeding) ·
`enrichment.py` (cluster selection + global grouping) · `consolidation.py`
(dedup/conflicts) · `graph_ops.py` (writes, `_write_standard_name_edges`,
`persist_refined_*`) · `parents.py` (derived-parent gate) · `derivation.py`
(`HAS_ARGUMENT`/`HAS_ERROR`) · `defaults.py` (constants) · `review/pipeline.py`
(RD-quorum) · `fanout/` (refine_name fan-out) · `orphan_sweep.py`. SN-eligibility
is owned by DD `node_category`, pre-filtered via `SN_SOURCE_CATEGORIES` in
`imas_codex/core/node_categories.py`.

### Schema & MCP

Nodes in `imas_codex/schemas/standard_name.yaml`; all edges, properties, and
`Review`/`LLMCost` fields are in `agents/schema-reference.md` (auto-generated).
MCP read tools: `search_standard_names` (semantic + per-segment grammar
filters), `fetch_standard_names`, `list_standard_names`,
`list_grammar_vocabulary` (discover valid tokens before filtering).
Config sections (`[tool.imas-codex.sn*]`) and accessors are in the table at the
top of this file.

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
