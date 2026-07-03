# Standard Name Generation Pipeline

> DD-enriched standard name generation: classify IMAS paths, batch by physics
> concept, compose names via LLM, review with an RD-quorum, refine low-scoring
> names and docs, and persist to the knowledge graph. Design rationale lives in
> [`standard-names-decisions.md`](standard-names-decisions.md); the architecture
> boundary with ISN is in [`boundary.md`](boundary.md).

This is the detailed reference. `AGENTS.md` carries only the orientation
(pool table, command index, lifecycle axes, module map) and the safety
tripwires; everything below is the authoritative depth behind it.

## Overview

Standard names give each IMAS Data Dictionary path a canonical, human-readable
physics identity — e.g. `electron_temperature` for
`core_profiles/profiles_1d/electrons/temperature`. They bridge the structural
DD namespace and the semantic physics vocabulary, enabling cross-facility and
cross-IDS data discovery.

Names follow a **composable grammar** (from `imas_standard_names`, ISN) where a
name is built from ordered segments:

```
{subject}_{qualifier1}_{qualifier2}_{physical_base}_{due_to_process}
```

Examples: `electron_temperature`, `toroidal_magnetic_field_at_magnetic_axis`,
`ion_deuterium_density`. ISN owns the grammar, vocabulary, and validation;
codex owns the pipeline, evaluation, and graph persistence.

## Seven-Pool Run Loop

`sn run` drives a **single concurrent loop of seven pools**. Each pool claims
eligible work by a stage gate, does one unit of work, and persists a durable
state transition. Pools run concurrently, weighted by `POOL_WEIGHTS`
(`imas_codex/standard_names/pools.py`); the loop orchestrator is
`run_sn_pools()` in `loop.py`.

| Pool | Label | Stage gate | Key operation |
|------|-------|------------|---------------|
| 1 | `GENERATE_NAME` | `StandardNameSource.status=pending` | LLM generates a name; new SN persisted at `name_stage='drafted'`. **Unit injected from DD — never from LLM.** |
| 2 | `ENRICH_PARENTS` | `origin='derived' AND description=placeholder AND has live child` | LLM synthesizes a real description for a placeholder derived parent by **generalizing over its children**, embeds it locally, and **accepts it structurally** (`name_stage → 'accepted'`, score inherited from accepted children — skips REVIEW_NAME). Breaks the derived-parent coverage deadlock (see [Derived / Structural Parents](#derived--structural-parents)). Childless derived parents are legitimately unscoped and skipped. Model: `get_model("sn-parent-enrich")` (compose-tier). |
| 3 | `REVIEW_NAME` | `name_stage='drafted'` | RD-quorum scores the name; atomic transition → `accepted` (rsn≥min) / `reviewed` / `exhausted`. Also admits `origin='derived'` parents via the admissibility gate; inadmissible accepted parents are deleted by `normalize_derived_parent_lifecycle` at startup. |
| 4 | `REFINE_NAME` | `name_stage='reviewed' AND rsn<min AND chain_length<cap` | Creates a NEW SN node; predecessor flipped to `superseded`; source edges migrated; `REFINED_FROM` edge added. |
| 5 | `GENERATE_DOCS` | `name_stage='accepted' AND docs_stage='pending'` | LLM generates documentation; `docs_stage → 'drafted'`. Cross-pipeline gate: fires only after the name is accepted. |
| 6 | `REVIEW_DOCS` | `docs_stage='drafted'` | RD-quorum scores docs; atomic transition → `accepted` / `reviewed` / `exhausted`. **On promotion to `accepted`, the doc's bare `[name]` brackets are normalized at source** (`_normalize_bare_doc_links` scoped to the node) — link if the target is a live SN, else strip — so no accepted doc carries a broken bracket regardless of when it was written. |
| 7 | `REFINE_DOCS` | `docs_stage='reviewed' AND rds<min AND docs_chain_length<cap` | Rewrites docs in-place; prior content snapshotted on a `DocsRevision` node via `DOCS_REVISION_OF`; `docs_stage → 'drafted'`. |

`ENRICH_PARENTS` is a name-axis producer: it runs under `--names-only` and
`--flush` (it drains the *existing* placeholder backlog, not new work), is
excluded under `--docs-only`, and shares the single `--cost-limit` budget pool.
Its weight and replica count live alongside the other pools in `POOL_WEIGHTS`
and `[tool.imas-codex.sn-pools].enrich-parents-replicas`.

**Acceptance always overrides cap.** Even at `chain_length == cap − 1`, a
passing score wins — there is no forced exhaustion on a good result.

**Escalation.** On the final refine attempt (`chain_length == cap − 1`) the pool
switches to `DEFAULT_ESCALATION_MODEL` (default
`openrouter/anthropic/claude-opus-4.6`), overridable via `--escalation-model`.

**Backlog throttle.** When the refine_name backlog exceeds 0.5 × the
generate_name backlog, `BudgetManager.pool_admit` dampens the generate_name
effective weight by 0.5× so refinement can catch up.

**Orphan sweep.** A `refining` claim that stalls reverts to `reviewed` after
`DEFAULT_ORPHAN_SWEEP_TIMEOUT_S` (600 s). The periodic sweep lives in
`orphan_sweep.py`.

**Loop termination.** The loop stops on zero eligible work, an exhausted
`--cost-limit` (a single shared budget pool across all seven pools), or a
per-pool admission threshold. On `Ctrl-C` it writes an audit `SNRun` node
(`cost`, pool counters, `min_score`, `rotation_cap`, `stop_reason`); `sn status`
surfaces the most recent run.

### Scope routing

`sn run` runs the full loop across all eligible work by default. Two narrower
modes:

- `--only <phase>` — run a single phase in isolation (e.g. `--only reconcile`
  to mark stale sources without composing).
- `--focus <path>` — route specific DD paths through the full seven-pool loop,
  scoped by a UUID `scope_run_id`. Use for iterative prompt development and
  quality investigation on individual paths without a full rotation. Three
  input forms: trailing positional args (`sn run path1 path2`), quoted
  space-separated (`--focus "path1 path2"`), or repeated flags
  (`--focus path1 --focus path2`). Single-pass focus seeding is handled by
  `pool_adapter.run_explicit_paths()`.

## EXTRACT → COMPOSE Sub-Pipeline

Inside `GENERATE_NAME` (and via `pool_adapter.run_explicit_paths()` on the
single-pass focus path) runs the extract→compose→validate→consolidate→persist
sub-pipeline:

```
 ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐     ┌───────────┐
 │  EXTRACT  │────▶│  COMPOSE  │────▶│ VALIDATE  │────▶│CONSOLIDATE│────▶│  PERSIST  │
 │ DD query  │     │ LLM call  │     │ ISN 3-    │     │ dedup     │     │ Neo4j     │
 │ classify  │     │ per-batch │     │ layer +   │     │ conflicts │     │ conflict  │
 │ enrich    │     │ unit from │     │ round-trip│     │ coverage  │     │ detection │
 │ group     │     │ DD only   │     │           │     │           │     │           │
 └───────────┘     └───────────┘     └───────────┘     └───────────┘     └───────────┘
```

### Classification (node_category)

SN-eligibility is **owned by the DD `node_category`** classification, not a
bespoke rule engine. EXTRACT pre-filters DD paths through
`SN_SOURCE_CATEGORIES` in `imas_codex/core/node_categories.py` — only physics
quantities and the categories declared there proceed to COMPOSE. Error fields,
pure metadata, structural nodes, and identifiers are filtered out by category.

### Enrichment (cluster selection + grouping)

`imas_codex/standard_names/enrichment.py`:

- **Primary cluster selection** — each path may belong to multiple semantic
  clusters; one primary is chosen by scope rank (IDS > domain > global), then
  highest `similarity_score`, then lexicographic tie-break.
- **Grouping cluster selection** — for batching context, the *broadest* scope
  is preferred (global > domain > IDS).
- **Global grouping by (cluster × unit)** — grouping is global across all IDSs,
  not per-IDS. If `electron_temperature` appears in `core_profiles`,
  `core_sources`, and `equilibrium`, all three land in the same batch so the
  LLM sees the full cross-IDS picture and assigns one name. Oversized groups
  split on `max_batch_size` (token-budget guard); unclustered paths sub-group
  by `parent_path`.

Each batch carries: cluster description, cross-IDS path summary, sibling paths,
and existing standard names for reuse.

### Validation (ISN 3-layer + grammar round-trip)

`_validate_via_isn()` runs three annotation-only layers:

1. **Pydantic** — `create_standard_name_entry()` fires its field validators.
2. **Semantic** — `run_semantic_checks()` runs grammar-semantic checks.
3. **Description** — `validate_description()` checks for metadata leakage.

Validation is **annotation-only**: entries are never rejected by semantic
warnings — they persist to `validation_issues` (tagged strings) and
`validation_layer_summary` (JSON per-layer counts). The only **hard
quarantine** triggers are: grammar round-trip failure
(`parse_standard_name()` rejects), Pydantic construction error, or detected
ambiguity (one name → multiple distinct physical quantities). See
[Validation Gating](#validation-gating).

### Consolidation (`consolidation.py`)

After VALIDATE, all candidates are consolidated in one pass.

**Dedup** — when batches produce the same name: keep the longest documentation,
**union** `imas_paths` and `tags`, keep the highest confidence.

**Conflict checks**, applied in order — conflicting entries are *filtered out*,
not failed (the pipeline makes partial progress rather than aborting):

| Check | Type | Catches |
|-------|------|---------|
| 1 | `unit_mismatch` | Same name proposed with different units |
| 2 | `kind_mismatch` | Same name with different kind (scalar vs vector) |
| 3 | `duplicate_source` | One source path claimed by multiple names |
| 4 | Coverage gap | Source paths with no candidate mapping |
| 5 | Registry reuse | Existing accepted names → reuse instead of mint |

## Unit Safety Model

Units flow from the Data Dictionary to the graph — **never from LLM output**.

```
DD (HAS_UNIT relationship)
  → EXTRACT reads unit from graph
  → COMPOSE prompt marks unit "read-only, authoritative" (LLM output has no unit field)
  → worker injects DD unit into the candidate dict
  → PERSIST writes (StandardName)-[:HAS_UNIT]->(Unit)
```

If the LLM emits a unit anyway, the worker discards it and uses the DD value.
`cocos`, `cocos_transformation_type`, `dd_version`, and `physics_domain` are
likewise DD-authoritative and injected post-LLM — `physics_domain` comes
directly from `IMASNode.physics_domain` (falls back to `"general"`).

## Scoring

Review scoring uses integer per-dimension scores (0–20) normalized to a 0–1
aggregate. The **rubric-driven numeric `score` (0–1) is the sole accept/refine
signal** — there is no separate `verdict` field; the reviewer LLM produces
scores plus optional `revised_name` / `suggested_name`.

**Name axis (4 dims):** grammar, semantic, convention, completeness.
**Docs axis (4 dims):** description_quality, documentation_quality,
completeness, physics_accuracy.
**Benchmark (6 dims):** grammar, semantic, documentation, convention,
completeness, compliance (aggregate `sum / 120.0`).

Criteria are defined in `imas_codex/llm/config/sn_review_criteria.yaml`.

**Tier thresholds** (`review_tier`, derived from `reviewer_score_name`):

| Tier | Minimum |
|------|---------|
| outstanding | ≥0.85 |
| good | ≥0.60 |
| inadequate | ≥0.40 |
| poor | <0.40 |

## RD-Quorum Review

`sn review` (and the `REVIEW_NAME` / `REVIEW_DOCS` pools) use a
**Rational-Disagreement quorum** for high-confidence axis scores. The two axes
(`names`, `docs`) are independent, each with its own rubric, model chain, and
per-axis score columns.

```
 cycle 0: primary   (BLIND)  ──┐
 cycle 1: secondary (BLIND)  ──┼─ disagree on any dim? → cycle 2
 cycle 2: escalator (SEES 0+1) ┘  (only if disputed items exist AND len(models)==3)
```

- **Cycle 0** (primary, blind) scores with `models[0]`.
- **Cycle 1** (secondary, blind) re-scores with `models[1]` — blindness
  enforced (no `prior_reviews` block in the prompt).
- An item is **disputed** if any per-dimension diff between cycles 0 and 1
  exceeds `disagreement-threshold` (default 0.15, normalized 0–1).
- **Cycle 2** (escalator, context-aware) runs only if disputed items exist AND
  `len(models) == 3`; it receives both prior critiques and is authoritative for
  the disputed items.

**Immediate persistence:** each cycle's `Review` nodes are written before the
next cycle starts (crash-safety). After all cycles,
`update_review_aggregates` picks the most-recent winning group and mirrors final
scores onto the SN axis slots.

**Partial-failure ladder:** both missing → retry → `retry_item` (quarantine);
one missing → `single_review`.

**`resolution_method` (on `Review` nodes):**

| Value | Meaning |
|-------|---------|
| `quorum_consensus` | Cycles 0+1 agreed; final = mean |
| `authoritative_escalation` | Cycle 2 ran; authoritative for disputed items |
| `max_cycles_reached` | 0+1 disagreed, no escalator; mean + disagreement flag |
| `retry_item` | Both cycles failed; item quarantined |
| `single_review` | Only one cycle produced a score |

**1:1 scoring invariant:** each submitted name receives exactly one score per
cycle. Names are sent in batches; the response is matched back by ID; unmatched
names are auto-retried individually so none is silently dropped on LLM
truncation.

**Configuration:** `[tool.imas-codex.sn-review]` (`disagreement-threshold`,
`max-cycles`, `active-profile`) + `[tool.imas-codex.sn-review.{names,docs}]`
(`models = [cycle0, cycle1, cycle2]`). Named profiles live under
`[tool.imas-codex.sn-review.names.profiles.*]`. Accessors:
`get_sn_review_{names,docs}_models()`, `get_sn_review_max_cycles()`,
`get_sn_review_disagreement_threshold()`, `get_sn_review_active_profile()`.

```toml
[tool.imas-codex.sn-review.names]
models = [
  "openrouter/deepseek/deepseek-v4-flash",     # cycle 0 primary (blind, cheapest)
  "openrouter/openai/gpt-5.4",                 # cycle 1 secondary (blind, fast)
  "openrouter/google/gemini-3.1-pro-preview",  # cycle 2 escalator (highest quality)
]
```

**Budget:** `review_pipeline` reserves `batch_cost × num_models × 1.3` upfront
via `BudgetLease.reserve()`, then charges per cycle — prevents secondary-cost
leaks on a mid-cycle crash.

**Axis-split review storage:** name and docs reviews persist to independent
column families so a docs pass cannot clobber name-only data and vice-versa.
Paired slots per axis: `reviewer_score_{axis}`, `reviewer_scores_{axis}`
(per-dim JSON), `reviewer_comments_{axis}`,
`reviewer_comments_per_dim_{axis}`, `reviewer_model_{axis}`. Same-axis
re-review requires `--force`; the guard is `_axis_overwrite_blocked` in
`review/pipeline.py`. **Review never demotes** — a low-scoring `valid` name
stays `valid` and routes to a refine pool.

**`Review` node fields (RD-quorum):** `review_axis`, `cycle_index` (0/1/2),
`review_group_id` (UUID per quorum session), `resolution_role`
(primary/secondary/escalator), `resolution_method`. `id` format:
`{sn_id}:{axis}:{review_group_id}:{cycle_index}`.

## Structured Fan-Out

`imas_codex/standard_names/fanout/` runs a bounded
Proposer → Executor → Synthesizer fan-out for `refine_name`. The Proposer emits
a closed-catalog `FanoutPlan` (Pydantic discriminated union on `fn_id`, all
bounds enforced at parse time); a pure-Python executor runs it in parallel via
`asyncio.to_thread`; the call-site's LLM call ingests the rendered evidence
block. No agentic loop, no runtime function generation.

- **Runtime gating lives in config** (`[tool.imas-codex.sn-fanout]`). Current
  rollout is enabled for `refine_name`; if disabled, `run_fanout()` is a true
  no-op.
- **One `GraphClient` per refine cycle** — the worker passes its `gc` in;
  runners never instantiate their own.
- **Cost ownership:** the Proposer call is charged to the caller's
  `BudgetLease` as a sub-event with `batch_id=fanout_run_id`; callers stamp the
  same id on the Synthesizer charge so the `Fanout` ↔ `LLMCost` graph join
  works.
- **`Fanout` node** is telemetry-only, runtime-written (like `LLMCost`); exempt
  from LinkML schema management.

## Derived / Structural Parents

Structural parent SNs are inferred from the ISN grammar peel and enter with
`origin='derived'`, `name_stage='pending'`. The placeholder node is created at
write time by `_write_standard_name_edges` **only if it passes the two-clause
admission gate** in `imas_codex/standard_names/parents.py`
(`is_admissible_parent_name`):

- **Clause A — IR specificity.** `parse_standard_name(name).ir` has at least one
  non-empty qualifier, operator, projection, locus, or mechanism. This admits
  `elongation_of_plasma_boundary` and rejects bare bases like `pressure`.
- **Clause B — vector-like topology.** The candidate already has ≥2 inbound
  `HAS_PARENT` edges with `operator_kind='projection'` and distinct `axis`
  values. This admits bare-base vectors like `magnetic_field` and
  `electric_field` on topological grounds. (Clause B is callable without a live
  graph via a stub `gc`.)

Names failing both clauses are never created; their children remain parentless
and group via shared `physical_base` / `qualifier` grammar fields in Cypher.

**Why bare-base scalars are excluded:** category labels like `pressure`,
`density`, `temperature` *are* the `physical_base` grammar segment — making
parent nodes for them would populate the catalog with trivially generic entries
carrying no scientific identity. Clause A rejects exactly these.

Once created, derived parents flow through the standard
`REVIEW_NAME → GENERATE_DOCS → REVIEW_DOCS` path. Derived parents are skipped from the refine loop (their names are structurally
fixed). Admissibility is enforced by `is_admissible_parent_name`, and
`normalize_derived_parent_lifecycle` deletes any inadmissible accepted derived
parent — with its `HAS_PARENT` edges and review scaffolding — on every `sn run`
startup; description-quality failures route to `REFINE_DOCS`. The
`apply_derived_parent_migration` idempotent self-healing
pass runs on every `sn run` startup to migrate legacy `origin='deterministic'`
entries. There is no separate `sn parents` CLI — all parent handling is folded
into `sn run`.

### Breaking the coverage deadlock — `ENRICH_PARENTS`

A freshly materialized derived parent carries the
`DETERMINISTIC_PARENT_DESCRIPTION_PLACEHOLDER` until a real description is
written. That placeholder is a **deadlock**: `claim_review_name_batch` excludes
it (review needs a real description for the semantic-similarity check), so it
never earns a `reviewer_score_name`; and `claim_generate_docs_batch` gates on
`reviewer_score_name IS NOT NULL`, so it never earns docs either. The parent is
parked at `name_stage='accepted'` (held) and is invisible to both axes.

The `ENRICH_PARENTS` pool breaks this loop. For each claimed placeholder parent
with ≥1 live child it:

1. fetches the parent's live `HAS_PARENT` children (the concrete instances) via
   `fetch_derived_parent_children`,
2. renders the `sn/enrich_parent_{system,user}` prompts — which ground strictly
   on the children's accepted descriptions + names, never invented physics —
   and calls `get_model("sn-parent-enrich")` (compose-tier; `generate_docs`
   rewrites the full long-form documentation downstream),
3. embeds the synthesized description locally (free), and
4. persists via `persist_enriched_parent`: writes `description` + `embedding`,
   stamps `parent_enriched_at` / `parent_enrich_model`, and **accepts the
   parent structurally** — `name_stage → 'accepted'` with `reviewer_score_name`
   inherited from its accepted children (the min — "as valid as its weakest
   accepted child"; falls back to `DEFAULT_MIN_SCORE`) and
   `reviewer_model_name='structural-inheritance'`.

**Why skip REVIEW_NAME.** A derived parent's name is a deterministic grammar
peel that already passed the admission gate, and its description generalizes
over already-RD-quorum-accepted children — so it inherits name validity by
construction. Routing it through the name quorum is not just wasteful (~$0.27
each) but actively harmful: the quorum systematically marks a parent down for
being *less specific than its children* — measured ~66% of derived parents
scored <0.85 (mean 0.778) despite clean grammar, and a structurally-fixed name
cannot be improved by refine_name, so each rejection becomes a futile
refine→exhaust that strands the parent with no docs. Description quality is
still fully gated on the docs axis (`GENERATE_DOCS` → `REVIEW_DOCS`). The
`reviewer_model_name='structural-inheritance'` marker keeps these acceptances
auditably distinct from real LLM reviews; any prior review is preserved in the
`StandardNameReview` history.

The physics domain was already inherited from the children at materialize time
(`_materialize_derived_parent_rows`), so enrichment leaves it untouched.
**Childless derived parents are legitimately unscoped — they are never claimed
and never fabricated.** The pool is **not throttled** (it no longer feeds the
review_name bottleneck; it is cheap, drains a finite backlog, and the accepted
parents queue durably in `generate_docs`-pending) and runs under `--flush`, so a
cost-capped `sn run --flush` cleanly drains the existing backlog.

**`MAGNITUDE_OF` edge:** when the pipeline composes a `magnitude_of_<X>` name
from a real DD/signal source AND `<X>` is an admitted `kind='vector'` parent,
`_emit_magnitude_of_edges` emits `(magnitude_sn)-[:MAGNITUDE_OF]->(vector_sn)`.
**Passive** — never created speculatively. This is an algebraic-sibling
relationship, not hierarchical like `HAS_PARENT`.

## Prompt Architecture

### Static-first ordering

Two-message pattern optimized for prompt caching:

1. **System message** (static, ~6k tokens) — grammar rules, canonical segment
   order, full vocabulary, curated examples, tokamak ranges, composition rules,
   output schema. Cached by OpenRouter across calls. Rendered from
   `sn/compose_system.md`.
2. **User message** (dynamic, per-batch) — unit policy, batch context (IDS,
   cluster, siblings), existing names for reuse, per-path detail. Rendered from
   `sn/compose_dd.md`.

In Jinja templates, `{% include %}` schema/rules blocks go *before* dynamic
variables to maximize the cacheable prefix. `build_compose_context()` in
`context.py` assembles the static context (module-level cache).

### Per-item DD context injection

Each DD path item is enriched with four channels during EXTRACT:

1. **Hybrid DD search neighbours** (`hybrid_neighbours`) — concept-similar DD
   paths via vector + keyword search (`_hybrid_search_neighbours` in
   `workers.py`, backed by `hybrid_dd_search` in `graph/dd_search.py`).
2. **Related DD paths** (`related_neighbours`) — cross-IDS structural siblings
   via explicit graph relationships (cluster membership, shared coordinates,
   matching units, identifier schemas, COCOS transformation type), via
   `find_related_dd_paths`.
3. **Error companions** (`error_fields`) — uncertainty fields (`_error_upper`,
   `_error_lower`, `_error_index`) for each path.
4. **Identifier enum values** (`identifier_values`) — allowed enumeration
   values (name, index, description) when a path references an identifier
   schema.

**Compose retry with expanded context:** on grammar/validation failure the
compose worker retries up to `retry_attempts` times (default 1), re-enriching
with expanded hybrid search (`search_k=retry_k_expansion`, default 12).
Configurable via `[tool.imas-codex.sn]` or `IMAS_CODEX_SN_RETRY_*` env vars.

**Scored-example injection:** compose and review prompts include dynamically
selected exemplar StandardName nodes at target thresholds `(1.0, 0.8, 0.65,
0.4)`, graph-backed and chosen at runtime by the example loader. Context keys:
`compose_scored_examples`, `review_scored_examples`.

### Shared prompt fragments

- `{% include "sn/_grammar_reference.md" %}` — grammar vocabulary + segment
  order (used in `compose_system.md`).
- `llm/prompts/shared/sn/_scoring_rubric.md` — shared scoring-rubric reference.
- `llm/config/sn_review_criteria.yaml` — dimensions and tiers (via
  `load_prompt_config()`).
- ISN context keys (`quick_start`, `common_patterns`, `critical_distinctions`)
  rendered in the compose prompt.
- Axis review prompts `review_names.md` / `review_docs.md` share a
  `{% if prior_reviews %}…{% endif %}` block rendered **only** for the cycle-2
  escalator — cycles 0 and 1 never see it (blindness enforced).

**Response model:** `StandardNameComposeBatch` (`candidates`, `skipped`,
`vocab_gaps`). `vocab_gaps` capture cases where the grammar lacks a needed
token; see [Grammar vocabulary](#grammar-vocabulary).

## Lifecycle

### Four StandardName axes

| Axis | States | Set by | Notes |
|---|---|---|---|
| `name_stage` / `docs_stage` | `pending → drafted → reviewed → {accepted \| refining → drafted \| exhausted \| superseded}` | Pool workers | Cross-pipeline gate: `GENERATE_DOCS` fires only when `name_stage='accepted'`. `refining` reverts to `reviewed` after 600 s (orphan sweep). `chain_length` / `docs_chain_length` track refinement depth (root = 0). `superseded` = predecessor in a `REFINED_FROM` chain; source edges migrate to the latest. |
| `name_stage` | `pending → drafted → reviewed → accepted` (`refining`/`exhausted`/`superseded` side states) | `sn run` → `sn release --export-only` → `sn import` | Name-pipeline + catalog round-trip state. |
| `status` | `draft → published → deprecated` | Catalog import | ISN vocabulary lifecycle; pipeline defaults to `draft`. Deprecated names link via `superseded_by` ↔ `deprecates`. |
| `validation_status` | `pending → valid \| quarantined` | Compose worker | Gates `sn review`, consolidation, and `sn release --export-only`. Critical failures (grammar round-trip, Pydantic, ambiguity) quarantine; semantic warnings stay `valid`. |

**`origin`:** `pipeline` (LLM-generated), `catalog_edit` (human-edited via
catalog PR), or `derived` (structural parent from the admission gate).
`filter_protected()` skips `PROTECTED_FIELDS` on `catalog_edit` names unless
overridden via `sn run --override-edits <name>`. PR provenance:
`catalog_pr_number`, `catalog_pr_url`, `catalog_commit_sha`; round-trip
timestamps: `exported_at`, `imported_at`.

### StandardNameSource lifecycle

`StandardNameSource` nodes track individual DD-path / facility-signal
extraction. Written by the extract worker, updated by the compose worker:

```
extracted → composed | attached | vocab_gap | failed | stale
```

- **extracted** — queued for composition.
- **composed** — LLM generated a new name for this source.
- **attached** — auto-attached to an existing name (no LLM call).
- **vocab_gap** — a grammar vocabulary gap prevented naming.
- **failed** — composition failed (LLM error, validation rejection).
- **stale** — backing DD path / signal no longer exists (set by reconcile).

**ID format:** `dd:{full_dd_path}` or `signals:{facility}:{signal_id}` — the
`dd:` prefix is the canonical URI scheme (e.g.
`dd:equilibrium/time_slice/profiles_1d/psi`).

**Reconciliation:** `sn run --only reconcile` marks sources whose backing path
no longer exists as `stale`.

### Validation gating

`validation_status` is independent of review and gates names before review,
consolidation, or export:

| Status | Set by | Meaning |
|--------|--------|---------|
| `pending` | PERSIST phase | Default; awaiting validation |
| `valid` | VALIDATE phase | Passed critical checks; eligible downstream |
| `quarantined` | VALIDATE phase | Failed a critical check; excluded |

**Critical (→ quarantine):** grammar round-trip failure, Pydantic construction
error, detected ambiguity. **Non-critical (→ valid):** semantic warnings,
description-quality hints — recorded in `validation_issues`, surfaced to the
reviewer, but never quarantine.

## Graph Persistence

**Module:** `imas_codex/standard_names/graph_ops.py`.

### Write semantics

- **`write_standard_names()` (build path):** `coalesce(b.field, sn.field)` for
  all fields — passing `None` preserves existing data. Persists
  `validation_issues` and `validation_layer_summary`. Safe for re-runs:
  imported catalog data is never clobbered by a subsequent `sn run`.
- **`_write_catalog_entries()` (import path):** catalog fields SET directly
  (catalog is authoritative). Graph-only fields (embedding, model,
  generated_at) preserved via coalesce.
- **Review write path:** each RD-quorum cycle persists its `Review` nodes
  immediately; `update_review_aggregates` then mirrors final axis scores onto
  the SN slots.
- **Conflict-detecting writes:** before writing, the build path checks for unit
  conflicts and filters out (does not overwrite) any entry whose unit differs
  from an existing node's.

Both paths call shared `_write_standard_name_edges(gc, names)` as a tail pass
after node MERGE; forward-reference targets are MERGEd as bare placeholder
nodes.

### Structural edges

| Edge | From → To | Source / Derivation |
|------|-----------|---------------------|
| `HAS_ARGUMENT` | derived SN → parent SN | ISN parser: outermost unary prefix/postfix or projection layer; props `{operator, operator_kind, [role, separator, axis, shape]}` |
| `HAS_PARENT` | child SN → parent SN | ISN grammar peel; direction matches `(IMASNode)-[:HAS_PARENT]->(IMASNode)`; props `{operator, operator_kind, …}`. Emitted only when the parent passes the admission gate (`_filter_admissible_parents`). |
| `MAGNITUDE_OF` | magnitude SN → vector SN | Passive; only when a sourced SN composes to `magnitude_of_<X>` and `<X>` is an admitted vector parent. |
| `HAS_ERROR` | inner SN → uncertainty sibling | ISN parser: `upper_/lower_/uncertainty_index` prefix; direction inverted; `{error_type}` |
| `HAS_PREDECESSOR` / `HAS_SUCCESSOR` | SN → SN | `predecessor`/`successor` (pipeline) or `deprecates`/`superseded_by` (catalog) |
| `IN_CLUSTER` | SN → `IMASSemanticCluster` | `primary_cluster_id` |
| `HAS_PHYSICS_DOMAIN` | SN → `PhysicsDomain` | `physics_domain` slug → seeded singleton |
| `HAS_UNIT` | SN → `Unit` | `unit` (both paths) |
| `HAS_COCOS` | SN → `COCOS` | `cocos` integer (pipeline path) |
| `REFINED_FROM` | new SN → predecessor SN | `persist_refined_name`; source edges (`PRODUCED_NAME`, `HAS_STANDARD_NAME`) migrate to the new SN in the same transaction |
| `DOCS_REVISION_OF` | SN → `DocsRevision` | `persist_refined_docs`; snapshots prior docs+reviewer state before in-place rewrite |

`HAS_ARGUMENT` / `HAS_ERROR` derivation lives in
`imas_codex/standard_names/derivation.py` (`derive_edges(name)`, pure logic).
Each name is peeled one layer only; the inner name runs its own derivation when
written. Unparseable names silently produce no derived edges.

### Source-tracking relationships

```
(IMASNode)-[:HAS_STANDARD_NAME]->(StandardName)
(FacilitySignal)-[:HAS_STANDARD_NAME]->(StandardName)
(StandardNameSource)-[:FROM_DD_PATH]->(IMASNode)
(StandardNameSource)-[:FROM_SIGNAL]->(FacilitySignal)
(StandardNameSource)-[:PRODUCED_NAME]->(StandardName)
```

### Cost & budget

Cost is graph-backed via `LLMCost` nodes written async by `BudgetManager`.
`SNRun.status`: `started → completed | interrupted | failed | degraded`. The
only charge API is `lease.charge_event(cost, event)` (soft, never raises).
Start the manager with `await shared_mgr.start()`; finalize each run with
`drain_pending()` + `get_total_spent()` in a `finally` block. (LLMCost node
properties and canonical cost queries are documented in `AGENTS.md` →
Graph Operations.)

## PR-Driven Round-Trip

The graph is authoritative for pipeline state; the catalog (ISNC) is
authoritative for human-reviewed editorial fields.

```
sn release --export-only → sn preview → sn release -m "msg" → GitHub Pages / PR review → PR merged → sn import
```

1. **`sn release --export-only`** — runs only the export leg: reads validated
   SN nodes, applies quality gates (`reviewer_score_name ≥ 0.65` + description
   sub-score), writes YAML to `<staging>/standard_names/<domain>/<name>.yml`
   (default staging `~/.cache/imas-codex/staging`) and stops (no tag/push).
2. **`sn preview`** — auto-exports and launches a local MkDocs dev server via
   ISN's CatalogRenderer. `--no-export` serves an existing staging dir. Tunnel:
   `ssh -L 8000:localhost:8000 <host>`.
3. **`sn release -m "msg"`** — auto-exports, copies staging YAML into the ISNC
   checkout, commits, tags the next semver, pushes. RC → origin (fork); final
   (`--final`) → upstream. The tag push triggers ISNC CI → GitHub Pages. For
   custom filtering, run `sn release --export-only` (with the `[export]`
   scoping flags) first, then `--skip-export`.
4. **PR review on GitHub** — edits description, documentation, tags, kind,
   links, status, etc. Merged to ISNC main.
5. **`sn import`** — reads ISNC YAML (auto-discovered from `isnc-dir`, or
   `--isnc`), diffs against the graph, and flips `origin=catalog_edit` on any
   name whose `PROTECTED_FIELDS` were edited.

**Protection model:** `PROTECTED_FIELDS` = {description, documentation, kind,
tags, links, status, deprecates, superseded_by, validity_domain, constraints}.
Pipeline writers call `filter_protected()` before graph writes — `catalog_edit`
names have these stripped from pipeline updates unless `override=True` or the
name is in `override_names` (`sn run --override-edits <name>`, repeatable).

**Idempotent re-run:** `ImportWatermark` (singleton) records the last imported
`catalog_commit_sha`; `ImportLock` (singleton) prevents concurrent imports.

## Reset & Clear Semantics

- **`sn run --reset-to {drafted|extracted}`** — re-process existing nodes
  (`drafted`) or clear matching SN nodes for a full re-run (`extracted`). Clears
  transient fields (embedding, model, generated_at) and removes
  `HAS_STANDARD_NAME` / `HAS_UNIT` edges. Scoped to `--source`; narrow with
  `--since`, `--before`, `--below-score`, `--tier`, `--retry-quarantined`.
- **`sn run --reset-only`** — perform the cleanup then exit (requires
  `--reset-to`).
- **`sn run --from-model <substring>`** — re-generate names produced by a
  specific model.
- **`sn clear`** — full-subsystem wipe (SN + Review + StandardNameSource +
  VocabGap + SNRun + grammar tree) with auto re-seed from ISN. No scoping.
- **`sn prune`** — relationship-first safe delete. Requires `--stage` or
  `--all`; `--include-sources` also drops StandardNameSource nodes.
- **Grammar sync is automatic** — `sn run` syncs the installed ISN grammar
  into the graph at startup when the active version differs (idempotent
  no-op otherwise; best-effort — a sync failure logs and continues), and
  `sn clear` re-seeds it after a wipe (`--no-reseed` to skip).

**Chain history is permanent.** `--reset-to` leaves `REFINED_FROM` chains and
`DocsRevision` snapshots in place. **Safety guard:** `--reset-to` and
`sn prune` require `--include-accepted` to touch `name_stage=accepted`
names (catalog-authoritative). `sn clear` has no such guard.

## Family Harmonization

Docs are generated per-name, so sibling families (a vector's projections,
per-locus / per-species / per-zone variants) can drift apart in their
documentation structure even when each member individually passes review.
Two mechanisms address this:

**In-pipeline (always on).** `fetch_sibling_family()` (`context.py`) injects
each name's HAS_PARENT family — parent, anchor, live members with their doc
openings — into the generate_docs, review_docs and refine_docs prompts. The
family = children sharing a parent via `operator_kind ∈ {projection,
qualifier, coordinate, locus}`; the anchor (template authority) is the parent
when its docs are accepted and real, else the best-scored docs-accepted
member. All three prompts carry a parallel-structure directive: one opening
noun-phrase template per family, varying only member-specific tokens, symbols
and genuinely member-specific physics — faithfulness outranks uniformity, so
distinct per-member physics (e.g. an unsigned radial mode index next to
signed Fourier harmonics) must never be flattened. The review rubric docks
description/documentation quality for gratuitous template divergence from
accepted siblings, citing the sibling ids.

**Link-integrity gate (accept path).** `persist_reviewed_docs` scans an
accepting doc's markdown links: a `[label](name:target)` whose snake_cased
label is itself an existing StandardName id different from the target is a
referencing error reviewers systematically miss. The doc is demoted to
`reviewed` (score clamped under the accept threshold) with a
`link_integrity` per-dimension comment listing the mismatches, so the next
refine pass fixes the labels. At the rotation cap the doc accepts anyway —
a link nit never exhausts a doc.

**Idempotent done-state (automatic).** Each family carries
`harmonized_at` + `harmonized_group_signature` (a sha256 over sorted live
member ids and per-member content hashes) on members and parent. Every
`sn run` post-drain reconcile calls `restamp_harmonized_families()`: families
whose live members are ALL docs-accepted get a fresh stamp when membership
or content changed; families with any non-accepted member wait for a later
run. A new sibling composing therefore updates its family's stamp
automatically once its docs pass review — no operator action.

**Curative wave (`sn run --families`).** For docs accepted with drifted or
defective content, `sn run --families "<parent …>" --include-accepted` is a
one-shot wave: it resolves each parent's live family, snapshots every
member's docs to a `DocsRevision` (same `{id}#rev-{chain}` scheme as
refine), resets the family's docs pipeline, drains ONLY those names through
the docs pools (docs-only flush, scoped internally), and the post-drain
reconcile restamps. `--dry-run` previews the member set. The deterministic
drift metric (first-6-token opening signature; drift = 1 − max_cohort/n)
counts member-specific tokens as drift and therefore over-flags — treat
`sn status`'s drift worklist as a RANKING and confirm genuine structural
drift (LLM triage or human read) before re-opening accepted docs.

**Inspection.** `sn status` prints the Sibling Families block (counts,
stamped, awaiting docs, drift worklist + top entries);
`sn status --family <seed>` prints one assembled family (parent, anchor,
members). The `harmonize.py` library (`build_worklist`, `assemble_family`,
`lint_links`, `mark_families_for_regen`, `restamp_harmonized_families`)
remains the programmatic surface.


## CLI Commands

| Command | Purpose | Key options |
|---------|---------|-------------|
| `sn run` | Run the seven-pool loop. Auto-seeds all eligible domains; `--domain` restricts. `--focus` routes specific paths; `--only` runs a single phase; `--flush` drains without composing; `--rename OLD:NEW` short-circuits to the parent-rename cascade (no LLM; pair with `--dry-run`). | `--source {dd,signals}`, `--domain` (multi), `--facility`, `--focus` (multi), `--limit`, `--max-sources`, `-c/--cost-limit`, `--dry-run`, `--force`, `--reset-to`, `--reset-only`, `--from-model`, `--since`, `--before`, `--below-score`, `--tier`, `--retry-quarantined`, `--retry-skipped`, `--retry-vocab-gap`, `--min-score` (0.80), `--rotation-cap` (3), `--escalation-model`, `--review-name-backlog-cap`, `--review-docs-backlog-cap`, `--skip-review`, `--only`, `--override-edits`, `--flush`, `--rename`, `--include-accepted`, `--scope-run-id`, `--families` |
| `sn review` | Score existing valid names via RD-quorum (3-layer: audits → batched LLM → consolidation) | `--ids`, `--physics-domain`, `--stage`, `--unreviewed`, `--force`, `--models`, `--batch-size`, `--neighborhood`, `--target`, `--reviewer-profile` |
| `sn preview` | Auto-export + local MkDocs preview | `--export/--no-export`, `--staging`, `--port`, `--host` |
| `sn release` | Release to ISNC catalog (RC→origin, final→upstream). `--export-only` runs just the graph→staging export leg and stops (no tag/push). | `-m`, `--bump`, `--final`, `--remote`, `--isnc`, `--staging`, `--skip-export`, `--dry-run`, `--export-only`, `--names-only`, and `[export]` scoping (`--min-score`, `--include-unreviewed`, `--min-description-score`, `--gate-only`, `--gate-scope {all,a,b,c,d}`, `--domain`, `--force`, `--skip-gate`, `--override-edits`, `--include-sources/--no-include-sources`) |
| `sn import` | Import reviewed YAML back into the graph | `--isnc`, `--accept-unit-override`, `--accept-cocos-override`, `--dry-run` |
| `sn status` | StandardName + StandardNameSource statistics, sibling-family harmonization state | `--family` |
| `sn coverage` | DD/signal coverage by domain, cluster, IDS | `--domain`, `--ids`, `--format` |
| `sn clear` | Full-subsystem wipe + auto re-seed of ISN grammar | `--dry-run`, `--force`, `--no-comment-export`, `--no-reseed` |
| `sn prune` | Scoped delete (relationship-first) | `--stage`, `--all`, `--source`, `--ids`, `--include-accepted`, `--include-sources`, `--dry-run` |
| `sn bench` | Benchmark LLM models on generation quality | `--models`, `--max-candidates`, `--runs`, `--temperature`, `--output`, `--reviewer-model`, `--reviewer-models` |

## Benchmark

`sn bench` uses the same prompt pipeline as `sn run` (system/user split via
`build_compose_context()`). Model lists default from
`[tool.imas-codex.sn-benchmark]`. The output table includes a **Cache %**
column (provider-side via OpenRouter). Scoring is 6-dimensional (see
[Scoring](#scoring)).

**Qualified compose models** (equilibrium + core_profiles + magnetics):

| Role | Model | Avg quality | Notes |
|------|-------|-------------|-------|
| Recommended | `openai/gpt-5.5` | 78.2 | Best overall, strong grammar + docs |
| Alt | `anthropic/claude-sonnet-4.6` | 76.5 | 32% Outstanding, best grammar + docs |
| Budget | `google/gemini-3.1-pro-preview` | 74.6 | Near-top, good consistency |
| Light | `anthropic/claude-haiku-4.5` | 61.2 | Adequate for bulk generation |

**Reviewer models** (7-model benchmark, GPT-5.5 compose, 4 items, 2026-05-15):

| Role | Model | Names | Docs | Cost/run | Notes |
|------|-------|-------|------|----------|-------|
| Cycle 0 primary | `deepseek/deepseek-v4-flash` | 0.756 | 0.992 | $0.002 | Cheapest; negative rank-corr (−0.5) maximises escalator |
| Cycle 1 secondary | `openai/gpt-5.4` | 0.672 | 0.881 | $0.033 | Fastest (31s), cross-vendor |
| Cycle 2 escalator | `google/gemini-3.1-pro-preview` | 0.884 | 1.000 | $0.469 | Highest quality, authoritative |
| Eliminated | `anthropic/claude-sonnet-4.6` | 0.581 | 0.875 | $0.079 | Lowest commercial names score |
| Eliminated | `deepseek/deepseek-v4-pro` | 0.519 | 0.931 | $0.040 | Lowest names, very slow (259s) |
| Eliminated | `moonshot/kimi-k2.6` | 0.694 | 0.966 | $0.066 | No advantage |
| Failed | `alibaba/qwen3.6-plus` | 0.703 | 0.000 | — | 100% docs structured-output failure |

Projected cost at 10K sources: $659 (vs prior Sonnet→GPT→Opus $2,578, **−83%**).

**GPT-5.x compatibility:** requires `strict: false` JSON-schema wrapping
(handled in `llm.py`) and cannot use `temperature=0.0` (handled in benchmark).

## MCP Tools

| Tool | Purpose |
|------|---------|
| `search_standard_names` | Semantic + keyword search; filters: `kind`, `physics_domain`, `name_stage`, `cocos_type`, `k`, and exact per-segment grammar filters (`physical_base`, `subject`, `component`, `coordinate`, `position`, `process`, `region`, `geometric_base`, `device`) |
| `fetch_standard_names` | Fetch full entries by name ID (space/comma separated) |
| `list_standard_names` | List with `physics_domain` / `kind` / `name_stage` / `cocos_type` filters |
| `list_grammar_vocabulary` | Distinct tokens + usage counts for a grammar segment |

Grammar-segment filters match exactly against the parsed `sn.<segment>`
property — use `list_grammar_vocabulary` to discover valid tokens first. The
`physics_domain` filter (canonical scalar + `source_domains` list) is pushed
into Cypher in all three search branches, so it does not lose results below
`LIMIT $k`.

## Module / File Map

| File | Purpose |
|------|---------|
| `pools.py` | Pool specs: `POOL_WEIGHTS`, `POOL_NAMES`, `_build_pool_specs`, backlog throttle |
| `loop.py` | Six-pool loop orchestrator (`run_sn_pools()`) |
| `workers.py` | Claim/process/persist for all seven pools |
| `pool_adapter.py` | Routes `--focus` through pool compose; explicit-path seeding |
| `enrichment.py` | Primary/grouping cluster selection; global (cluster × unit) grouping |
| `consolidation.py` | Cross-batch dedup, conflict checks, coverage accounting |
| `graph_ops.py` | Neo4j read/write with unit-conflict detection; `persist_refined_name`, `persist_refined_docs`; `_write_standard_name_edges` |
| `parents.py` | Derived-parent admission gate (`is_admissible_parent_name`, `recompute_parent_kind`) |
| `derivation.py` | `HAS_ARGUMENT`/`HAS_ERROR` edge derivation (`derive_edges`) |
| `defaults.py` | Constants: `DEFAULT_MIN_SCORE=0.80`, `DEFAULT_ORPHAN_SWEEP_TIMEOUT_S=600`, `DEFAULT_REFINE_ROTATIONS`, `DEFAULT_ESCALATION_MODEL` |
| `models.py` | Pydantic response models (`StandardNameComposeBatch`, `StandardNameAttachment`) |
| `source_paths.py` | Encode/parse/split/merge for SN source paths |
| `context.py` | Grammar context builder (single ISN import boundary) |
| `search.py` | Vector search for similar existing SN nodes (collision avoidance) |
| `orphan_sweep.py` | Reverts stale `refining` claims after timeout |
| `example_loader.py` | Graph-backed scored-example selection for prompts |
| `review/pipeline.py` | RD-quorum review; `_axis_overwrite_blocked` guard |
| `fanout/` | Bounded Proposer → Executor → Synthesizer fan-out for refine_name |
| `benchmark.py` | LLM model quality benchmarking |
| `imas_codex/core/node_categories.py` | `SN_SOURCE_CATEGORIES` pre-filter |
| `imas_codex/graph/dd_search.py` | `hybrid_dd_search`, `find_related_dd_paths` |
| `imas_codex/llm/prompts/sn/*` | Compose + review prompts |
| `imas_codex/llm/config/sn_review_criteria.yaml` | Review scoring config |
| `imas_codex/schemas/standard_name.yaml` | LinkML schema |

## Architecture Boundary

ISN owns grammar, vocabulary, and validation; codex owns the pipeline,
evaluation, and graph persistence. Full detail in [`boundary.md`](boundary.md).

**Import boundary** (ISN ≥0.8.0rc7) — the only public surface codex imports:
`get_grammar_context()` (single entry point, 19 keys),
`create_standard_name_entry()` (Pydantic), `run_semantic_checks()`,
`validate_description()`, `parse_standard_name()` / `compose_standard_name()`
(round-trip). Never import ISN private modules. Never hardcode grammar rules —
pull from `get_grammar_context()`. Review criteria live in codex
(`sn_review_criteria.yaml`).

## Grammar Vocabulary

Closed-vocabulary system: **physical bases** (~78, irreducible dimensional
quantities, CI-gated), **qualifiers** (~92, prefix modifiers stripped
recursively by the parser), **processes** (~90, suffix-only via
`_due_to_{token}`). All segments — including `physical_base` (~100 tokens) —
have closed vocabularies defined in ISN's `SEGMENT_TOKEN_MAP`.

**Composition order:** `{subject}_{qualifier1}_{qualifier2}_{physical_base}_{due_to_process}`.

**Rules:** (1) Never add compounds to `physical_bases` — use qualifiers.
(2) Qualifier order is insertion-order, preserved through round-trip.
(3) Subjects win over qualifiers (parser stage 3 before stage 5). (4) Process
tokens as prefixes are qualifiers, not processes. Never hardcode vocabulary
tokens in Python — pull from ISN.

**VocabGap nodes** record missing grammar tokens from composition (linked via
`HAS_SN_VOCAB_GAP` from `IMASNode`/`FacilitySignal`; carry segment, token,
reason). Pseudo segments like `grammar_ambiguity` are filtered at write time
(`imas_codex.standard_names.segments.filter_closed_segment_gaps`); reviewers
audit the `physical_base` slot via the decomposition rule
(`sn_review_criteria.yaml` I4.6). `VocabGap` nodes accumulate automatically
during composition; query them directly from the graph when assembling an ISN
vocabulary rotation. A compose-time component-token reuse check
(`vocab_semantic_dedup.nearest_registered_token`, threshold
`[sn-compose].dedup-similarity-threshold`) flags a proposed new token that is a
near-synonym of a registered same-segment token and chains a refine so the
agent reuses it or confirms it distinct — the confirmation is stamped on the
`VocabGap` (`dedup_decision`) so the rotation does not re-litigate it.

### Vocabulary rotation — ISN fork RC workflow

When rotations surface `VocabGap` nodes blocked by the current ISN vocabulary,
add tokens on the fork (`~/Code/imas-standard-names` →
`Simon-McIntosh/IMAS-Standard-Names`) and cut an RC release. Upstream is
`iterorganization/IMAS-Standard-Names`; the dep in imas-codex pins to a git tag
on the fork, so RC tags on origin suffice.

**Vocab-addition rules** (apply *before* editing YAML):

- Every legitimate physics quantity deserves a token regardless of frequency.
  Classify each gap first: TRUE_GAP (add), COMPOSE_ERROR (fix the prompt
  instead), REJECT (not a valid grammar concept).
- Single-word base preferred; physical-quantity semantics only; no overlap with
  existing tokens; not a unit or geometry primitive.
- **No compound tokens that subsume `physical_base` words** —
  `trapped_particle` as a subject greedily consumes "particle" and breaks
  grouping with `particle_density`. Use `trapped` alone.
- **Prefer atomic qualifiers.** Orbit class (`trapped`, `co_passing`) and
  species (`fast_particle`, `electron`) are independent axes — never combine.
- **Grouping invariant:** all orbit/species variants of the same physical
  quantity MUST parse to the same `physical_base`.
- **Round-trip every existing name** containing the candidate string:
  `assert compose_standard_name(parse_standard_name(name)) == name`.

**Release procedure:**

```bash
cd ~/Code/imas-standard-names
uv run pytest && git push origin main           # edit vocab YAML, test, commit
uv run standard-names release status            # inspect state-machine state
uv run standard-names release -m "feat: ..."    # increment RC (NEVER tag manually)
# --bump minor/major starts a new series; --final finalizes to upstream.

cd ~/Code/imas-codex                            # bump the dep (appears twice)
sed -i 's|@v0\.8\.0rc[0-9]\+|@v0.8.0rc<NN>|g' pyproject.toml
uv sync && uv run pytest tests/standard_names/ -x -q
git commit -am "deps: bump imas-standard-names to v0.8.0rc<NN>" && git push origin main
```

Multi-RC chains are normal during bootstrap. Re-rotate vocab-bound domains
after each bump to measure lift.
