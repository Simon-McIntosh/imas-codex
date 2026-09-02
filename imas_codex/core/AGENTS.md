This file governs the `imas_codex/core/` subtree — durable node state, physics-domain and
node categorization. It also carries the Standard Names pipeline reference: SN source
eligibility and the grammar/domain principles are anchored in `core` (node_categories,
node_classifier, physics_domain), so its operational record lives here beside them.



## Standard Names

> **Full reference:** [`docs/architecture/standard-names.md`](docs/architecture/standard-names.md)
> (pipeline, RD-quorum, fanout, derived parents, prompt architecture, graph
> edges, write semantics, CLI flag detail, benchmark results) and
> [`standard-names-decisions.md`](docs/architecture/standard-names-decisions.md)
> (rationale). This section is **orientation + tripwires only** — flags are in
> `--help`, schema in `agents/schema-reference.md`, live stats in `sn status`.

### Pipeline (seven-pool `sn run` loop)

| Pool | Stage gate | Operation |
|------|------------|-----------|
| `GENERATE_NAME` | `StandardNameSource.status=pending` | LLM generates name; new SN at `name_stage='drafted'`. **Unit from DD, never LLM.** Runs the EXTRACT→COMPOSE→VALIDATE→CONSOLIDATE→PERSIST sub-pipeline. |
| `ENRICH_PARENTS` | `origin='derived' AND description=placeholder AND has live child` | LLM synthesizes a real description for a placeholder derived parent by **generalizing over its children**, embeds it locally, and **accepts it structurally** (`name_stage→'accepted'`, `reviewer_score_name` inherited from accepted children, `reviewer_model_name='structural-inheritance'`) — it **skips REVIEW_NAME**: a structurally-fixed abstraction is systematically penalized by the name quorum for being less specific than its children (measured ~66% scored <0.85), so review only sends it to a futile refine→exhaust. Description quality is still gated on the docs axis. Breaks the coverage deadlock (placeholder → excluded from review → no score → excluded from docs). Childless parents are unscoped and skipped. Model: `get_model("sn-parent-enrich")` (compose-tier). |
| `REVIEW_NAME` | `name_stage='drafted'` | RD-quorum scores → `accepted`/`reviewed`/`exhausted`. Derived parents add a `specificity` dim. |
| `REFINE_NAME` | `name_stage='reviewed' AND rsn<min AND refine_attempts<cap` | New SN node; predecessor `superseded`; `REFINED_FROM` edge; source edges migrate. The cap counts CLAIMED attempts (charged before the model call, inherited by the successor), not `chain_length`, which counts only successors that persisted — an attempt the graph refuses spends budget instead of re-claiming forever. Last rotation runs the escalation seat; a decided refusal parks the name with `refine_stop_reason`. Detail: `imas_codex/standard_names/AGENTS.md`. |
| `GENERATE_DOCS` | `name_stage='accepted' AND docs_stage='pending'` | LLM docs → `docs_stage='drafted'`. Cross-gate: fires only after name accepted. |
| `REVIEW_DOCS` | `docs_stage='drafted'` | RD-quorum scores → `accepted`/`reviewed`/`exhausted`. **Accept-path link hygiene:** on promotion to `accepted` the doc's bare `[name]` brackets are normalized at source (link or strip), so no accepted doc carries a broken bracket regardless of when it was written. |
| `REFINE_DOCS` | `docs_stage='reviewed' AND rds<min AND docs_chain_length<cap` | Rewrites docs in-place; prior snapshot on `DocsRevision` via `DOCS_REVISION_OF`. |

Pools run concurrently weighted by `POOL_WEIGHTS`. **Acceptance overrides cap**
(a passing score wins even at the final rotation). **Escalation:** the final
refine attempt switches to `--escalation-model` (default: the local
compose model — override for a paid frontier final attempt). **Backlog throttle:** refine_name
backlog > 0.5 × generate_name backlog dampens generate weight 0.5×.
`--cost-limit` is a single shared budget pool; `Ctrl-C` writes an audit `SNRun`.
Scope routing: `--only <phase>` (single phase, e.g. `--only reconcile`),
`--focus <path>` (specific paths through the full loop, UUID-scoped).
Mid-pipeline names are durable state any later run continues — size `-c` so a
cohort completes; `--flush` gates new work to drain the backlog (pre-audit/
release convergence, not recovery).

### Family harmonization (automatic)

Sibling families (projections / per-locus / per-species variants sharing a
HAS_PARENT parent) must read as a matched set. This is enforced with NO
dedicated command:

- Every generate/review/refine docs call injects the sibling family + a
  parallel-structure directive (always on).
- The docs accept path gates link integrity: a `[label](name:target)` whose
  label names a DIFFERENT existing standard name demotes the doc to
  `reviewed` with a `link_integrity` comment so refine fixes it.
- Every `sn run` post-drain reconcile restamps family idempotency signatures
  (`harmonized_at` + `harmonized_group_signature`) for families whose live
  members are all docs-accepted — a new member's docs landing updates the
  family automatically on the next run.
- `sn status` reports family/drift state; `sn status --family <seed>` shows
  one family.
- Curative re-open of ACCEPTED docs is the only manual act:
  `sn run --families "<parent …>" --include-accepted` (one-shot: snapshot →
  reset → scoped docs drain → restamp). Never edit docs by hand.


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
  `--include-accepted` to touch `name_stage=accepted` (catalog-
  authoritative) names. `sn clear` has no guard — it wipes everything.
- **Review never demotes:** a low-scoring `valid` name stays `valid` and routes
  to a refine pool; it is not quarantined.
- **Derived-parent coverage:** a derived parent (`origin='derived'`) is born
  with a placeholder description that excludes it from BOTH review (placeholder
  ≠ real description) and docs (no `reviewer_score_name`) — a deadlock. The
  `ENRICH_PARENTS` pool breaks it by synthesizing a children-grounded
  description; it grounds on the parent's **children**, never invented physics.
  Childless derived parents are legitimately unscoped — never fabricate a
  description for them. **Derived parents skip REVIEW_NAME** and are accepted
  structurally (score inherited from accepted children,
  `reviewer_model_name='structural-inheritance'`): the name is a deterministic
  grammar peel and the description generalizes over already-accepted children,
  so name validity is inherited by construction; the quorum would otherwise
  reject ~66% for being abstractions. Quality is gated on the docs axis. Drain
  the backlog with `sn run --flush` (enrich is an unthrottled producer that
  runs under flush; the existing `--cost-limit` caps it).
- **Bare-bracket link hygiene is fixed at source:** docs are normalized on the
  REVIEW_DOCS accept path (`persist_reviewed_docs`), so an accepted doc never
  carries a bare `[name]` bracket. The post-drain `resolve_doc_links` reconcile
  remains as a belt-and-suspenders net — the per-cycle manual sweep is no longer
  needed.
- **Import boundary (ISN ≥0.8.0rc7):** import only the public surface
  (`get_grammar_context()`, `create_standard_name_entry()`,
  `run_semantic_checks()`, `validate_description()`, `parse_standard_name()` /
  `compose_standard_name()`). Never import ISN private modules; never hardcode
  grammar rules or vocabulary tokens — pull from `get_grammar_context()`.
  Review criteria live in codex (`sn_review_criteria.yaml`). Boundary detail:
  `docs/architecture/boundary.md`.
- **ISN OWNS ALL GRAMMAR VOCABULARY — NEVER redefine it in codex Python (binding).**
  The set of grammar segments, and the tokens within each segment (subjects,
  physical/geometric bases, channels, populations, orbits, aggregations, zones,
  qualifiers, states, processes, coordinate axes, loci, operators), are defined
  **only** in the `imas-standard-names` project (its `grammar/vocabularies/*.yml`
  + generated `SEGMENT_TOKEN_MAP`). A codex `.py` file must never hold a literal
  list/set/dict of ISN grammar tokens or the ISN segment names — that duplicate
  silently drifts the instant ISN adds/renames/removes one (e.g. a new `state`
  segment, a renamed base). Derive every such set at runtime from
  `get_grammar_context()` (tokens per segment, and the segment list itself).
  - ❌ `_SHAPE_PARAMETER_BASES = frozenset({"triangularity", "elongation", "squareness"})`
    — ISN physical_base tokens hardcoded in codex.
  - ❌ `TIER1_SEGMENTS = frozenset({"physical_base", "subject", ...})` — hardcodes
    the ISN segment-name set; a new ISN segment is silently untiered.
  - ✅ derive from `get_grammar_context()["grammar"]` / `SEGMENT_TOKEN_MAP`; codex
    may still attach codex-only POLICY (search tier, shape-surface flag) keyed by
    those segment names, but the universe of names/tokens comes from ISN, and a
    test must assert every ISN segment is covered so drift fails loudly.
  - **NOT covered by this rule** (legitimately codex): IMAS-DD *path* tokens
    (`rho_tor_norm`, `psi`, `adc`, `ids_properties` in `core/node_classifier.py`),
    raw DD-leaf skip lists (`node_classifier`/`workers` non-nameable coordinates
    like `time`/`delay`/`count`), and DD-path→ISN-token *translation maps* — but
    the ISN-token *side* of any translation map must be validated against the
    ISN vocabulary at load, never assumed.
  When you find a violation, fix it by deriving from ISN (or flag + track it if
  the ISN accessor doesn't exist yet — request the accessor on the ISN side).
- **Closed segments:** *all* grammar segments — including `physical_base` — are
  closed (ISN `SEGMENT_TOKEN_MAP`). A composer "missing token" report against
  `physical_base` is not a real gap; pseudo segments (`grammar_ambiguity`) are
  filtered at write time. When a true gap blocks naming, follow the vocab
  rotation workflow in the architecture doc (add tokens on the ISN fork, cut an
  RC, bump the dep — appears twice in `pyproject.toml`).
- **Propose changes THROUGH `sn edit`, never hand-edit graph text.** A wrong
  name or docs string is fixed by `imas-codex sn edit <name> (--hint TEXT |
  --rename NAME | --docs TEXT) --reason TEXT`, not a Cypher `SET` — hand
  editing bypasses grammar validation, RD-quorum review, and scoring. `--hint`
  steers generate/refine under the grammar; `--rename`/`--docs` skip straight
  to review with a full replacement. `--reason` is mandatory and is shown to
  the reviewer as intent context so a deliberate edit isn't penalized for
  differing from a prior variant — review still scores independently and can
  reject it. See the `.claude/skills/sn-edit` skill and
  [Edit Side-Car](docs/architecture/standard-names.md#edit-side-car) for
  scope/cascade rules and the worked example.

### CLI commands

`sn run` (seven-pool loop), `review`, `preview`, `release`, `import`,
`status`, `coverage`, `clear`, `prune`, `bench`, `edit`. Run
`uv run imas-codex sn <cmd> --help` for flags; semantics and the full flag
matrix are in the architecture doc. Grammar sync is automatic (`sn run`
startup + `sn clear` re-seed); the graph→staging export leg is
`sn release --export-only`.

### Lifecycle axes

Four independent axes on each `StandardName` (full state tables in the doc):

| Axis | States | Driver |
|------|--------|--------|
| `name_stage` / `docs_stage` | `pending → drafted → reviewed → {accepted \| refining → drafted \| exhausted \| superseded}` | pool workers (`refining` reverts after 600 s orphan sweep) |
| `name_stage` | `pending → drafted → reviewed → accepted` (`refining`/`exhausted`/`superseded` side states) | name pipeline + `export` → `import` (catalog round-trip) |
| `status` | `draft → active → {deprecated \| superseded}` | catalog import (ISN vocabulary lifecycle) |
| `validation_status` | `pending → valid \| quarantined` | compose worker (gates review/consolidation/export) |

`origin`: `pipeline` | `catalog_edit` (human-edited; `filter_protected()` skips
`PROTECTED_FIELDS` unless `--override-edits`) | `derived` (structural parent
from the `parents.py` admission gate). `StandardNameSource`:
`extracted → composed | attached | vocab_gap | failed | stale`; ID scheme
`dd:{path}` or `signals:{facility}:{id}`.

### Acceptance & recovery (binding)

- **Never direct-accept a name.** Acceptance is earned only through the
  RD-quorum review pool (`REVIEW_NAME`). Promoting a name to
  `name_stage='accepted'` by hand — a Cypher `SET`, a "blind accept", or any
  code path that sets accepted without a fresh quorum score — is a banned
  anti-pattern, **even when the name is structurally identical to accepted
  siblings**. The single sanctioned structural accept is `ENRICH_PARENTS` for
  placeholder derived parents (systematically penalised by the quorum by
  construction; documented at its emission point) — nothing else.
- **`exhausted` is recoverable, not a dead end.** A sound name that reaches
  the refine cap and lands `exhausted`/`reviewed` on quorum variance is
  recovered with `sn rescore` (`rescore_name` → `stage_name_for_rescore`):
  it reverts the name to `drafted` and resubmits the *same* name for a fresh
  quorum draw — never a reword, never a hand-accept. A name whose siblings
  accept at 0.96+ typically clears on a fresh draw. **Repeated exhaustion of
  structurally-sound names is a pipeline signal to fix** (prompt, quorum
  composition, or threshold), not to paper over with an accept.

### Naming principle (binding)

- **As short as possible while fully retaining semantic meaning, and no
  shorter.** There is no arbitrary character cap — a name is exactly as long
  as its physics requires. (The former 70-char `length_soft_cap` audit was an
  anti-pattern and has been removed.)
- **Ordered sample positions never enter identity.** First/second/third and
  start/end endpoint labels remain in DD provenance, while the Standard Name
  retains the quantity, carrier, representation, owner, axis, mechanism, and
  locus. An unavailable non-ordinal identity is a vocabulary gap, never a
  nearest-object substitution.
- **US spelling throughout.** Names and prose use American spelling
  (`normalized`, `gage`); `american_spelling_check` enforces this from the
  breame UK→US map and quarantines British forms for regeneration.

### Key modules

> **Working inside `imas_codex/standard_names/`?** Read
> [`imas_codex/standard_names/AGENTS.md`](imas_codex/standard_names/AGENTS.md)
> first — the naming-hygiene keep-list (which physics/vocabulary tokens
> legitimately match the plan-label patterns), the attachment-guard failure modes,
> unit authority, and the acceptance rules that bite when editing these files.

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
