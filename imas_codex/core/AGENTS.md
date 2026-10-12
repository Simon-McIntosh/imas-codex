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
- `sn harmonize` reports pattern families read-only (members, anchor, drift,
  lint counts by class, scope total) over one of `--batch`, `--family`,
  `--concept`, `--all`; it writes nothing. Never edit docs by hand.


### Correcting a Standard Name fault

**Choose by total operator effort, not by avoiding model charges.** Under the
production routing, name generation and composition use the local model and are
free; reads, censuses and dry runs are also free. Name review is paid, as are
documentation generation, documentation review and both name and documentation
refinement. A thirteen-name correction can consume roughly a day and dozens of
coordination actions while one rescore that releases three blocked rows records
only $0.49 in paid pipeline spend. Regenerate names freely when regeneration is
the right mechanism; concentrate paid review work into batches, and do not
replace a pipeline operation with manual graph surgery.

Use this route in order:

1. **Classify the fault.** A wrong spelling of an existing name is a rename; a
   source with no name is generation; a source bound to the wrong name is an
   attachment repair; a stale scalar, impossible claim or tangled lineage is an
   infrastructure repair. These operations preserve different provenance, so
   mixing them turns one fault into several.
2. **Repair infrastructure before data.** If a legitimate class is refused by
   a claim predicate, migration compare-and-set or lifecycle guard, repair that
   mechanism first and then apply the whole class. Repeatedly steering individual
   rows around the same refusal buys no correctness and multiplies review work.
3. **Stage in bulk, review once.** `sn edit --stage-only` is the bulk/scripted
   migration route. Stage the complete rename cohort, verify the dry-run and
   queued-cohort census, then drain it in one scoped batch instead of starting
   an inline paid review run for every row. Twenty-eight renames queued together
   require one batch review rather than twenty-eight inline review cycles.
4. **Then spend on the pipeline.** Once the spellings, bindings and guards are
   correct, run `sn run` and let each pool perform the work it owns. Do not
   hand-route or hand-accept an item that a pool claims.

The tool boundaries are part of the route:

| Tool | Use it for | Boundary that must stay visible |
|------|------------|---------------------------------|
| `sn run --only compose` | Run name composition work. | `compose` is a broad phase label: it selects `generate_name`, `generate_docs` and `enrich_parents`. It is not the name-composition pool alone. |
| `sn run --only review` | Drain review-oriented work, commonly with `--edits` after bulk staging. | `review` is also broad: it selects `review_name`, `refine_name`, `generate_docs`, `review_docs`, `refine_docs` and `enrich_parents`. Use `--only review_name` when exactly the single name-review pool is intended. |
| `sn run --flush` | Drain already-queued downstream work. | Flush skips auto-seeding and structurally excludes only `generate_name`; it still runs `review_name`, `refine_name`, every docs pool and `enrich_parents`. It cannot mint a name for a source that has none. |
| `sn edit --hint` | Steer regeneration through generate → review → score. Use `--axis both` when the name and its short description must both change. | The pipeline still composes the result. This is the regeneration route, not a literal replacement. |
| `sn edit --rename` | Queue a complete successor spelling and enter `review_name` without generation. | It is a provenance-preserving rename, not a short-description editor. A successor inherits documentation only when its parsed IR is equal to the predecessor's, or differs solely in `locus.token` with every other locus and IR field unchanged. |
| `sn edit --docs` | Replace long-form `documentation` and enter `review_docs`. | It does not replace the short `description`. There is no direct short-description replacement mode; change that field by regeneration. |
| `sn edit --kind` | Repair the unchanged identity's structural kind after `derive_kind()` agrees. | This fourth, non-pipeline edit mode previews by default and writes only with `--apply`; it changes neither wording nor lifecycle axes. |
| `sn rescore NAME` | Give one stranded, non-accepted name a fresh review quorum. | Only `reviewed` and `exhausted` names are eligible. It moves the same identity to `drafted`, preserves lineage and spent refine attempts, and runs exactly `review_name`; it cannot revive an accepted, superseded or already-live name. |
| `sn detach DD_PATH NAME` | Remove one semantically wrong source-to-name realization, including its DD-side projection, and record the judgement. | It refuses to orphan a name with only that attachment. The only ordinary exception is a derived parent anchored by live children; a retired identity is not an exception and its historical binding must not be detached. |
| `sn attach DD_PATH NAME` | Bind an unbound, already-extracted DD source to the existing stable name it realizes. | It writes the provenance edge, DD projection and scalar/cache mirrors together, consults the attachment guard, and refuses to repoint a source already bound to a live name. Repoint explicitly as detach, then attach. |
| `sn remove-source-backing SOURCE DD_PATH` | Remove one wrong `StandardNameSource -[:FROM_DD_PATH]-> IMASNode` backing while retaining at least one other backing. | It does not remove a source-to-name `PRODUCED_NAME` edge. Using it to drop a name binding is a category error. |
| `sn remove-lineage SUCCESSOR PREDECESSOR` | Remove one incorrect directed `REFINED_FROM` edge with a recorded reason. | Direction is checked. A superseded predecessor must retain another incoming successor; outgoing predecessor history cannot satisfy that guard. |
| `sn recover-terminal-attachments` | Return an exact manifest-bound cohort whose sources were finalized against terminal names to fresh composition. | The default is a zero-write plan. Apply requires the exact manifest SHA-256 and atomically removes the terminal realization, resets the source and writes both retry and change receipts; any cohort or compare-and-set mismatch refuses the whole operation. |

### Resume state unless a reset is the intended repair

A name at `drafted`, `reviewed`, `refining`, or a docs-axis intermediate state
is ordinary pipeline state. Continue its eligible pool with a bounded `sn run`;
do not erase its reviews, lineage, source binding, or already-paid attempts just
because the cohort has not yet completed. The command forms below name the few
cases where changing state is the actual repair. Their implementations live in
`imas_codex/cli/sn.py` and `imas_codex/standard_names/graph_ops.py`.

| Situation | Command form | Use it when | Do not use it when |
|---|---|---|---|
| A source failed or was deliberately skipped, and a concrete repair now makes a fresh composition meaningful. | `uv run --no-sync imas-codex sn retry --failed <dd-path> --reason "<evidence>" --dry-run` | The exact source is blocked and the reason records why a retry is justified. `--failed` also releases extracted sources at the compose-attempt cap. | The source is merely progressing through ordinary pipeline lifecycle states; a retry is a lifecycle transition, not a way to hurry an in-flight claim. |
| A drafted cohort needs a deliberate re-compose without starting generation. | `uv run --no-sync imas-codex sn run --reset-to drafted --since <timestamp> --before <timestamp> --reset-only` | The intended operation is to reset the explicitly bounded drafted cohort, inspect the outcome, and then run the next bounded pass. | You only need the eligible review or refine pool to continue. Resetting discards useful in-progress state rather than using it. |
| A bounded cohort must be rebuilt from its authoritative source facts. | `uv run --no-sync imas-codex sn run --reset-to extracted --since <timestamp> --before <timestamp> --reset-only` | The matching name nodes must be cleared so composition starts again from their DD sources. | A spelling is known. Use `sn edit --rename` for a successor or `sn edit --hint` for regeneration; `--force` is not a rename route. |
| A catalog-authoritative accepted name is deliberately included in a reset or deletion cohort. | Add `--include-accepted` to the guarded reset or prune command after a dry run. | The operation was explicitly reviewed as affecting export-eligible identities. | A routine recovery can avoid accepted names. `sn clear` has no equivalent guard and is a full subsystem wipe, never a targeted recovery tool. |

Every reset preserves lineage and documentation history; it changes claimability
or stage, not the evidence trail. Dry-run first, then state the exact selected
cohort and the reason in the applied operation.

Name generation is source-claim driven, not an in-place renderer. The
`generate_name` claim takes only `StandardNameSource` rows at `status='extracted'`
with `attempt_count < 5`, then charges the attempt when it claims the row. A
normally attached source is therefore not reclaimed. The claim explicitly
freezes linked `approved` and `contested` names, not every `accepted` name; a
reset or reseed that makes the source claimable can create another candidate
identity rather than re-rendering the existing node. In particular, do not use
`--force` as a rename primitive: the DD pool route does not pass that flag into
its pool call. Use `sn edit --rename` for a known successor spelling or
`sn edit --hint` when regeneration is required.

Migration preflights compare a source's **live** bindings and ignore
`superseded` or `exhausted` bindings. Those retired edges are generation and
lineage provenance: preserve them, and never detach them merely to make a
migration pass. If an apply still refuses after the live-binding preflight is
clean, treat that as an infrastructure guard failure and repair the
compare-and-set; deleting history is not a workaround.

No `sn` verb repairs a `status` scalar that disagrees with the same node's
`name_stage`. Stop on that state and repair the lifecycle writer or guarded
reconciliation path that produced it. Do not conceal the disagreement with a
hand-written Cypher `SET`.


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
