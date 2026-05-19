# Research: Deterministic-parent design in the sn run pipeline

> **STATUS:** Implementation in progress as of 2026-05-19. See commits 091b1ece onwards. Phases 0/1/2/3/7 complete; Phases 4/5/6/8/9/11 in progress.

**Status:** Research — early exploration, NOT ready for implementation
**Date:** 2026-05-19
**Pipeline version at write time:** ISNC `v0.2.0rc12`, ISN `v0.8.0rc24`, imas-codex `main @ 2b4ea3c4`
**Author:** investigation session triggered by user feedback on rc7/rc12 parent rendering

> **Note for the next session:** this is a research note, not an implementation
> plan. Several design choices and one or two structural assumptions remain
> open. Treat the "Proposed direction" section as a sketch — read the "Open
> questions" section before doing anything.

---

## 1. Problem statement

Catalog entries like `upper_elongation_of_plasma_boundary` and
`lower_elongation_of_plasma_boundary` shipped with the wrong structural parent
(`elongation` instead of the natural sibling-grouping `elongation_of_plasma_boundary`).
Investigation in the same session fixed the derivation layer (see commits
`74d93ee4`, `2b4ea3c4` in imas-codex; `0748502`, `2649754` in ISN; the
companion catalog/dataset patch shipped as `v0.8.0rc24`). The fix peels one
ISN-grammar layer per `COMPONENT_OF` edge — qualifier and locus layers are now
recognised in addition to the existing operator and projection layers.

That fix exposed a deeper issue, which is what this note is about:

> **The structural derivation pipeline creates and accepts catalog entries
> for parents that are too generic to deserve a standard-name entry.**

By "too generic" we mean:

- `pressure` as a structural parent of `electron_pressure`, `ion_pressure`,
  `thermal_electron_pressure`, …
- `density`, `current`, `area`, `volume`, `temperature`, `radius`, … all
  surface as bare-base parents of qualifier-peeled children.

These are useful as **graph nodes** (they group siblings and answer "what's
the base of this family?") but they are **NOT useful as catalog entries**.
A user looking up `pressure` in the catalog deserves to be told "be more
specific" — not to see an LLM-written paragraph trying to define pressure
in the abstract.

The user's framing: ​*"plain pressure should never get a standard name as
they are too broad … We want to capture parents such as elongation of
plasma boundary, but want to avoid SNs for names such as density and
pressure as these are too vague."*

---

## 2. Current state — what actually happens today

### 2.1 How structural parents are born

Three triggers create a `StandardName` placeholder node for a parent that
nobody composed:

1. **`_write_standard_name_edges`** (`imas_codex/standard_names/graph_ops.py`
   line ~1232) — `MERGE (src:StandardName {id: b.from_name}) MERGE
   (tgt:StandardName {id: b.to_name})` creates a bare node for the target
   if it does not exist. This is how `elongation` was first materialised
   into the graph as a side-effect of writing the `upper_elongation_of_plasma_boundary`
   structural edge.
2. **`rederive_structural_edges`** (`graph_ops.py` line ~1360) — calls
   `_write_standard_name_edges` for every existing SN. After the
   rc11/rc12 derivation fix this added 53 new placeholder parent nodes
   to the graph.
3. **Catalog import / `sn import`** — names that already exist in the
   catalog YAML get MERGE'd in. Less relevant here; the issue is the
   derivation-created ones.

### 2.2 How they get promoted

`backfill_deterministic_parent_origin()` (`graph_ops.py` line ~1389) runs
as a self-healing idempotent pass on every `sn run` start. It performs
three transformations:

1. **Origin stamp.** Any parent that has at least one inbound
   `COMPONENT_OF` edge but no `origin` set is stamped `origin='deterministic'`
   AND snapped directly to `name_stage='accepted'`.
2. **Promotion from reviewed.** Any `origin='deterministic'` parent still
   at `name_stage='reviewed'` is promoted to `'accepted'` (left over from
   a transient routing policy that briefly sent deterministic parents
   through REVIEW_NAME with synthetic low scores).
3. **Description placeholder reset.** Resets `docs_stage IN ['pending',
   'drafted']` deterministic parents to a canonical placeholder description
   so the export's "docs not finalised" guard treats them uniformly.

**The defect** is in step 1: structural parents bypass `GENERATE_NAME` and
`REVIEW_NAME` entirely. Whatever the LLM-name-quality bar is for compose-path
entries, deterministic parents are exempt. They land in the accepted set on
structural existence alone.

### 2.3 What scoring data already exists

Audit of the 74 deterministic parents in the live graph
(`MATCH (sn:StandardName) WHERE sn.origin='deterministic' …`):

| Metric | Count | Fraction |
|---|---:|---:|
| Total deterministic parents | 74 | — |
| With a `reviewer_score_name` | 21 | 28% |
| Without a `reviewer_score_name` | 53 | 72% |
| With a `reviewer_score_docs` | 17 | 23% |
| Without a `reviewer_score_docs` | 57 | 77% |

The 21 with a name score were stamped before the auto-accept policy
firmed up. Sample scores for those:

```
beta                                          name=0.30  (would-fail at 0.65 threshold)
average_magnetic_field                        name=0.30
average_magnetic_flux_due_to_external_coil    name=0.45  docs=0.88  (awkward — see Open Q 4)
```

So the **plumbing** for routing deterministic parents through RD-quorum
review (the `[tool.imas-codex.sn-review]` profile + the `Review` node
machinery + the `reviewer_score_name` slot on `StandardName`) **already
exists**. It's the policy in `backfill_deterministic_parent_origin` step 1
that prevents the existing infrastructure from being applied. The redesign
is therefore primarily a policy and lifecycle change, not new infrastructure.

### 2.4 The orthogonal "kind" taxonomies

Three concepts use the word "kind" — confusing on first encounter, important
to keep separate when reading the codebase. See `tests/standard_names/test_derivation.py`
for examples of each.

#### 2.4.1 Catalog `kind` — algebraic shape of the quantity

- **Source of truth:** ISN `imas_standard_names/models.py` (Pydantic
  discriminator `Literal[...]` on each `StandardNameXxxEntry` subclass).
- **Also declared:** `imas-codex/imas_codex/schemas/standard_name.yaml::StandardNameKind`
  (LinkML enum), generated into `imas_codex/graph/models.py`.
- **Where seen:** `kind:` field in every catalog YAML entry;
  `StandardName.kind` graph property; SPA `n.kind` overwritten by the
  SPA's structural kind (§2.4.2) at dataset-build time.
- **Auto-derivation:** `imas-codex/imas_codex/standard_names/kind_derivation.py:derive_kind(name)`
  runs after LLM compose, pattern-matching the name tokens. LLM defaults
  to `scalar`; this enforces the structurally correct kind.

| Kind | Meaning | Trigger pattern |
|---|---|---|
| **scalar** | Rank-0 quantity. Default. Includes single named components of a vector (a component is a scalar projection). | Default; explicit when name starts with an axis token (`radial_`, `toroidal_`, …) |
| **vector** | Rank-1 quantity. Has a `.magnitude` property derived as `magnitude_of_<name>`. | LLM-assigned for multi-component data |
| **tensor** | Rank-2+ (metric, stress, …). | Contains `_tensor` |
| **eigenfunction** | Eigenfunction of a linear operator (MHD, wave, …). | Contains `_eigenfunction` |
| **spectrum** | Per-mode / per-frequency decomposition. | Ends with `_spectrum` |
| **complex** | Complex-valued (real/imag or magnitude/phase). | Contains `real_part` / `imaginary_part` |
| **metadata** | Non-measurable identifier — no unit required, no provenance. | LLM-assigned for descriptive / structural identifiers |

#### 2.4.2 SPA `kind` — display classification (UI only)

- **Source of truth:** `imas-standard-names/imas_standard_names/catalog/dataset.py:_structural_kind()`.
- **Where seen:** **only** `data.json` at SPA build time; not stored in
  graph or YAML. Overwrites the catalog `kind` on the SPA's `n.kind`
  field.
- **Derivation:** ISN grammar parse, inspects the IR, returns one of:

| SPA kind | Meaning | Trigger |
|---|---|---|
| **location** | Pass-through from catalog `kind == "metadata"` | `kind == "metadata"` |
| **at_point** | Quantity evaluated at a locus (`safety_factor_at_magnetic_axis`) | IR carries `locus` |
| **component** | Axis projection of a vector (`toroidal_magnetic_field`) | IR carries `projection` |
| **global** | Reduction / subject-scalar over a whole region (`total_plasma_current`, `minimum_safety_factor`) | `total_*` prefix OR reduction qualifier OR global-subject qualifier |
| **base** | Field-valued, no locus / projection / reduction. Default. | Otherwise |

Used by the SPA's KindBadge glyph (≡/⇕/⊙/▢/⛬). Not relevant to the
parent-redesign discussion except as a vocabulary nuisance.

#### 2.4.3 `ArgumentRef.operator_kind` — structural-edge layer label

- **Source of truth:** ISN `models.py::ArgumentRef.operator_kind` (Literal).
- **Where seen:** every `arguments[*].operator_kind` in catalog YAML;
  graph `COMPONENT_OF.operator_kind` edge property.
- **Set by:** `imas-codex/imas_codex/standard_names/derivation.py:derive_edges()`.

| operator_kind | Meaning | Other fields |
|---|---|---|
| **unary_prefix** | Outermost is unary-prefix operator (`maximum_of_x`, `time_derivative_of_x`) | All other forbidden |
| **unary_postfix** | Outermost is unary-postfix operator (`x_magnitude`, `x_moment`) | All other forbidden |
| **binary** | Binary operator (`ratio_of_x_to_y`) — emits two edges | `role ∈ {a,b}`, `separator ∈ {and,to}` required |
| **projection** | Axis projection (`toroidal_magnetic_field`) | `axis`, `shape ∈ {component,coordinate}` required |
| **qualifier** | Outermost qualifier (`upper_…`, `electron_…`, `volume_averaged_…`) — added rc11 | All other forbidden |
| **locus** | Locus suffix peel (`…_of_plasma_boundary`, `…_at_magnetic_axis`) — added rc11 | All other forbidden |

The structural-parent design lives entirely on the `operator_kind` axis.
The catalog `kind` (§2.4.1) and SPA `kind` (§2.4.2) are unaffected by the
redesign.

---

## 3. What "too broad" looks like in practice

The 70 names affected by the rc11/rc12 derivation fix (i.e. their parents
got fixed) decompose into these structural parents. Children-count is the
in-degree of `COMPONENT_OF` after backfill. Sorted by children desc:

```
pressure                            (12 children)   ← bare base, too broad
density                             (8 children)    ← bare base, too broad
area                                (3 children)    ← bare base, too broad
current                             (3 children)    ← bare base, too broad
volume                              (3 children)    ← bare base, too broad
radius_of_poloidal_field_coil       (3 children)    ← specific, keep
elongation_of_plasma_boundary       (2 children)    ← specific, keep
inductance                          (2 children)    ← bare base, edge case
plasma_current                      (2 children)    ← specific enough?
plasma_energy                       (2 children)    ← specific enough?
ion_temperature                     (2 children)    ← qualifier-only, edge case
temperature                         (2 children)    ← bare base, too broad
velocity                            (2 children)    ← bare base, too broad
inner_squareness                    (2 children)    ← qualifier-only, edge case
outer_squareness                    (2 children)    ← qualifier-only, edge case
major_radius_of_flux_surface        (2 children)    ← specific, keep
… 38 single-child parents
```

There is no clean syntactic predicate that separates "keep" from "drop"
purely from the name string. Some bare bases (`pressure`, `density`) are
clearly too broad; others (`triangularity`, `safety_factor`) are perhaps
specific enough to deserve a definition. Some qualifier-only names
(`ion_temperature`, `inner_squareness`) sit awkwardly between. **A
deterministic syntactic rule will misclassify in both directions.** The
redesign therefore uses the existing RD-quorum name reviewer as the
classifier.

---

## 4. Proposed direction — sketch

> **This section is a sketch.** Open questions remain about thresholds,
> recovery, and the exact lifecycle vocabulary. The next session should
> treat this as a starting point, not a specification.

### 4.1 Goals

1. **Filter generic parents.** A name that fails the name-axis review at
   the configured threshold should be **quarantined** — kept in the graph
   for structural traversal, never exported, never linked from children's
   prose.
2. **Keep specific parents.** A name that passes review is accepted and
   exported normally.
3. **Decouple name review from docs refinement.** Once a parent's name
   passes review, the name axis is **frozen** for that node. Subsequent
   refinement only edits the docs. `REFINE_NAME` does not apply to
   structurally-derived names — if the name is wrong, the issue is in
   the grammar or the child name, not in this entry.

### 4.2 Lifecycle (proposed)

Replace the current "stamp `deterministic` + snap to `accepted`" with a
two-step admission:

```
NEW node states for deterministic parents:

  origin=deterministic-pending,  name_stage=pending
    ← initial state when _write_standard_name_edges creates the placeholder
       ↓ GENERATE_NAME-LITE (no LLM call — the name is structurally given)

  origin=deterministic-pending,  name_stage=drafted,
                                 reviewer_score_name=None
       ↓ REVIEW_NAME (same RD-quorum, structural-context prompt extension)

  origin=deterministic-pending,  name_stage=reviewed,
                                 reviewer_score_name=F
       ├─ F ≥ threshold → origin=deterministic,
       │                  name_stage=accepted
       │                  (flows into GENERATE_DOCS as today)
       │
       └─ F < threshold → origin=deterministic-quarantined,
                          name_stage=quarantined
                          (excluded from export; prose links scrubbed)

  origin=deterministic,  docs_stage=drafted/reviewed/exhausted
       ↓ REFINE_DOCS (existing pool; docs-only — name is frozen)

  origin=deterministic,  docs_stage=accepted    ← terminal good state

NEVER applied to deterministic parents:
  • REFINE_NAME
  • GENERATE_NAME's LLM compose path
```

### 4.3 What needs to change vs what gets reused

**Reused as-is:**

- RD-quorum review pipeline (`imas_codex/standard_names/review/pipeline.py`)
- `Review` node schema and write semantics
- `reviewer_score_name` storage on `StandardName`
- Cost accounting (`LLMCost`, `BudgetManager`)
- REFINE_DOCS pool and prompts
- Export gates (with one extra filter — quarantined origin)

**Net new:**

- Two new `origin` values: `deterministic-pending`,
  `deterministic-quarantined`. Adds to `standard_name.yaml` LinkML;
  regenerate.
- Structural-context channel in the REVIEW_NAME prompt — the reviewer
  sees which children reference this name, what layer was peeled, the
  inner/outer entries. Lets the reviewer judge specificity properly.
- Quarantine cascade — when a parent quarantines, walk
  `(child)-[:COMPONENT_OF]->(parent)` and rewrite any `(name:<parent>)`
  references in `child.documentation` to plain anchor text. Re-uses the
  existing `resolve_doc_links` dead-link removal path; only the
  predicate changes from "superseded" to "quarantined".
- `backfill_deterministic_parent_origin` retired in favour of
  `backfill_deterministic_parent_review` — the new pass stamps
  `deterministic-pending` (NOT `accepted`).

**Removed / disabled:**

- The current auto-`accepted` policy in step 1 of
  `backfill_deterministic_parent_origin`.
- The REFINE_NAME pool's eligibility for `origin LIKE 'deterministic%'`
  rows. (Today refine_name never sees them because they're already
  accepted; under the new lifecycle they still shouldn't.)

### 4.4 Migration

For the 74 existing deterministic parents in the live graph:

- **53 without a name score** → reset to `origin=deterministic-pending,
  name_stage=pending`. They'll go through REVIEW_NAME on the next
  `sn run`.
- **21 with an existing name score** → apply the threshold immediately:
  - `≥ threshold` → keep `origin=deterministic`, `name_stage=accepted`
  - `< threshold` → flip to `origin=deterministic-quarantined`,
    `name_stage=quarantined`; run the quarantine cascade to scrub
    children's prose
- One-shot Cypher pass; idempotent; safe to re-run.

### 4.5 Code-level surface (rough mapping)

| Concern | File | Change shape |
|---|---|---|
| New origin enum values | `imas_codex/schemas/standard_name.yaml` | LinkML edit; regenerate models |
| Placeholder stamping | `imas_codex/standard_names/graph_ops.py:_write_standard_name_edges` | Replace inline `MERGE (tgt {id:...})` with one that stamps `origin=deterministic-pending, name_stage=pending` for newly created nodes |
| Backfill retirement | `imas_codex/standard_names/graph_ops.py:backfill_deterministic_parent_origin` | Replace with `backfill_deterministic_parent_review`; remove the auto-accept step |
| REVIEW_NAME structural context | `imas_codex/standard_names/review/contextualiser.py` (or wherever the prompt context is built) | New channel `structural_origin` derived from live COMPONENT_OF edges |
| Review prompt | `imas_codex/llm/prompts/sn/review_names.md` | `{% if structural_origin %}…{% endif %}` block |
| Quarantine cascade | new `imas_codex/standard_names/quarantine.py` | Calls existing `resolve_doc_links` machinery with a new predicate |
| Export filter | `imas_codex/standard_names/export.py:_get_export_candidates` | Add `origin <> 'deterministic-quarantined'` to the gate |
| Pool wiring | `imas_codex/standard_names/pools.py`, `workers.py` | `REFINE_NAME` claim queries must exclude `origin LIKE 'deterministic%'`. `GENERATE_NAME` must short-circuit for `origin='deterministic-pending'` (no LLM, just transition to drafted) |
| SPA fallback | `imas-standard-names/imas_standard_names/catalog/dataset.py:_parent_token` | **No change needed** — already prefers `entry["arguments"]` (which atomic-drops when target is missing from export) over local IR peel |

### 4.6 What I deliberately did NOT propose

- **No new pool taxonomy.** Reuse REVIEW_NAME / REFINE_DOCS as-is; only
  gate them differently for deterministic-pending entries.
- **No LLM compose for deterministic-pending.** The name IS the
  structural id. Skipping compose saves cost and avoids the
  "LLM tries to refine the structural name" failure mode.
- **No permanent storage of the structural-origin context.** Derive it
  at review time from the live `COMPONENT_OF` edges. Cheap and always
  fresh; no stale-cache problem.
- **No catalog-edit override semantics.** If a deterministic-quarantined
  name ever needs to come back, that's a user-driven action (see Open Q 2),
  not an automated catalog round-trip.

---

## 5. Open questions

> **Resolve before implementing.** Each of these is load-bearing.

### Q1 — What threshold?

`[tool.imas-codex.sn-review]` currently uses `min_score=0.65` for the
compose-path name axis. Open: do we use the same bar for structural
parents, or a different one? Arguments either way:

- **Same (0.65).** Single config value, single mental model. The
  reviewer prompt already knows the scoring rubric for catalog-worthy
  names — structural origin doesn't change that.
- **Higher (e.g. 0.75) for structural parents.** A structural parent's
  job is to anchor a family, not to describe a measurement. The bar
  should be higher because the prose is purely curatorial — no DD
  paths anchor it.

Need empirical data before deciding. The 21 already-scored deterministic
parents straddle the band: `beta=0.30`, `average_magnetic_field=0.30`,
`average_magnetic_flux_due_to_external_coil=0.45` would all quarantine at
0.65 (and intuitively all three SHOULD quarantine — they're too
generic). Higher scores in the 0.6–0.8 band are where intuition
diverges.

### Q2 — Quarantine recovery

If a parent quarantines but the reviewer was wrong (false positive
quarantine), how do we get it back?

- **Option A: never automatic.** Once quarantined, requires a manual
  `sn parent-unquarantine <name>` CLI invocation or a graph edit.
- **Option B: re-review on threshold change.** When `min_score` is
  lowered or model chain changes, quarantined parents re-enter the
  review queue. Risk: surprises the user; same name oscillates between
  states as config evolves.
- **Option C: re-review on vocabulary change.** If the ISN grammar
  changes such that the structural derivation produces a different
  edge set, affected parents auto-rederive. Quarantine survives across
  unrelated changes.

The user's prompt language ("discarded (or quarantined)") suggests
they're comfortable with quarantine being relatively sticky. I lean
toward Option A + Option C.

### Q3 — Do bare bases (`pressure`, `density`, `temperature`) count as
parents at all?

The new derivation never **directly** emits a `COMPONENT_OF →
pressure` edge from any child — `pressure` arises only as the
**inner** of `_local_ir_peel("electron_pressure")` (one qualifier
peel). Today `_write_standard_name_edges` MERGE-creates a placeholder
StandardName node for `pressure` to anchor the edge.

The redesign has two viable paths:

- **Path A: treat bare bases as quarantine candidates** (current
  proposal). Let the reviewer judge `pressure` against the threshold
  and quarantine it if too broad. Single rule for all parents.
- **Path B: never create a StandardName node for IR-leaf parents** —
  treat them as "category labels" not "catalog entries." Skip placeholder
  creation in `_write_standard_name_edges` when the target is an
  IR leaf with no other structural layers. The COMPONENT_OF edge still
  carries the structural information; the leaf just isn't a catalog
  entry.

Path B is more decisive but loses optionality (you can't ever have a
catalog entry for `pressure`, even if a reviewer decided you should).
Path A defers the decision to the reviewer per case. I lean toward A
for that reason — but the engineering cost of B is lower.

### Q4 — Awkward mixed scores

`average_magnetic_flux_due_to_external_coil` scored `name=0.45,
docs=0.88`. Rich documentation under a poor-quality structural name.
Three handling options:

- **Quarantine outright** — the docs go with the name. Loses good prose.
- **Demote to structural-only** — keep the graph edge for traversal,
  suppress export. Docs survive in the graph for future re-promotion;
  catalog stays clean.
- **Treat as accept** — the docs prove the entry has value; trust the
  prose over the name score.

The redesign currently assumes quarantine. The user's framing
("**if the SN scores low (below threshold) on the name alone**, then
it should be discarded") supports quarantine — but acknowledges the
"alone" qualifier. Worth a discussion.

### Q5 — What does the reviewer prompt look like?

Today's `review_names.md` is tuned for LLM-composed names where the
LLM "had a chance" to write something correct. For structurally-derived
names the reviewer needs different framing:

> "This name was derived structurally by peeling the `upper` qualifier
> off `upper_elongation_of_plasma_boundary`. The result, `elongation_of_plasma_boundary`,
> would anchor a family of 2 children currently in the catalog. Is the
> derived name `elongation_of_plasma_boundary` itself worth a catalog
> entry — or is it too broad / inherently a category-label?"

Need to draft this prompt and run a small benchmark — the existing
`sn-benchmark` tooling can score the prompt variations.

### Q6 — Does the SPA need any changes?

After the rc11/rc12 derivation + dataset fix, `_parent_token` already
prefers `entry["arguments"][0]["name"]`. When a parent quarantines, its
`arguments` row drops out of the export atomically (RC-mode rule),
which means the child's YAML drops `arguments` — and `_parent_token`
falls through to `_local_ir_peel(name)`, which still surfaces the
correct structural parent string. **Net: zero SPA changes** if we
believe the local IR peel is good enough as a display-only fallback.

But: the SPA will show a parent link pointing to a name that isn't in
the catalog — clicking it renders the "Not yet in catalog" state. Is
that acceptable UX, or do we want the SPA to suppress the parent link
entirely when the target is quarantined? Today `NameLink.missing` is
already styled differently (gray, no click). That may be enough.

---

## 6. Pointers for the next session

### 6.1 Reproducing the rc12 graph state

```bash
cd ~/Code/imas-codex
uv run python -c "
from imas_codex.graph.client import GraphClient
with GraphClient() as gc:
    # The 74 deterministic parents
    rows = list(gc.query('''
        MATCH (sn:StandardName) WHERE sn.origin='deterministic'
        OPTIONAL MATCH (child)-[r:COMPONENT_OF]->(sn)
        RETURN sn.id AS id,
               sn.reviewer_score_name AS s_name,
               sn.reviewer_score_docs AS s_docs,
               count(r) AS in_degree
        ORDER BY in_degree DESC, sn.id
    '''))
    for r in rows: print(r)
"
```

### 6.2 Key files to read first

1. **`imas_codex/standard_names/derivation.py`** — the per-layer peel.
   This is the source of every `COMPONENT_OF` edge. The qualifier and
   locus layers were added in commits `74d93ee4` and `2b4ea3c4`. Read
   `_derive_structural` end-to-end before changing anything.
2. **`imas_codex/standard_names/graph_ops.py`** — three functions matter:
   - `_write_standard_name_edges` (line ~1161) writes the
     `COMPONENT_OF` edges and MERGE-creates placeholder parents.
   - `rederive_structural_edges` (line ~1360) backfills the graph.
   - `backfill_deterministic_parent_origin` (line ~1389) is the
     function this redesign retires.
3. **`imas_codex/standard_names/review/pipeline.py`** — the RD-quorum
   review pool. The new lifecycle reuses this verbatim; only the claim
   query changes (include `origin=deterministic-pending` in the eligible
   set).
4. **`imas_codex/standard_names/export.py`** — `_derive_arguments_for_entry`
   (line ~569) builds the YAML `arguments:` block from graph
   `COMPONENT_OF` edges. The RC-mode atomic-drop rule lives in
   `_run_gate_b` (line ~291).
5. **`imas-standard-names/imas_standard_names/catalog/dataset.py:_parent_token`** —
   confirms the SPA already does the right thing as a side-effect of
   the rc11/rc12 fix. Do NOT undo that change.
6. **AGENTS.md** sections "RD-Quorum Review" and "Standard Names" — the
   policy guardrails this redesign must respect (e.g. axis-split review
   storage, the cost-budget model).

### 6.3 Don't-skim list

Read these even if they seem off-topic:

- `tests/standard_names/test_derivation.py` (D26-D29) — the
  per-layer peel invariants
- `tests/standard_names/test_link_dedup.py` — the dedup machinery the
  quarantine cascade will reuse
- `plans/research/standard-names/10-implementation-review.md` — the
  previous implementation-review note; useful framing for "should this
  go to features/?" decisions
- `docs/architecture/standard-names.md` and
  `docs/architecture/standard-names-decisions.md` — the canonical
  architecture docs; the redesign must remain consistent with the
  "axis-split review storage" and "score-canonical policy" stated
  there

### 6.4 What to do before implementing

1. **Pick a threshold (Q1)** — sample 10 deterministic parents from
   the live graph, run them through `review_names.md` manually (paste
   the prompt + the structural context into a chat with the current
   reviewer-model chain), and look at the scores. Decide whether 0.65
   is well-calibrated or you need a higher bar for structural-only
   entries.
2. **Decide Path A vs Path B (Q3)** — re-read the user prompt
   verbatim; it implies Path A but Path B is mentioned obliquely
   ("these names would fail review"). One sentence in the user's
   prompt could resolve this.
3. **Draft the reviewer prompt (Q5)** — propose a `review_names.md`
   diff that adds the `{% if structural_origin %}` block. Run on the
   10-sample set from step 1.
4. **Plan the migration (§4.4)** — write the Cypher migration as
   `imas_codex/standard_names/migrations/<n>_deterministic_review.py`
   following the pattern of existing migrations. Idempotent + dry-run
   support.
5. **Confirm SPA UX (Q6)** — load `data.json` from a hypothetical
   post-quarantine state (manually edit a copy: remove `arguments` from
   a few entries, remove the quarantined parents themselves) and walk
   the SPA. Confirm the missing-parent state is acceptable.

Only THEN write the first implementation commit.

---

## 7. Out of scope

The following are not part of this redesign:

- **Renaming `elongation` to `elongation_of_closed_flux_surface`** etc.
  That's a vocabulary decision living in ISN's grammar / vocab files,
  not in imas-codex's pipeline. Surfaced by the user in the same
  message but architecturally orthogonal — solve separately if at all.
- **Changing the `kind` enums (§2.4).** Quarantine is purely about
  `origin` and `name_stage`. The algebraic-shape and SPA-display kinds
  are unaffected.
- **The 70 child entries that benefit from the rc11/rc12 fix.** Those
  ship correctly already; this redesign only governs the parents they
  link to.

---

## 8. Sign-off checklist (for the implementing session)

Before merging the implementation commit:

- [ ] Migration is idempotent (re-run produces zero changes)
- [ ] Migration dry-run mode reports affected counts without writing
- [ ] All existing `test_derivation.py` cases still pass
- [ ] New tests cover: the lifecycle transitions; the quarantine
      cascade; the prompt's `structural_origin` block rendering
- [ ] AGENTS.md `## Standard Names` section updated to reflect the new
      lifecycle states
- [ ] `imas_codex/schemas/standard_name.yaml::origin` enum extended;
      models regenerated; `agents/schema-reference.md` rebuilt by
      `uv sync`
- [ ] No commit is on a branch other than `main`
- [ ] Each commit ends with `git push origin main`
- [ ] No `git stash` anywhere in the history (per AGENTS.md ban)

---

## Appendix A — Audit query (replay in any session)

```cypher
// Deterministic parents missing a name score
MATCH (sn:StandardName) WHERE sn.origin = 'deterministic'
OPTIONAL MATCH (child)-[r:COMPONENT_OF]->(sn)
RETURN sn.id AS id,
       sn.reviewer_score_name AS s_name,
       sn.reviewer_score_docs AS s_docs,
       count(r) AS in_degree
ORDER BY in_degree DESC, sn.id
```

## Appendix B — Affected releases

| Release | Repo | Change |
|---|---|---|
| `IMAS-Standard-Names@v0.8.0rc23` | ISN fork | `ArgumentRef.operator_kind` widened with `qualifier`, `locus` |
| `IMAS-Standard-Names@v0.8.0rc24` | ISN fork | `entry_schema.json` regen'd to match |
| `imas-codex@74d93ee4` | imas-codex | qualifier + locus peel layers in `derive_edges` |
| `imas-codex@2b4ea3c4` | imas-codex | migration self-loop guard |
| `imas-standard-names-catalog@v0.2.0rc12` | ISNC | first RC with both Catalog Site and Release CI green; deployed at https://simon-mcintosh.github.io/imas-standard-names-catalog/v0.2.0rc12/ |
