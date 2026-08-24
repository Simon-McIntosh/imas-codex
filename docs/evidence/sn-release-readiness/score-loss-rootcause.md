# Names-axis score-loss root cause

## Verdict

The live 96-identity class closes with no residual:

| Outcome | Count | What happened | Would the symmetric catalog-provenance guard at `40966ff0` prevent it? |
|---|---:|---|---|
| Active scalar strip | **6** | `write_name_review_results()` accepts a result without `reviewer_score` or `reviewed_at`, maps both missing values to `None`, and unconditionally writes those nulls while preserving the stage, review relationships, per-dimension scores, comments, and existing model. | **No.** This is a pipeline axis writer, not a catalog writer; the guard is never called. |
| Winning review exists but scalar projection is missing | **82** | Structural acceptance writes `name_stage='accepted'` and `reviewer_model_name='structural-inheritance'` but deliberately omits `reviewer_score_name`, even when the row already has a schema-declared winning names-review group. | **No.** The incoming structural update omits the scalar; the catalog guard detects replacement or clearing of existing authority, not a missing projection, and this path does not call it. |
| Null is correct because no winning group exists | **8** | Each row has one names-axis review, but its `resolution_method` is null. None has a `single_review`, `quorum_consensus`, or `authoritative_escalation` winner to project. | **No, and it should not.** There is no score authority to mirror. The structural authority, rather than a fabricated score, is the valid acceptance basis. |
| **Total** | **96** | **6 + 82 + 8 = 96** | **The guard prevents none of this third mechanism.** |

This is neither the July catalog-import provenance wipe nor the documentation resolution-method mirror omission. The import class had no surviving names-axis review records and is now refused symmetrically. The class here is defined by the opposite fact: every row has at least one names-axis `StandardNameReview` while its scalar is null.

## Live measurement

The read-only snapshot selected:

```cypher
MATCH (sn:StandardName)
WHERE sn.name_stage = 'accepted'
  AND sn.reviewer_score_name IS NULL
  AND EXISTS {
    MATCH (sn)-[:HAS_REVIEW]->(review:StandardNameReview)
    WHERE review.review_axis = 'names'
  }
RETURN sn
```

It reproduced the measured class exactly:

- accepted: **96**;
- superseded with the same review-plus-null-score shape: **63**;
- pending with the shape: **1**;
- `StandardName` count before: **4,666**;
- `StandardName` count after: **4,666**;
- graph delta: **0**;
- LLM calls: **0**.

The winning methods were read from the LinkML `ReviewResolutionMethod` authority through `_winning_review_resolution_methods()`, not hard-coded into the census: `single_review`, `quorum_consensus`, and `authoritative_escalation`.

The accepted population's write fingerprints are:

| Fingerprint | Winner? | Count | Classification |
|---|---:|---:|---|
| `reviewer_model_name` is a real reviewer model; no structural authority; scalar and `reviewed_name_at` null while per-dimension scores/comments/model survive | yes | **6** | active strip |
| `reviewer_model_name='structural-inheritance'`; structural authority present | yes | **78** | missing projection |
| `reviewer_model_name='structural-inheritance'`; no structural authority; winning review present | yes | **4** | missing projection on the legacy childless structural rows |
| `reviewer_model_name='structural-inheritance'`; structural authority present | no | **8** | correct null |

Thus the projection bucket is **78 + 4 = 82**. Of the 90 structural-marker rows, 89 lack `parent_enriched_at` and carry the startup structural-promotion fingerprint; one, `efficiency_of_plant_system`, carries `parent_enriched_at` and the enrichment-persistence fingerprint.

### Six active strips

The six identities are:

- `iron_density_at_plasma_boundary`
- `toroidal_deuterium_tritium_velocity_at_plasma_boundary`
- `x_minor_axis_unit_vector_of_shatter_cone`
- `y_direction_unit_vector_of_shatter_cone`
- `z_direction_unit_vector_of_camera`
- `z_minor_axis_unit_vector_of_shatter_cone`

All six are `name_stage='accepted'`, have a winning names-review group, retain non-null `reviewer_scores_name`, comments, and `reviewer_model_name='openrouter/qwen/qwen3.7-max'`, but have both `reviewer_score_name=NULL` and `reviewed_name_at=NULL`. Representative retained per-dimension values are:

- `iron_density_at_plasma_boundary`: grammar 20.0, semantic 19.5, convention 20.0, completeness 19.5; attached winning score 1.0;
- `toroidal_deuterium_tritium_velocity_at_plasma_boundary`: grammar 20.0, semantic 19.5, convention 20.0, completeness 20.0; attached winning scores include 1.0;
- `z_minor_axis_unit_vector_of_shatter_cone`: all four dimensions 20.0; attached winning score 1.0.

That asymmetric survival is the exact write fingerprint at `imas_codex/standard_names/graph_ops.py:6294-6335`:

- line 6299 unconditionally sets `sn.reviewer_score_name = b.reviewer_score_name`;
- line 6300 unconditionally sets `sn.reviewed_name_at = b.reviewed_name_at`;
- lines 6301-6305 coalesce the per-dimension score, comments, and model fields, preserving their prior values;
- lines 6315-6317 build the first two payload values with `e.get(...)`, so a malformed or partial entry silently turns both into `None`;
- the query does not constrain or update `name_stage` and does not remove `HAS_REVIEW`.

This path **NULLS existing scalar state**. It is not merely a failure to fill a new field.

### Eight correct nulls

These eight structurally-authorized identities have no winning group and therefore no score that may legitimately be projected:

| Identity | Attached reviews | Resolution methods | Stored review score |
|---|---:|---|---:|
| `fast_neutral_pressure` | 1 | none | 0.8000 |
| `flux_due_to_beam_beam_fusion` | 1 | none | 0.6375 |
| `ion_momentum_flux` | 1 | none | 0.5125 |
| `magnetic_vector_potential` | 1 | none | 0.7375 |
| `neutral_momentum_convection_velocity` | 1 | none | 0.8375 |
| `normalized_gyrocenter_perturbed_pressure` | 1 | none | 0.7500 |
| `poloidal_perturbed_vacuum_magnetic_field` | 1 | none | 0.7125 |
| `wavelength_of_filter` | 1 | none | 0.6125 |

The numeric review values are evidence that a reviewer ran; they are not a winning-group scalar authority. Projecting any of them would convert an unresolved single seat into an accepted score.

## Responsible write paths

### 1. Active strip: `write_name_review_results`

`imas_codex/standard_names/graph_ops.py:6259-6335` is the active strip. Its contract says entries contain `reviewer_score`, but it does not validate that requirement. A missing key becomes `None` at line 6316 and is written without a compare-and-set or stage guard at line 6299. The same occurs for `reviewed_at` at lines 6300 and 6317. Review history is a separate relationship and is untouched.

The current pool path also exposes why the split must be guarded atomically: `imas_codex/standard_names/workers.py:8366-8374` writes all `StandardNameReview` records first and only then calls the scalar/stage writer. A failure, malformed second payload, or no-op in the second half leaves durable reviews without their scalar projection.

### 2. Missing projection: structural acceptance

Two callers pass an accepted transition to the common structural-authority writer while omitting `reviewer_score_name`:

- `persist_enriched_parent()` at `imas_codex/standard_names/graph_ops.py:25590-25612` writes `name_stage='accepted'`, the structural reviewer marker, and the reviewed timestamp, but its `parent_updates` contains no name score;
- `structural_accept_derived_parents()` at `imas_codex/standard_names/graph_ops.py:25665-25698` promotes a non-terminal derived parent to accepted through the same omission. Its eligibility does not ask whether an attached names-review group already has a winner.

The common statement at `imas_codex/standard_names/graph_ops.py:25424-25528` applies only the supplied `parent_updates` and creates the structural-authority graph. It neither selects a winning review group nor projects the score. It also leaves all existing `HAS_REVIEW` edges intact. This is a **FAILS TO WRITE** mechanism, not an active null assignment.

The startup promotion is the regrowth path that can undo a repass: a repair can project a winning score; a later rescore/reset can legitimately return the parent to a non-terminal stage with a cleared score; then `structural_accept_derived_parents()` promotes it back to accepted without re-projecting the still-present winning group.

### 3. No winning group

The selector itself can correctly find no winner. The current schema-derived selector is implemented at `imas_codex/standard_names/graph_ops.py:6094-6124`. The eight rows above each carry only one review with `resolution_method=NULL`, so the winning set is empty. Leaving their scalar untouched is correct; the defect is only that the same structural write also ignores winners when they do exist.

## Candidate-surface audit

- `context.py` does not write either scalar. Its occurrences at lines 773 and 911 read `reviewer_score_docs` to choose documentation context.
- `harmonize.py` does not write a scalar. Lines 143-194 read and carry `reviewer_score_docs` while selecting a documentation-family anchor.
- `edit.py` does not create this accepted shape. The accepted-cohort transition at lines 3031-3046 requires the score already be null and moves `name_stage` from accepted to drafted while preserving review fields.
- The other explicit name-score nullers in `graph_ops.py` move the row away from accepted in the same statement: focus reseed at lines 1650-1662, pinned-rename resubmission at lines 23361-23371, and rescore at lines 23415-23443. They account for **0** of the accepted 96-row class.

## Reproductions

The mock reproduction in `path-reproduction.log` ran the real functions without a live-graph write.

For the active strip, calling `write_name_review_results([{'id': 'already_accepted'}])` produced:

```text
written= 1
payload_reviewer_score_name= None
payload_reviewed_name_at= None
cypher_sets_score_unconditionally= True
cypher_touches_name_stage= False
cypher_touches_has_review= False
```

On a real accepted node with a score and `HAS_REVIEW`, that statement clears the scalar and timestamp and leaves both acceptance and review history in place.

For the structural omission, calling the real `persist_enriched_parent()` against a controlled graph client produced:

```text
stage= accepted
parent_updates_name_stage= accepted
parent_updates_reviewer_model_name= structural-inheritance
parent_updates_has_reviewer_score_name= False
```

The existing regression tests independently pin the same behavior:

```text
tests/standard_names/test_enrich_parents.py::test_persist_accepts_structurally PASSED
tests/standard_names/test_structural_authority_persist.py::test_accept_and_authority_are_one_guarded_graph_statement PASSED
tests/standard_names/test_pipeline_authority_guard.py::test_reconcile_refuses_score_and_model_clear_on_scored_name PASSED
3 passed
```

The important contrast is that the third test proves the catalog guard works on its own call surface, while the first two prove structural acceptance bypasses that surface and intentionally omits the score.

## Guard scope

The guard at `imas_codex/standard_names/protection.py:77-129` compares only authority keys explicitly supplied by a catalog payload. Its module contract at lines 8-12 says catalog writers call it. `catalog_import.guard_catalog_write_payloads()` is the adapter at `imas_codex/standard_names/catalog_import.py:37-57`, and catalog reconcile reads current review state at `imas_codex/standard_names/catalog_reconcile.py:48-72` before applying only editorial scalar deltas.

Consequences for this 96-row class:

- the six active strips never enter the catalog adapter, so the guard cannot see them;
- the 82 missing projections supply no `reviewer_score_name` key, so even a generic invocation would treat the field as “leave unchanged” rather than detect the omission;
- the eight correct nulls contain no existing scalar authority to protect.

The guard is therefore narrower than a statement that it prevents all post-rescore score loss. It prevents the catalog-import mechanism only.

## Recommended repair — not applied

1. Make the axis writer fail closed. `write_name_review_results()` must reject any entry lacking a finite `reviewer_score` and a non-null reviewed timestamp before issuing its first graph statement. Required review values must use indexed access/validation, never `dict.get()` followed by an unconditional `SET`. Add a regression in which an accepted scored node with `HAS_REVIEW` receives a partial payload and remains byte-for-byte unchanged after the writer refuses it.
2. Add one schema-derived names-axis winning-group projection, analogous in intent to the documentation mirror but projecting score, model, timestamp, per-dimension scores, and resolution method from one exact winning group. Selection must use the LinkML winning-method set, prefer a canonical group, break ties deterministically, and return a receipt naming the group and review node.
3. Invoke that selector inside the same guarded structural-accept transaction. If a winning names group exists, structural acceptance must project it atomically. If none exists, it must retain the null and rely on the signed structural authority. Do not use `max(review.score)` and do not fabricate a score from an unresolved group.
4. Make `write_reviews()` plus the scalar/stage transition atomic, or record a durable incomplete-projection state that the startup reconciliation must finish before accepting the row. Review-first and scalar-second is currently a crash/no-op window.
5. Gate the repair with the exact live partition: active strip 6, missing projection 82, correct no-winner 8, residual 0. After the sanctioned backfill, the 88 rows with winners must have group-derived scalar receipts; the eight no-winner rows must remain null and structurally authorized.

No repair, source edit, graph mutation, or LLM call was made by this node.

## Evidence files

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T043348336779-n-scoreloss/live-graph-audit.log` — full 96-row read-only snapshot, review-group details, stage census, and 4,666/4,666 StandardName sentinel.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T043348336779-n-scoreloss/path-reproduction.log` — mock executions of the two responsible current functions.
