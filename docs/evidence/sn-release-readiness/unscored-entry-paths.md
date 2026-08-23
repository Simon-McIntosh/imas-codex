# Unscored accepted-entry paths

## Result

The live export-eligible name cohort contains **1,469** accepted names with no
`reviewer_score_name`. It partitions exactly into the four populations fixed by
the readiness audit: **1,181** `origin=catalog_edit` rows with no reviewer,
**221** rows stamped `reviewer_model_name=structural-inheritance`, **65**
`origin=derived` rows with no reviewer, and **2** rows carrying a non-structural
reviewer model but a null score. The partition sums to 1,469, so the residual
unattributed name count is **0**.

The independent documentation-axis query finds **441** rows at
`docs_stage=accepted` with no `reviewer_score_docs`. All 441 have
`origin=catalog_edit` and the same creation timestamp,
`2026-07-04T21:20:38.632Z`. Of these, 434 have neither a docs reviewer model nor
a `StandardNameReview` record; seven have one or both forms of review history.
The documentation population is therefore fully attributed at the path-class
level, but 434 rows lack enough durable review history to identify the exact
individual invocation that admitted or later erased each score. That
row-by-row provenance residual is stated explicitly below rather than inferred.

This was a read-only investigation. `StandardNameChange` was **7,900 before and
7,900 after (delta 0)**. `PRODUCED_NAME` was **5,779 before and 5,779 after
(delta 0)**.

## Scope and population definitions

The 1,469-name cohort is the release-facing population, not every intermediate
accepted node:

```cypher
MATCH (sn:StandardName {
  name_stage: 'accepted',
  validation_status: 'valid',
  docs_stage: 'accepted'
})
WHERE sn.reviewer_score_name IS NULL
RETURN sn
```

For context, removing the `validation_status=valid` and `docs_stage=accepted`
release gates yields 1,620 accepted/name-unscored nodes: 1,275 catalog rows, 255
structural-inheritance rows, 84 other derived rows, and six rows with another
reviewer model. Those 151 additional intermediate rows are not mixed into the
required 1,469-row attribution.

The docs-axis population is deliberately independent of name-stage and export
eligibility:

```cypher
MATCH (sn:StandardName {docs_stage: 'accepted'})
WHERE sn.reviewer_score_docs IS NULL
RETURN sn
```

## Live name-axis attribution

| Live population | Count | Durable evidence | Producing path attribution | Residual |
|---|---:|---|---|---:|
| `origin=catalog_edit`, reviewer model null | 1,181 | All 1,181 have the same `created_at` (`2026-07-04T21:20:38.632Z`), zero name-axis `StandardNameReview` records, and the historical catalog origin. | Historical bulk import `imas_codex/standard_names/catalog_import.py@4b949931^:_write_import_entries` set `name_stage='accepted'` unconditionally at historical line 435 and stamped `origin=catalog_edit`; the code was deleted by `4b949931`. A then-live export exemption (`b815f3a8`) explicitly treated the catalog as the review record. | 0 |
| `reviewer_model_name=structural-inheritance` | 221 | 130 carry `parent_enriched_at`; 91 do not. Origins are 129 derived and 92 catalog-edit, showing that origin was subsequently rewritten for some structurally admitted names. The structural reviewer marker is the stronger path discriminator. | The 130 timestamped rows are direct products of `graph_ops.py:persist_enriched_parent`. The remaining 91 are the live output shape of `graph_ops.py:structural_accept_derived_parents` or its predecessor maintenance invocation: accepted stage plus the structural marker, without the enrichment timestamp. Both paths intentionally omit a score. | 0 at path-class level; 91 cannot be split between current and predecessor maintenance invocations from surviving fields. |
| `origin=derived`, reviewer model null | 65 | All 65 have zero name-axis review records and no `parent_enriched_at`. Representative identities are `average_external_magnetic_flux`, `co_passing_fast_current_density`, `co_passing_fast_electron_torque_density_due_to_collisions`, `co_passing_fast_ion_torque_density_due_to_collisions`, and `co_passing_thermal_ion_torque_density_due_to_collisions`. | `graph_ops.py:_materialize_derived_parent_rows` and its batch executor `_materialize_derived_parent_batch` explicitly choose accepted for an unreviewed placeholder parent; `seed_parent_sources` and `normalize_derived_parent_lifecycle` keep this path live. | 0 |
| non-structural reviewer model, score null | 2 | `iron_density_at_plasma_boundary` has three name review records; `toroidal_deuterium_tritium_velocity_at_plasma_boundary` has five. Both carry `openrouter/qwen/qwen3.7-max` plus rubric/comments, proving review happened while the scalar projection is null. | `graph_ops.py:write_name_review_results` writes `e.get('reviewer_score')` directly to `reviewer_score_name` without a non-null guard or stage demotion. The same projection shape can be expressed by `signed_manifest.py:_apply_mutation` through generic `set_properties`. The ordinary review caller normally supplies a score, so the live rows demonstrate a projection/persistence hole, not an unreviewed acceptance decision. | 0 at path-class level; surviving data cannot distinguish a null review-result projection from a later generic property mutation. |

The 221 structural rows include 78 with historical name review records. Those
records do not supply an own scalar score and do not change the attribution:
the structural acceptance writers overwrite the lifecycle marker while
deliberately leaving `reviewer_score_name` null.

## Live docs-axis attribution

| Live population | Count | Durable evidence | Path attribution | Residual |
|---|---:|---|---|---:|
| Catalog cohort with no docs review record/model | 434 | Same catalog origin and exact creation timestamp; zero docs-axis review records. Name stages are 136 accepted, four exhausted, and 294 superseded. | These are the pre-authority catalog cohort. The historical catalog import preserved graph-only review state while stamping catalog provenance; their docs admission predates durable per-review evidence. Once accepted, `write_docs_review_results` could also write a missing `reviewer_score` as null without demoting docs. | 434 invocations are not individually recoverable; the graph has no review event or change record that can distinguish unscored admission from later scalar erasure. |
| Catalog cohort with docs review history/model | 7 | Review-record counts are 2, 2, 6, 8, 9, 10, and 12. `krypton_density_at_magnetic_axis` remains name-accepted; the other six are name-superseded. | `graph_ops.py:write_docs_review_results` is the direct fail-open projection path: it assigns `e.get('reviewer_score')` to `reviewer_score_docs` while leaving an existing accepted docs lifecycle untouched. Generic signed-manifest property mutation is the other reachable erasure path. | 0 at path-class level; exact writer invocation is not retained. |

The current scored accept writer, `graph_ops.py:persist_reviewed_docs`, is not
the source of an unscored admission by itself: it accepts only after computing
the target from a required numeric score and writes `reviewer_score_docs` and
`docs_stage` together. The 441 live rows therefore require either historical
pre-evidence admission or a later score-erasure path.

## Complete admitting-path inventory

“Reachable” below distinguishes the ordinary seven-pool pipeline from an
operator/API path. A path is listed if it can either set an axis to accepted
without checking that axis's score, or erase an axis score without demoting an
already accepted lifecycle.

| File and symbol | Axis/mechanism | Current reachability | Live grounding |
|---|---|---|---|
| `imas_codex/standard_names/catalog_import.py@4b949931^:_write_import_entries` (historical lines 377–446; accepted assignment at 435) | Set every imported name accepted without a score. | **Deleted**; current catalog import is check-only. | Primary producer of the 1,181 catalog/name-unscored rows. The July export exemption confirms the intended historical policy. |
| `imas_codex/standard_names/graph_ops.py:_materialize_derived_parent_rows` (accepted branches at 3885–3896) and `_materialize_derived_parent_batch` (4061–4072) | Set an unreviewed placeholder derived parent accepted. | **Yes, ordinary maintenance/seeding** through `seed_parent_sources` and `normalize_derived_parent_lifecycle`. | 65 derived/no-reviewer rows. |
| `imas_codex/standard_names/graph_ops.py:persist_enriched_parent` (25051; accepted at 25111) | Structurally accept an enriched derived parent, stamp `structural-inheritance`, explicitly omit the score. | **Yes, ordinary `ENRICH_PARENTS` pool**. | 130 structural rows carry its unique `parent_enriched_at` marker. |
| `imas_codex/standard_names/graph_ops.py:structural_accept_derived_parents` (25162; accepted at 25199) | Maintenance promotion to structural acceptance without a score. | **Yes, ordinary startup maintenance**. | Remaining 91 structural rows have the marker but no enrichment timestamp; this path and its predecessor produce that shape. |
| `imas_codex/standard_names/graph_ops.py:mark_for_refine_docs` (24569; accepted at 24619) | Restore a derived parent to name-accepted while writing only a synthetic docs score. | **Yes, ordinary review-name pre-step**. | No separately identifiable release-facing row; any output without the unique `desc_name_similarity` marker belongs to the materializer cohort above. Live attributable count: 0. |
| `imas_codex/standard_names/graph_ops.py:write_name_review_results` (6099; scalar assignment at 6139/6156) | Null-capable name-score projection; stage is not demoted. | **Yes, ordinary review pipeline**, though its typed caller is expected to supply a score. | Two reviewed/model-stamped name rows have null scalar scores and 3/5 review records. |
| `imas_codex/standard_names/graph_ops.py:write_docs_review_results` (6182; scalar assignment at 6260/6274) | Null-capable docs-score projection; stage is not demoted. | **Yes, ordinary review pipeline**, though its typed caller is expected to supply a score. | Seven docs rows retain review history/model with a null scalar; it is also compatible with the 434 history-free catalog rows. |
| `imas_codex/standard_names/merge.py:revert_contested` (572; accepted at 581) | Return contested name to accepted without checking score. | **Yes, operator merge API**, not an ordinary pool transition. | No unique marker links a live required-cohort row to this path; attributable count 0. |
| `imas_codex/standard_names/merge.py:undo_merge` (904; accepted at 936 and 951) | Return approved/contested names to accepted without checking score. | **Yes, operator merge API**, not an ordinary pool transition. | No unique marker links a live required-cohort row to this path; attributable count 0. |
| `imas_codex/standard_names/provenance_lifecycle.py:cancel_staged_rename` (82; accepted branch at 163–166) | Restore predecessor to accepted when either a score clears threshold **or merely** `catalog_approved_at` exists. | **Yes, operator edit-cancellation path**. | Compatible with catalog-approved unscored nodes, but no required-cohort row retains a unique cancellation marker; attributable count 0. |
| `imas_codex/standard_names/graph_ops.py:reset_standard_names` (7807; parameterized stage assignment at 7966–7973) | Caller-controlled `to_stage` can be accepted without setting a score. | **Callable API; current CLI callers use drafted**, so not reachable as an accepted transition in the present command pipeline. | Attributable live count 0. |
| `imas_codex/standard_names/graph_ops.py:_claim_sn_atomic` (13797; parameterized stage assignment at 13886–13890) | Generic claim primitive permits any `to_stage`. | **Callable internal API; current callers use refining**, not accepted. | Attributable live count 0. |
| `graph_ops.py:release_enrich_failed_claims`, `release_review_names_failed_claims`, and `release_review_docs_failed_claims` (22451, 22558, 22653) | Caller-controlled failure rollback can set either lifecycle axis to accepted. | **Callable internal APIs; current worker callers use drafted**. | Attributable live count 0. |
| `imas_codex/standard_names/signed_manifest.py:_apply_mutation` (3944; generic property merge at 4046–4082), reached by `apply_signed_manifest` | Signed generic `set_properties` may set an accepted stage without a score or clear a score after acceptance. | **Yes, sanctioned authority/operator path**, not an ordinary pool transition. | Compatible with the two name and seven docs projection anomalies; no unique manifest/change marker survives on those rows, so no exact split is claimed. |

## Score-carrying accepted paths that do not admit the defect alone

These are important negative controls: they set accepted, but their predicates
or atomic writes carry a non-null score. They should remain permitted after the
invariant is enforced.

| File and symbol | Why it is safe in isolation |
|---|---|
| `graph_ops.py:persist_reviewed_name` (14766 onward) | Receives a required numeric score, computes acceptance from that score, and writes `reviewer_score_name` plus `name_stage` in the same fenced transaction. |
| `graph_ops.py:persist_reviewed_docs` (15385 onward) | Receives a required numeric score, computes acceptance from that score, and writes `reviewer_score_docs` plus `docs_stage` atomically. |
| `graph_ops.py:promote_stranded_reviewed` (11379) | The name predicate requires `reviewer_score_name >= min_score`; the docs predicate requires `reviewer_score_docs >= min_score` plus a docs resolution method before either accepted assignment. |

These writers are safe only if later generic or projection writers cannot erase
their scores. The proposed invariant therefore belongs at every graph write
boundary, not only in the two normal persistence functions.

## Invariant and schema-driven regression test

The graph invariant is:

> For each review axis declared by the `StandardName` schema, an accepted
> lifecycle requires a non-null scalar score, a non-placeholder reviewer model,
> and at least one `StandardNameReview` on that same axis whose terminal review
> group supports the stored score. There is no catalog, derived-parent, signed
> manifest, merge, recovery, or rollback exemption.

In property terms, at minimum:

```text
name_stage = accepted
  => reviewer_score_name IS NOT NULL
     AND reviewer_model_name IS NOT NULL
     AND reviewer_model_name <> structural-inheritance
     AND a HAS_REVIEW record exists for review_axis = names

docs_stage = accepted
  => reviewer_score_docs IS NOT NULL
     AND reviewer_model_docs IS NOT NULL
     AND a HAS_REVIEW record exists for review_axis = docs
```

The stronger production check must also verify that the latest terminal review
group's resolution and aggregate score agree with the scalar projection. A
non-null stale or synthetic number is not sufficient authority.

The schema-driven test should attach review-authority annotations to the two
lifecycle slots in `imas_codex/schemas/standard_name.yaml`, for example the
required score slot, model slot, review axis, and forbidden placeholder model.
The existing `tests/graph` schema loader can then enumerate every annotated
accepted state rather than hard-code the two axes. For each annotation it should
construct a zero-row query covering:

1. accepted with null score;
2. accepted with null or forbidden reviewer model;
3. accepted with no same-axis `HAS_REVIEW` record;
4. accepted with no terminal resolution or with a scalar that disagrees with
   the terminal review group's aggregate.

The assertion is **zero violations on both annotated axes**. Adding a third
reviewed lifecycle in the schema would automatically add a third graph check.
The same schema metadata should drive a centralized pre-commit validator called
by ordinary persistence, manifest application, merge/undo, lifecycle recovery,
generic stage setters, and review-result projection. Neo4j property constraints
cannot express this cross-property/relationship rule alone; every mutating
transaction must refuse a post-state that violates it.

## Read-only evidence artifacts

- `live-before.json`: initial live population and counters.
- `live-corrected.json`: full corrected-axis snapshot (`review_axis='names'` /
  `'docs'`) used for the population and review-history attribution.
- `live-after.json`: final live population and counters after this evidence file
  was authored.
- All three were produced by `audit_live_graph.py` using `GraphClient` and
  read-only `MATCH`/`RETURN` queries only.
