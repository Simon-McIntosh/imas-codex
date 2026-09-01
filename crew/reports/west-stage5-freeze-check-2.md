NEEDS-HELP: The frozen-stage guard makes every active Standard Name pool claim fail because its atomic read-back query references an unbound `$frozen_name_stages` parameter.
tried: Took the required ordered baseline snapshot, launched the one authorized unscoped `sn run --flush --cost-limit 5`, and observed all five non-empty pools repeatedly fail with `Neo.ClientError.Statement.ParameterMissing`; after 1,343 claim failures, zero processed names, unchanged pending counts, and no circuit breaker, interrupted the runaway invocation, let its reconciliation/finalization complete, and took the identical after-snapshot.
options: (1) add the missing `frozen_name_stages` binding to the `_claim_sn_atomic` read-back transaction, add an execution-level parameter-binding regression test, then redispatch this exact freeze-check; (2) apply only the one-line binding repair and redispatch immediately, accepting another source-inspection-only testing gap; (3) treat the zero frozen-row diff from this interrupted, zero-pool-work run as passing evidence and continue.
leaning: Option 1, because it fixes the exact runtime defect and closes the test gap that allowed a source-inspection test to report green while the live query was unexecutable; option 3 is not valid freeze evidence.
cost-if-wrong: Choosing option 2 may require another repair and another full capped run if another transaction parameter is still unbound; choosing option 3 could promote a cohort whose frozen-stage guard has never survived ordinary live pool activity, requiring graph recovery and repetition of the approval rehearsal.

# WEST folded-catalog freeze-check repeat

## Result

**BLOCKED — the frozen cohort was byte-identical, but the ordinary pipeline
never became executable.** The same ordered projection returned 375 approved
and 1 contested Standard Name before and after, with zero added, removed, or
modified frozen identities and zero modified projected fields. That numerical
result cannot pass the freeze gate because every non-empty Standard Name pool
failed before returning a claimed item.

The single required command was launched once. It had to be interrupted after
the five non-empty pools accumulated **1,343 identical claim failures** while
pending counts remained 9 review-name, 7 refine-name, 2 generate-docs, 15
review-docs, and 1 enrich-parent. It then finalized normally enough to write
`SNRun` `cba34c44-0eac-4cd9-981d-dc5b24033eda`, but that receipt correctly
records status and stop reason `interrupted`, exit status 1, zero processed
names, and exact spend **$0.00 / $5.00**.

## Snapshot instrument

The identical ordered Cypher projection from the first freeze-check was run
immediately before launch and immediately after finalization:

```cypher
MATCH (sn:StandardName)
WHERE sn.name_stage IN ['approved', 'contested']
RETURN sn.id AS id,
       sn.name_stage AS name_stage,
       sn.docs_stage AS docs_stage,
       sn.description AS description,
       sn.reviewer_score_name AS reviewer_score_name,
       sn.reviewer_scores_name AS reviewer_scores_name,
       sn.reviewer_score_docs AS reviewer_score_docs,
       sn.reviewer_scores_docs AS reviewer_scores_docs,
       sn.reviewer_score_secondary AS reviewer_score_secondary,
       sn.reviewer_scores_secondary AS reviewer_scores_secondary,
       sn.review_mean_score AS review_mean_score,
       sn.review_disagreement AS review_disagreement,
       sn.reviewer_disagreement AS reviewer_disagreement,
       sn.catalog_pr_number AS catalog_pr_number,
       sn.catalog_pr_url AS catalog_pr_url,
       sn.catalog_reviewer_actor AS catalog_reviewer_actor,
       toString(sn.catalog_approved_at) AS catalog_approved_at,
       sn.catalog_merge_commit_sha AS catalog_merge_commit_sha,
       sn.catalog_commit_sha AS catalog_commit_sha
ORDER BY sn.id
```

As in the first check, Neo4j did not provide `sha256()`, so each returned
`description` was replaced client-side by its UTF-8 SHA-256 before canonical
JSON comparison. The remaining 18 projected scalars were compared directly.
A baseline positive control proved the query was aimed at populated fields:
4,682 `StandardName` candidates, 4,682 with `id`, 4,682 with `name_stage`,
4,656 with `description`, and 374 with `catalog_pr_number`; the target predicate
returned 375 approved and 1 contested.

## Before and after

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| Snapshot rows | 376 | 376 | 0 |
| `name_stage=approved` | 375 | 375 | 0 |
| `name_stage=contested` | 1 | 1 | 0 |
| Rows with a non-null description hash | 376 | 376 | 0 |
| Rows carrying `catalog_pr_number=3` | 374 | 374 | 0 |
| Added frozen identities | 0 | 0 | 0 |
| Removed frozen identities | 0 | 0 | 0 |
| Modified frozen identities | 0 | 0 | **0** |
| Modified projected fields | 0 | 0 | **0** |

Both snapshots have canonical digest
`fe4d32b4544879cb7dfdef1c5faaa594ea8233064fb4ef9f237c6916ba45596b`.
The row-level diff over all 376 identities and all 19 projected fields is empty.

### Drafted-docs positive control

`breakdown_initial_time` remained the intended positive control because it was
approved while still carrying `docs_stage=drafted`. Every projected value was
identical:

| Field | Before | After |
|---|---|---|
| `name_stage` | `approved` | `approved` |
| `docs_stage` | `drafted` | `drafted` |
| description SHA-256 | `7ffaab2d25de671ec4af7276745f052f26fdc133d73637ce1a7e5ebe69775944` | same |
| `reviewer_score_name` | 0.925 | 0.925 |
| `reviewer_score_docs` | 0.7083333333333334 | 0.7083333333333334 |
| `review_mean_score` | 0.903125 | 0.903125 |
| `review_disagreement` | true | true |
| `catalog_approved_at` | `2026-09-01T21:28:54.72Z` | same |
| remaining `catalog_*` provenance | null | null |

This is not a successful live positive control: the review-docs pool never
returned any claim because the shared read-back query failed for frozen and
non-frozen candidates alike. The unchanged row therefore proves only that the
failed transaction rolled back, not that ordinary pool work can proceed while
excluding it.

## Required command and receipt

Executed exactly once, unscoped:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync imas-codex sn run --flush --cost-limit 5
```

- Process exit status: **1**
- `SNRun`: `cba34c44-0eac-4cd9-981d-dc5b24033eda`
- Run interval: 2026-09-01 22:10:03.038788Z to 22:15:22.676Z
- Recorded status / stop reason: `interrupted` / `interrupted`
- Recorded pipeline elapsed time: 265.209 seconds
- Exact spend: **$0.00 / $5.00**; no budget was consumed because no claim
  reached an LLM call
- Counters: 0 composed, 0 enriched, 0 reviewed, 0 regenerated
- Active-pool claim failures: review-name 411, refine-name 214,
  generate-docs 197, review-docs 416, enrich-parents 105; total **1,343**
- Rotated full logs:
  `/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log.1`
  (SHA-256 `ba1972c26ada62edbab581e19e74fb107385b74244a8993cb790e75dc8ec5af3`)
  and `/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log`
  (SHA-256 `380567e27dc62b7ad5266aec2da5b35e902324ce726ffa850fa38f3beab4de0e`)

## Non-frozen activity

The pre-pool and post-interrupt reconciliation phases did persist changes to
**45 distinct non-frozen Standard Names**: 0 created, 0 deleted, and 45 existing
rows modified. These are bookkeeping and normalization effects, not successful
pool processing. The most frequent changed fields were `harmonized_at` on 22
rows, `harmonized_group_signature` on 18, `source_paths` on 13, and source
resolution snapshot fields on 9. Representative identities include
`alpha_parameter` (`source_paths`), `co_passing_fast_particle_pressure`
(harmonization stamp), `effective_resistivity_of_passive_loop` (source
resolution snapshot), `net_electron_power_density` (grammar and source
metadata), and `neutral_internal_state_energy_flux_at_wall` (documentation
normalization).

This activity proves that the whole invocation was not a no-op, but it does not
exercise the pool boundary the guard was intended to protect.

## Runtime defect and test gap

The seed and expansion queries in `_claim_sn_atomic` build a `params` mapping
that includes `frozen_name_stages`. The token read-back query at
`imas_codex/standard_names/graph_ops.py:14175` also references
`$frozen_name_stages`, but its call at lines 14199–14200 passes only `token` and
`drain_scope_id`. Neo4j therefore rejects the query before a claim can be
returned and the transaction rolls back.

The focused regression at
`tests/standard_names/test_frozen_stage_pool_guard.py:29` inspects source text
and counts occurrences of `IN $frozen_name_stages`; it never executes the
atomic primitive or verifies that every referenced parameter is bound. That is
why 10 focused tests and the 282-test subset could be green while the first live
invocation failed immediately.

## Gate verdict and next action

The required numerical diff is zero—**0 of 376 frozen rows and 0 of 19 fields
changed**—but the freeze gate remains **BLOCKED**, not passing. A new repair node
must bind `frozen_name_stages` in the read-back transaction and add an
execution-level regression that reaches this step. After that repair lands, a
fresh worker must repeat the same single unscoped capped invocation and the
same ordered before/after projection. No catalog promotion or undo should use
this interrupted run as freeze evidence.
