# WEST folded-catalog freeze-check after claim-binding repair

## Result

**PASS — every approved and contested Standard Name remained frozen while the
ordinary pipeline executed live claims and persisted non-frozen work.** The
same ordered projection returned 375 approved and 1 contested row before and
after. The row-level comparison found **0 added, 0 removed, and 0 modified
frozen identities**, with **0 modified fields across 376 rows × 19 projected
fields**.

The single unscoped command completed naturally at the shared budget boundary.
It exited 1 because the terminal `SNRun` is deliberately `degraded` with
`stop_reason=budget_exhausted`, not because the pipeline crashed. The repaired
atomic claim path returned live candidates, all pools reported zero errors at
shutdown, and the exact run-log segment contains **0 `ParameterMissing`
occurrences**. Meanwhile the invocation touched **51 distinct non-frozen
Standard Names**: 1 created, 0 deleted, and 50 existing rows modified.

## Snapshot instrument

The identical ordered Cypher projection from the preceding freeze-check was
run immediately before launch and immediately after finalization:

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

Neo4j did not provide `sha256()`, so each returned `description` was replaced
client-side by its UTF-8 SHA-256 before canonical JSON comparison. The
remaining 18 projected scalars were compared directly. Canonical JSON used
sorted keys, compact separators, and string conversion only for non-JSON
values.

The baseline positive control proved the projection was aimed at populated
properties: 4,682 `StandardName` candidates, 4,682 with `id`, 4,682 with
`name_stage`, 4,656 with `description`, and 374 with `catalog_pr_number`; the
target predicate returned 375 approved and 1 contested. After the run there
were 4,683 candidates, 4,683 with `id`, 4,683 with `name_stage`, 4,657 with
`description`, and the same 374 with catalog PR provenance; the frozen
predicate still returned 375 approved and 1 contested.

## Before and after

| Measure | Before | After | Delta |
|---|---:|---:|---:|
| All `StandardName` rows | 4,682 | 4,683 | +1 |
| Frozen snapshot rows | 376 | 376 | 0 |
| `name_stage=approved` | 375 | 375 | 0 |
| `name_stage=contested` | 1 | 1 | 0 |
| Frozen rows with a non-null description hash | 376 | 376 | 0 |
| Rows carrying `catalog_pr_number=3` | 374 | 374 | 0 |
| Added frozen identities | 0 | 0 | **0** |
| Removed frozen identities | 0 | 0 | **0** |
| Modified frozen identities | 0 | 0 | **0** |
| Modified projected fields | 0 | 0 | **0** |

Both snapshots have canonical digest
`fe4d32b4544879cb7dfdef1c5faaa594ea8233064fb4ef9f237c6916ba45596b`.
The ordered row-level diff over all 376 identities and all 19 projected fields
is empty.

### Drafted-docs positive control

`breakdown_initial_time` explicitly exercises the repaired boundary because it
remains approved while carrying the legacy `docs_stage=drafted` state that the
unrepaired docs pool previously claimed. Ordinary docs claims executed during
this run, but every projected value on this frozen row remained identical:

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

This is a live positive control rather than a no-op check: atomic pool claims
returned candidates; the run persisted 27 reviews, 3 documentation generations
or enrichments, and 3 refinements/regenerations while leaving the approved
drafted-docs row outside every claim and aggregate write.

## Required command and receipt

Executed exactly once, unscoped:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync imas-codex sn run --flush --cost-limit 5
```

- Process exit status: **1** (`budget_exhausted`, expected degraded terminal)
- `SNRun`: `220c6abf-45e6-4722-837d-77399c549d04`
- Run interval: 2026-09-01 22:49:24.555345Z to 23:14:20.746Z
- Budget stop signal: 2026-09-01 23:11:02.552168Z
- Recorded status / stop reason: `degraded` / `budget_exhausted`
- Recorded elapsed time: 1,297.997 seconds
- Exact spend: **$5.099992 / $5.00** (101.99984% of cap, $0.099992 or
  1.99984% overshoot from already-funded calls completing after the stop
  signal); `cost_is_exact=true`
- `LLMCost` source-of-truth cross-check: 72 recorded calls sum to
  $5.099992000000001 (floating-point representation of the same receipt)
- Counters: 0 composed, 3 enriched, 27 reviewed, 3 regenerated
- Pool work: review-name processed 9, refine-name 1, generate-docs 3,
  review-docs 18, refine-docs 2, enrich-parents 0; all pools exited with
  `error_count=0`
- Full rotating log:
  `/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log`
  (SHA-256 `e2658f90fd91a53a36647dbe3970eb4b26f823638bca2d5452e33329629cb5b7`)
- Exact run-log segment: byte offset 435,973 through EOF byte 833,569,
  397,596 bytes, SHA-256
  `332867774907f2857514874f722e47ce117e7aa7a45ca559dc46bb4765d77569`
- `ParameterMissing` occurrences in that exact run-log segment: **0**

The byte boundary matters because the rotating file retained earlier
`ParameterMissing` traces from the blocked predecessor run before offset
435,973. The file inode stayed `43006726` and did not rotate during this run,
so the append-only segment precisely isolates this invocation without
miscounting old failures.

## Non-frozen activity

The same canonical whole-node fingerprint comparison (excluding embeddings)
found **51 distinct non-frozen Standard Names touched**: 1 created, 0 deleted,
and 50 existing identities changed.

- Created: `net_electron_energy_source_rate`, the persisted refinement of
  `net_electron_power_density`.
- Review/refine examples: `fast_ion_charge_state_torque_due_to_collisions`
  changed its name-review score and terminal stage;
  `ion_diamagnetic_momentum_convection_velocity` and
  `ipb98y2_confinement_time` changed docs-review fields;
  `net_electron_power_density` gained the refinement lineage and source
  migration state.
- Reconciliation examples: `co_passing_fast_particle_pressure`,
  `counter_passing_fast_particle_pressure`, and
  `diamagnetic_momentum_convection_velocity` received harmonization stamps.
- Most frequent changed fields were `claim_seq` on 30 rows,
  `harmonized_at` on 25, `llm_cost`, `review_count`, and
  `review_mean_score` on 24 each, `harmonized_group_signature` on 19, and
  `docs_stage` on 18.

This non-frozen activity proves that the command exercised the ordinary live
pipeline and its repaired claim binding while the folded cohort remained
frozen.

## Gate verdict

**PASS.** The repaired ordinary pipeline produced live claims and persisted
non-frozen results with **0 `ParameterMissing` errors**, while the identical
ordered snapshot proved **0 of 376 approved or contested rows and 0 of 19
projected fields changed**. `breakdown_initial_time` specifically remained
approved with drafted docs and byte-identical projected state. The next catalog
operation may rely on this run as freeze evidence, subject to its own authority
and gates.
