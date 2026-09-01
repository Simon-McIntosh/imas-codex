# WEST folded-catalog freeze check

## Result

**FAIL — the approved/contested cohort was not row-frozen.** The ordinary,
unscoped flush preserved the cohort census at 375 approved and 1 contested,
and it did not add or remove a frozen identity. It nevertheless changed two
persisted fields on the approved `breakdown_initial_time` row:
`review_mean_score` changed from `0.9375` to `0.903125`, and
`review_disagreement` changed from `false` to `true`.

The command ran once and terminated after 374.185 seconds with exit status 1.
Its durable `SNRun` is `43ffe187-f79a-4772-89b8-e23fb9a82cb2`, status
`degraded`, stop reason `budget_exhausted`. Exact recorded spend was $5.439569
against the $5.00 ceiling: $0.439569 (8.79%) over the requested cap because
already-launched reviews completed after the budget signal.

## Snapshot instrument

The same ordered Cypher projection was executed immediately before and after
the run:

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

Neo4j 2026 does not expose a `sha256()` Cypher function in this deployment, so
the Cypher result's `description` value was replaced client-side with
`SHA-256(UTF-8(description))` before canonical JSON hashing and comparison.
Every other returned scalar was compared directly. A positive-control query
proved the projection was aimed at populated properties before the baseline:
4,675 `StandardName` candidates, 4,675 with `id`, 4,675 with `name_stage`,
4,649 with `description`, and 374 with `catalog_pr_number`; its target predicate
returned the expected 375 approved and 1 contested rows.

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
| Modified frozen identities | 0 | 1 | **+1** |
| Modified projected fields | 0 | 2 | **+2** |

The canonical snapshot digest changed from
`549dd1844610fce3b98d1610ba36f802846c469278928116825fe8b46a2fb188`
to
`fe4d32b4544879cb7dfdef1c5faaa594ea8233064fb4ef9f237c6916ba45596b`.

### Row-level diff

| Standard Name | Field | Before | After | Unchanged supporting identity |
|---|---|---:|---:|---|
| `breakdown_initial_time` | `review_mean_score` | 0.9375 | 0.903125 | approved; docs drafted; name score 0.925; docs score 0.7083333333333334 |
| `breakdown_initial_time` | `review_disagreement` | false | true | description SHA-256 `7ffaab2d25de671ec4af7276745f052f26fdc133d73637ce1a7e5ebe69775944` |

All other 375 approved/contested rows and all other projected fields were
byte-for-byte equal after canonicalization. In particular, no catalog
provenance field changed. `breakdown_initial_time` retained
`catalog_approved_at=2026-09-01T21:28:54.72Z` with the remaining catalog
provenance fields null. The sole contested row, `pulse_duration`, remained
contested/docs-drafted with description hash
`45900155931d662027330974fb8bae40d70ba94360b3cd10015b4152e5953a7b`,
name score 0.88125, docs score 0.5083333333333333, and null catalog provenance.

The run log showed docs review activity for `breakdown_initial_time`, followed
by a stage/token mismatch that made the final docs persistence a no-op. The
Standard Name row's aggregate review fields nevertheless changed, so the
ordinary pipeline's protection is incomplete even though its stage and
description did not change.

## Ordinary run and non-frozen activity

Command (executed exactly once, unscoped):

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync imas-codex sn run --flush --cost-limit 5
```

- Exit status: **1**
- CLI completion: status `degraded`; stop reason `budget_exhausted`
- Run interval: 2026-09-01 21:46:13.388698Z to 21:53:18.7Z
- Exact spend: **$5.439569 / $5.00**
- Run counters: 0 names composed, 15 enriched, 5 reviewed, 11 regenerated
- Full auto-log: `/home/ITER/mcintos/.local/share/imas-codex/logs/sn_sn-compose.log`
- Log SHA-256: `3302a5b43c24e6617a19b0d528e0ef5de2a58fb3e33d2ac8ef746a94790a8551`

For the required activity control, all `StandardName` property maps were
snapshotted with the vector payload excluded but all persisted lifecycle,
review, catalog, embedding-metadata, source, and provenance properties retained.
The run changed **135 distinct non-frozen Standard Names** in their persisted
end state:

| Persisted outcome | Distinct names |
|---|---:|
| Created | 12 |
| Deleted | 5 |
| Existing row changed | 118 |
| **Union** | **135** |

Representative created names were
`normalized_saturated_permeability_of_ferritic_element`,
`net_electron_power_density`, `total_incident_neutral_energy_flux_at_wall`,
`vertical_outline_of_control_surface`, and
`weight_of_interferometer_beam`. Representative deleted names were
`current_density_due_to_collisions`, `normalized_pressure_at_flux_surface`,
and `temperature_at_separatrix`. Representative changed existing names were
`current_density_due_to_distribution_function_driven`,
`ion_diamagnetic_momentum_convection_velocity`, `electron_power_density`, and
`ipb98y2_confinement_time`.

## Verdict and next action

The activity control fired strongly, but the freeze gate failed quantitatively:
**1 of 376 frozen rows changed, across 2 projected fields**. Do not treat the
folded state as isolated from the ordinary pipeline. The next repair should
prevent review-side aggregate writes for `approved` and `contested` names (or
defer every aggregate write until the same stage/token guard that protects
final persistence succeeds), then repeat this identical single-run snapshot
test. The catalog cohort should not be promoted on this evidence.
