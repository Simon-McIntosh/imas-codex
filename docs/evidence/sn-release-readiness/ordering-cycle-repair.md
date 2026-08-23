# Spectrometer ordering-cycle repair reversibility receipt

Applied to the production `codex` graph on 2026-08-23 from source commit
`d37a598a`. The repair changed exactly one relationship, changed no node, made
no model call, and spent USD 0.

## Applied mutation

Exactly this relationship was deleted:

| Source node id | Relationship type | Complete property set | Target node id |
|---|---|---|---|
| `spectral_signal_to_noise_ratio_of_spectrometer_channel` | `HAS_PARENT` | `{"operator": "spectral", "operator_kind": "qualifier"}` | `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` |

The relationship was the reversed qualifier peel identified by the preceding
read-only disposition. Removing a `spectral` qualifier cannot introduce a
`logarithm_of_` prefix or point from a base quantity back to its own child.

The mutation was fail-closed. Immediately before deletion it re-asserted the
two exact endpoint ids, the complete property maps of both directed edges, the
two-edge pair inventory, all three population sentinels, and the single
graph-wide bidirectional `HAS_PARENT` pair. The write returned one deleted
relationship with the property map recorded above.

## Exact recreation data

If this disposition proves wrong, the deleted relationship can be recreated
exactly from this receipt through the governed graph-repair path:

```cypher
MATCH
  (source:StandardName {
    id: 'spectral_signal_to_noise_ratio_of_spectrometer_channel'
  }),
  (target:StandardName {
    id: 'logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel'
  })
CREATE (source)-[:HAS_PARENT {
  operator: 'spectral',
  operator_kind: 'qualifier'
}]->(target)
```

Before recreation, assert that both endpoint ids still resolve uniquely, that
this exact relationship is absent, and that the legitimate reverse edge below
still exists. The statement is recovery data, not authority for an unreviewed
write.

## Before and after sentinels

| Population or invariant | Before | After | Delta | Required result |
|---|---:|---:|---:|---|
| `StandardName` nodes | 4,395 | 4,395 | **0** | pass |
| `HAS_PARENT` relationships | 61,921 | 61,920 | **-1** | pass |
| `HAS_ERROR` relationships | 31,281 | 31,281 | **0** | pass |
| Bidirectional `HAS_PARENT` pairs | 1 | 0 | **-1** | pass |

Both identities survived unchanged as `name_stage='accepted'` and
`validation_status='valid'`; both retained their null catalog `status`. Neither
identity was accepted, superseded, renamed, revalidated, or otherwise edited by
this repair.

## Legitimate edge survival assertion

The only relationship remaining between the pair is:

| Source node id | Relationship type | Complete property set | Target node id |
|---|---|---|---|
| `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` | `HAS_PARENT` | `{"operator": "logarithm", "operator_kind": "unary_prefix"}` | `spectral_signal_to_noise_ratio_of_spectrometer_channel` |

This is the legitimate operator peel: removing the outer unary `logarithm`
prefix from `logarithm_of_X` yields `X`. Its post-write existence and complete
property set were asserted directly. Deleting it would have failed this repair.

## Catalog-ordering check

The same live export path that previously stopped with `OrderingError` on
these two identities was rerun with `min_score=0.85`, `skip_gate=True`,
`force=True`, and `include_sources=False`. It completed with `EXIT=0`; catalog
ordering no longer reported an unemitted cycle.

The full dry-run accounting also closed:

| Measure | Result |
|---|---:|
| Accepted population | 2,535 |
| Emitted | 534 |
| Excluded, all named reasons | 2,001 |
| Accounted total | 2,535 |
| Exclusion-accounting gate | **pass** |

The named exclusions were `below_name_score=1`,
`documentation_not_accepted=192`, `documentation_review_unresolved=1,758`,
`invalid_catalog_entry=2`, `invalid_validation_status=47`, and
`name_review_quorum_shortfall=1`. The two invalid catalog entries and 677
dangling-link prunes remain visible in the dry-run log; neither reintroduced an
ordering failure.

Full check log:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T101117494677-n-cyclerepair/catalog-ordering-check.log`.

The redundant logarithm identity remains live. Its sanctioned catalog
lifecycle disposition is deliberately outside this repair.
