# Origin derivation validation

## Verdict

**REFUSAL — no origin-derivation rule is validated for mutation.** The
`source_types` anomaly is explained, but the catalog-import fingerprint does
not re-derive the declared origin of every accepted control. The required
misclassification count is zero; the measured count is **5**, so this node
writes no `origin` value and no `status` value.

The read-only validation ran against the live graph on 2026-08-25. Its full
output is retained at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T141822953928-n-originrule/logs/origin-rule-validation.log`.

## Schema sanity

Every result below was returned with the same schema-positive census. The
graph held **4,658** `StandardName` nodes; **4,658** carried the schema identity
property `id`, **4,658** carried `name_stage`, **3,567** carried `origin`,
**3,267** carried `source_types`, **2,104** carried `imported_at`, and **2,053**
carried `catalog_commit_sha`. These non-zero property counts prove that the
queries were aimed at properties that exist; none of the reported zeroes is a
silent missing-property result.

The accepted control predicate is exactly `name_stage = 'accepted'`. It
reproduced the held cohort sizes: **1,372** declared `catalog_edit`, **426**
declared `pipeline`, and **278** declared no origin. This also resolves the
apparent 1,367/422/271 discrepancy produced by additionally requiring
`docs_stage = 'accepted'`: documentation acceptance is not part of the held
name-control definition.

## The `source_types` anomaly

The first of the two proposed readings holds: **`source_types` is not the
catalog marker that its mere presence appears to be.** It is a multivalued set
of contributing source kinds. The earlier 0.98 comparison measured only
whether the property existed, discarding its value.

The value-aware measurement decides the issue:

| Declared origin | `source_types` value | Count | Fraction of cohort |
|---|---:|---:|---:|
| null | `['dd']` | 272 | 0.9784 |
| null | absent | 6 | 0.0216 |
| null | contains `catalog` | **0** | **0.0000** |
| `catalog_edit` | `['catalog']` | 1,275 | 0.9293 |
| `catalog_edit` | `['dd']` | 97 | 0.0707 |
| `pipeline` | absent | 394 | 0.9249 |
| `pipeline` | `['dd']` | 30 | 0.0704 |
| `pipeline` | `['catalog']` | 2 | 0.0047 |

Thus the null cohort's 0.98 property-presence rate is entirely DD provenance:
272 identities carry `['dd']`, and none carries the `catalog` value. The
property itself is also not a lossless editorial-origin classifier: 97 accepted
`catalog_edit` identities carry `['dd']`, while two accepted `pipeline`
identities carry `['catalog']`.

The same value-aware result holds across the complete **1,091** null-origin
cohort: **1,085** carry `['dd']`, six have no `source_types`, and **zero** carry
`catalog`. Its lifecycle split is 278 accepted, 13 drafted, 41 exhausted, 54
reviewed, and 705 superseded. All 1,091 have both `imported_at` and
`catalog_commit_sha` absent. The 705 superseded identities remain separate from
any proposed live mutation, as required.

## Candidate rule and held-answer check

The strongest simple rule supported by the import evidence is:

```text
if imported_at is present or catalog_commit_sha is present:
    derive catalog_edit
else:
    derive pipeline
```

This rule projects all 278 accepted null-origin identities, and all 1,091 null
origins more broadly, to `pipeline`. That projection is not trusted until the
rule re-derives every held answer.

| Held answer | Control size | Derived correctly | Misclassified | Required |
|---|---:|---:|---:|---:|
| `catalog_edit` | 1,372 | 1,372 | **0** | 0 |
| `pipeline` | 426 | 421 | **5** | 0 |
| **Total** | **1,798** | **1,793** | **5** | **0** |

The five counterexamples are representative identities rather than anonymous
counts:

| Declared `pipeline` identity | `source_types` | Import evidence |
|---|---|---|
| `electron_density` | `['catalog']` | both markers present |
| `normalized_toroidal_flux_coordinate` | `['dd']` | both markers present |
| `safety_factor` | `['catalog']` | both markers present |
| `toroidal_magnetic_field` | `['dd']` | both markers present |
| `vertical_coordinate_of_camera` | `['dd']` | both markers present |

All five share `imported_at = 2026-07-04T21:21:17.079Z` and
`catalog_commit_sha = a2f8831cf9d14af2f7120969c728f990bdd923cf`, yet their
held origin is `pipeline`. Two even carry `source_types=['catalog']`. Therefore
neither catalog-import markers nor the exact source-type value can recover the
held editorial origin without exceptions.

## Refusal reason and next evidence needed

The anomaly does **not** demonstrate mixed provenance in the null cohort: exact
source-type values and both import markers consistently point away from a
catalog round-trip. The refusal instead arises at the mandatory positive
control. The graph contains five accepted pipeline identities with the same
catalog-import evidence used to recognize catalog edits, so the available
markers do not encode a zero-error inverse of declared origin.

Before any backfill, those five counterexamples need a durable provenance
explanation that distinguishes why their catalog round-trip did not make them
`catalog_edit`, or a different schema-owned marker/history source must be shown
to classify all 1,798 accepted controls with **0** errors. Until that evidence
exists, the safe result is no rule and no mutation.
