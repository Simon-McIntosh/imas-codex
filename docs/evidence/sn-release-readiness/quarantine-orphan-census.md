# Quarantine and orphan census

provisional: true

Measured 2026-09-21 against the live production graph through
`GraphClient()`, resolved URI `bolt://98dci4-gpu-0002:7687`. Every statement
below is `MATCH`/`RETURN` only — no property, relationship or node was
written. `run_audits` was deliberately **not** used as the validator: a prior
audit called 23 of 35 quarantines stale and a revalidation cleared none, so
the instrument is named beside every number here and the re-check uses
`validate_name_candidate`, the gate that wrote the status in the first place.

## 1. The quarantine figure reproduces exactly

Query:

```cypher
MATCH (sn:StandardName) WHERE sn.validation_status='quarantined'
RETURN count(sn) AS n
```

| Measure | Live | Query |
|---|---:|---|
| All `StandardName` | 5,130 | `MATCH (sn:StandardName) RETURN count(sn)` |
| `validation_status='quarantined'` | **660** | above |
| `validation_status='valid'` | 4,457 | group-by on `validation_status` |
| `validation_status='pending'` | 11 | same |
| `validation_status IS NULL` | 2 | same |

660 is confirmed, not carried forward. The status axis partitions the label
completely (660+4,457+11+2 = 5,130), so nothing is hiding in a fourth value.

### By lifecycle stage

```cypher
MATCH (sn:StandardName) WHERE sn.validation_status='quarantined'
RETURN coalesce(sn.name_stage,'<null>') AS stage, count(*) AS n ORDER BY n DESC
```

| `name_stage` | Quarantined | Exportable stage? |
|---|---:|---|
| superseded | 330 | no |
| exhausted | 266 | no |
| **accepted** | **46** | **yes** |
| drafted | 11 | no |
| reviewed | 6 | no |
| pending | 1 | no |

**596 of the 660 (90.3%) are already terminal** — superseded or exhausted —
and are unreachable by any export gate regardless of their validation status.
The live population that matters to a release is the 46 accepted rows, which
is the same class the 2026-08-23 census measured at 47.
