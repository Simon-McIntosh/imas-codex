# Ancestor supersession instrument evidence

Date: 2026-08-18

## Contract

`supersede_into_ancestor` folds a live Standard Name only into an accepted,
valid node already reachable through its outgoing `REFINED_FROM` history. It
also requires strict ancestry: the target must not be the descendant and must
not be reverse-reachable from it through a directed cycle. The instrument does
not create or delete a lineage relationship. The exact preview signs the two
lifecycle records, the complete connected lineage edge set, every source bound
to the descendant, every backing node, and each backing projection relationship.

Sources bound only to the descendant are re-validated against the ancestor and
moved by `retarget_standard_name_sources`. Sources already bound to both names
with their scalar selecting the ancestor are re-validated and lose only the
redundant descendant binding under an exact compare-and-set. Other source
shapes are refused. Apply requires the preview digest; a deterministic ledger
record makes replay return `already_applied` with zero writes. Before apply,
the transaction locks every signed source, backing node, and projection
relationship and then recomputes the complete authority hash under those locks.

## Transactional regression evidence

The red run exited 2 during collection because the instrument and its conflict
type did not yet exist. The final run used an isolated, auth-disabled Neo4j
2026.01.4 instance on `bolt://127.0.0.1:28687`, never the project graph:

```text
.........                                                                [100%]
9 passed in 6.90s
```

The nine tests cover a direct ancestor, a multi-hop ancestor with zero-write
idempotent replay, non-ancestor refusal, self-target refusal, directed-cycle
refusal, attachment re-validation refusal with rollback, scalar-disagreement
refusal, exact deduplication of a source already bound to both names, and a
concurrent source-scalar plus backing-projection drift between hash computation
and locking. The direct and multi-hop assertions prove the original lineage
remains and no reverse edge is created.

Replay is proven write-free by canonical byte equality over every participant
node's labels and properties and every incident relationship's identity, type,
properties, and complete endpoint labels and properties. This includes the
signed names, source, backing node, lineage, source binding, backing projection,
and ledger-link closure; it is not inferred from a change-row count.

## Live preview

The authorized live operation was preview-only. The transaction rolled back
and no apply was invoked.

| Field | Value |
|---|---:|
| Descendant | `atomic_count_of_ion_state` |
| Ancestor | `atomic_count` |
| Manifest SHA-256 | `e51c191e06fb4d8df8d8e32ff374304e0e02a94cdcdeef1238ddbb9a22019a57` |
| Sources signed | 76 |
| Backing nodes signed | 76 |
| Backing projections signed | 115 |
| Singly-bound sources to retarget | 37 |
| Dual-bound sources to deduplicate | 39 |
| Preserved lineage edges | 6 |
| Persistent changes | 0 |

The observed 39 dual-bound sources agree exactly with the lineage
adjudication. Their scalar mirrors already select `atomic_count`; the remaining
37 source rows are exclusively bound to `atomic_count_of_ion_state` and are the
cohort the guarded retarget primitive would migrate during a separately
authorized apply.

Full logs and the complete signed manifest are stored in the worker run
envelope:

- `red-proof.log`
- `corrective-red.log`
- `corrective-final-test.log`
- `live-dry-run-corrective.log`
