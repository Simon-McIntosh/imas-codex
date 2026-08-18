# Ancestor supersession instrument evidence

Date: 2026-08-18

## Contract

`supersede_into_ancestor` folds a live Standard Name only into an accepted,
valid node already reachable through its outgoing `REFINED_FROM` history. It
does not create or delete a lineage relationship. The exact preview signs the
two lifecycle records, the complete connected lineage edge set, every source
bound to the descendant, and each source's backing projections.

Sources bound only to the descendant are re-validated against the ancestor and
moved by `retarget_standard_name_sources`. Sources already bound to both names
with their scalar selecting the ancestor are re-validated and lose only the
redundant descendant binding under an exact compare-and-set. Other source
shapes are refused. Apply requires the preview digest; a deterministic ledger
record makes replay return `already_applied` with zero writes.

## Transactional regression evidence

The red run exited 2 during collection because the instrument and its conflict
type did not yet exist. The final run used an isolated, auth-disabled Neo4j
2026.01.4 instance on `bolt://127.0.0.1:28687`, never the project graph:

```text
.....                                                                    [100%]
5 passed in 10.85s
```

The five tests cover a direct ancestor, a multi-hop ancestor with idempotent
replay, non-ancestor refusal, attachment re-validation refusal with rollback,
and exact deduplication of a source already bound to both names. The direct and
multi-hop assertions prove the original lineage remains and no reverse edge is
created.

## Live preview

The authorized live operation was preview-only. The transaction rolled back
and no apply was invoked.

| Field | Value |
|---|---:|
| Descendant | `atomic_count_of_ion_state` |
| Ancestor | `atomic_count` |
| Manifest SHA-256 | `857296594056a3c99475f249356aeec905cc3fe4614e3d0f9944fc7bf4567a07` |
| Sources signed | 76 |
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
- `disposable-neo4j-test-final.log`
- `live-dry-run.log`
