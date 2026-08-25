# Orphan-parent source reconciliation guard

## Outcome

`reconcile_orphan_parent_sources` now refuses both unsafe admission shapes that
allowed a migrated structural source to revive an obsolete identity:

- a parent whose `name_stage` is terminal (`superseded`, `exhausted`, or
  `contested`); and
- a canonical `derived:<parent>` source that is already bound to a different
  live target.

The predicates run during candidate selection and are repeated in the mutation
query, so a parent or source that changes after selection is refused atomically.
The batched reconciliation path applies the same already-bound check before it
writes any member of its cohort. A stale source remains refused by the existing
lifecycle rule.

## Disposable-graph reproduction

`tests/standard_names/test_orphan_parent_guard.py` constructs the production
failure shape on a disposable Neo4j 2026.01.4 instance:

- terminal predecessor `conductivity` at `name_stage='superseded'`;
- migrated, non-stale `derived:conductivity` bound to the live target
  `electrical_conductivity`;
- a live child retaining `HAS_PARENT` to the terminal predecessor; and
- an independent live parent whose canonical derived source is already bound
  to another live target, plus a terminal parent with no existing source, so
  each refusal predicate is tested independently.

| Measure | Count |
|---|---:|
| Terminal `conductivity` bindings before the unguarded write | 0 |
| Terminal bindings after replaying the unguarded write shape | 1 |
| Terminal bindings after removing the reproduced edge | 0 |
| Candidates selected by the guarded selector across all three unsafe parents | 0 |
| Sources seeded when the three rows are forced past selection into the guarded writer | 0 |
| Terminal bindings after guarded reconciliation | 0 |
| Migrated sources whose live target and scalar remained unchanged | 2 of 2 |

Thus the before-and-after regrowth count is **1 under the unguarded path versus
0 under guarded reconciliation**. No project or production graph was contacted.

## Verification

The final focused command ran the new disposable-graph case, the existing
`reconcile_orphan_parent_sources` disposable-graph test, the existing provenance
rebuild tests, and `tests/graph/test_cypher_property_check.py` against the same
isolated endpoint: **21 passed, 0 failed, 0 skipped**. The pre-existing
`tests/standard_names/test_orphan_parent_source_reconcile.py` remained
byte-unchanged.

Full logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T081258158767-n-orphanparentguard/logs/focused-final.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T081258158767-n-orphanparentguard/logs/disposable-neo4j-final.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T081258158767-n-orphanparentguard/neo4j-runtime/logs/neo4j.log`
