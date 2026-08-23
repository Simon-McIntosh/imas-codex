# Documentation review-resolution mirror: blocked preflight

## Outcome

No graph mutation was made. The live population and production export baseline
were reproduced, but the requested backfill cannot be performed through a
current sanctioned persistence function. The assigned write fence permits only
this evidence file, so repairing or adding the missing projection mechanism is
outside this node's authority. A raw Cypher `SET` would write the scalar outside
the mechanism that owns documentation-review persistence and was therefore not
used.

## Live preflight

The graph was read through `GraphClient` with the worktree linked to the main
checkout's authenticated environment. No model was called and no LLM cost was
incurred.

| Measure | Live value |
|---|---:|
| All `StandardName` nodes | **4,666** |
| `name_stage=accepted`, `docs_stage=accepted`, null `docs_review_resolution_method` | **1,516** |
| Same population with both `reviewer_score_docs` and `reviewed_docs_at` | **1,408** |
| Accepted `catalog_edit` rows with no docs-axis review at all | **107** |
| Null-method rows with neither a docs score nor a docs timestamp | **108** |
| Of those 108, rows with no docs-axis review at all | **107** |

The remaining row in the 108-row no-score/no-timestamp population is
`krypton_density_at_magnetic_axis`. It has six attached docs-axis review rows
across three consensus groups but no mirrored docs score or review timestamp;
it is not part of the measured 1,408-row reviewed-and-passed cohort.

The reachable docs-axis terminal methods across the null-method population
also reproduce the dispatch measurements. Counts are distinct identities per
method, so an identity with several historical groups may contribute to more
than one row.

| Reachable terminal method | Names |
|---|---:|
| `quorum_consensus` | **1,234** |
| `authoritative_escalation` | **491** |
| `single_review` | **226** |
| `max_cycles_reached` | **2** |

The winning set was derived from the `ReviewResolutionMethod` enum description
in `imas_codex/schemas/standard_name.yaml`, not from a locally authored method
list. The schema-derived set is `quorum_consensus`,
`authoritative_escalation`, and `single_review`. Runtime assertions verified
that both `max_cycles_reached` and `retry_item` are excluded. This is fail
closed: an enum value not explicitly named by the schema as eligible is not
silently admitted.

The two identities carrying a reachable `max_cycles_reached` docs group are:

- `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_electron_density_at_pedestal_maximum`
- `tritium_density`

Per the node contract, both remain null. No non-winning terminal group was
copied, and neither identity was made export-eligible by a review that failed
to reach a verdict.

## Production export baseline

The baseline was measured through the real `run_export` path with
`min_score=0.85`, `skip_gate=True`, `force=True`, and
`include_sources=False`. It was not reconstructed from a hand predicate.

| Export measure | Before | After |
|---|---:|---:|
| Accepted/approved export population | **2,335** | **not measured: no write** |
| EXPORT-ELIGIBLE / emitted | **537** | **not measured: no write** |
| Exclusion accounting | **passed** | **not measured: no write** |

The before-run exclusion ledger included **1,508**
`documentation_review_unresolved` identities. The complete pre-write ledger
was: 1 bound-adjacent name, 267 documentation-not-accepted names, 1,508
documentation-review-unresolved names, 2 invalid catalog entries, 19 invalid
validation statuses, and 1 name-review quorum shortfall. Two further candidates
were rejected while constructing catalog entries, leaving 537 emitted
identities. The durable report is at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T163911466698-n-docsmirror/before-export/.export_report.json`.

## Exact blocker

The scalar's current owner is `persist_reviewed_docs` in
`imas_codex/standard_names/graph_ops.py`. Its compare-and-set query writes
`docs_review_resolution_method`, but only while transitioning a claimed
`docs_stage=drafted` row. Every target here is already
`docs_stage=accepted`; using that function would require staging accepted
documentation for rescore, clearing existing docs-review projections, and then
replaying a lifecycle transition. That is not a denormalized projection
backfill and would change more state than authorized.

Two apparent alternatives do not provide the missing sanctioned path:

- `update_review_aggregates` says in its docstring that it mirrors a winning
  group, but its implementation only writes `review_count`,
  `review_mean_score`, and `review_disagreement`. It never reads
  `review_axis`, never selects a review group, and never writes either
  resolution-method scalar.
- `write_docs_review_results` mirrors docs scores and timestamps, but does not
  accept or write `docs_review_resolution_method`.

Consequently there is no current function that can take a schema-derived,
docs-axis, canonical winning group and idempotently project only its resolution
method onto an already accepted name. Adding that mechanism and its regression
test requires scope for at least
`imas_codex/standard_names/graph_ops.py` and a focused file under
`tests/standard_names/`. The test must derive the winning set from the LinkML
schema and assert that `max_cycles_reached` and `retry_item` remain excluded.
Once that lands, this operational node can be rerun to produce the required
per-row receipt, the second-run zero-write proof, the 4,666-to-4,666 identity
sentinel, and the authoritative post-write `run_export` count.

## Conservation and non-actions

- `StandardName` count at preflight: **4,666**.
- `StandardName` count after this node: **4,666** because no mutation ran.
- Rows mirrored: **0**.
- Null-method population after this node: **1,516**.
- The **107** accepted `catalog_edit` rows carrying no docs-axis review remain
  excluded; no review basis was invented for them.
- Idempotency replay: **not executable** until the sanctioned projection path
  exists.
- Receipt rows: **none**, because emitting a receipt for an unapplied write
  would be false evidence.
- LLM calls: **0**.
