NEEDS-HELP: The live three-child cohort is confirmed, but the signed operator has no closed HAS_PARENT reparent program, so the required one-invocation mutation cannot be executed within this node's documentation-only write fence.

tried: Read the live plan at version 249 and SHA-256 `9cc1bf7e6831a968ba9d2b27781a06d66c237021792ed451d5dcc3a4a13f92af`, inspected the prior signed refusal and the current operator registry, and ran two read-only production preflights. The graph still has exactly 3 live `HAS_PARENT` children of `area_of_flux_surface`, matching the previously observed 3. The canonical target `poloidal_plane_cross_sectional_area_of_flux_surface` is accepted and valid with 21 producing sources. `area_of_flux_surface` is accepted and valid with one derived producer. No graph mutation was attempted; `StandardNameChange` stayed at 7,841 and `LLMCost` at 27,631 across the preflight.

options: (1) Extend the existing signed-manifest operator with a closed, tested `StandardName` `HAS_PARENT` reparent-and-supersede program, then rerun this node with source/test paths added to its fence. (2) Add an equivalently closed transaction to an existing sanctioned structural operator, with the same signed cohort, receipts, replay, collateral, and refusal guarantees. (3) Use raw Cypher to move the edges and supersede the node; this is rejected because it bypasses the signed operator, tested compare-and-set closure, and receipt contract.

leaning: Option 1. The generic signed envelope already owns authority bytes, participant locking, canonical manifest hashing, admitted/refused accounting, `StandardNameChange` receipts, collateral fingerprints, and replay. A narrowly registered child-reparent program adds only the missing typed mutation semantics and keeps this repair on the same authority surface as the earlier supersede refusal.

cost-if-wrong: A broader or bespoke path would require new schema and replay review, might invalidate previously signed authorities, and could leave children reparented without the umbrella supersede or leave the umbrella's derived source attached to a superseded identity. Raw Cypher would require a separate forensic reconstruction because it would produce no signed row receipts.

# Flux-surface umbrella retirement preflight

## Required outcome versus observed state

The requested terminal operation is one signed invocation that derives the live
children of `area_of_flux_surface`, relocates every child to
`poloidal_plane_cross_sectional_area_of_flux_surface`, proves the umbrella has
zero live children, and only then supersedes the umbrella into the same
canonical identity. The invocation must refuse the supersede if any child move
is refused, emit one receipt per mutated row, preserve every verbatim refusal,
and replay at `already_applied` with zero persistent writes.

The read-only live preflight derived exactly the expected cohort:

| Child derived from live `HAS_PARENT` closure | Lifecycle | Producing sources | Current edge properties |
|---|---|---:|---|
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | accepted, valid | 1 (`attached`) | `operator_kind=unary_prefix`; `operator=derivative_with_respect_to_normalized_poloidal_flux_coordinate` |
| `derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface` | accepted, valid | 1 (`composed`) | `operator_kind=unary_prefix`; `operator=derivative_with_respect_to_toroidal_flux_coordinate` |
| `surface_area_of_flux_surface` | accepted, valid, catalog status `draft` | 20 (15 `attached`, 5 `composed`) | `operator_kind=qualifier`; `operator=surface` |

Each child has exactly one current `HAS_PARENT` edge, and that edge targets
`area_of_flux_surface`. The child count is **3 live rows against the previously
observed 3**. The preflight captured each child's complete producing-source
properties and lifecycle so a future applying invocation can state
quantitatively whether both are unchanged after relocation.

The umbrella itself currently has one producing source:
`derived:area_of_flux_surface`, at lifecycle `composed`, whose scalar target is
still `area_of_flux_surface`. The final closed program must account for this
source explicitly; merely setting the umbrella lifecycle to `superseded` would
leave a producer pointing at a terminal identity.

## Exact operator gap

`apply_signed_manifest` currently admits the following relationship programs:

- `delete_relationship` only as a closed
  `StandardNameSource-[:PRODUCED_NAME]->StandardName` reconciliation;
- `add_relationship` only as a closed derived-source revival that creates
  `StandardNameSource-[:PRODUCED_NAME]->StandardName`;
- paired delete/add/set only as an ordinary producing-source migration.

The executor hard-codes those relationship types and endpoint labels. It cannot
delete a signed `StandardName-[:HAS_PARENT]->StandardName` edge, recreate it on
another parent while preserving its exact properties, or make the final
supersede conditional on all child rows being admitted. An authority containing
such mutations is rejected by the closed-program validators before graph
access.

The two other public structural paths do not satisfy the contract:

- `reconcile_structural_edges_for_standard_names` deterministically re-derives
  each child's current grammar parent, which is still `area_of_flux_surface`;
  it would recreate the present topology rather than relocate it.
- `supersede_into_ancestor` requires a live `REFINED_FROM` path from the
  successor to the predecessor. No path exists in either direction between
  `area_of_flux_surface` and
  `poloidal_plane_cross_sectional_area_of_flux_surface`, and that operator does
  not pre-relocate incoming structural-child edges.

The earlier builder-emitted supersede authority therefore remains correctly
refused with the verbatim reason `target has a live structural child`. Running
it again before a sanctioned child relocation would repeat the same refusal;
moving the relationships with raw Cypher would violate the repository's
Standard Names mutation boundary and would not meet the signed receipt or replay
measure.

## Required scope extension

A corrective node needs exclusive write scope over the signed-manifest operator
and focused tests. The smallest safe program should:

1. derive the live child closure and the final umbrella row inside one applying
   invocation;
2. sign each existing `HAS_PARENT` relationship, both endpoint snapshots, and
   exact edge properties;
3. lock and re-derive the cohort before any mutation;
4. admit or refuse every child relocation independently, but apply no row if
   any relocation is refused;
5. prove the umbrella live-child count is zero inside the same transaction
   before admitting its supersede row;
6. preserve every child's lifecycle and complete producing-source closure;
7. account for `derived:area_of_flux_surface` without leaving it attached to a
   superseded target;
8. emit one immutable `StandardNameChange` per mutated logical row and verify
   receipt count, live ledger delta, collateral fingerprint, and write-free
   replay.

Until that program exists, the requested quantitative apply measures are
honestly **not produced**: admitted/refused partition, mutated rows, receipt
rows, ledger delta, replay outcome, and post-apply child count are all absent,
not inferred. Production state remains unchanged.

## Evidence files

- Live cohort and child source/lifecycle closure:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T091811642017-umbrellamove/live-preflight.log`
- Lineage and umbrella-source preflight:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T091811642017-umbrellamove/lineage-preflight.log`
- Prior signed refusal authority:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T085559805386-supersede6/legacy-spelling-supersede-authority.json`
- Prior machine receipt containing the three-child refusal:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T085559805386-supersede6/legacy-spelling-supersede-result.json`
