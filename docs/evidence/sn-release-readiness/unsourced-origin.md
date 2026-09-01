# Origin of the 72 never-grounded chain-cap identities

## Verdict

The live population still contains **72 distinct `StandardName` identities** at
`chain_length >= 3` that have neither a current `PRODUCED_NAME` binding nor a
source-bearing `REFINED_FROM` ancestor. They were minted by exactly two root
write paths:

| Root-minting path | Identities | Live discriminator |
|---|---:|---|
| Legacy generated-name persistence | **37** | At least one chain root has `generated_at`, no `imported_at`, and a generation model. |
| Removed bulk catalog-import fold | **35** | The root has `origin='catalog_edit'`, `generated_at` and `model` null, `created_at=2026-07-04T21:20:38.632Z`, and `imported_at=2026-07-04T21:21:17.079Z`. |
| **Total** | **72** | Exact identity-collapsed population. |

The dominant path is therefore the legacy generated-name writer, by 37 to 35:
`imas_codex/standard_names/graph_ops.py::write_standard_names` (currently at
the definition beginning near line 5036). The catalog-fold path was
`imas_codex/standard_names/catalog_import.py::_write_import_entries`, removed
by commit `4b949931`; the parent revision shows its `MERGE
(sn:StandardName {id: b.id})` and its deliberate absence of both a
`StandardNameSource` node and a `PRODUCED_NAME` edge.

This is **not** a population of 72 names that the current composer can mint
without an input source. The current generate path,
`persist_generated_name_winners`, drops candidates without `source_id` and
`source_types`, requires a claimed `StandardNameSource`, and finalizes the
`PRODUCED_NAME` reservation in the same transaction as the rich name write.
What remains permissive is refinement: `persist_refined_name` collects the
predecessor's source cohort and calls `retarget_standard_name_sources(...,
_allow_empty_noop=True)`. An already-unsourced predecessor therefore produces
an already-unsourced successor without refusal. That is how roots created by
the two historical paths propagated to the chain cap.

## Live measurement and positive control

All graph operations were read-only `MATCH ... RETURN` queries through
`GraphClient` on 2026-09-01. The positive control returned:

| Property or relationship instrument | Live result |
|---|---:|
| `StandardName` candidates | 4,675 |
| candidates with `id` | 4,675 |
| candidates with non-null `chain_length` | 3,104 |
| candidates with `chain_length >= 3` | 126 |
| candidates with non-null `origin` | 3,576 |
| distinct `StandardNameSource` nodes | 9,678 |
| `PRODUCED_NAME` relationship rows | 5,299 |
| roots in this cohort currently receiving `HAS_PARENT` | 8 |
| roots/chains in this cohort carrying change-ledger operations | positive; operations listed below |

Thus the negative source predicate is aimed at populated labels, properties,
and relationship directions. It is not a zero produced by a guessed key.
`consolidated_at` is declared on `StandardName` and written only by
`graph_ops.mark_names_consolidated`; the live graph has **0** populated values,
including **0/72** in this cohort. Consolidation therefore supplies no minting
provenance for any row and cannot be used to explain the roots.

The identity-collapsed ancestry query returned **72 identities, 71 with one
root and 1 with two roots**. The two-root identity is
`flux_surface_normal_surface_integrated_net_energy_flux_at_plasma_boundary`,
whose roots are `plasma_power_at_plasma_boundary` and
`power_at_plasma_boundary`. It is counted once throughout this report.

## Root path 1: legacy generated-name persistence — 37

All 37 identities have a generated root rather than the bulk-import signature:
`generated_at` is populated, `imported_at` is null, and the root carries a
generation model. Representative early roots carry
`hosted_vllm/deepseek-v4-flash`; later roots carry configured OpenRouter
models. Their creation dates span the pre-ledger generation period beginning
2026-06-17.

The source-level trace is `graph_ops.write_standard_names`. Its contract
requires each input dict to carry `source_types` and `source_id`; it first
MERGEs the `StandardName`, then writes the historical direct backing projection
`(:IMASNode|FacilitySignal)-[:HAS_STANDARD_NAME]->(sn)`. That path predates the
present authoritative `StandardNameSource-[:PRODUCED_NAME]->StandardName`
ledger. Therefore these 37 were composer-generated from source-bearing input,
but the root mint itself did not establish the relationship the current census
uses as authority. Subsequent refinement could see an empty
`StandardNameSource` cohort and accept it as a no-op.

The full-chain ledger corroborates later fold/dedup/refine activity rather than
a source-free model proposal: 33 of the 37 carry
`supersede_exhausted_orphan`, 32 carry `backfill_refine`, two carry
`regenerate`, and the sole two-root identity carries both `regenerate` and
`human_edit`. These markers overlap and are not used as an exclusive count.
No generated root currently receives `HAS_PARENT`, and no generated root has
`consolidated_at`.

## Root path 2: removed catalog-import fold — 35

All 35 roots share the same bulk-write fingerprint: `origin='catalog_edit'`,
the same `created_at` and `imported_at` timestamps, null `generated_at`, and
null model. Historical source inspection at `4b949931^` identifies the exact
writer as `catalog_import._write_import_entries`. It MERGEd a name and set its
catalog/editorial fields but, as its own `run_import` contract stated, created
no `StandardNameSource` and no `PRODUCED_NAME`; catalog `sources:` were ignored.
This is a fold-back path, not composer output.

The `HAS_PARENT` join divides these 35 roots into **8** that currently receive
one or more structural child edges and **27** that do not. The shared import
timestamps prove that even the eight structural roots were minted by the
catalog fold; `HAS_PARENT` topology was attached or retained later and is not
their node-creation path. There are **0** roots with `origin='derived'` and
**0** cohort roots with consolidation provenance.

The full-chain ledger shows source migration/refinement after import: 34 of the
35 catalog-fold identities carry `source_migration_manifest`; one instead
carries the older `backfill_refine`/`human_edit` path. Again these are later
chain operations, not alternative root minters.

## Mechanism and implication

The 72 are not explained by cross-batch consolidation, and only eight have
current derived-parent topology. The exact explanation is:

1. **37** roots entered through a legacy composer persistence representation
   that required source-bearing input but did not yet make
   `StandardNameSource/PRODUCED_NAME` the authoritative root binding.
2. **35** roots entered through a catalog fold that explicitly created no
   source ledger at all.
3. Refinement then propagated either kind of unsourced predecessor because
   `persist_refined_name` deliberately permits an empty source-cohort no-op.

So the answer to the plan's binary question is qualified but decisive:
**the current composer cannot mint an unbound name; not every root arrived via
fold, dedup, or derived materialization, because 37 are legacy composer writes.
The storage representation of those writes was nevertheless pre-authoritative,
and the remaining defect is that refinement treats absence as a valid cohort.**
The 35 catalog roots did arrive through a fold. One identity has an explicit
multi-root regenerate/dedup history. Eight catalog roots have derived topology,
but zero roots were minted with `origin='derived'`.

## Reproduction predicates

The population predicate was:

```cypher
MATCH (sn:StandardName)
WHERE sn.chain_length >= 3
  AND NOT EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }
  AND NOT EXISTS {
    MATCH (sn)-[:REFINED_FROM*1..]->(ancestor:StandardName)
    WHERE EXISTS {
      MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(ancestor)
    }
  }
```

Roots were selected per identity by walking `REFINED_FROM*0..` and retaining
nodes with no outgoing `REFINED_FROM`. Each root was then joined to `origin`,
`generated_at`, `imported_at`, `model`, `source_paths`, `consolidated_at`,
incoming and outgoing `HAS_PARENT`, and every linked
`StandardNameChange.operation`. Counts were performed after collecting roots
per terminal identity, so the one two-root chain contributes one, not two.
