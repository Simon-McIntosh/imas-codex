# Unsourced chain-cap identities, partitioned by recoverability

## Authority and method

The semantic authority is the live `imas-codex:sn-release-readiness` plan,
version 151, section §12 ("Two defect classes a rescore surfaced before
spending"). The question the plan fixes is narrow: of the `chain_length >= 3` standard names carrying no
producing source, how many can have a source recovered from their refinement
ancestry, and how many cannot — because that split decides whether the beat is a
repair or a rebuild, and how large each part is.

Measured live on **2026-09-17** through `imas_codex.graph.client.GraphClient`
from this worktree, against the active graph (`graph_name = codex`). Every
statement is a **read**: no graph mutation was performed, no node, edge or
property was written.

- Cohort definition — `sn.chain_length >= 3`, counting `StandardName`
  identities.
- Unsourced definition, verbatim from the plan and from
  `unsourced-origin-cause.md`:
  `NOT EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }`.
- Recoverable definition — a sourced ancestor exists at any depth:
  `EXISTS { MATCH (sn)-[:REFINED_FROM*1..]->(a:StandardName)
  WHERE EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(a) } }`.

The producer predicate is expressed as `EXISTS` subqueries rather than an
`OPTIONAL MATCH`, so each `StandardName` contributes exactly one row and the
counts below are **identity counts, not row counts**. This matters: the earlier
ancestry census double-counted one identity because
`flux_surface_normal_surface_integrated_net_energy_flux_at_plasma_boundary`
carries two immediate `REFINED_FROM` predecessors, and that identity is not in
this cohort. The sum is asserted below to make the collapse visible.

## Live results

| Measure | Live value |
|---|---:|
| `chain_length >= 3` identities (cohort) | **174** |
| — carrying a producing source | 72 |
| — carrying no producing source (unsourced) | **102** |
| unsourced with a sourced ancestor at any depth | **10** |
| unsourced with no sourced ancestor at all | **92** |
| — of which no `REFINED_FROM` predecessor exists | 1 |
| — of which a sourced *immediate* predecessor exists | 6 |
| — of which the sourced ancestor is at depth 2–3 only | 4 |

### Positive controls

A negative count from a predicate that can never fire proves nothing, so the
same producer predicate was run where it must fire:

| Control | Live value |
|---|---:|
| `EXISTS { (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }` over the whole label | **2,789** |
| the same predicate over the `chain_length >= 3` cohort (known-sourced part) | **72** |
| `(:StandardNameSource)-[r:PRODUCED_NAME]->(:StandardName)` edge count | **5,493** |

The predicate returns 2,789 over the label and 72 over the cohort's sourced
half, so it demonstrably fires; the 102 unsourced is an absence the predicate
can see through, not an inability to match. Both properties used by the negative
tests — a populated `chain_length` and a live `PRODUCED_NAME` ledger — exist and
are non-empty.

## Verdict against the plan and against the audit

| Source | Asserted | Live | Verdict |
|---|---|---|---|
| Plan §12 / `f-srr-unsourced`: unsourced chain-cap | 78 | **drifted** (102) |
| Plan §12: unrebuildable (no sourced ancestor at any depth) | 72 | 92 | **drifted** |
| Plan figure audit (2026-09-14 re-measure): cohort at `chain_length >= 3` | 174 | **current** (174) |
| Plan figure audit: unsourced chain-cap | 102 | 102 | **current** (102) |

The plan's own section text quotes the 78/72 split; the audit's re-measured
174/102 still describes the live graph exactly, so the audit is the figure the
rescope should carry and the plan's section text is drifted by 92 percent on the
unrebuildable part. Against the section's own cohort of 126 identities, the
cohort has grown 38 percent (126 → 174) and the unrebuildable class 28 percent
(72 → 92) while the section sat. The audit's re-measure is the current figure.

## The partition, and the sum asserted

```text
10 with a sourced ancestor at any depth + 92 with none = 102 unsourced
 6 sourced immediate predecessor         + 4 sourced only deeper = 10 recoverable
72 sourced + 102 unsourced = 174 chain-cap
```

The two parts of the unsourced cohort sum to the whole (10 + 92 = 102), and the
recoverable part splits exactly into its immediate and deeper halves (6 + 4 =
10). The recoverable partition is the **smaller** partition (10 of 102) and
every identity in it is listed below. The remaining 92 have no authority cohort
to replay at any depth and belong to the rebuild class; only one of them has no
`REFINED_FROM` predecessor at all.

### The recoverable partition — every identity with a sourced ancestor, by id

| Identity | depth to sourced ancestor | sourced ancestor | disposition |
|---|---:|---|---|
| `absorbed_plasma_heating_power` | 1 | `total_plasma_heating_power` | immediate repair |
| `inverse_of_spectral_surface_curvature_of_optical_element` | 1 | `spectral_surface_curvature_of_optical_element` | immediate repair |
| `parallel_neutral_state_convection_velocity` | 1 | `parallel_neutral_particle_convection_velocity` | immediate repair |
| `particle_count_accumulated_at_pellet_path_due_to_pellet_injection` | 1 | `accumulated_particle_count_at_pellet_path_due_to_pellet_injection` | immediate repair |
| `radial_plasma_momentum_source` | 1 | `radial_momentum_source` | immediate repair |
| `tritium_density_flux_surface_averaged_at_plasma_boundary` | 1 | `flux_surface_averaged_tritium_density_at_plasma_boundary` | immediate repair |
| `total_momentum_flux_normalized_due_to_perturbed_parallel_vector_potential` | 2 | `normalized_total_momentum_flux_due_to_perturbed_parallel_vector_potential` | deeper adjudication |
| `vertical_coordinate_of_plasma_filament` | 2 | `vertical_outline_of_plasma_filament` | deeper adjudication |
| `neutral_internal_state_atomic_power_density_due_to_collisions` | 3 | `neutral_state_power_density` | deeper adjudication |
| `surface_thickness_of_cryostat` | 3 | `thickness_of_cryostat` | deeper adjudication |

Six identities sit one `REFINED_FROM` hop above a sourced predecessor, so a
predecessor-cohort replay is available for these; four reach their sourced
ancestor only at depth 2–3, so copying the remote binding would skip the
intervening semantic refinements and each needs row-level lineage adjudication
rather than a blind replay. (Depth is the shortest `REFINED_FROM` path to *any*
sourced ancestor, so a row may also have a nearer unsourced predecessor.)

The one predecessor-less row is named here rather than counted, because it is
terminal and carries no predecessor to repair from:

- `toroidal_diamagnetic_magnetic_flux_at_flux_surface` — `name_stage =
  superseded`, `origin = pipeline`.

It is not release-relevant as it stands.

## What this decides for the beat

The unsourced chain-cap cohort is **rebuild-dominant**: 92 of 102 identities
have no sourced ancestor anywhere, so no history to replay. Only 10 carry a
recoverable ancestor, 6 of them immediate. A remediation sized on the plan's 72
would under-scope the rebuild class by 20 identities; the live figure is 92, and
the audit's 102 for the whole unsourced cohort is current.

## Reproduction

Scripts and logs on disk for this run:

- `/tmp/unsourced-cohort/census.py` → `/tmp/unsourced-cohort/census.log`,
  exit 0 (cohort, unsourced, controls, partition, id list).
- `/tmp/unsourced-cohort/depth.py` → `/tmp/unsourced-cohort/depth.log`,
  exit 0 (depth to nearest sourced ancestor, immediate-predecessor sourcing).
- `/tmp/unsourced-cohort/boundary.py` → `/tmp/unsourced-cohort/boundary.log`,
  exit 0 (the predecessor-less row, with stage and origin).

Those paths are session scratch and are quoted for this run only; the exact
queries are inlined below so the queries, rather than the scratch files, are the
durable record.

```cypher
// cohort total
MATCH (sn:StandardName) WHERE sn.chain_length >= 3 RETURN count(sn)

// unsourced
MATCH (sn:StandardName) WHERE sn.chain_length >= 3
  AND NOT EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }
RETURN count(sn)

// recoverable: exists a sourced ancestor at any depth
MATCH (sn:StandardName) WHERE sn.chain_length >= 3
  AND NOT EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }
  AND EXISTS { MATCH (sn)-[:REFINED_FROM*1..]->(a:StandardName)
               WHERE EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(a) } }
RETURN count(sn)

// unrecoverable: no sourced ancestor at any depth
MATCH (sn:StandardName) WHERE sn.chain_length >= 3
  AND NOT EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }
  AND NOT EXISTS { MATCH (sn)-[:REFINED_FROM*1..]->(a:StandardName)
                   WHERE EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(a) } }
RETURN count(sn)

// depth to the nearest sourced ancestor, per recoverable identity
MATCH p = (sn:StandardName)-[:REFINED_FROM*1..5]->(a:StandardName)
WHERE sn.chain_length >= 3
  AND NOT EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }
  AND EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(a) }
RETURN sn.id AS id, min(length(p)) AS depth_to_sourced ORDER BY sn.id
```

The figure on the cumulative landed record
(`docs/evidence/archive/sn-release-readiness-landed.html#unsourced-cohort-partition`)
shows the three-tier partition; the identity list above is the table it cannot list.