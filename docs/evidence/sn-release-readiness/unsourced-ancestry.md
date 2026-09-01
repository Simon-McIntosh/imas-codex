# Unsourced refine-product ancestry census

## Authority and method

The semantic authority was the live `imas-codex:sn-release-readiness` plan,
version 81, section 12. The four read-only Cypher queries preserved in
`unsourced-refines.md` were executed unchanged against the live graph through
the imas-codex graph REPL on 2026-09-01. No graph mutation was attempted.

The preserved ancestry query has one cardinality defect that matters to its
interpretation: its optional match produces one row per immediate
`REFINED_FROM` predecessor, not one row per unsourced identity. The exact query
was still run and its raw aggregate is reported below. A supplemental read-only
identity-collapsed query was then used to state the requested partition without
double-counting.

## Positive control

The first preserved query returned:

| Measure | Live result |
|---|---:|
| `StandardName` nodes | 4,675 |
| nodes with non-null `chain_length` | 3,104 |
| nodes with `chain_length >= 3` | 126 |
| distinct bound `StandardNameSource` nodes | 5,195 |
| `PRODUCED_NAME` relationship rows | 5,299 |

This is a firing positive control, not a null-property inference:
`chain_length` is populated on 3,104 nodes, 126 nodes satisfy the chain-cap
predicate, and the graph contains 5,299 authoritative source relationships.
Both properties used by the negative tests therefore demonstrably exist.

## Reproduced live population

The second preserved query measured the current `chain_length >= 3` population
as follows; these are live results, not the plan's carried figures:

| Population | Live count |
|---|---:|
| total | 126 |
| unsourced by absent `PRODUCED_NAME` | 78 |
| sourced by present `PRODUCED_NAME` | 48 |
| non-empty denormalized `source_paths` | 54 |

Thus the current authoritative-edge population is **126 total = 78 unsourced +
48 sourced**. The measured 78 reproduces the earlier plan observation, but is
asserted here from the current query. The 54 non-empty scalar projections are
reported separately and are not treated as source authority.

## Exact preserved ancestry and identity-manifest results

The third preserved query returned `unsourced=79`,
`recoverable_from_immediate_predecessor=4`,
`sourced_ancestor_at_any_depth=6`,
`no_sourced_ancestor_anywhere=73`, and `no_predecessor=1`.

The fourth preserved identity-manifest query likewise returned **79 rows but
78 distinct identities**. The extra row is not graph churn: the unsourced
identity
`flux_surface_normal_surface_integrated_net_energy_flux_at_plasma_boundary`
has two immediate `REFINED_FROM` predecessors,
`surface_integrated_net_lost_energy_flux_at_plasma_boundary` and
`power_at_plasma_boundary`. Both preserved queries expand that identity twice.
The exact third-query aggregates therefore count one already-unsourced identity
twice and cannot be used as the requested identity partition without collapse.

## Identity-collapsed ancestry partition

The supplemental query retained the same live predicates but expressed
immediate and any-depth ancestry as `EXISTS` subqueries before counting each
`sn` once. It returned:

| Disposition | Distinct identities |
|---|---:|
| source-bearing ancestor anywhere | **6** |
| of which immediate-predecessor repair is available | **4** |
| of which source exists only at greater depth | **2** |
| no source-bearing ancestor anywhere | **72** |
| of which no `REFINED_FROM` predecessor exists | **1** |
| total unsourced identities | **78** |

The required accounting closes exactly:

```text
4 immediate repair + 2 greater-depth adjudication + 72 rebuild = 78 unsourced
6 with sourced ancestry + 72 with no sourced ancestry = 78 unsourced
```

The four immediate-repair candidates and their source-bearing predecessor
bindings are:

- `absorbed_plasma_heating_power` from
  `total_plasma_heating_power`, source
  `derived:total_plasma_heating_power`.
- `inverse_of_spectral_surface_curvature_of_optical_element` from
  `spectral_surface_curvature_of_optical_element`, source
  `derived:spectral_surface_curvature_of_optical_element`.
- `parallel_neutral_state_convection_velocity` from
  `parallel_neutral_particle_convection_velocity`, sources
  `dd:edge_transport/model/ggd/neutral/particles/v_parallel/values` and
  `dd:plasma_transport/model/ggd/neutral/particles/v_parallel/values`.
- `radial_plasma_momentum_source` from `radial_momentum_source`, source
  `dd:plasma_sources/source/ggd/neutral/momentum/r`.

The two greater-depth-only cases are not safe for a blind immediate repair:

- `neutral_internal_state_atomic_power_density_due_to_collisions` reaches
  sourced ancestor `neutral_state_power_density`, bound to
  `dd:plasma_sources/source/profiles_1d/neutral/state/energy`.
- `poloidal_magnetic_field_of_magnetic_field_probe` reaches sourced ancestor
  `poloidal_magnetic_field`, with DD bindings including
  `dd:magnetics/b_field_pol_probe/field` and the equilibrium constraint paths.

They require row-level lineage adjudication because copying a remote ancestor's
binding would skip intervening semantic refinements. The 72 with no sourced
ancestor have no authority cohort to replay and therefore belong to the rebuild
class. Representative rebuild rows include the two plan exemplars
`flux_surface_normal_neutral_energy_diffusion_coefficient` and
`flux_surface_normal_momentum_convection_velocity`; the sole row with no
predecessor at all is `toroidal_diamagnetic_magnetic_flux_at_flux_surface`.

Verdict: **REBUILD 72; REPAIR 4; ADJUDICATE 2** — the unsourced chain-cap population is rebuild-dominant, with only four exact predecessor cohorts safely replayable.
