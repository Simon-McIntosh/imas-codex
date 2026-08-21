# Integrity ratchet regrowth triage

Measured live on 2026-08-21 at `2026-08-21T10:32:45.044Z`, using the two
queries verbatim from `tests/graph/test_sn_integrity_ratchets.py`.

## Verdict

Both ceilings should **hold** at their 2026-08-20 values.

| Integrity class | Recorded ceiling | Current | Standing refusals | Transient reconcile artifacts | Genuine regrowth | Recommendation |
|---|---:|---:|---:|---:|---:|---|
| Sources with multiple live targets | 23 | 27 | 23 | 4 | 0 | **Hold at 23** |
| Stale sources retaining live bindings | 3 | 9 | 3 | 4 | 2 | **Hold at 3** |

The multiple-target ceiling should not be raised to 27: all four excess rows
share one temporarily live target, `toroidal_momentum_flux`, which was
`superseded` immediately before an interrupted scoped reset and is now only
`drafted`. No new standing refusal was authorized. It should not yet be lowered
because all 23 originally measured source/target sets remain live and unchanged.

The stale-binding ceiling should not be raised to 9: four excess rows are the
same interrupted `toroidal_momentum_flux` stage artifact, while the other two
are proven reattachments of edges explicitly detached the previous day. It
should not yet be lowered because the original three named stale residues remain
live and unchanged.

Across the two result sets, **36 rows = 26 named standing refusals + 8 transient
reconcile artifacts + 2 genuine regrowth rows**. The categories are per ratchet
row: the eight transient rows comprise four multiple-target rows and four stale
rows caused by the same temporarily live name.

## Classification rules and evidence chain

- **Named standing refusal** means the exact source id and exact live-target set
  were already present in the frozen 2026-08-20 census from which the ceiling
  was recorded. The current graph matches all 23 multiple-target rows and all 3
  stale rows exactly; none is new.
- **Transient reconcile artifact** means an old edge became countable only
  because its target's lifecycle stage changed during an interrupted scoped
  run. The pre-run snapshot records `toroidal_momentum_flux` as `superseded`.
  It is now `drafted`, carries focus scope
  `25c083b5-d385-4a0e-bfd0-b0da1e1d1e67`, and has no reviewer score. The
  interrupted run is `f1168b3b-f3d9-47b3-809d-99b5dbca7106`. The four
  multiple-target source scalars still select their other, accepted target; the
  four stale sources still select `toroidal_momentum_flux`. Thus these eight
  rows reflect target liveness during an unfinished reset, not eight new
  attachment decisions.
- **Genuine regrowth** requires evidence that a row was absent after a deliberate
  change and later returned. The `StandardNameChange` ledger records explicit
  `detach_stale_source_binding` changes for `derived:electron_diffusivity` and
  `derived:ion_diffusivity` at `2026-08-20T17:01:00.406Z`. Both source nodes
  remain `status='stale'`, yet both now again have a live `PRODUCED_NAME` edge.

Durable inputs:

- Current graph snapshot:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T102848563114-ratchetdiag/live-census.json`
  — SHA-256
  `98725ae9819873d72610c85b92c04cc718907636ec672560d99786c75771a082`.
- Focused live-graph test:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T102848563114-ratchetdiag/focused-ratchets.log`
  — SHA-256
  `afac1b6811b089a7e71c0ace5cd853fd637b63fc6338680aa43b9bddbc1c72bf`.
- Frozen 2026-08-20 baseline:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260820T202243350344-sgwi-integrity-invariant-ratchets/preflight-census.log`
  — SHA-256
  `05290b19c27a2d64a2e40890ce044c86bc249aa66fd2c6b256134836e5abe23c`.
- Pre-reset lifecycle snapshot:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T085507035497-sgwi-orphan-regeneration/pre-run-snapshot.json`
  — SHA-256
  `cc5757aca25f0c664b79e143f9cafd636d22bc15eb61a62dd38e91474b1bf29b`.
- Interrupted-run reconcile log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T085507035497-sgwi-orphan-regeneration/sn-run.log`
  — SHA-256
  `e6690226454c1d24cdbe6da4cb201db3ae4d72ed08337b7244ca8dcc75c13b7c`.

## All 27 multiple-live-target sources

The target list in every row is the live graph evidence returned by
`_MULTIPLE_LIVE_TARGETS_QUERY`. For standing refusals, the evidence column also
records the current source status and scalar mirror; exact membership in the
frozen baseline is independently preserved in the cited census.

| Source | Live targets | Marking | Graph evidence |
|---|---|---|---|
| `dd:core_sources/source/profiles_1d/ion/momentum/radial` | `momentum_source`, `radial_ion_momentum_source` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `radial_ion_momentum_source`. |
| `dd:edge_profiles/ggd/mass_density/values` | `mass_density`, `total_plasma_mass_density` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `attached`, scalar `mass_density`. |
| `dd:edge_profiles/ggd/neutral/velocity/phi` | `toroidal_neutral_velocity`, `toroidal_neutral_momentum_convection_velocity` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `toroidal_neutral_velocity`. |
| `dd:edge_sources/source/ggd/ion/momentum/r` | `radial_ion_momentum`, `radial_ion_momentum_source` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `radial_ion_momentum`. |
| `dd:edge_sources/source/ggd/neutral/momentum/phi` | `toroidal_neutral_momentum_source`, `toroidal_momentum_flux` | transient reconcile artifact | Extra target `toroidal_momentum_flux` is drafted in scope `25c083b5…`; scalar remains `toroidal_neutral_momentum_source`. |
| `dd:edge_transport/model/ggd/neutral/state/particles/d_pol/values` | `poloidal_neutral_state_particle_diffusivity`, `neutral_state_particle_diffusivity` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `poloidal_neutral_state_particle_diffusivity`. |
| `dd:equilibrium/time_slice/profiles_1d/mass_density` | `mass_density`, `total_plasma_mass_density` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `attached`, scalar `mass_density`. |
| `dd:equilibrium/time_slice/profiles_1d/squareness_upper_outer` | `outer_squareness_of_flux_surface`, `squareness_of_flux_surface`, `upper_outer_squareness_of_flux_surface` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `upper_outer_squareness_of_flux_surface`. |
| `dd:gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter_bessel_1/j_parallel` | `parallel_normalized_perturbed_current_density_bessel_1`, `parallel_normalized_perturbed_current_density`, `normalized_perturbed_current_density`, `perturbed_current_density` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `perturbed_current_density`. |
| `dd:langmuir_probes/embedded/surface_area` | `area_of_langmuir_probe`, `wetted_area_of_langmuir_probe` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `area_of_langmuir_probe`. |
| `dd:langmuir_probes/reciprocating/surface_area` | `area_of_langmuir_probe`, `wetted_area_of_langmuir_probe` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `attached`, scalar `area_of_langmuir_probe`. |
| `dd:mhd/ggd/mass_density/values` | `mass_density`, `total_plasma_mass_density` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `attached`, scalar `mass_density`. |
| `dd:mhd_linear/time_slice/toroidal_mode/plasma/phi_potential_perturbed/imaginary` | `perturbed_electrostatic_potential_imaginary_part`, `electrostatic_potential_imaginary_part` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `perturbed_electrostatic_potential_imaginary_part`. |
| `dd:plasma_profiles/ggd/mass_density/values` | `mass_density`, `total_plasma_mass_density` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `mass_density`. |
| `dd:plasma_profiles/ggd/neutral/velocity/phi` | `toroidal_neutral_velocity`, `toroidal_neutral_momentum_convection_velocity` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `attached`, scalar `toroidal_neutral_velocity`. |
| `dd:plasma_sources/source/ggd/ion/momentum/phi` | `toroidal_momentum_flux`, `toroidal_ion_torque_density` | transient reconcile artifact | Extra target `toroidal_momentum_flux` is drafted in scope `25c083b5…`; scalar remains `toroidal_ion_torque_density`. |
| `dd:plasma_sources/source/ggd/ion/momentum/radial` | `radial_ion_momentum`, `radial_ion_momentum_source` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `attached`, scalar `radial_ion_momentum`. |
| `dd:plasma_sources/source/ggd/momentum/phi` | `toroidal_momentum_flux`, `toroidal_torque_density` | transient reconcile artifact | Extra target `toroidal_momentum_flux` is drafted in scope `25c083b5…`; scalar remains `toroidal_torque_density`. |
| `dd:plasma_sources/source/ggd/neutral/momentum/phi` | `toroidal_momentum_flux`, `toroidal_neutral_torque_density` | transient reconcile artifact | Extra target `toroidal_momentum_flux` is drafted in scope `25c083b5…`; scalar remains `toroidal_neutral_torque_density`. |
| `dd:plasma_sources/source/profiles_1d/ion/momentum/radial` | `radial_ion_momentum`, `radial_ion_momentum_source` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `attached`, scalar `radial_ion_momentum`. |
| `dd:plasma_transport/model/ggd/momentum/flux/radial` | `radial_momentum_flux`, `radial_momentum` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `radial_momentum_flux`. |
| `dd:plasma_transport/model/ggd/neutral/energy/v_parallel/values` | `parallel_neutral_species_energy_convection_velocity`, `neutral_species_energy_convection_velocity` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `parallel_neutral_species_energy_convection_velocity`. |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/poloidal` | `poloidal_neutral_state_momentum_flux`, `poloidal_linear_neutral_internal_state_momentum_flux` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `poloidal_neutral_state_momentum_flux`. |
| `dd:plasma_transport/model/ggd/neutral/state/momentum/flux/radial` | `radial_neutral_internal_state_momentum_flux`, `radial_neutral_state_momentum_flux` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `attached`, scalar `radial_neutral_internal_state_momentum_flux`. |
| `dd:plasma_transport/model/profiles_1d/neutral/state/momentum/flux/poloidal` | `poloidal_neutral_state_momentum_flux`, `poloidal_linear_neutral_internal_state_momentum_flux` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `poloidal_neutral_state_momentum_flux`. |
| `dd:plasma_transport/model/profiles_1d/neutral/state/momentum/flux_limiter/z` | `vertical_coordinate_of_active_limiter_point`, `vertical_neutral_state_momentum_flux_limiter_coefficient` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `attached`, scalar `vertical_coordinate_of_active_limiter_point`. |
| `dd:runaway_electrons/global_quantities/volume_average/current_density` | `parallel_runaway_electron_current_density`, `parallel_volume_averaged_runaway_electron_current_density`, `volume_averaged_runaway_electron_current_density` | named standing refusal | Exact source/target-set match to 2026-08-20 census; status `composed`, scalar `volume_averaged_runaway_electron_current_density`. |

Count check: **23 standing + 4 transient + 0 regrowth = 27**.

## All 9 stale sources retaining live bindings

The target list in every row is the live graph evidence returned by
`_STALE_LIVE_BINDINGS_QUERY`.

| Source | Live targets | Marking | Graph evidence |
|---|---|---|---|
| `dd:core_transport/model/profiles_1d/momentum_tor/flux` | `toroidal_momentum_flux` | transient reconcile artifact | Only target is scoped, drafted `toroidal_momentum_flux`; it was superseded before the interrupted reset. |
| `dd:ece/channel/t_e_voltage` | `voltage_of_diagnostic_antenna` | named standing refusal | Exact source/target-set match to 2026-08-20 stale census. |
| `dd:edge_transport/model/ggd/ion/momentum/flux/toroidal` | `toroidal_momentum_flux` | transient reconcile artifact | Only target is scoped, drafted `toroidal_momentum_flux`; it was superseded before the interrupted reset. |
| `dd:equilibrium/time_slice/boundary_separatrix/closest_wall_point/distance` | `gap_at_plasma_boundary` | named standing refusal | Exact source/target-set match to 2026-08-20 stale census. |
| `dd:equilibrium/time_slice/profiles_1d/b_average` | `flux_surface_average_magnetic_field_magnitude` | named standing refusal | Exact source/target-set match to 2026-08-20 stale census. |
| `dd:plasma_transport/model/ggd/momentum/flux/toroidal` | `toroidal_momentum_flux` | transient reconcile artifact | Only target is scoped, drafted `toroidal_momentum_flux`; it was superseded before the interrupted reset. |
| `dd:plasma_transport/model/profiles_1d/momentum_tor/flux` | `toroidal_momentum_flux` | transient reconcile artifact | Only target is scoped, drafted `toroidal_momentum_flux`; it was superseded before the interrupted reset. |
| `derived:electron_diffusivity` | `electron_diffusivity` | genuine regrowth | Ledger detached this edge on 2026-08-20; source is still `stale`, but the live edge exists again. |
| `derived:ion_diffusivity` | `ion_diffusivity` | genuine regrowth | Ledger detached this edge on 2026-08-20; source is still `stale`, but the live edge exists again. |

Count check: **3 standing + 4 transient + 2 regrowth = 9**.

## Changes that introduced the genuine regrowth

Both genuine rows share one introducing mechanism and one runtime event. The
2026-08-20 detach ledger proves the edges were absent by deliberate action. At
`2026-08-21T09:02:24Z`, the interrupted ordinary run logged
`reconcile_orphan_parent_sources: seeded 3 missing parent provenance source(s)`.
That reconciler reuses a pre-existing `derived:<parent>` source with `MERGE`,
sets `status = coalesce(status, 'composed')`, and then merges the
`PRODUCED_NAME` edge. For a pre-existing stale source, `coalesce` preserves
`stale` while the edge is recreated. The behavior originates in commit
`78c3508c` (`reconcile_orphan_parent_sources`) and was exercised by run
`f1168b3b-f3d9-47b3-809d-99b5dbca7106`.

| Regrowth source | Prior removing change | Introducing change |
|---|---|---|
| `derived:electron_diffusivity` | `sn-change:stale-source-detach:89b01809c63d6f1f0edc6ead80c10367d02ad848b394d904b9f25eabefcd2080`; detached from `electron_diffusivity` at `2026-08-20T17:01:00.406Z` | `reconcile_orphan_parent_sources` during run `f1168b3b-f3d9-47b3-809d-99b5dbca7106` at `2026-08-21T09:02:24Z`; reused the stale source and merged the edge |
| `derived:ion_diffusivity` | `sn-change:stale-source-detach:5eb45a560eeb61b4fcee2d71edce42e4f7c1bae8e293caaa4ca7e27373482001`; detached from `ion_diffusivity` at `2026-08-20T17:01:00.406Z` | `reconcile_orphan_parent_sources` during run `f1168b3b-f3d9-47b3-809d-99b5dbca7106` at `2026-08-21T09:02:24Z`; reused the stale source and merged the edge |

This is a lifecycle bug, not a reason to bless six more stale bindings. A
subsequent repair should make structural-source seeding refuse stale source
nodes or explicitly reactivate them under reviewed authority, and it should
prevent a detached stale scalar mirror from silently recreating an edge. That
code and graph repair is outside this read-only node's write scope.

## Verification and non-mutation statement

The focused module measured exactly **27**, **9**, **36**, and **1** rows for
its four ratchets: the first two failed their 23 and 3 ceilings, while the
unsourced-name and explicit-axis ratchets passed. Result: **2 failed, 2 passed,
1 warning in 9.48 seconds**. Graph connection and test setup completed normally;
no slow connect was interpreted as a service fault.

This node issued only read-only graph queries and wrote only this evidence
record plus external run logs. It performed no graph mutation, no Cypher
`SET`/`CREATE`/`MERGE`/`DELETE`, no CLI graph operation, and no pipeline run. It
did not edit either ceiling constant or any test source.
