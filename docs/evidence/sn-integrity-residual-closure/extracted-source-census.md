# Extracted-source terminal-target census

## Outcome

**The difference from the previous zero is definitional, not evidence that the
previously measured semantic-source class regrew.** One read-only production
invocation at commit `328a0816bfbbb3ee48c7fb376c751726eac15d8c`, from
2026-08-22 20:51:17.266818 UTC through 20:51:17.490482 UTC, selected every
`StandardNameSource` with at least one `PRODUCED_NAME` target and with every
such target at `name_stage IN ['superseded', 'exhausted']`. It returned exactly
**140 = 91 stale + 46 extracted + 3 failed**. The partition sum is therefore
**140**, exactly equal to the no-live-target total measured by that invocation.

The previous close used this source predicate verbatim:

```cypher
source.status IN ['composed', 'attached']
```

Within that scope it formed `live_targets` with the terminal stages removed
and selected `size(live_targets) = 0`. Re-running that complete selection in
this invocation returned **0**, matching the previous close. All 46 rows below
have `source.status = 'extracted'`, so none satisfies the previous predicate.
Their `PRODUCED_NAME` targets are all terminal history, not current semantic
targets: **19 exhausted target records and 28 superseded target records** (47
target records because one source has two terminal targets). The extracted
subset comprises **7 DD sources and 39 derived sources**.

Accordingly, every row is classified `definitional-difference`. That label is
narrow: it means the row cannot be evidence against the earlier zero because
the earlier invariant deliberately scoped current semantic sources to
`composed` and `attached`, and treats superseded/exhausted targets as dead
history. It does not grant mutation authority or claim that retaining terminal
history is the preferred cleanup shape. Any later terminal-edge hygiene review
is a separate class and must not rewrite this zero into a regression.

## In-invocation nonmutation proof

The same process read both counters before the cohort query and again after
the row retrieval. Neither changed.

| Counter | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` nodes | 7,873 | 7,873 | **0** |
| `PRODUCED_NAME` relationships | 5,774 | 5,774 | **0** |

The machine-readable invocation result, including all 140 rows, the exact
historical selection, timestamps, partition, and counter snapshots, is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T204834042192-n-extcensus/live-census.json`.

## Complete extracted-status partition

`DD path` is the source's semantic DD path. A derived source has no DD backing,
which is stated explicitly rather than fabricating a path. Target entries give
the exact Standard Name identity followed by its live `name_stage` value from
the invocation.

| DD path | Source row | Every `PRODUCED_NAME` target and stage | Classification |
|---|---|---|---|
| core_profiles/profiles_1d/momentum_phi | dd:core_profiles/profiles_1d/momentum_tor | `toroidal_total_plasma_angular_momentum` (`exhausted`) | definitional-difference |
| core_transport/model/profiles_1d/ion/momentum/diamagnetic/flow_damping_rate | dd:core_transport/model/profiles_1d/ion/momentum/diamagnetic/flow_damping_rate | `ion_diamagnetic_momentum_damping_rate` (`exhausted`) | definitional-difference |
| edge_transport/model/ggd/momentum/flux_limiter/diamagnetic | dd:edge_transport/model/ggd/momentum/flux_limiter/diamagnetic | `momentum_coefficient_due_to_diamagnetic_drift` (`exhausted`) | definitional-difference |
| pf_active/coil/element/geometry/oblique/z | dd:pf_active/coil/element/geometry/oblique/z | `vertical_coordinate_of_poloidal_field_coil` (`superseded`) | definitional-difference |
| pf_plasma/element/geometry/oblique/z | dd:pf_plasma/element/geometry/oblique/z | `vertical_coordinate_of_diagnostic_component_centre` (`superseded`) | definitional-difference |
| plasma_transport/model/profiles_1d/ion/momentum/diamagnetic/flow_damping_rate | dd:plasma_transport/model/profiles_1d/ion/momentum/diamagnetic/flow_damping_rate | `ion_diamagnetic_momentum_damping_rate` (`exhausted`) | definitional-difference |
| waves/coherent_wave/profiles_2d/power_density_n_phi | dd:waves/coherent_wave/profiles_2d/power_density_n_tor | `per_toroidal_mode_flux_surface_average_total_absorbed_power_density` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:count_at_detector_pixel | `count_at_detector_pixel` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:density_of_pellet | `density_of_pellet` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:energy_flux_at_first_wall | `energy_flux_at_first_wall` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:fast_neutral_internal_state_pressure | `fast_neutral_internal_state_pressure` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:fast_particle_pressure | `fast_particle_pressure` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:flux_surface_averaged_current_density | `flux_surface_averaged_current_density` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:flux_surface_averaged_metric | `flux_surface_averaged_metric` (`superseded`); `flux_surface_normal_contravariant_flux_surface_averaged_metric` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:ion_charge_state_momentum_diffusion_coefficient | `ion_charge_state_momentum_diffusion_coefficient` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:ion_charge_state_torque_density_due_to_collisions | `ion_charge_state_torque_density_due_to_collisions` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:ion_charge_state_velocity_due_to_diamagnetic_drift | `ion_charge_state_velocity_due_to_diamagnetic_drift` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:ion_energy_convection_velocity | `ion_energy_convection_velocity` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:ion_state_energy_convection_velocity | `ion_state_energy_convection_velocity` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:ion_state_energy_flux | `ion_state_energy_flux` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:ion_state_momentum_flux | `ion_state_momentum_flux` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:ion_state_particle_convection_velocity | `ion_state_particle_convection_velocity` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:launched_power_of_electron_cyclotron_launcher | `launched_power_of_electron_cyclotron_launcher` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:length_of_antenna_strap | `length_of_antenna_strap` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:magnetic_field_magnitude | `magnetic_field_magnitude` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:neutral_internal_state_diffusion_coefficient | `neutral_internal_state_diffusion_coefficient` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:neutral_internal_state_diffusivity | `neutral_internal_state_diffusivity` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:neutral_internal_state_energy_diffusivity | `neutral_internal_state_energy_diffusivity` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:neutral_internal_state_momentum_convection_velocity | `neutral_internal_state_momentum_convection_velocity` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:neutral_internal_state_momentum_diffusivity | `neutral_internal_state_momentum_diffusivity` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:neutral_internal_state_momentum_flux | `neutral_internal_state_momentum_flux` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:neutral_internal_state_momentum_flux_limiter_coefficient | `neutral_internal_state_momentum_flux_limiter_coefficient` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:neutral_state_momentum_flux | `neutral_state_momentum_flux` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:neutral_state_momentum_source | `neutral_state_momentum_source` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:normalized_momentum_flux_due_to_perturbed_parallel_vector_potential | `normalized_momentum_flux_due_to_perturbed_parallel_vector_potential` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:particle_convection_velocity | `particle_convection_velocity` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:particle_source_rate | `particle_source_rate` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:poloidal_angle | `poloidal_angle` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:power_of_lower_hybrid_antenna | `power_of_lower_hybrid_antenna` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:prefill_gas_count | `prefill_gas_count` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:runaway_electron_current | `runaway_electron_current` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:spectral_surface_curvature_of_optical_element | `spectral_surface_curvature_of_optical_element` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:torque_density_due_to_diamagnetic_drift | `torque_density_due_to_diamagnetic_drift` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:total_ion_energy_diffusion_coefficient | `total_ion_energy_diffusion_coefficient` (`exhausted`) | definitional-difference |
| — (derived source; no DD path) | derived:total_plasma_heating_power | `total_plasma_heating_power` (`superseded`) | definitional-difference |
| — (derived source; no DD path) | derived:voltage_of_neutron_detector | `voltage_of_neutron_detector` (`exhausted`) | definitional-difference |

## Classification check

The table contains **46 rows: 46 definitional-difference + 0 damage**. Each
row is outside the previous source-status predicate, and every listed target
is terminal. No row was promoted, detached, rewritten, or otherwise mutated
to reach this conclusion.
