# Reference-edge parity refresh

## Outcome

The `links` scalar and the `REFERENCES` relationship now agree in both
directions. Measured live on the `codex` graph at
`bolt://98dci4-gpu-0002:7687`, before the write at `2026-09-17T15:29:09Z` and
after it at `2026-09-17T15:54:09Z`:

| Measure | Before | After | Verdict |
|---|---:|---:|---|
| `REFERENCES` edges, directed | 6,557 | **6,913** | equal to the resolvable entry count |
| Resolvable `links` entries | 6,913 | 6,913 | unchanged |
| Resolvable entries carrying no edge | **439** | **0** | **pass** |
| Edges carrying no `links` entry | **83** | **0** | **pass** |
| Unresolvable entries | 123 | 123 | recounted, each with a stated reason |
| Malformed entries, that is no `name:` prefix | 0 | 0 | pass |
| Edges with a non-`StandardName` endpoint | 0 | 0 | pass |
| Names with a non-empty `links` scalar | 3,122 | 3,122 | unchanged |
| Total scalar entries | 7,036 | 7,036 | unchanged |

Two bounded, idempotent writes ran at `2026-09-17T15:53:35Z`: **439 edges
materialised**, each with the scalar authoritative for it, and **83 edges
removed, each adjudicated against the committed scalar producer before removal.
No scalar value was written: the scalar census snapshot carries the same SHA-256
before and after
(`52cffd6f17f7f77eca303aae15cb94494d4c095fef3174f9156734d48ded968b`), so the
refresh demonstrably consumed nothing it read.

The derive beat for this pair is **not** performed here. Parity is its
precondition, not its substitute.

![Parity in both directions, before and after the refresh](/imas-codex/figures/scalar-edge-authority/links-parity-refresh.svg)

## The recorded 356 is an arithmetic remainder; the measured gap is 439

The plan's 2026-09-17 figure audit and the followup this node answers both record
**356** resolvable entries carrying no edge. The measured figure is **439**.

The 356 is the inference `6,913 resolvable entries minus 6,557 edges`. That
subtraction is only valid if every edge is backed by a distinct scalar entry, and
83 of the 6,557 are not. Measured per row as a set difference:

- `6,913 resolvable = 6,474 matched + 439 carrying no edge`
- `6,557 edges = 6,474 matched + 83 carrying no entry`

The inference understated the gap by exactly the orphan count: 356 of 6,913, and
439 of 6,913. Both are recorded because the gate this node answers is stated
against the 356, and a reader comparing them is entitled to know which instrument
moved. The replacement instrument is stronger on the point that matters: it
enumerates the set difference per row rather than deriving a cohort from a count
comparison, which is why the four ledger rows below can be attributed
individually.

## The 83 edges: adjudicated, then removed

Every edge carrying no scalar entry was adjudicated before any of them was
deleted, because such an edge is either a scalar that has lost a target or an edge
built from a target the documentation no longer names — and deleting on sight
would discard the only record that the target was once intended, the trap the plan
names for `produced_sn_id`.

The instrument is the committed scalar producer itself:
`_extract_links_from_docs` at `imas_codex/standard_names/graph_ops.py:515`,
re-run over each source's *current* `documentation` and compared both against the
stored scalar and against the orphan entry.

| Verdict | Rows |
|---|---:|
| Stale scalar, the producer still emits the entry | **0** |
| Fossil edge only, the producer no longer names the target | **83** |
| Sources whose recomputed scalar differs from the stored one | **0** of 65 |

That last row is why removal is the durable choice rather than a judgement call.
For all 65 sources the stored scalar already equals what the committed producer
emits from the current documentation, so the scalar is not out of date: the
documentation no longer carries those targets. Restoring them as scalar entries
could not hold, because the scalar is a pure function of the documentation text
and the next regeneration would drop the entry again, while nothing in the
repository recreates a deleted edge. The intent record is kept below as a ledger,
so the rows are removed rather than lost.

### The 83 removed edges

| # | Source standard name | Removed target id | Reason |
|---:|---|---|---|
| 1 | `cold_neutral_temperature` | `hot_neutral_temperature` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 2 | `count_at_detector_pixel` | `extent_of_detector_pixel` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 3 | `count_at_detector_pixel` | `normalized_count_at_detector_pixel` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 4 | `count_at_detector_pixel` | `time_derivative_of_count_at_detector_pixel` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 5 | `counter_passing_thermal_ion_charge_state_power_density_due_to_collisions` | `thermal_ion_charge_state_power_due_to_collisions` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 6 | `diamagnetic_current_density` | `velocity_due_to_e_cross_b_drift` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 7 | `diamagnetic_current_density` | `vertical_diamagnetic_current_density` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 8 | `electron_density_at_pedestal_maximum` | `density_at_pedestal_maximum` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 9 | `electron_density_at_pedestal_maximum` | `derivative_of_electron_density_at_pedestal_maximum_with_respect_to_normalized_poloidal_flux_coordinate` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 10 | `electron_deposited_power` | `electron_absorbed_wave_power` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 11 | `first_local_tangential_coordinate_of_optical_element` | `first_local_tangential_coordinate_of_aperture` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 12 | `first_local_tangential_coordinate_of_optical_element` | `second_local_tangential_coordinate_of_aperture` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 13 | `first_local_tangential_width_of_neutron_detector` | `second_local_tangential_width_of_neutron_detector` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 14 | `flux_due_to_eddy_current` | `energy_flux_at_wall_due_to_surface_emission` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 15 | `flux_due_to_eddy_current` | `energy_flux_due_to_eddy_current` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 16 | `flux_surface_averaged_time_averaged_normalized_perturbed_parallel_magnetic_field_magnitude` | `normalized_perturbed_parallel_magnetic_field_magnitude` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 17 | `focal_length_of_line_of_sight` | `diameter_of_line_of_sight` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 18 | `hot_neutral_velocity` | `hot_neutral_temperature` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 19 | `incident_neutral_particle_flux_at_wall` | `neutral_particle_flux_at_wall_due_to_pumping` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 20 | `ion_charge_state_diamagnetic_momentum_convection_velocity` | `ion_velocity_due_to_diamagnetic_drift` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 21 | `ion_pressure` | `total_ion_pressure` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 22 | `ion_state_momentum_diffusivity` | `ion_charge_state_diffusion_coefficient` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 23 | `maximum_gas_flow` | `gas_flow` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 24 | `minimum_gas_flow` | `gas_flow` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 25 | `neutral_temperature` | `hot_neutral_temperature` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 26 | `neutral_velocity_due_to_diamagnetic_drift` | `neutral_velocity_magnitude_due_to_diamagnetic_drift` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 27 | `normalized_perturbed_pressure` | `parallel_normalized_particle_perturbed_pressure` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 28 | `parallel_conductivity` | `parallel_electric_field` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 29 | `parallel_electric_field` | `vertical_electric_field` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 30 | `parallel_fast_neutral_internal_state_pressure` | `fast_neutral_pressure` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 31 | `parallel_ion_charge_state_momentum_flux_limiter_coefficient` | `poloidal_ion_charge_state_momentum_flux_limiter_coefficient` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 32 | `parallel_ion_momentum_flux_limiter_coefficient` | `radial_ion_momentum_flux_limiter_coefficient` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 33 | `parallel_neutral_state_momentum_diffusion_coefficient` | `parallel_neutral_internal_state_diffusivity` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 34 | `perpendicular_normalized_perturbed_pressure` | `normalized_perturbed_pressure` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 35 | `plasma_pressure_imaginary_part` | `plasma_pressure` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 36 | `plasma_temperature` | `plasma_pressure` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 37 | `poloidal_angle_of_electron_cyclotron_beam` | `poloidal_angle_of_beam_tracing_beam` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 38 | `poloidal_angle_of_electron_cyclotron_beam` | `poloidal_angle_of_flux_surface` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 39 | `poloidal_angle_of_electron_cyclotron_beam` | `vertical_wave_vector_of_beam_tracing_beam` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 40 | `poloidal_current_weighted_average_external_magnetic_flux` | `external_magnetic_flux` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 41 | `poloidal_ion_momentum_flux` | `poloidal_momentum_flux` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 42 | `poloidal_ion_state_momentum_flux_limiter_coefficient` | `poloidal_ion_charge_state_momentum_flux_limiter_coefficient` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 43 | `poloidal_ion_velocity` | `poloidal_electron_velocity` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 44 | `poloidal_ion_velocity` | `poloidal_ion_charge_state_velocity` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 45 | `poloidal_magnetic_flux_at_constraint_position` | `poloidal_magnetic_flux_at_flux_surface` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 46 | `poloidal_momentum_flux` | `poloidal_ion_momentum_flux` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 47 | `radial_coordinate_of_antenna_strap` | `toroidal_angle_of_antenna_strap` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 48 | `radial_coordinate_of_antenna_strap` | `vertical_coordinate_of_antenna_strap` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 49 | `radial_coordinate_of_coil_conductor_element` | `radial_coordinate_of_conductor_cross_section` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 50 | `radial_coordinate_of_shunt` | `vertical_coordinate_of_shunt` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 51 | `radial_current_density_due_to_anomalous_transport` | `radial_current_density` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 52 | `radial_current_density_due_to_parallel_viscosity` | `parallel_current_density_due_to_parallel_viscosity` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 53 | `radial_current_density_due_to_parallel_viscosity` | `vertical_current_density_due_to_parallel_viscosity` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 54 | `radial_ion_charge_state_momentum_diffusivity` | `radial_ion_charge_state_momentum_diffusion_coefficient` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 55 | `radial_ion_charge_state_momentum_diffusivity` | `radial_ion_charge_state_momentum_flux` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 56 | `radial_ion_charge_state_momentum_flux` | `radial_ion_charge_state_particle_flux` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 57 | `radial_ion_particle_diffusivity` | `radial_ion_charge_state_diffusivity` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 58 | `radial_ion_particle_diffusivity` | `radial_ion_diffusion_coefficient` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 59 | `radial_magnetic_field` | `vertical_magnetic_field` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 60 | `radial_neutral_state_momentum_flux` | `vertical_neutral_state_momentum_flux` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 61 | `radial_neutral_state_momentum_flux_limiter_coefficient` | `parallel_neutral_state_momentum_flux_limiter_coefficient` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 62 | `radial_neutral_velocity` | `radial_neutral_internal_state_velocity` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 63 | `radial_neutral_velocity` | `radial_neutral_momentum` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 64 | `radial_outline_of_passive_loop_element` | `radial_coordinate_of_passive_loop_element` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 65 | `radial_outline_of_passive_loop_element` | `vertical_outline` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 66 | `radial_vector_potential` | `radial_magnetic_field` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 67 | `radius_of_neutron_detector` | `first_local_tangential_width_of_diagnostic_aperture` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 68 | `radius_of_neutron_detector` | `height_of_neutron_detector` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 69 | `radius_of_neutron_detector` | `radius_of_diagnostic_aperture` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 70 | `radius_of_poloidal_field_coil` | `radius_of_iron_core_segment` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 71 | `thermal_plasma_pressure` | `plasma_pressure` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 72 | `toroidal_coordinate_of_optical_element` | `toroidal_coordinate_of_aperture` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 73 | `toroidal_current_due_to_wave_driven_current_drive` | `poloidal_magnetic_flux` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 74 | `toroidal_ion_charge_state_momentum_flux` | `radial_ion_charge_state_momentum_flux` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 75 | `total_plasma_pressure` | `plasma_pressure` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 76 | `total_power_due_to_ion_cyclotron_heating` | `launched_power_due_to_ion_cyclotron_heating` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 77 | `vertical_coordinate_of_antenna_strap` | `toroidal_angle_of_antenna_strap` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 78 | `vertical_coordinate_of_cryostat` | `vertical_outline_of_cryostat` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 79 | `vertical_coordinate_of_ion_cyclotron_heating_antenna` | `radial_outline_of_antenna_strap` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 80 | `vertical_current_density_due_to_anomalous_transport` | `toroidal_current_density_due_to_anomalous_transport` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 81 | `vertical_current_density_due_to_anomalous_transport` | `vertical_diamagnetic_current_density` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 82 | `vertical_magnetic_field` | `toroidal_magnetic_field` | the committed scalar producer no longer emits the entry for the source's current documentation |
| 83 | `volume_averaged_neutral_temperature` | `hot_neutral_temperature` | the committed scalar producer no longer emits the entry for the source's current documentation |

## Unresolvable entries: recounted, each with a stated reason

The census found **123** entries naming a target id with no `StandardName` node,
against the 123 then recorded. The cohort is **identical before and after** the
refresh: no row appeared, no row disappeared, and no row's stated reason changed.
Each row was re-verified against the graph independently of the census
instrument, so the count is not the same instrument agreeing with itself.

| Check | Result |
|---|---:|
| Unresolvable entries before | 123 |
| Unresolvable entries after | 123 |
| Rows present in both censuses | 123 |
| Rows appearing, and rows disappearing | 0, 0 |
| Target ids matching a node under any label | 0 |
| Rows whose stated reason fails | 0 |

A zero from a capped probe is not an empty world, so the target check asked
`MATCH (n) WHERE n.id = tid` across every label rather than only
`StandardName`, and the entry-still-present half was read from the live scalar
rather than inferred. Every row below carries the same reason, and that reason was
confirmed per row rather than assumed from the cohort.

### The 123 unresolvable entries

| # | Source standard name | Absent target id | Reason |
|---:|---|---|---|
| 1 | `argon_density_at_plasma_boundary` | `argon_density_at_pedestal` | no node under any label has the target id; the entry is still present in the source's scalar |
| 2 | `bulk_plasma_velocity_due_to_diamagnetic_drift_magnitude` | `bulk_plasma_velocity_due_to_diamagnetic_drift` | no node under any label has the target id; the entry is still present in the source's scalar |
| 3 | `coolant_temperature` | `temperature_at_inlet` | no node under any label has the target id; the entry is still present in the source's scalar |
| 4 | `coolant_temperature` | `temperature_at_outlet` | no node under any label has the target id; the entry is still present in the source's scalar |
| 5 | `critical_momentum_due_to_avalanche` | `critical_electric_field` | no node under any label has the target id; the entry is still present in the source's scalar |
| 6 | `current_due_to_ohmic_induction` | `plasma_current_due_to_ohmic_induction` | no node under any label has the target id; the entry is still present in the source's scalar |
| 7 | `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | `neutron_flux_due_to_beam_beam_fusion` | no node under any label has the target id; the entry is still present in the source's scalar |
| 8 | `deuterium_tritium_neutron_flux_due_to_beam_beam_fusion` | `neutron_flux_due_to_beam_beam_fusion` | no node under any label has the target id; the entry is still present in the source's scalar |
| 9 | `effective_charge_at_plasma_boundary` | `charge_at_plasma_boundary` | no node under any label has the target id; the entry is still present in the source's scalar |
| 10 | `efficiency_of_spectrometer_channel` | `spectral_etendue_of_spectrometer_channel` | no node under any label has the target id; the entry is still present in the source's scalar |
| 11 | `electron_energy_flux_at_wall` | `energy_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 12 | `electron_particle_flux_at_wall` | `particle_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 13 | `electron_temperature_at_midplane` | `temperature_at_midplane` | no node under any label has the target id; the entry is still present in the source's scalar |
| 14 | `fast_electron_source_rate_due_to_hot_tail` | `electron_source_rate_due_to_hot_tail` | no node under any label has the target id; the entry is still present in the source's scalar |
| 15 | `fast_neutral_internal_state_number_density` | `neutral_internal_state_number_density` | no node under any label has the target id; the entry is still present in the source's scalar |
| 16 | `flux_due_to_beam_beam_fusion` | `neutron_flux_due_to_beam_beam_fusion` | no node under any label has the target id; the entry is still present in the source's scalar |
| 17 | `flux_due_to_recombination` | `particle_flux_at_wall_due_to_recombination` | no node under any label has the target id; the entry is still present in the source's scalar |
| 18 | `flux_surface_averaged_carbon_density` | `ratio_of_carbon_density_to_electron_density` | no node under any label has the target id; the entry is still present in the source's scalar |
| 19 | `flux_surface_averaged_deuterium_tritium_density` | `flux_surface_averaged_total_ion_density` | no node under any label has the target id; the entry is still present in the source's scalar |
| 20 | `fraction_of_neutron_detector_converter` | `atomic_fraction_of_neutron_detector_converter` | no node under any label has the target id; the entry is still present in the source's scalar |
| 21 | `gradient_of_radial_electron_density` | `radial_electron_density` | no node under any label has the target id; the entry is still present in the source's scalar |
| 22 | `ion_average_temperature_at_magnetic_axis` | `average_temperature_at_magnetic_axis` | no node under any label has the target id; the entry is still present in the source's scalar |
| 23 | `ion_particle_flux_at_wall` | `particle_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 24 | `ion_state_kinetic_energy_flux_at_wall_due_to_surface_emission` | `energy_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 25 | `ion_state_momentum_convection_velocity` | `radial_plasma_effective_momentum_convection_velocity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 26 | `ion_state_particle_flux` | `ion_state_energy_flux` | no node under any label has the target id; the entry is still present in the source's scalar |
| 27 | `ion_state_particle_flux_at_wall` | `particle_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 28 | `ion_state_temperature` | `edge_ion_average_temperature` | no node under any label has the target id; the entry is still present in the source's scalar |
| 29 | `ion_temperature` | `ion_temperature_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 30 | `ion_temperature_at_midplane` | `temperature_at_midplane` | no node under any label has the target id; the entry is still present in the source's scalar |
| 31 | `maximum_of_energy_flux_at_first_wall` | `energy_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 32 | `maximum_of_energy_flux_at_limiter` | `energy_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 33 | `maximum_power_at_inner_divertor_target` | `power_at_inner_divertor_target` | no node under any label has the target id; the entry is still present in the source's scalar |
| 34 | `maximum_power_at_outer_divertor_target` | `power_at_outer_divertor_target` | no node under any label has the target id; the entry is still present in the source's scalar |
| 35 | `neon_prefill_count` | `xenon_prefill_count` | no node under any label has the target id; the entry is still present in the source's scalar |
| 36 | `net_absorbed_power_of_plant_system` | `absorbed_power_of_plant_system` | no node under any label has the target id; the entry is still present in the source's scalar |
| 37 | `neutral_energy_flux_at_wall` | `energy_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 38 | `neutral_internal_state_momentum_flux_limiter_coefficient_over_edge_region` | `internal_state_momentum_flux_limiter_coefficient_over_edge_region` | no node under any label has the target id; the entry is still present in the source's scalar |
| 39 | `neutral_particle_flux_at_wall` | `particle_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 40 | `neutral_power_at_wall_due_to_recombination` | `power_at_wall_due_to_recombination` | no node under any label has the target id; the entry is still present in the source's scalar |
| 41 | `neutral_state_energy_convection_velocity` | `energy_convection_velocity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 42 | `neutral_state_energy_flux` | `ion_state_energy_flux` | no node under any label has the target id; the entry is still present in the source's scalar |
| 43 | `neutral_state_energy_flux_at_wall` | `energy_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 44 | `neutral_state_energy_flux_due_to_recombination` | `energy_flux_due_to_recombination` | no node under any label has the target id; the entry is still present in the source's scalar |
| 45 | `neutral_state_particle_flux_at_wall` | `particle_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 46 | `neutron_flux` | `deuterium_deuterium_neutron_flux` | no node under any label has the target id; the entry is still present in the source's scalar |
| 47 | `neutron_flux` | `tritium_tritium_neutron_flux` | no node under any label has the target id; the entry is still present in the source's scalar |
| 48 | `neutron_rate_of_neutron_detector` | `rate_of_neutron_detector` | no node under any label has the target id; the entry is still present in the source's scalar |
| 49 | `neutron_source_rate_due_to_beam_beam_fusion` | `neutron_flux_due_to_beam_beam_fusion` | no node under any label has the target id; the entry is still present in the source's scalar |
| 50 | `normalized_atomic_count_of_pellet` | `atomic_count_of_pellet` | no node under any label has the target id; the entry is still present in the source's scalar |
| 51 | `normalized_atomic_count_of_pellet` | `count_of_pellet` | no node under any label has the target id; the entry is still present in the source's scalar |
| 52 | `normalized_perturbed_density_imaginary_part` | `normalized_perturbed_density` | no node under any label has the target id; the entry is still present in the source's scalar |
| 53 | `normalized_total_particle_perturbed_pressure_of_gyrokinetic_eigenmode` | `normalized_gyrocenter_perturbed_pressure` | no node under any label has the target id; the entry is still present in the source's scalar |
| 54 | `nuclear_power_density_of_breeder_blanket_module` | `power_density_of_breeder_blanket_module` | no node under any label has the target id; the entry is still present in the source's scalar |
| 55 | `outer_atomic_count_of_pellet` | `atomic_count_of_pellet` | no node under any label has the target id; the entry is still present in the source's scalar |
| 56 | `oxygen_density_at_limiter` | `ion_density_at_limiter` | no node under any label has the target id; the entry is still present in the source's scalar |
| 57 | `parallel_effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | `effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | no node under any label has the target id; the entry is still present in the source's scalar |
| 58 | `parallel_momentum_flux_due_to_perturbed_parallel_vector_potential` | `parallel_momentum_flux_due_to_perturbed_parallel_magnetic_field` | no node under any label has the target id; the entry is still present in the source's scalar |
| 59 | `parallel_momentum_flux_due_to_perturbed_parallel_vector_potential` | `perpendicular_momentum_flux_due_to_perturbed_parallel_vector_potential` | no node under any label has the target id; the entry is still present in the source's scalar |
| 60 | `parallel_normalized_perturbed_vector_potential_amplitude` | `parallel_normalized_perturbed_vector_potential` | no node under any label has the target id; the entry is still present in the source's scalar |
| 61 | `peak_voltage_of_neutron_detector` | `requested_upper_voltage_of_neutron_detector` | no node under any label has the target id; the entry is still present in the source's scalar |
| 62 | `peak_voltage_of_neutron_detector` | `voltage_of_neutron_detector` | no node under any label has the target id; the entry is still present in the source's scalar |
| 63 | `perpendicular_momentum_flux_due_to_perturbed_parallel_magnetic_field` | `parallel_momentum_flux_due_to_perturbed_parallel_magnetic_field` | no node under any label has the target id; the entry is still present in the source's scalar |
| 64 | `perpendicular_normalized_gyrocenter_perturbed_pressure` | `normalized_gyrocenter_perturbed_pressure` | no node under any label has the target id; the entry is still present in the source's scalar |
| 65 | `perturbed_vector_potential` | `normalized_perturbed_vector_potential` | no node under any label has the target id; the entry is still present in the source's scalar |
| 66 | `perturbed_vector_potential` | `parallel_normalized_perturbed_vector_potential` | no node under any label has the target id; the entry is still present in the source's scalar |
| 67 | `plasma_velocity_due_to_diamagnetic_drift` | `bulk_plasma_velocity_due_to_diamagnetic_drift` | no node under any label has the target id; the entry is still present in the source's scalar |
| 68 | `poloidal_current_density_due_to_collisions` | `current_density_due_to_collisions` | no node under any label has the target id; the entry is still present in the source's scalar |
| 69 | `poloidal_current_density_due_to_viscosity` | `current_density_due_to_viscosity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 70 | `poloidal_ion_momentum` | `poloidal_momentum` | no node under any label has the target id; the entry is still present in the source's scalar |
| 71 | `poloidal_ion_state_energy_diffusivity` | `poloidal_ion_state_energy_diffusion_coefficient` | no node under any label has the target id; the entry is still present in the source's scalar |
| 72 | `poloidal_ion_state_momentum` | `poloidal_momentum` | no node under any label has the target id; the entry is still present in the source's scalar |
| 73 | `poloidal_ion_state_momentum_flux` | `ion_state_momentum_flux` | no node under any label has the target id; the entry is still present in the source's scalar |
| 74 | `poloidal_ion_state_velocity_due_to_e_cross_b_drift` | `parallel_ion_state_velocity_due_to_e_cross_b_drift` | no node under any label has the target id; the entry is still present in the source's scalar |
| 75 | `poloidal_neutral_state_momentum` | `poloidal_momentum` | no node under any label has the target id; the entry is still present in the source's scalar |
| 76 | `power_at_wall_due_to_conduction` | `energy_flux_at_wall` | no node under any label has the target id; the entry is still present in the source's scalar |
| 77 | `power_of_divertor_due_to_fusion` | `power_due_to_fusion` | no node under any label has the target id; the entry is still present in the source's scalar |
| 78 | `power_of_divertor_due_to_radiation` | `power_due_to_radiation` | no node under any label has the target id; the entry is still present in the source's scalar |
| 79 | `power_of_neutral_beam_injector` | `absorbed_power_of_neutral_beam_injector` | no node under any label has the target id; the entry is still present in the source's scalar |
| 80 | `radial_centroid_of_electron_cyclotron_launcher_mirror` | `angle_of_electron_cyclotron_launcher_mirror` | no node under any label has the target id; the entry is still present in the source's scalar |
| 81 | `radial_current_density_due_to_viscosity` | `current_density_due_to_viscosity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 82 | `radial_derivative_of_poloidal_ion_state_velocity` | `second_radial_derivative_of_poloidal_ion_state_velocity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 83 | `radial_effective_total_ion_energy_convection_velocity` | `radial_ion_momentum_effective_convection_velocity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 84 | `radial_energy_convection_velocity` | `energy_convection_velocity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 85 | `radial_ion_state_momentum_flux` | `ion_state_momentum_flux` | no node under any label has the target id; the entry is still present in the source's scalar |
| 86 | `radial_neutral_state_energy_diffusion_coefficient` | `radial_ion_state_energy_diffusion_coefficient` | no node under any label has the target id; the entry is still present in the source's scalar |
| 87 | `radial_neutral_state_momentum_convection_velocity` | `radial_neutral_species_momentum_convection_velocity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 88 | `radial_plasma_momentum_source` | `plasma_momentum_source` | no node under any label has the target id; the entry is still present in the source's scalar |
| 89 | `radiance_at_spectral_line` | `motional_stark_radiance_at_spectral_line` | no node under any label has the target id; the entry is still present in the source's scalar |
| 90 | `ratio_of_coolant_mass_to_time` | `coolant_mass` | no node under any label has the target id; the entry is still present in the source's scalar |
| 91 | `reference_calibration_wavelength_of_spectrometer_channel` | `wavelength_of_spectrometer_channel` | no node under any label has the target id; the entry is still present in the source's scalar |
| 92 | `second_radial_derivative_of_poloidal_ion_velocity` | `second_radial_derivative_of_poloidal_ion_state_velocity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 93 | `second_radial_derivative_of_toroidal_ion_state_velocity` | `toroidal_ion_state_velocity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 94 | `source_rate_due_to_thermal_fusion` | `neutron_power` | no node under any label has the target id; the entry is still present in the source's scalar |
| 95 | `thermal_electron_power` | `absorbed_power` | no node under any label has the target id; the entry is still present in the source's scalar |
| 96 | `time_derivative_of_electron_temperature` | `time_derivative_of_ion_state_temperature` | no node under any label has the target id; the entry is still present in the source's scalar |
| 97 | `time_derivative_of_electron_temperature` | `time_derivative_of_ion_temperature` | no node under any label has the target id; the entry is still present in the source's scalar |
| 98 | `time_derivative_of_total_electron_density` | `time_derivative_of_total_ion_density` | no node under any label has the target id; the entry is still present in the source's scalar |
| 99 | `time_derivative_of_total_ion_state_density` | `time_derivative_of_fast_ion_state_density` | no node under any label has the target id; the entry is still present in the source's scalar |
| 100 | `time_derivative_of_total_ion_state_density` | `time_derivative_of_total_ion_density` | no node under any label has the target id; the entry is still present in the source's scalar |
| 101 | `toroidal_beryllium_velocity_at_plasma_boundary` | `toroidal_beryllium_velocity_at_pedestal` | no node under any label has the target id; the entry is still present in the source's scalar |
| 102 | `toroidal_co_passing_thermal_ion_state_torque_density_due_to_collisions` | `co_passing_thermal_ion_state_torque_density_due_to_collisions` | no node under any label has the target id; the entry is still present in the source's scalar |
| 103 | `toroidal_ion_momentum` | `ion_momentum` | no node under any label has the target id; the entry is still present in the source's scalar |
| 104 | `toroidal_ion_state_momentum_flux_limiter_coefficient` | `poloidal_ion_state_momentum_coefficient` | no node under any label has the target id; the entry is still present in the source's scalar |
| 105 | `toroidal_ion_state_momentum_flux_limiter_coefficient` | `toroidal_neutral_momentum_coefficient` | no node under any label has the target id; the entry is still present in the source's scalar |
| 106 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `neutral_state_momentum_coefficient` | no node under any label has the target id; the entry is still present in the source's scalar |
| 107 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `parallel_neutral_state_momentum_coefficient` | no node under any label has the target id; the entry is still present in the source's scalar |
| 108 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `poloidal_neutral_state_momentum_coefficient` | no node under any label has the target id; the entry is still present in the source's scalar |
| 109 | `toroidal_neutral_state_momentum_flux_limiter_coefficient` | `toroidal_neutral_momentum_coefficient` | no node under any label has the target id; the entry is still present in the source's scalar |
| 110 | `toroidal_neutral_state_velocity_due_to_diamagnetic_drift` | `effective_neutral_internal_state_velocity_due_to_diamagnetic_drift` | no node under any label has the target id; the entry is still present in the source's scalar |
| 111 | `toroidal_trapped_thermal_ion_state_torque_density_due_to_collisions` | `toroidal_trapped_fast_ion_state_torque_density_due_to_collisions` | no node under any label has the target id; the entry is still present in the source's scalar |
| 112 | `total_plasma_energy` | `plasma_energy` | no node under any label has the target id; the entry is still present in the source's scalar |
| 113 | `total_power_of_neutral_beam_injector` | `absorbed_power_of_neutral_beam_injector` | no node under any label has the target id; the entry is still present in the source's scalar |
| 114 | `total_power_of_plant_system` | `absorbed_power_of_plant_system` | no node under any label has the target id; the entry is still present in the source's scalar |
| 115 | `tritium_tritium_neutron_source_rate_due_to_thermal_fusion` | `deuterium_deuterium_neutron_source_rate_due_to_thermal_fusion` | no node under any label has the target id; the entry is still present in the source's scalar |
| 116 | `velocity_due_to_diamagnetic_drift` | `bulk_plasma_velocity_due_to_diamagnetic_drift` | no node under any label has the target id; the entry is still present in the source's scalar |
| 117 | `vertical_coordinate_of_divertor_target` | `vertical_coordinate_of_inner_divertor_target` | no node under any label has the target id; the entry is still present in the source's scalar |
| 118 | `vertical_coordinate_of_divertor_target` | `vertical_coordinate_of_outer_divertor_target` | no node under any label has the target id; the entry is still present in the source's scalar |
| 119 | `vertical_ion_state_momentum_flux` | `ion_state_momentum_flux` | no node under any label has the target id; the entry is still present in the source's scalar |
| 120 | `vertical_neutral_state_momentum_convection_velocity` | `vertical_neutral_momentum_convection_velocity` | no node under any label has the target id; the entry is still present in the source's scalar |
| 121 | `x1_coordinate_of_electron_cyclotron_launcher_mirror` | `angle_of_electron_cyclotron_launcher_mirror` | no node under any label has the target id; the entry is still present in the source's scalar |
| 122 | `x2_coordinate_of_electron_cyclotron_launcher_mirror` | `angle_of_electron_cyclotron_launcher_mirror` | no node under any label has the target id; the entry is still present in the source's scalar |
| 123 | `xenon_density_at_internal_transport_barrier` | `ion_density_at_internal_transport_barrier` | no node under any label has the target id; the entry is still present in the source's scalar |

## The 123 against the materialisation ledger of 127

The materialisation node carried 127 unresolvable entries. Four moved and none
were added, so the population reads 123 now. Each movement is itemised below,
because a dropped entry and a target that appeared have opposite effects on
parity and a bare count cannot tell them apart.

The four movers account for the whole of the difference: 127 prior rows, 4 moved,
123 still unresolvable. That final figure is the same 123 rows the census found,
so the two instruments reproduce each other rather than merely agreeing on a
count.

### The four rows that moved

| Source standard name | Target id | Movement |
|---|---|---|
| `electron_pressure_at_pedestal_top` | `electron_density_at_pedestal_top` | target node created since the reading: the entry now resolves |
| `energy_density` | `plasma_energy` | entry no longer in the scalar |
| `normalized_perturbed_pressure` | `perturbed_pressure` | entry no longer in the scalar |
| `perpendicular_momentum_flux_due_to_perturbed_parallel_magnetic_field` | `momentum_flux_due_to_perturbed_parallel_magnetic_field` | target node created since the reading: the entry now resolves |

Two of the four rows now resolve and already carry an edge: the target node was
created after the earlier reading while the entry stayed in the scalar, so both
are among the 439 edges this refresh materialised. The other two have left the
scalar entirely and no target node exists, so both sides are silent and parity is
unaffected either way. A dropped entry needs no edge, which is why the direction
of each movement is carried here rather than summarised as a net figure.

## What the instrument is

The same script ran before and after against the same graph: it enumerates the
entry set and the edge set and reports the set difference per direction per row.

Controls, non-zero, and deliberately unsatisfiable, in both censuses:

| Control | Before | After | Why it is here |
|---|---:|---:|---|
| `PRODUCED_NAME` edges | 5,493 | 5,493 | an independent populated relationship |
| `HAS_REVIEW` edges | 28,880 | 28,880 | unrelated to this pair, and populated |
| `StandardName` nodes | 5,130 | 5,130 | the population under test |
| `size(n.links) < 0` | 0 | 0 | unsatisfiable against the same cohort, so the filters are not matching everything |

A zero is a measurement only when the same instrument is shown to count
something present, which is why the three populated relationships above are
reported beside it. The unresolvable audit was also run against two different
readings rather than one: the first invocation compared the pre-census with
itself, which would have made its `identical sets` result vacuous, so it was
re-run against the post-census and returned the same 123 rows.

## Measurement boundary

The censuses, the adjudication and the write ran on the login node, which is the
only placement that can reach the live graph: the URI resolves through a
login-local tunnel that a compute allocation cannot establish. Every query began
from `StandardName`, bounded its work to the populated `links` cohort, and
returned aggregates or complete row sets; the slowest query was 0.808 s. The
write ran in two batches, one merging the 439 and one deleting the 83, each
capped at 500 pairs.

The scalar was captured as an ordered `id` plus `links` snapshot at both
censuses, and the two digests are equal. That equality is the negative half of the
control: a pass that had read the scalar and blanked it would produce a plausible
edge count and every other figure here would look the same.

This is the gate for this node only. No product test suite was run here; merged
verification belongs to the separately dispatched test node. A failure found
outside this node's declared scope is reported under follow-ons rather than fixed
here.