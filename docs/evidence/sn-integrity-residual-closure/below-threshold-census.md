# Below-threshold Standard Name producing-source census

## Outcome

The live graph contains **76 non-terminal, below-threshold names**: **75 are
`hint-steerable`** because they hold at least one producing source, and
**1 is `blocked-on-source`**. The sole blocked live row is
`poloidal_neutral_internal_state_momentum_convected_velocity`, at score
**0.3625**, with **0** producing sources and **0** refinement attempts.

A further **68 below-threshold names are already exhausted**. They remain in the
complete table under the required `already-exhausted` disposition, but are not
live provenance targets and cannot re-enter generation through
`sn edit --hint`. The table therefore contains **144 rows total**:

| Disposition | Count |
|---|---:|
| `hint-steerable` | **75** |
| `blocked-on-source` | **1** |
| `already-exhausted` | **68** |
| **Total listed** | **144** |

Five listed rows have no producing source: the one live blocked row and four
already-exhausted rows. A zero source count does not override exhaustion; the
terminal lifecycle is the operative disposition.

## Census boundary and classification

The threshold is strict: `reviewer_score_name < 0.85`. The operational
below-threshold cohort excludes `accepted` and `superseded` names. This is
the only boundary consistent with the requested disposition set and with the
actual `sn edit --hint` contract:

1. `name_stage='exhausted'` → `already-exhausted`;
2. otherwise, producing-source count greater than zero →
   `hint-steerable`;
3. otherwise → `blocked-on-source`.

“Producing-source count” is the number of distinct
`StandardNameSource` nodes connected by `PRODUCED_NAME`. This is the exact
precondition used by the name-axis hint planner: it counts that edge without a
source-status filter. Exhausted names are retained in this evidence because the
requested closed disposition set explicitly distinguishes them, while the
repository’s canonical provenance-ledger definition treats them as terminal
rather than live.

The graph also has **14 accepted names** whose stored score is below 0.85.
They are listed separately below, not mixed into the steering census:
ordinary review does not demote an accepted name, and the hint planner explicitly
refuses `name_stage='accepted'`. A source edge therefore does **not** make a
hint available to those accepted rows.

Of the three identities named in the plan hand-off,
`minimum_of_safety_factor` and
`toroidal_line_integrated_impurity_ion_velocity` are now superseded and
therefore outside the live cohort. The third,
`poloidal_neutral_internal_state_momentum_convected_velocity`, is the sole
live `blocked-on-source` row.

## Complete operational cohort

| Standard name | name_stage | Score | Producing sources | refine_attempts | Disposition |
|---|---|---:|---:|---:|---|
| `accumulated_prefill_gas_count` | `reviewed` | 0.81875 | 2 | 0 | `hint-steerable` |
| `angle_of_plasma_boundary_gap` | `exhausted` | 0.5875 | 1 | 0 | `already-exhausted` |
| `angular_width_of_camera` | `reviewed` | 0.65625 | 1 | 0 | `hint-steerable` |
| `beta` | `reviewed` | 0.3 | 1 | 0 | `hint-steerable` |
| `deuterium_tritium_flux` | `exhausted` | 0.8 | 1 | 0 | `already-exhausted` |
| `effective_neutral_internal_state_velocity_due_to_e_cross_b_drift` | `reviewed` | 0.75 | 1 | 0 | `hint-steerable` |
| `electron_convection_velocity` | `exhausted` | 0.7125 | 1 | 0 | `already-exhausted` |
| `electron_density_at_pedestal_top` | `reviewed` | 0.80625 | 3 | 0 | `hint-steerable` |
| `electron_pressure_at_pedestal_top` | `reviewed` | 0.84375 | 3 | 0 | `hint-steerable` |
| `fast_ion_charge_state_torque_due_to_collisions` | `exhausted` | 0.79375 | 1 | 0 | `already-exhausted` |
| `fast_particle_pressure` | `exhausted` | 0.825 | 1 | 0 | `already-exhausted` |
| `field_aligned_surface_tilt_angle_of_langmuir_probe` | `reviewed` | 0.5375 | 2 | 0 | `hint-steerable` |
| `flux_surface_averaged_velocity` | `exhausted` | 0.575 | 1 | 0 | `already-exhausted` |
| `flux_surface_normal_contravariant_flux_surface_averaged_metric` | `exhausted` | 0.75 | 1 | 0 | `already-exhausted` |
| `forward_power_of_lower_hybrid_antenna_row` | `reviewed` | 0.65625 | 1 | 0 | `hint-steerable` |
| `gradient_of_normalized_pressure_at_flux_surface` | `exhausted` | 0.7 | 1 | 0 | `already-exhausted` |
| `heat_power_over_halo_region` | `reviewed` | 0.775 | 1 | 0 | `hint-steerable` |
| `inertial_current_density_due_to_diamagnetic_drift` | `exhausted` | 0.5875 | 1 | 0 | `already-exhausted` |
| `ion_average_temperature_at_plasma_boundary` | `exhausted` | 0.80625 | 1 | 0 | `already-exhausted` |
| `ion_charge_state_energy_diffusion_coefficient_due_to_diffusion` | `reviewed` | 0.75625 | 2 | 0 | `hint-steerable` |
| `ion_charge_state_torque_density` | `exhausted` | 0.5875 | 0 | 3 | `already-exhausted` |
| `ion_charge_state_torque_density_due_to_collisions` | `exhausted` | 0 | 1 | 0 | `already-exhausted` |
| `ion_convection_velocity` | `exhausted` | 0.6875 | 1 | 0 | `already-exhausted` |
| `ion_diamagnetic_momentum_convection_velocity` | `exhausted` | 0.7125 | 1 | 0 | `already-exhausted` |
| `ion_diamagnetic_momentum_damping_rate` | `exhausted` | 0.6125 | 2 | 0 | `already-exhausted` |
| `ion_energy_convection_velocity` | `exhausted` | 0.6125 | 1 | 0 | `already-exhausted` |
| `ion_state_average_charge_number` | `exhausted` | 0.84375 | 1 | 0 | `already-exhausted` |
| `ion_state_energy_diffusivity` | `reviewed` | 0.74375 | 1 | 0 | `hint-steerable` |
| `ion_state_particle_convection_velocity_magnitude` | `reviewed` | 0.71875 | 1 | 0 | `hint-steerable` |
| `ion_velocity` | `reviewed` | 0.63125 | 2 | 0 | `hint-steerable` |
| `magnetic_field_magnitude` | `exhausted` | 0.65 | 1 | 0 | `already-exhausted` |
| `magnetic_shear_at_pedestal` | `reviewed` | 0.65 | 1 | 0 | `hint-steerable` |
| `measured_voltage_of_spectrometer_channel` | `exhausted` | 0.81875 | 2 | 3 | `already-exhausted` |
| `minor_length_of_antenna_strap` | `reviewed` | 0.75625 | 1 | 0 | `hint-steerable` |
| `minor_length_of_conductor_cross_section` | `reviewed` | 0.69375 | 1 | 0 | `hint-steerable` |
| `momentum_coefficient_due_to_diamagnetic_drift` | `exhausted` | 0.5625 | 1 | 0 | `already-exhausted` |
| `momentum_power` | `reviewed` | 0.4875 | 1 | 0 | `hint-steerable` |
| `mutual_inductance` | `exhausted` | 0.68125 | 13 | 0 | `already-exhausted` |
| `neutral_convection_velocity` | `exhausted` | 0.58125 | 1 | 0 | `already-exhausted` |
| `neutral_diamagnetic_momentum_flux` | `reviewed` | 0.75625 | 1 | 0 | `hint-steerable` |
| `neutral_internal_state_diffusivity` | `exhausted` | 0.78125 | 1 | 0 | `already-exhausted` |
| `neutral_internal_state_momentum_flux` | `exhausted` | 0.75 | 1 | 0 | `already-exhausted` |
| `neutral_internal_state_torque_density` | `exhausted` | 0.8 | 1 | 0 | `already-exhausted` |
| `neutral_state_diamagnetic_momentum_diffusivity` | `reviewed` | 0.825 | 2 | 0 | `hint-steerable` |
| `neutral_state_particle_flux` | `reviewed` | 0.84375 | 1 | 0 | `hint-steerable` |
| `neutral_velocity_due_to_diamagnetic_drift_magnitude` | `reviewed` | 0.78125 | 1 | 0 | `hint-steerable` |
| `normal_wave_electric_field` | `reviewed` | 0.7375 | 1 | 0 | `hint-steerable` |
| `normalized_beam_power` | `reviewed` | 0.5875 | 1 | 0 | `hint-steerable` |
| `normalized_neutron_flux` | `exhausted` | 0.575 | 1 | 0 | `already-exhausted` |
| `normalized_perpendicular_gyroaveraged_perturbed_energy` | `exhausted` | 0.75625 | 1 | 0 | `already-exhausted` |
| `normalized_toroidal_flux_coordinate_of_measurement_position` | `reviewed` | 0.83125 | 1 | 0 | `hint-steerable` |
| `normalized_toroidal_hard_xray_peak_lower_bound_width` | `exhausted` | 0.3 | 1 | 0 | `already-exhausted` |
| `parallel_fast_ion_state_pressure` | `reviewed` | 0.74375 | 1 | 0 | `hint-steerable` |
| `parallel_fast_neutral_state_pressure` | `reviewed` | 0.75625 | 1 | 0 | `hint-steerable` |
| `parallel_ion_state_momentum` | `reviewed` | 0.625 | 1 | 0 | `hint-steerable` |
| `parallel_momentum_flux` | `exhausted` | 0.66875 | 1 | 0 | `already-exhausted` |
| `particle_convection_velocity` | `exhausted` | 0.55 | 1 | 0 | `already-exhausted` |
| `peak_incident_heat_flux_at_limiter` | `exhausted` | 0.5625 | 1 | 0 | `already-exhausted` |
| `peak_wave_current_of_antenna_strap_amplitude` | `exhausted` | 0.6875 | 1 | 3 | `already-exhausted` |
| `per_toroidal_mode_flux_surface_average_total_absorbed_power_density` | `exhausted` | 0.625 | 2 | 0 | `already-exhausted` |
| `perturbed_major_radius` | `reviewed` | 0.58125 | 1 | 0 | `hint-steerable` |
| `plasma_breakdown_time` | `exhausted` | 0.65625 | 1 | 0 | `already-exhausted` |
| `plasma_power_at_wall` | `exhausted` | 0.6875 | 1 | 0 | `already-exhausted` |
| `poloidal_convection_velocity` | `exhausted` | 0.65 | 1 | 0 | `already-exhausted` |
| `poloidal_ion_charge_state_energy_velocity_due_to_convection` | `reviewed` | 0.71875 | 1 | 0 | `hint-steerable` |
| `poloidal_ion_energy_convection_velocity` | `reviewed` | 0.7 | 1 | 0 | `hint-steerable` |
| `poloidal_ion_state_momentum_convection_velocity` | `reviewed` | 0.73125 | 1 | 0 | `hint-steerable` |
| `poloidal_momentum_neutral_internal_state_flux_limiter_coefficient` | `reviewed` | 0.575 | 1 | 0 | `hint-steerable` |
| `poloidal_neutral_internal_state_momentum_convected_velocity` | `reviewed` | 0.3625 | 0 | 0 | `blocked-on-source` |
| `poloidal_neutral_state_energy_convection_velocity` | `reviewed` | 0.70625 | 3 | 0 | `hint-steerable` |
| `poloidal_neutral_state_momentum_convection_velocity` | `reviewed` | 0.78125 | 2 | 0 | `hint-steerable` |
| `poloidal_turn_count` | `reviewed` | 0.7875 | 1 | 0 | `hint-steerable` |
| `power_due_to_fusion_reactions` | `exhausted` | 0.8125 | 4 | 0 | `already-exhausted` |
| `power_of_lower_hybrid_antenna` | `exhausted` | 0.6125 | 1 | 1 | `already-exhausted` |
| `radial_coordinate_of_arc_of_circle_center` | `reviewed` | 0.64375 | 1 | 0 | `hint-steerable` |
| `radial_coordinate_of_electron_cyclotron_launcher_mirror` | `reviewed` | 0.7 | 1 | 0 | `hint-steerable` |
| `radial_coordinate_of_flux_surface` | `exhausted` | 0.7 | 1 | 0 | `already-exhausted` |
| `radial_coordinate_of_launching_position` | `exhausted` | 0.7 | 1 | 0 | `already-exhausted` |
| `radial_current_density` | `reviewed` | 0.65625 | 3 | 0 | `hint-steerable` |
| `radial_derivative_of_toroidal_particle_velocity` | `exhausted` | 0.4625 | 1 | 0 | `already-exhausted` |
| `radial_effective_thermal_ion_charge_state_energy_velocity_due_to_convection` | `reviewed` | 0.7125 | 2 | 0 | `hint-steerable` |
| `radial_ion_charge_state_convection_velocity` | `reviewed` | 0.75 | 1 | 0 | `hint-steerable` |
| `radial_ion_momentum` | `exhausted` | 0.575 | 1 | 1 | `already-exhausted` |
| `radial_ion_state_energy_convection_velocity` | `reviewed` | 0.825 | 1 | 0 | `hint-steerable` |
| `radial_ion_state_momentum_convection_velocity` | `reviewed` | 0.78125 | 1 | 0 | `hint-steerable` |
| `radial_ion_state_particle_convection_velocity` | `reviewed` | 0.80625 | 1 | 0 | `hint-steerable` |
| `radial_ion_velocity` | `reviewed` | 0.8375 | 12 | 0 | `hint-steerable` |
| `radial_momentum` | `reviewed` | 0.50625 | 10 | 0 | `hint-steerable` |
| `radial_momentum_diffusion_coefficient` | `reviewed` | 0.7375 | 1 | 0 | `hint-steerable` |
| `radial_momentum_source` | `reviewed` | 0.8 | 1 | 0 | `hint-steerable` |
| `radial_neutral_momentum` | `reviewed` | 0.68125 | 1 | 0 | `hint-steerable` |
| `radial_neutral_state_energy_convection_velocity` | `reviewed` | 0.7875 | 1 | 0 | `hint-steerable` |
| `radial_neutral_state_momentum_convection_velocity` | `exhausted` | 0.65 | 1 | 3 | `already-exhausted` |
| `radial_neutral_state_momentum_diffusion_coefficient` | `reviewed` | 0.74375 | 2 | 0 | `hint-steerable` |
| `radial_outline_of_cryostat` | `reviewed` | 0.84375 | 1 | 0 | `hint-steerable` |
| `radius` | `reviewed` | 0.4875 | 1 | 0 | `hint-steerable` |
| `radius_of_antenna_strap` | `reviewed` | 0.5625 | 2 | 0 | `hint-steerable` |
| `radius_of_aperture` | `reviewed` | 0.6375 | 2 | 0 | `hint-steerable` |
| `radius_of_diagnostic_aperture` | `reviewed` | 0.83125 | 12 | 0 | `hint-steerable` |
| `ratio_of_charge_of_conductor_to_voltage_of_conductor` | `exhausted` | 0.4125 | 0 | 3 | `already-exhausted` |
| `ratio_of_critical_alpha_parameter_to_alpha_parameter_at_pedestal` | `exhausted` | 0.675 | 1 | 0 | `already-exhausted` |
| `ratio_of_deposited_power_at_divertor_target_to_total_incident_power_of_divertor` | `reviewed` | 0.8 | 1 | 0 | `hint-steerable` |
| `ratio_of_ion_velocity_to_magnetic_field` | `reviewed` | 0.625 | 3 | 0 | `hint-steerable` |
| `ratio_of_magnetic_field_to_current_of_poloidal_field_coil` | `exhausted` | 0.675 | 5 | 0 | `already-exhausted` |
| `ratio_of_plasma_vorticity_to_major_radius` | `exhausted` | 0.8125 | 12 | 0 | `already-exhausted` |
| `ratio_of_vorticity_to_major_radius` | `reviewed` | 0.66875 | 11 | 0 | `hint-steerable` |
| `reference_beta` | `exhausted` | 0.3 | 1 | 0 | `already-exhausted` |
| `safety_factor_at_pedestal` | `reviewed` | 0.66875 | 1 | 0 | `hint-steerable` |
| `spectral_calibration_factor_of_spectrometer_channel` | `reviewed` | 0.63125 | 1 | 0 | `hint-steerable` |
| `spun_wavelength_of_fiber_optic_current_sensor` | `exhausted` | 0.7125 | 1 | 0 | `already-exhausted` |
| `tendency_of_diamagnetic_energy` | `reviewed` | 0.83125 | 1 | 0 | `hint-steerable` |
| `thermal_radiative_power_of_divertor_target` | `reviewed` | 0 | 1 | 0 | `hint-steerable` |
| `tilt_angle_of_antenna_strap` | `exhausted` | 0.7375 | 1 | 0 | `already-exhausted` |
| `time_derivative_of_electron_density` | `reviewed` | 0.83125 | 1 | 0 | `hint-steerable` |
| `toroidal_angle` | `reviewed` | 0.8125 | 2 | 0 | `hint-steerable` |
| `toroidal_angle_of_along_pellet_path` | `exhausted` | 0.725 | 0 | 0 | `already-exhausted` |
| `toroidal_angle_of_secondary_x_point` | `exhausted` | 0.825 | 1 | 0 | `already-exhausted` |
| `toroidal_coordinate` | `reviewed` | 0.78125 | 1 | 0 | `hint-steerable` |
| `toroidal_coordinate_of_visible_camera` | `reviewed` | 0.83125 | 1 | 0 | `hint-steerable` |
| `toroidal_current_density_due_to_distribution_function_driven` | `exhausted` | 0.5875 | 2 | 0 | `already-exhausted` |
| `toroidal_ion_velocity_due_to_diamagnetic_drift` | `reviewed` | 0.625 | 1 | 0 | `hint-steerable` |
| `toroidal_lithium_velocity_at_separatrix` | `exhausted` | 0 | 1 | 0 | `already-exhausted` |
| `toroidal_momentum_convection_velocity` | `exhausted` | 0.65 | 8 | 0 | `already-exhausted` |
| `toroidal_momentum_flux` | `exhausted` | 0.75 | 8 | 3 | `already-exhausted` |
| `toroidal_neutral_state_momentum_source` | `exhausted` | 0.8 | 1 | 0 | `already-exhausted` |
| `toroidal_particle_current` | `exhausted` | 0.75625 | 1 | 0 | `already-exhausted` |
| `toroidal_total_plasma_angular_momentum` | `exhausted` | 0.6625 | 3 | 0 | `already-exhausted` |
| `total_diamagnetic_current_density` | `exhausted` | 0.81875 | 1 | 0 | `already-exhausted` |
| `total_ion_energy_diffusion_coefficient` | `exhausted` | 0.75 | 1 | 0 | `already-exhausted` |
| `total_ion_energy_diffusivity` | `exhausted` | 0.81875 | 2 | 0 | `already-exhausted` |
| `total_power_due_to_fusion` | `exhausted` | 0.84375 | 1 | 3 | `already-exhausted` |
| `total_size_of_camera` | `exhausted` | 0.525 | 1 | 0 | `already-exhausted` |
| `trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | `exhausted` | 0.7 | 0 | 3 | `already-exhausted` |
| `turn_count_of_correction_coil` | `reviewed` | 0.84375 | 1 | 0 | `hint-steerable` |
| `vertical_coordinate_of_bolometer` | `reviewed` | 0.84375 | 1 | 0 | `hint-steerable` |
| `vertical_coordinate_of_outlet_due_to_gas_injection` | `reviewed` | 0.6875 | 1 | 0 | `hint-steerable` |
| `vertical_momentum_convection_velocity` | `reviewed` | 0.775 | 4 | 0 | `hint-steerable` |
| `vertical_neutral_state_momentum_flux` | `reviewed` | 0.625 | 3 | 0 | `hint-steerable` |
| `vertical_outline` | `reviewed` | 0.625 | 10 | 0 | `hint-steerable` |
| `vertical_total_force_of_poloidal_field_coil` | `exhausted` | 0.5875 | 1 | 0 | `already-exhausted` |
| `voltage_of_spectrometer_channel` | `reviewed` | 0.61875 | 2 | 0 | `hint-steerable` |
| `voltage_of_wall` | `reviewed` | 0.8375 | 1 | 0 | `hint-steerable` |
| `wave_curvature_of_beam_tracing_beam` | `exhausted` | 0.6875 | 2 | 0 | `already-exhausted` |
| `wave_curvature_of_wave_beam` | `exhausted` | 0.7875 | 1 | 0 | `already-exhausted` |

## Accepted below-threshold boundary rows

These rows are reported to make the denominator explicit. They are not assigned
one of the three operational dispositions because none would be truthful:
`sn edit --hint` refuses an accepted name even when it has producers.

| Standard name | Score | Producing sources | refine_attempts |
|---|---:|---:|---:|
| `absorbed_power_of_plant_system` | 0.8125 | 1 | 0 |
| `atomic_mass_of_wall_material` | 0.775 | 1 | 0 |
| `bulk_plasma_velocity_due_to_diamagnetic_drift` | 0.8125 | 1 | 0 |
| `ion_heating_power` | 0.575 | 1 | 0 |
| `line_integrated_electron_number_density` | 0.8125 | 5 | 0 |
| `motional_stark_photon_radiance_at_spectral_line` | 0.8125 | 1 | 0 |
| `normalized_effective_particle_energy` | 0.7625 | 1 | 0 |
| `parity_of_gyrokinetic_eigenmode` | 0.775 | 1 | 0 |
| `pfirsch_schlueter_current_density_due_to_diamagnetic_drift` | 0.825 | 1 | 0 |
| `plasma_current_due_to_ohmic_induction` | 0.8125 | 1 | 0 |
| `ratio_of_line_averaged_hydrogen_density_to_line_averaged_total_hydrogenic_density` | 0.8125 | 1 | 0 |
| `safety_factor_at_plasma_boundary` | 0.7625 | 2 | 0 |
| `spectral_etendue_of_spectrometer_channel` | 0.6625 | 1 | 0 |
| `target_atomic_fraction_of_neutron_detector_converter` | 0.8 | 1 | 0 |

## Exact nonmutation proof

The census ran as one read-only graph session. It captured both requested global
write counters, executed the cohort and boundary reads, then captured the same
counters again. Both deltas are exactly zero:

| Graph write measure | Before | After | Delta |
|---|---:|---:|---:|
| `StandardNameChange` nodes | **7,873** | **7,873** | **0** |
| `PRODUCED_NAME` relationships | **5,774** | **5,774** | **0** |

Every graph statement used only `MATCH`, `OPTIONAL MATCH`, aggregation and
`RETURN`; no provider call or write API was invoked. The machine-readable
query result is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T204851859123-n-threshcensus/below-threshold-census-query.json`. It records capture time
`2026-08-22T20:52:34.854999+00:00`, the two counter snapshots, all 144 cohort
rows, the 14 accepted boundary rows, and the asserted disposition totals.

Source checkout commit: `328a0816bfbbb3ee48c7fb376c751726eac15d8c`.

