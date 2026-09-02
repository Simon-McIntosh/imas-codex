# Stored Standard Name kind census

Read-only live-graph census taken 2026-09-02 from the active graph. The scan targeted exactly `StandardName` nodes with `name_stage = 'accepted'`, projected `id` and stored `kind`, and compared each row with `derive_kind(id)` in the current checkout. No graph write was issued.

Positive control: 2335 accepted candidates; 2335 have `id`; 2335 have stored `kind`. Mismatches: 175 (3 stored-to-derived pairs).

## Kind-pair summary

| Stored | Derived | Count |
|---|---|---:|
| `scalar` | `vector` | 38 |
| `tensor` | `scalar` | 2 |
| `vector` | `scalar` | 135 |

## Triggering WEST identities and source bindings

- `flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude`: stored `tensor`, derived `scalar`; source paths: `equilibrium/time_slice/profiles_1d/gm3`.
- `plasma_electrical_conductivity`: stored `tensor`, derived `scalar`; source paths: `conductivity`.

## Complete mismatch census

### scalar -> vector (38)

- `breakdown_magnetic_field`
- `cold_neutral_velocity`
- `current_density_due_to_wave_driven_current_drive`
- `diamagnetic_current_density_due_to_heat_viscosity`
- `diamagnetic_current_density_due_to_ion_neutral_friction`
- `diamagnetic_current_density_due_to_parallel_viscosity`
- `diamagnetic_current_density_due_to_perpendicular_viscosity`
- `diamagnetic_momentum_convection_velocity`
- `electron_energy_convection_velocity`
- `energy_velocity_due_to_convection`
- `fast_current_density`
- `fluctuating_saturated_ion_current_density`
- `force_of_poloidal_field_coil`
- `heat_convection_velocity`
- `hot_neutral_velocity`
- `ion_charge_state_energy_velocity_due_to_convection`
- `ion_charge_state_particle_convection_velocity`
- `ion_diamagnetic_momentum_convection_velocity`
- `left_hand_circularly_polarized_wave_electric_field`
- `magnetic_field_at_pedestal_top_low_field_side`
- `maximum_force_of_poloidal_field_coil`
- `maximum_magnetic_field`
- `minimum_force_of_poloidal_field_coil`
- `neutral_internal_state_energy_convection_velocity`
- `neutral_internal_state_momentum_velocity_due_to_convection`
- `neutral_internal_state_velocity_due_to_convection`
- `per_toroidal_mode_left_hand_circularly_polarized_wave_electric_field`
- `pfirsch_schlueter_current_density_due_to_diamagnetic_drift`
- `plasma_magnetic_field`
- `product_of_current_density_and_major_radius`
- `ratio_of_parallel_ion_velocity_to_magnetic_field_magnitude`
- `ratio_of_toroidal_ion_velocity_to_magnetic_field_magnitude`
- `right_hand_circularly_polarized_wave_electric_field`
- `saturated_ion_current_density`
- `stray_breakdown_magnetic_field`
- `thermal_ion_charge_state_energy_velocity_due_to_convection`
- `toroidal_flux_coordinate_gradient`
- `velocity_due_to_e_cross_b_drift`

### tensor -> scalar (2)

- `flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude`
- `plasma_electrical_conductivity`

### vector -> scalar (135)

- `co_passing_fast_ion_charge_state_torque_density_due_to_collisions`
- `co_passing_fast_ion_torque_density_due_to_collisions`
- `co_passing_thermal_electron_torque_density_due_to_collisions`
- `co_passing_thermal_ion_charge_state_torque_density_due_to_collisions`
- `co_passing_thermal_ion_torque_density_due_to_collisions`
- `co_passing_torque_density`
- `co_passing_torque_density_due_to_j_cross_b_force`
- `counter_passing_fast_electron_torque_density_due_to_collisions`
- `counter_passing_fast_ion_charge_state_torque_density_due_to_collisions`
- `counter_passing_fast_ion_torque_density_due_to_collisions`
- `counter_passing_thermal_electron_torque_density_due_to_collisions`
- `counter_passing_thermal_ion_charge_state_torque_density_due_to_collisions`
- `counter_passing_thermal_ion_torque_density_due_to_collisions`
- `counter_passing_torque_density`
- `counter_passing_torque_density_due_to_j_cross_b_force`
- `cumulative_inside_flux_surface_torque`
- `current_weighted_average_external_magnetic_flux`
- `density_at_pedestal_maximum`
- `edge_plasma_momentum_diffusivity`
- `effective_electron_diffusivity`
- `effective_ion_diffusivity`
- `effective_neutral_diffusivity`
- `electron_density_at_pedestal_maximum`
- `electrostatic_potential_amplitude`
- `energy_diffusion_coefficient`
- `explicit_ion_torque`
- `fast_electron_pressure`
- `fast_electron_torque_density_due_to_collisions`
- `fast_ion_charge_state_pressure`
- `fast_ion_pressure`
- `fast_ion_torque_density_due_to_collisions`
- `fast_neutral_pressure`
- `fast_particle_torque_density_due_to_coulomb_collisions_with_electrons`
- `fast_particle_torque_due_to_j_cross_b_force`
- `fast_torque_due_to_collisions`
- `first_measurement_direction_unit_vector_of_shatter_cone`
- `flux_due_to_e_cross_b_drift`
- `flux_limiter_coefficient`
- `flux_surface_averaged_inverse_of_square_of_magnetic_field_magnitude`
- `flux_surface_averaged_magnetic_field_magnitude`
- `gyrocenter_frequency`
- `halo_current`
- `heat_flux`
- `ion_charge_state_momentum_damping_rate`
- `ion_charge_state_momentum_diffusivity`
- `ion_charge_state_momentum_flux`
- `ion_charge_state_momentum_flux_limiter_coefficient`
- `ion_charge_state_momentum_source`
- `ion_charge_state_particle_flux`
- `ion_charge_state_rotation_frequency`
- `ion_charge_state_torque_density`
- `ion_momentum_damping_rate`
- `ion_momentum_diffusion_coefficient`
- `ion_momentum_flux`
- `ion_momentum_flux_limiter_coefficient`
- `ion_momentum_source`
- `ion_particle_diffusivity`
- `ion_rotation_frequency`
- `ion_torque`
- `ion_torque_density`
- `kinetic_energy_density`
- `linear_mhd_mode_reference_phase`
- `linear_neutral_internal_state_momentum_flux`
- `mach_number`
- `magnetic_flux`
- `magnetic_flux_due_to_diamagnetic_drift`
- `mhd_mode_number`
- `momentum`
- `momentum_diffusion_coefficient`
- `momentum_diffusivity`
- `momentum_flux`
- `momentum_flux_due_to_e_cross_b_drift`
- `momentum_flux_limiter_coefficient`
- `momentum_source`
- `net_ion_momentum_source`
- `neutral_internal_state_momentum_diffusion_coefficient`
- `neutral_internal_state_momentum_flux_limiter_coefficient_over_edge_region`
- `neutral_internal_state_momentum_source`
- `neutral_momentum_diffusion_coefficient`
- `neutral_momentum_diffusivity`
- `neutral_momentum_flux`
- `neutral_momentum_flux_limiter_coefficient`
- `neutral_momentum_source`
- `neutral_particle_diffusivity`
- `neutral_species_energy_flux`
- `neutral_species_particle_diffusivity`
- `neutral_state_particle_diffusivity`
- `neutral_torque_density`
- `neutron_source_rate_due_to_beam_beam_fusion`
- `normalized_gyrocenter_perturbed_pressure`
- `normalized_momentum_flux_due_to_e_cross_b_drift`
- `normalized_particle_perturbed_energy`
- `normalized_particle_perturbed_pressure`
- `particle_diffusivity`
- `particle_torque_density_due_to_coulomb_collisions_with_electrons`
- `perturbed_electrostatic_potential_amplitude`
- `perturbed_pressure_bessel_1`
- `plasma_momentum_diffusivity`
- `plasma_momentum_flux`
- `refractive_index`
- `rotation_frequency_due_to_e_cross_b_drift`
- `runaway_electron_diffusivity`
- `runaway_electron_particle_flux`
- `source_due_to_diamagnetic_drift`
- `source_rate_due_to_beam_beam_fusion`
- `suprathermal_neutral_internal_state_pressure`
- `thermal_electron_torque_density_due_to_collisions`
- `thermal_electron_torque_due_to_collisions`
- `thermal_ion_charge_state_energy_diffusion_coefficient`
- `thermal_ion_charge_state_torque_due_to_collisions`
- `thermal_ion_torque_density_due_to_collisions`
- `thermal_ion_torque_density_due_to_thermalization`
- `thermal_ion_torque_due_to_collisions`
- `torque_density`
- `torque_density_due_to_j_cross_b_force`
- `torque_due_to_neutral_beam_shinethrough`
- `total_fast_ion_pressure`
- `total_ion_energy_flux`
- `total_momentum_flux`
- `total_momentum_source`
- `total_neutral_momentum_diffusivity`
- `total_plasma_momentum`
- `total_runaway_electron_current`
- `total_thermal_ion_energy_diffusion_coefficient`
- `trapped_fast_ion_charge_state_torque_density_due_to_collisions`
- `trapped_fast_ion_torque_density_due_to_collisions`
- `trapped_thermal_electron_torque_density_due_to_collisions`
- `trapped_thermal_ion_charge_state_torque_density_due_to_collisions`
- `trapped_thermal_ion_torque_density_due_to_collisions`
- `trapped_torque_density`
- `trapped_torque_density_due_to_j_cross_b_force`
- `wave_current_of_antenna_strap`
- `wave_mode_number`
- `wavelength_of_visible_camera`
- `weight_of_interferometer_beam`

