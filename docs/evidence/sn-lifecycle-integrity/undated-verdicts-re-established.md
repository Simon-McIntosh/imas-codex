# Deterministic validation restored the observed verdicts

## Result

The deterministic admission gate re-established 525 of the 538 initially
unstamped `valid` verdicts. It did not backfill a timestamp: every changed
row passed through `drain_validation_for_ids`, which writes its verdict,
issues, and `validated_at` together.

The fourteen names in the review cut were processed first. Their exact result
was `validated=14`, `quarantined=0`, `requarantined_ids=[]`, and
`cleared_ids=[radial_outline_of_antenna_strap,
toroidal_coordinate_of_camera, vertical_outline_of_plasma_facing_component,
poloidal_length_of_flux_surface, lower_bound_photon_energy,
minimum_over_flux_surface_magnetic_field_magnitude,
toroidal_angle_of_poloidal_magnetic_field_probe,
incident_soft_xray_radiance, total_power_due_to_ohmic_dissipation,
ipb98y2_confinement_time, radiative_temperature_at_ece_channel,
intensity_at_spectral_line, total_electron_count,
vertical_coordinate_of_ece_channel]`. No review-cut identity re-quarantined.

The remaining cohort returned `validated=511`, `quarantined=37`,
`requarantined_ids` containing 37 identities, and 474 `cleared_ids`. The exact
per-call arrays, including every cleared identity, are preserved in the
durable drain logs named in the worker manifest. The 37 re-quarantines are the
finding the prior scalar could not supply; they were not washed back to valid.

The re-quarantined identities are:

`derivative_with_respect_to_normalized_minor_radius_of_logarithm_of_density`,
`first_local_tangential_coordinate`,
`impurity_ion_photon_radiance_of_spectral_line_due_to_charge_exchange`,
`flux_surface_normal_ion_charge_state_particle_flux`,
`flux_surface_normal_momentum_diffusivity`,
`inverse_of_tangential_surface_curvature_of_optical_element`,
`inverse_of_second_local_tangential_front_surface_curvature_of_optical_element`,
`inverse_of_curvature_of_iron_core_segment`,
`inverse_of_first_local_tangential_back_surface_curvature_of_optical_element`,
`flux_surface_normal_bulk_plasma_momentum_diffusivity`,
`flux_surface_normal_plasma_momentum_diffusivity`,
`inverse_of_second_local_tangential_back_surface_curvature_of_optical_element`,
`magnetic_field_at_pedestal_top_high_field_side_magnitude`,
`normalized_electron_larmor_radius_at_pedestal_top_high_field_side`,
`normalized_molecular_gas_count_due_to_gas_injection`,
`normalized_saturated_permeability_of_ferritic_element`,
`parallel_normalized_gyrocenter_momentum_flux_of_gyrokinetic_eigenmode_due_to_e_cross_b_drift`,
`normalized_electron_larmor_radius_at_pedestal_top_low_field_side`,
`perturbed_pressure_bessel_1`, `pressure_bessel_1`,
`perturbed_magnetic_field_of_wave_beam`, `plasma_internal_energy`,
`per_toroidal_and_poloidal_mode_number_surface_current_of_ion_cyclotron_heating_antenna`,
`root_mean_square_of_fluctuating_floating_electrostatic_potential`,
`toroidal_coordinate_of_bragg_crystal`,
`root_mean_square_of_variation_of_vacuum_wavelength_of_spectrometer_channel`,
`tendency_of_derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface`,
`root_mean_square_of_difference_of_wavelength_of_spectrometer_channel_and_reference_wavelength_of_spectrometer_channel`,
`time_derivative_of_radial_width_of_neoclassical_tearing_mode`,
`thermal_plasma_internal_energy`, `toroidal_line_averaged_plasma_velocity`,
`vertical_coordinate_of_fibre_bundle`, `vertical_outline_of_control_surface`,
`total_thermal_plasma_internal_energy`, `vertical_coordinate_of_hard_xray_detector`,
`voltage_of_ion_cyclotron_heating_antenna_amplitude`, and
`volume_integrated_electron_power_density`.

| Population | Before | After |
|---|---:|---:|
| `valid` with null `validated_at` | 538 | 13 |
| Review-cut identities in that population | 14 | 0 |
| Valid identities re-established by the gate | 0 | 488 |
| Re-quarantined by the gate | 0 | 37 |

## Residual blocker

The remaining 13 rows still make the invariant false. Each has
`validation_status='valid'`, a null `validated_at`, no active claim, and a
null description. The gate is intentionally unable to admit them because
`claim_ids_for_validation` requires `sn.description IS NOT NULL` before taking
its claim. They therefore cannot receive an honest new observation through
this route, and writing `validated_at` directly would create the false record
this work avoids.

| Name stage | Identities |
|---|---|
| `superseded` | `breakdown_time`, `calibration_spectral_coefficient_of_line_of_sight`, `coolant_delay`, `radial_gap`, `ratio_of_ion_velocity_to_magnetic_field_strength` |
| `pending` | `line_averaged_neon_density`, `poloidal_ion_state_momentum_diffusion_coefficient`, `radial_coordinate_of_reflector`, `radius_of_soft_xray_detector`, `ratio_of_diamagnetic_vorticity_to_major_radius`, `toroidal_coordinate_of_spectrometer`, `toroidal_tritium_velocity`, `vertical_coordinate_of_reflector` |

The next repair must decide how a no-description identity is classified before
it may hold any current validation verdict. It belongs in the validation claim
or lifecycle owner, not in this evidence-only node. The export fence continues
to withhold every undated `valid` row; none of the residual rows is currently
accepted or approved.

## Measurement

The before census found 5,048 identities, 538 `valid` rows without an
observation, 261 otherwise publishable unstamped rows, and 2,058 equivalent
stamped valid rows. Its two bounded graph queries took 0.038 and 0.015
seconds. The final residual census took 0.043 seconds and found 13 unstamped
valid rows, 4,378 stamped valid rows, and 42 stamped quarantined rows.

Each graph operation ran on the login node because the authenticated graph
route is local to that host. Priority-drain elapsed time was 7.484 seconds;
the eleven remaining-drain calls each used at most 50 named identities and
completed in 7.126 to 10.353 seconds.
