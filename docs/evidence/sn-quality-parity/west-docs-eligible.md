# WEST documentation eligibility census

This is a read-only snapshot of the live `codex` graph at
**2026-08-23T16:21:57.759Z**. It resolves the historical 231-identity WEST
manifest against live `StandardName` nodes and removes the exact 20-identity
intersection with the concurrent below-bar name-refine cohort. It is evidence
for campaign scoping, not graph-mutation or apply authority.

## Headline result

| Measure | Result |
|---|---:|
| Live `StandardName` total | **4665** |
| WEST manifest identities | **231** |
| Join on `StandardName.id` | **231** |
| Join on `StandardName.name` | **0** |
| Nodes carrying a `name` property | **0** |
| Concurrent-refine exclusions | **20** |
| Docs-eligible after exclusion | **211** |
| Exclusion overlap remaining in eligible subset | **0** |

The sound join key is `StandardName.id`: it is the snake_case Standard Name
identity. The class has no `name` property, so the alternative predicate is
valid Cypher but evaluates to null and silently returns zero rows. Reporting
**231 beside 0** prevents that zero from being mistaken for a clean no-overlap
result.

“Docs-eligible” here means the resolved WEST identity set after the plan-mandated
20-identity exclusion only. The live `name_stage` is retained beside every row
for downstream safety; this census does not silently add a second lifecycle
filter.

Live documentation-stage counts are
**accepted=195, drafted=4, exhausted=1, pending=25, reviewed=6** across all resolved WEST
identities and **accepted=175, drafted=4, exhausted=1, pending=25, reviewed=6** after
the exclusion.

## Evidence inputs

- Below-bar refine receipt: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T142524040287-n-drainfinish/final-receipt.json`
  (SHA-256 `e54b31cf9c581839c4898a7503a1e05889518ce98ba8240f2d29cafc1d7238a2`), filtered at
  `final_score < 0.85`: **273**
  distinct identities.
- WEST manifest: `/home/ITER/mcintos/.local/share/imas-codex/receipts/standard-names/portfolio-census-2026-08-16/west-gate-manifest.json`
  (SHA-256 `a613ecb74bae12b8f576769994e1b7df37e68a0c90a0168c6441d62dbb35d000`):
  **231** distinct `name_dispositions`
  identities.
- Exact intersection: **20** identities.

## Excluded concurrent-refine identities

| Standard Name id | Live `name_stage` | Live `docs_stage` |
|---|---|---|
| `current_of_poloidal_field_coil` | `exhausted` | `accepted` |
| `energy_confinement_time` | `superseded` | `accepted` |
| `gap_of_antenna_strap` | `superseded` | `accepted` |
| `hot_neutral_temperature` | `reviewed` | `accepted` |
| `ion_atomic_number` | `superseded` | `accepted` |
| `length_of_plasma_boundary` | `superseded` | `accepted` |
| `lower_photon_energy` | `superseded` | `accepted` |
| `neutral_pressure` | `exhausted` | `accepted` |
| `power_due_to_ohmic_dissipation` | `superseded` | `accepted` |
| `power_of_ion_cyclotron_heating_antenna` | `superseded` | `accepted` |
| `radiative_temperature` | `exhausted` | `accepted` |
| `spectral_radiance_of_soft_xray_detector` | `superseded` | `accepted` |
| `surface_temperature` | `superseded` | `accepted` |
| `thermal_energy_of_plant_component_port` | `exhausted` | `accepted` |
| `thermal_power_of_plant_component_port` | `superseded` | `accepted` |
| `toroidal_angle_of_magnetic_field_probe` | `superseded` | `accepted` |
| `toroidal_width_of_antenna_strap` | `exhausted` | `accepted` |
| `vertical_coordinate_of_plasma_boundary` | `superseded` | `accepted` |
| `vertical_outline_of_wall_material` | `superseded` | `accepted` |
| `voltage_amplitude` | `exhausted` | `accepted` |

## Every resolved WEST identity and live documentation stage

| Standard Name id | `name_stage` | `docs_stage` | Excluded | Docs-eligible |
|---|---|---|---:|---:|
| `accumulated_deposited_energy_of_plasma_facing_component` | `accepted` | `accepted` | no | yes |
| `accumulated_total_particle_count_due_to_gas_injection` | `accepted` | `accepted` | no | yes |
| `area_of_diagnostic_aperture` | `accepted` | `drafted` | no | yes |
| `area_of_poloidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `area_of_toroidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `atomic_count_of_ion_state` | `superseded` | `accepted` | no | yes |
| `atomic_mass` | `accepted` | `accepted` | no | yes |
| `breakdown_initial_time` | `accepted` | `accepted` | no | yes |
| `capacitance` | `superseded` | `pending` | no | yes |
| `cold_neutral_fraction` | `accepted` | `accepted` | no | yes |
| `cold_neutral_temperature` | `accepted` | `accepted` | no | yes |
| `coolant_temperature_at_inlet` | `accepted` | `accepted` | no | yes |
| `coolant_temperature_at_outlet` | `accepted` | `accepted` | no | yes |
| `coolant_transit_time_of_plant_component_port` | `accepted` | `accepted` | no | yes |
| `current_of_passive_loop` | `accepted` | `accepted` | no | yes |
| `current_of_poloidal_field_coil` | `exhausted` | `accepted` | yes | no |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | `accepted` | `reviewed` | no | yes |
| `derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_volume_of_flux_surface` | `accepted` | `accepted` | no | yes |
| `derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface` | `accepted` | `accepted` | no | yes |
| `derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface` | `accepted` | `accepted` | no | yes |
| `difference_of_total_plasma_heating_power_and_time_derivative_of_plasma_stored_energy` | `accepted` | `accepted` | no | yes |
| `difference_of_vacuum_poloidal_current_function_and_initial_vacuum_poloidal_current_function` | `accepted` | `accepted` | no | yes |
| `effective_charge` | `accepted` | `accepted` | no | yes |
| `effective_turn_count_of_coil_conductor_element` | `accepted` | `accepted` | no | yes |
| `effective_turn_count_of_passive_loop` | `accepted` | `accepted` | no | yes |
| `electron_density_at_divertor_target` | `accepted` | `accepted` | no | yes |
| `electron_density_at_magnetic_axis` | `accepted` | `accepted` | no | yes |
| `electron_density_at_plasma_boundary` | `accepted` | `accepted` | no | yes |
| `electron_temperature` | `accepted` | `accepted` | no | yes |
| `electron_temperature_at_divertor_target` | `accepted` | `accepted` | no | yes |
| `electron_temperature_at_magnetic_axis` | `accepted` | `accepted` | no | yes |
| `elongation_of_flux_surface` | `accepted` | `accepted` | no | yes |
| `elongation_of_plasma_boundary` | `accepted` | `accepted` | no | yes |
| `energy_confinement_enhancement_factor` | `accepted` | `accepted` | no | yes |
| `energy_confinement_time` | `superseded` | `accepted` | yes | no |
| `equilibrium_weight_of_flux_loop` | `accepted` | `accepted` | no | yes |
| `equilibrium_weight_of_interferometer_beam` | `accepted` | `drafted` | no | yes |
| `equilibrium_weight_of_poloidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `etendue_of_hard_xray_detector` | `accepted` | `accepted` | no | yes |
| `etendue_of_spectrometer_channel` | `accepted` | `accepted` | no | yes |
| `faraday_angle` | `accepted` | `accepted` | no | yes |
| `flux_surface_averaged_inverse_of_major_radius` | `accepted` | `accepted` | no | yes |
| `flux_surface_averaged_inverse_of_square_of_magnetic_field_magnitude` | `accepted` | `accepted` | no | yes |
| `flux_surface_averaged_inverse_of_square_of_major_radius` | `accepted` | `accepted` | no | yes |
| `flux_surface_averaged_magnetic_field_magnitude` | `accepted` | `accepted` | no | yes |
| `flux_surface_averaged_major_radius` | `accepted` | `accepted` | no | yes |
| `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_magnetic_field_magnitude` | `accepted` | `accepted` | no | yes |
| `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_major_radius` | `accepted` | `accepted` | no | yes |
| `flux_surface_averaged_square_of_magnetic_field_magnitude` | `accepted` | `accepted` | no | yes |
| `flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude` | `accepted` | `reviewed` | no | yes |
| `forward_power_of_ion_cyclotron_heating_antenna` | `accepted` | `accepted` | no | yes |
| `frequency_of_diagnostic_antenna` | `accepted` | `pending` | no | yes |
| `frequency_of_ion_cyclotron_heating_antenna` | `accepted` | `accepted` | no | yes |
| `gap_at_outboard_midplane` | `accepted` | `accepted` | no | yes |
| `gap_at_plasma_boundary` | `accepted` | `pending` | no | yes |
| `gap_of_antenna_strap` | `superseded` | `accepted` | yes | no |
| `gas_flow` | `accepted` | `accepted` | no | yes |
| `hard_xray_brightness` | `accepted` | `pending` | no | yes |
| `hard_xray_emissivity` | `accepted` | `pending` | no | yes |
| `height_of_poloidal_field_coil` | `accepted` | `accepted` | no | yes |
| `hot_neutral_fraction` | `accepted` | `accepted` | no | yes |
| `hot_neutral_temperature` | `reviewed` | `accepted` | yes | no |
| `initial_polarization_ellipticity_of_polarimeter_beam` | `accepted` | `exhausted` | no | yes |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | `drafted` | `pending` | no | yes |
| `ion_atomic_number` | `superseded` | `accepted` | yes | no |
| `ion_current_of_mass_spectrometer_channel` | `accepted` | `accepted` | no | yes |
| `launched_power_of_lower_hybrid_antenna` | `accepted` | `accepted` | no | yes |
| `length_of_interferometer_beam` | `accepted` | `accepted` | no | yes |
| `length_of_plasma_boundary` | `superseded` | `accepted` | yes | no |
| `length_of_poloidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `length_of_toroidal_magnetic_field_probe` | `accepted` | `reviewed` | no | yes |
| `line_averaged_effective_charge` | `accepted` | `pending` | no | yes |
| `line_averaged_electron_density` | `accepted` | `accepted` | no | yes |
| `line_integrated_electron_number_density` | `accepted` | `accepted` | no | yes |
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | `drafted` | `pending` | no | yes |
| `loop_voltage_at_plasma_boundary` | `accepted` | `accepted` | no | yes |
| `lower_bound_hard_xray_peak_width` | `superseded` | `pending` | no | yes |
| `lower_photon_energy` | `superseded` | `accepted` | yes | no |
| `lower_triangularity_of_flux_surface` | `accepted` | `accepted` | no | yes |
| `lower_triangularity_of_plasma_boundary` | `accepted` | `accepted` | no | yes |
| `lower_wavelength_of_filter` | `accepted` | `accepted` | no | yes |
| `magnetic_shear_at_flux_surface` | `accepted` | `accepted` | no | yes |
| `maximum_magnetic_field_magnitude` | `accepted` | `accepted` | no | yes |
| `maximum_of_energy_flux_at_divertor_target` | `accepted` | `accepted` | no | yes |
| `mhd_energy` | `accepted` | `accepted` | no | yes |
| `minimum_magnetic_field` | `accepted` | `accepted` | no | yes |
| `minimum_safety_factor` | `accepted` | `accepted` | no | yes |
| `minor_radius_of_plasma_boundary` | `accepted` | `accepted` | no | yes |
| `neutral_pressure` | `exhausted` | `accepted` | yes | no |
| `neutron_flux` | `accepted` | `accepted` | no | yes |
| `normalized_plasma_internal_inductance` | `accepted` | `accepted` | no | yes |
| `normalized_poloidal_flux_coordinate` | `reviewed` | `pending` | no | yes |
| `normalized_poloidal_flux_coordinate_of_plasma_boundary` | `accepted` | `accepted` | no | yes |
| `normalized_toroidal_flux_coordinate` | `accepted` | `accepted` | no | yes |
| `normalized_toroidal_flux_coordinate_at_measurement_position` | `accepted` | `pending` | no | yes |
| `normalized_toroidal_flux_coordinate_at_minimum_safety_factor` | `accepted` | `accepted` | no | yes |
| `normalized_toroidal_hard_xray_peak_lower_bound_width` | `exhausted` | `pending` | no | yes |
| `normalized_toroidal_plasma_beta` | `accepted` | `accepted` | no | yes |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | `accepted` | `reviewed` | no | yes |
| `photon_radiance_at_spectral_line` | `accepted` | `accepted` | no | yes |
| `plasma_beta` | `accepted` | `accepted` | no | yes |
| `plasma_breakdown_time` | `exhausted` | `pending` | no | yes |
| `plasma_current` | `accepted` | `accepted` | no | yes |
| `plasma_pressure` | `accepted` | `accepted` | no | yes |
| `poloidal_angle_of_flux_surface` | `accepted` | `accepted` | no | yes |
| `poloidal_angle_of_measurement_position` | `accepted` | `accepted` | no | yes |
| `poloidal_angle_of_toroidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `poloidal_beta` | `accepted` | `accepted` | no | yes |
| `poloidal_cross_sectional_area_of_flux_surface` | `superseded` | `accepted` | no | yes |
| `poloidal_cross_sectional_area_of_plasma_boundary` | `superseded` | `accepted` | no | yes |
| `poloidal_magnetic_field` | `accepted` | `accepted` | no | yes |
| `poloidal_magnetic_field_at_constraint_position` | `accepted` | `pending` | no | yes |
| `poloidal_magnetic_flux_at_flux_surface` | `accepted` | `pending` | no | yes |
| `poloidal_magnetic_flux_at_magnetic_axis` | `accepted` | `pending` | no | yes |
| `poloidal_magnetic_flux_at_measurement_position` | `accepted` | `accepted` | no | yes |
| `poloidal_magnetic_flux_at_plasma_boundary` | `accepted` | `pending` | no | yes |
| `poloidal_magnetic_flux_of_flux_loop` | `accepted` | `drafted` | no | yes |
| `power_due_to_ion_cyclotron_heating` | `accepted` | `accepted` | no | yes |
| `power_due_to_ohmic_dissipation` | `superseded` | `accepted` | yes | no |
| `power_of_ion_cyclotron_heating_antenna` | `superseded` | `accepted` | yes | no |
| `power_of_soft_xray_detector` | `accepted` | `accepted` | no | yes |
| `pressure_of_ion_cyclotron_heating_antenna` | `accepted` | `pending` | no | yes |
| `product_of_poloidal_current_function_and_derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_poloidal_current_function` | `accepted` | `accepted` | no | yes |
| `pulse_duration` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_at_inboard_midplane` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_at_outboard_midplane` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_of_camera` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_of_conductor_cross_section` | `accepted` | `reviewed` | no | yes |
| `radial_coordinate_of_detector_pixel` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_of_flux_loop` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_of_geometric_axis` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_of_line_of_sight` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_of_magnetic_axis` | `accepted` | `pending` | no | yes |
| `radial_coordinate_of_measurement_position` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_of_poloidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_of_strike_point` | `accepted` | `accepted` | no | yes |
| `radial_coordinate_of_x_point` | `accepted` | `accepted` | no | yes |
| `radial_derivative_of_poloidal_magnetic_flux` | `accepted` | `accepted` | no | yes |
| `radial_outline_of_antenna_strap` | `accepted` | `accepted` | no | yes |
| `radial_outline_of_limiter_tile` | `accepted` | `accepted` | no | yes |
| `radial_outline_of_plasma_boundary` | `accepted` | `accepted` | no | yes |
| `radial_outline_of_wall` | `accepted` | `accepted` | no | yes |
| `radiated_power_over_core_region` | `accepted` | `accepted` | no | yes |
| `radiative_temperature` | `exhausted` | `accepted` | yes | no |
| `radiative_temperature_at_magnetic_axis` | `accepted` | `accepted` | no | yes |
| `ratio_of_coolant_mass_to_time` | `accepted` | `accepted` | no | yes |
| `ratio_of_line_averaged_electron_density_to_greenwald_density` | `accepted` | `accepted` | no | yes |
| `ratio_of_line_averaged_hydrogen_density_to_line_averaged_total_hydrogenic_density` | `accepted` | `accepted` | no | yes |
| `ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density_and_neutral_density_of_isotope` | `accepted` | `accepted` | no | yes |
| `reference_major_radius` | `accepted` | `pending` | no | yes |
| `reflected_phase_of_ion_cyclotron_heating_antenna` | `accepted` | `accepted` | no | yes |
| `reflected_power_of_ion_cyclotron_heating_antenna` | `accepted` | `accepted` | no | yes |
| `safety_factor` | `accepted` | `accepted` | no | yes |
| `safety_factor_at_magnetic_axis` | `accepted` | `accepted` | no | yes |
| `safety_factor_at_normalized_poloidal_magnetic_flux_equal_to_0_95` | `accepted` | `accepted` | no | yes |
| `spectral_bremsstrahlung_radiance` | `accepted` | `accepted` | no | yes |
| `spectral_calibration_factor_at_line_of_sight` | `accepted` | `accepted` | no | yes |
| `spectral_flux_of_spectrometer_channel` | `accepted` | `pending` | no | yes |
| `spectral_radiance` | `accepted` | `pending` | no | yes |
| `spectral_radiance_of_soft_xray_detector` | `superseded` | `accepted` | yes | no |
| `spectral_signal_to_noise_ratio_of_spectrometer_channel` | `accepted` | `accepted` | no | yes |
| `spectral_wavelength_of_optical_element` | `accepted` | `reviewed` | no | yes |
| `surface_area_of_flux_surface` | `accepted` | `accepted` | no | yes |
| `surface_temperature` | `superseded` | `accepted` | yes | no |
| `temperature_of_soft_xray_detector` | `accepted` | `accepted` | no | yes |
| `thermal_electron_density` | `accepted` | `accepted` | no | yes |
| `thermal_electron_pressure_at_post_sawtooth_crash` | `accepted` | `accepted` | no | yes |
| `thermal_energy_of_plant_component_port` | `exhausted` | `accepted` | yes | no |
| `thermal_power_of_plant_component_port` | `superseded` | `accepted` | yes | no |
| `thickness_of_filter` | `accepted` | `accepted` | no | yes |
| `toroidal_angle_of_antenna_strap` | `accepted` | `accepted` | no | yes |
| `toroidal_angle_of_magnetic_field_probe` | `superseded` | `accepted` | yes | no |
| `toroidal_angle_of_measurement_position` | `accepted` | `drafted` | no | yes |
| `toroidal_angle_of_toroidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `toroidal_angular_width_of_limiter_tile` | `accepted` | `accepted` | no | yes |
| `toroidal_coordinate_of_detector_pixel` | `accepted` | `accepted` | no | yes |
| `toroidal_coordinate_of_line_of_sight` | `accepted` | `accepted` | no | yes |
| `toroidal_flux_coordinate` | `accepted` | `accepted` | no | yes |
| `toroidal_flux_coordinate_gradient_magnitude` | `accepted` | `accepted` | no | yes |
| `toroidal_flux_surface_averaged_current_density` | `accepted` | `accepted` | no | yes |
| `toroidal_magnetic_field` | `accepted` | `accepted` | no | yes |
| `toroidal_magnetic_field_at_magnetic_axis` | `accepted` | `accepted` | no | yes |
| `toroidal_magnetic_flux` | `accepted` | `accepted` | no | yes |
| `toroidal_magnetic_flux_due_to_diamagnetic_drift` | `accepted` | `accepted` | no | yes |
| `toroidal_vacuum_magnetic_field` | `accepted` | `pending` | no | yes |
| `toroidal_width_of_antenna_strap` | `exhausted` | `accepted` | yes | no |
| `total_electron_density` | `accepted` | `accepted` | no | yes |
| `total_electron_pressure` | `accepted` | `accepted` | no | yes |
| `total_external_heating_power` | `accepted` | `accepted` | no | yes |
| `total_neutral_source_rate_due_to_gas_injection` | `accepted` | `accepted` | no | yes |
| `total_plasma_radiated_power` | `accepted` | `accepted` | no | yes |
| `total_power_at_separatrix` | `accepted` | `accepted` | no | yes |
| `total_power_due_to_ion_cyclotron_heating` | `accepted` | `accepted` | no | yes |
| `triangularity_of_plasma_boundary` | `accepted` | `accepted` | no | yes |
| `turn_count_of_poloidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `turn_count_of_toroidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `upper_photon_energy` | `accepted` | `accepted` | no | yes |
| `upper_triangularity_of_flux_surface` | `accepted` | `accepted` | no | yes |
| `upper_triangularity_of_plasma_boundary` | `accepted` | `pending` | no | yes |
| `upper_wavelength_of_filter` | `accepted` | `accepted` | no | yes |
| `vacuum_poloidal_current_function` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_camera` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_conductor_cross_section` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_detector_pixel` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_flux_loop` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_geometric_axis` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_line_of_sight` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_magnetic_axis` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_measurement_position` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_plasma_boundary` | `superseded` | `accepted` | yes | no |
| `vertical_coordinate_of_poloidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_primary_x_point` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_strike_point` | `accepted` | `accepted` | no | yes |
| `vertical_coordinate_of_toroidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `vertical_outline_of_antenna_strap` | `accepted` | `accepted` | no | yes |
| `vertical_outline_of_limiter_tile` | `accepted` | `accepted` | no | yes |
| `vertical_outline_of_wall_material` | `superseded` | `accepted` | yes | no |
| `voltage_amplitude` | `exhausted` | `accepted` | yes | no |
| `voltage_of_mass_spectrometer_channel` | `accepted` | `accepted` | no | yes |
| `voltage_of_poloidal_magnetic_field_probe` | `accepted` | `accepted` | no | yes |
| `volume_averaged_electron_density` | `accepted` | `accepted` | no | yes |
| `volume_integrated_total_electron_density` | `accepted` | `accepted` | no | yes |
| `volume_of_flux_surface` | `accepted` | `pending` | no | yes |
| `volume_of_plasma_boundary` | `accepted` | `accepted` | no | yes |
| `wave_current_of_antenna_strap_amplitude` | `accepted` | `accepted` | no | yes |
| `wave_phase_of_antenna_strap` | `accepted` | `accepted` | no | yes |
| `wave_phase_of_ion_cyclotron_heating_antenna` | `accepted` | `accepted` | no | yes |
| `wave_phase_of_wave_beam` | `accepted` | `accepted` | no | yes |
| `wavelength_of_spectral_line` | `accepted` | `accepted` | no | yes |
| `wavelength_of_wave_beam` | `accepted` | `accepted` | no | yes |
| `width_of_poloidal_field_coil` | `accepted` | `accepted` | no | yes |

## Recorded command output

The command ran from the canonical checkout using its shared project
environment, with bytecode and uv cache writes disabled:

```text
env -u VIRTUAL_ENV PYTHONDONTWRITEBYTECODE=1 uv run --no-sync --no-cache python -
```

It read the two hashed inputs, computed the 273/231/20 sets, and issued one
Cypher statement containing the graph total plus both joins. Its assertions
required 231 id-joined rows, 0 name-joined rows, 0 nodes carrying `name`, 211
eligible rows, and zero excluded identities in the eligible set. Exact stdout:

```text
observed_at_utc=2026-08-23T16:21:57.759Z
standard_name_total=4665
name_property_node_count=0
below_bar_identity_count=273
west_identity_count=231
id_join_row_count=231
name_join_row_count=0
resolved_west_identity_count=231
excluded_identity_count=20
docs_eligible_identity_count=211
docs_eligible_excluded_overlap_count=0
resolved_docs_stage_counts=accepted=195, drafted=4, exhausted=1, pending=25, reviewed=6
docs_eligible_docs_stage_counts=accepted=175, drafted=4, exhausted=1, pending=25, reviewed=6
WEST	accumulated_deposited_energy_of_plasma_facing_component	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	accumulated_total_particle_count_due_to_gas_injection	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	area_of_diagnostic_aperture	name_stage=accepted	docs_stage=drafted	excluded=false	docs_eligible=true
WEST	area_of_poloidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	area_of_toroidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	atomic_count_of_ion_state	name_stage=superseded	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	atomic_mass	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	breakdown_initial_time	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	capacitance	name_stage=superseded	docs_stage=pending	excluded=false	docs_eligible=true
WEST	cold_neutral_fraction	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	cold_neutral_temperature	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	coolant_temperature_at_inlet	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	coolant_temperature_at_outlet	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	coolant_transit_time_of_plant_component_port	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	current_of_passive_loop	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	current_of_poloidal_field_coil	name_stage=exhausted	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface	name_stage=accepted	docs_stage=reviewed	excluded=false	docs_eligible=true
WEST	derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_volume_of_flux_surface	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	difference_of_total_plasma_heating_power_and_time_derivative_of_plasma_stored_energy	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	difference_of_vacuum_poloidal_current_function_and_initial_vacuum_poloidal_current_function	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	effective_charge	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	effective_turn_count_of_coil_conductor_element	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	effective_turn_count_of_passive_loop	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	electron_density_at_divertor_target	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	electron_density_at_magnetic_axis	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	electron_density_at_plasma_boundary	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	electron_temperature	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	electron_temperature_at_divertor_target	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	electron_temperature_at_magnetic_axis	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	elongation_of_flux_surface	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	elongation_of_plasma_boundary	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	energy_confinement_enhancement_factor	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	energy_confinement_time	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	equilibrium_weight_of_flux_loop	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	equilibrium_weight_of_interferometer_beam	name_stage=accepted	docs_stage=drafted	excluded=false	docs_eligible=true
WEST	equilibrium_weight_of_poloidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	etendue_of_hard_xray_detector	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	etendue_of_spectrometer_channel	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	faraday_angle	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	flux_surface_averaged_inverse_of_major_radius	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	flux_surface_averaged_inverse_of_square_of_magnetic_field_magnitude	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	flux_surface_averaged_inverse_of_square_of_major_radius	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	flux_surface_averaged_magnetic_field_magnitude	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	flux_surface_averaged_major_radius	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_magnetic_field_magnitude	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_major_radius	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	flux_surface_averaged_square_of_magnetic_field_magnitude	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude	name_stage=accepted	docs_stage=reviewed	excluded=false	docs_eligible=true
WEST	forward_power_of_ion_cyclotron_heating_antenna	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	frequency_of_diagnostic_antenna	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	frequency_of_ion_cyclotron_heating_antenna	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	gap_at_outboard_midplane	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	gap_at_plasma_boundary	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	gap_of_antenna_strap	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	gas_flow	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	hard_xray_brightness	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	hard_xray_emissivity	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	height_of_poloidal_field_coil	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	hot_neutral_fraction	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	hot_neutral_temperature	name_stage=reviewed	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	initial_polarization_ellipticity_of_polarimeter_beam	name_stage=accepted	docs_stage=exhausted	excluded=false	docs_eligible=true
WEST	inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width	name_stage=drafted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	ion_atomic_number	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	ion_current_of_mass_spectrometer_channel	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	launched_power_of_lower_hybrid_antenna	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	length_of_interferometer_beam	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	length_of_plasma_boundary	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	length_of_poloidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	length_of_toroidal_magnetic_field_probe	name_stage=accepted	docs_stage=reviewed	excluded=false	docs_eligible=true
WEST	line_averaged_effective_charge	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	line_averaged_electron_density	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	line_integrated_electron_number_density	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	line_integrated_spectral_wave_opacity_at_ece_channel_emission_position	name_stage=drafted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	loop_voltage_at_plasma_boundary	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	lower_bound_hard_xray_peak_width	name_stage=superseded	docs_stage=pending	excluded=false	docs_eligible=true
WEST	lower_photon_energy	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	lower_triangularity_of_flux_surface	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	lower_triangularity_of_plasma_boundary	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	lower_wavelength_of_filter	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	magnetic_shear_at_flux_surface	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	maximum_magnetic_field_magnitude	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	maximum_of_energy_flux_at_divertor_target	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	mhd_energy	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	minimum_magnetic_field	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	minimum_safety_factor	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	minor_radius_of_plasma_boundary	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	neutral_pressure	name_stage=exhausted	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	neutron_flux	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	normalized_plasma_internal_inductance	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	normalized_poloidal_flux_coordinate	name_stage=reviewed	docs_stage=pending	excluded=false	docs_eligible=true
WEST	normalized_poloidal_flux_coordinate_of_plasma_boundary	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	normalized_toroidal_flux_coordinate	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	normalized_toroidal_flux_coordinate_at_measurement_position	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	normalized_toroidal_flux_coordinate_at_minimum_safety_factor	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	normalized_toroidal_hard_xray_peak_lower_bound_width	name_stage=exhausted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	normalized_toroidal_plasma_beta	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive	name_stage=accepted	docs_stage=reviewed	excluded=false	docs_eligible=true
WEST	photon_radiance_at_spectral_line	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	plasma_beta	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	plasma_breakdown_time	name_stage=exhausted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	plasma_current	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	plasma_pressure	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	poloidal_angle_of_flux_surface	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	poloidal_angle_of_measurement_position	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	poloidal_angle_of_toroidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	poloidal_beta	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	poloidal_cross_sectional_area_of_flux_surface	name_stage=superseded	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	poloidal_cross_sectional_area_of_plasma_boundary	name_stage=superseded	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	poloidal_magnetic_field	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	poloidal_magnetic_field_at_constraint_position	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	poloidal_magnetic_flux_at_flux_surface	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	poloidal_magnetic_flux_at_magnetic_axis	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	poloidal_magnetic_flux_at_measurement_position	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	poloidal_magnetic_flux_at_plasma_boundary	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	poloidal_magnetic_flux_of_flux_loop	name_stage=accepted	docs_stage=drafted	excluded=false	docs_eligible=true
WEST	power_due_to_ion_cyclotron_heating	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	power_due_to_ohmic_dissipation	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	power_of_ion_cyclotron_heating_antenna	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	power_of_soft_xray_detector	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	pressure_of_ion_cyclotron_heating_antenna	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	product_of_poloidal_current_function_and_derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_poloidal_current_function	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	pulse_duration	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_at_inboard_midplane	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_at_outboard_midplane	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_camera	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_conductor_cross_section	name_stage=accepted	docs_stage=reviewed	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_detector_pixel	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_flux_loop	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_geometric_axis	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_line_of_sight	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_magnetic_axis	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_measurement_position	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_poloidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_strike_point	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_coordinate_of_x_point	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_derivative_of_poloidal_magnetic_flux	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_outline_of_antenna_strap	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_outline_of_limiter_tile	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_outline_of_plasma_boundary	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radial_outline_of_wall	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radiated_power_over_core_region	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	radiative_temperature	name_stage=exhausted	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	radiative_temperature_at_magnetic_axis	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	ratio_of_coolant_mass_to_time	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	ratio_of_line_averaged_electron_density_to_greenwald_density	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	ratio_of_line_averaged_hydrogen_density_to_line_averaged_total_hydrogenic_density	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density_and_neutral_density_of_isotope	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	reference_major_radius	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	reflected_phase_of_ion_cyclotron_heating_antenna	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	reflected_power_of_ion_cyclotron_heating_antenna	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	safety_factor	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	safety_factor_at_magnetic_axis	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	safety_factor_at_normalized_poloidal_magnetic_flux_equal_to_0_95	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	spectral_bremsstrahlung_radiance	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	spectral_calibration_factor_at_line_of_sight	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	spectral_flux_of_spectrometer_channel	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	spectral_radiance	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	spectral_radiance_of_soft_xray_detector	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	spectral_signal_to_noise_ratio_of_spectrometer_channel	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	spectral_wavelength_of_optical_element	name_stage=accepted	docs_stage=reviewed	excluded=false	docs_eligible=true
WEST	surface_area_of_flux_surface	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	surface_temperature	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	temperature_of_soft_xray_detector	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	thermal_electron_density	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	thermal_electron_pressure_at_post_sawtooth_crash	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	thermal_energy_of_plant_component_port	name_stage=exhausted	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	thermal_power_of_plant_component_port	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	thickness_of_filter	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_angle_of_antenna_strap	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_angle_of_magnetic_field_probe	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	toroidal_angle_of_measurement_position	name_stage=accepted	docs_stage=drafted	excluded=false	docs_eligible=true
WEST	toroidal_angle_of_toroidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_angular_width_of_limiter_tile	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_coordinate_of_detector_pixel	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_coordinate_of_line_of_sight	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_flux_coordinate	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_flux_coordinate_gradient_magnitude	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_flux_surface_averaged_current_density	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_magnetic_field	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_magnetic_field_at_magnetic_axis	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_magnetic_flux	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_magnetic_flux_due_to_diamagnetic_drift	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	toroidal_vacuum_magnetic_field	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	toroidal_width_of_antenna_strap	name_stage=exhausted	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	total_electron_density	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	total_electron_pressure	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	total_external_heating_power	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	total_neutral_source_rate_due_to_gas_injection	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	total_plasma_radiated_power	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	total_power_at_separatrix	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	total_power_due_to_ion_cyclotron_heating	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	triangularity_of_plasma_boundary	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	turn_count_of_poloidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	turn_count_of_toroidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	upper_photon_energy	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	upper_triangularity_of_flux_surface	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	upper_triangularity_of_plasma_boundary	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	upper_wavelength_of_filter	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vacuum_poloidal_current_function	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_camera	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_conductor_cross_section	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_detector_pixel	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_flux_loop	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_geometric_axis	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_line_of_sight	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_magnetic_axis	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_measurement_position	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_plasma_boundary	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	vertical_coordinate_of_poloidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_primary_x_point	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_strike_point	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_coordinate_of_toroidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_outline_of_antenna_strap	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_outline_of_limiter_tile	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	vertical_outline_of_wall_material	name_stage=superseded	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	voltage_amplitude	name_stage=exhausted	docs_stage=accepted	excluded=true	docs_eligible=false
WEST	voltage_of_mass_spectrometer_channel	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	voltage_of_poloidal_magnetic_field_probe	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	volume_averaged_electron_density	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	volume_integrated_total_electron_density	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	volume_of_flux_surface	name_stage=accepted	docs_stage=pending	excluded=false	docs_eligible=true
WEST	volume_of_plasma_boundary	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	wave_current_of_antenna_strap_amplitude	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	wave_phase_of_antenna_strap	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	wave_phase_of_ion_cyclotron_heating_antenna	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	wave_phase_of_wave_beam	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	wavelength_of_spectral_line	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	wavelength_of_wave_beam	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
WEST	width_of_poloidal_field_coil	name_stage=accepted	docs_stage=accepted	excluded=false	docs_eligible=true
```

