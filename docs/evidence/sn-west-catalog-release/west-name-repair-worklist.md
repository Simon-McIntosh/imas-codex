# WEST batch name repairs — one deduplicated worklist

provisional: true — rows are appended as they are compiled; the closing pass rewrites this line.

**61 repairs close 74 defective bindings.** This file aggregates and reconciles the seven landed
records of the WEST cohort; it judges nothing afresh and adds no defect the records do not contain.
No graph query was issued and no database was opened — every fact below is already in the seven
files.

## What the seven records supply

| Record | Draw | Rows judged | INCORRECT |
| --- | --- | --- | --- |
| [`west-name-audit.md`](west-name-audit) | every fourth, path-ordered | 86 | 10 |
| [`…-bremsstrahlung-to-equilibrium-boundary.md`](west-name-audit-bremsstrahlung-to-equilibrium-boundary) | contiguous, remainder 1–51 | 51 | 11 |
| [`…-equilibrium-boundary-to-profiles.md`](west-name-audit-equilibrium-boundary-to-profiles) | contiguous, remainder 52–102 | 51 | 8 |
| [`…-equilibrium-profiles-to-magnetics.md`](west-name-audit-equilibrium-profiles-to-magnetics) | contiguous, remainder 103–153 | 51 | 17 |
| [`…-magnetics-to-spectrometer-visible.md`](west-name-audit-magnetics-to-spectrometer-visible) | contiguous, remainder 154–204 | 51 | 14 |
| [`…-spectrometer-visible-to-wall.md`](west-name-audit-spectrometer-visible-to-wall) | contiguous, remainder 205–255 | 51 | 14 |
| **Totals** | | **341** | **74** |
| [`west-name-shared-identities.md`](west-name-shared-identities) | the 53 identities bound to more than one path | 53 groups, 164 bindings | 10 MUST-SPLIT over 24 bindings |

The 341 accepted bindings comprise **340 distinct name-and-path pairs**: the boundary-to-profiles
record's rows 67 and 68 are byte-identical in every field but `index`, the only duplicated pair in
the cohort. It is a graph-hygiene repair invisible to a catalog reader and is not counted as a name
defect below.

## How 74 bindings become 61 repairs

The five range records judge **bindings**; a repair acts on an **identity**. Three collapses take
the count from 74 to 61, and each is a case where one repair closes several rows:

1. **Nine identities are rejected on more than one binding** — 74 rows over 61 distinct identities,
   so 13 rows are second and third sightings of a defect already listed. Where two records reject
   the same identity, this file emits **one** row naming both records.
2. **Ten of those identities are MUST-SPLIT groups.** Where a split group and a range record
   describe the same defect, this file emits **one** row for the split rather than one per axis or
   per binding — so `radial_coordinate_of_strike_point`, rejected once by the every-fourth audit on
   the outer leg and once by the wall record on the inner, is one repair covering both.
3. **No identity is rejected for two different reasons.** Every one of the four multi-row rename
   identities carries the same proposed spelling in every record that rejects it, checked
   mechanically: `faraday_angle`, `line_integrated_electron_number_density`, `mhd_energy` and
   `upper_photon_energy` each appear in two or three records and each has exactly one proposal.

**61 = 51 renames + 10 splits.** The split is the number that sizes publication risk: a rename is a
relabelling a reader could in principle compensate for, and **a split cannot be performed by a
reader at all**, because the published name resolves to two different physical objects.

## Part one — 51 renames, one identity each

Every row here is closed by renaming one identity. The rejected and proposed spellings are read
from each record's verdict body, not pattern-matched on a format: the bremsstrahlung record states
the pair in a two-column table and the other five state it in prose, and both forms are read.
Every one of the 51 carries an explicit proposal in its source record — **no row here required a
spelling to be invented, and none is recorded as missing one.**

| rejected | proposed | source path(s) | record and row | class |
| --- | --- | --- | --- | --- |
| `accumulated_total_gas_count` | `accumulated_total_neutral_particle_count_due_to_gas_injection` | `summary/gas_injection_accumulated/total/value` | spec-visible→wall 219 | not self-descriptive |
| `area_of_diagnostic_aperture` | `area_of_diagnostic_detector` | `hard_x_rays/channel/detector/surface` | audit 37 | bound to the wrong object |
| `area_of_poloidal_magnetic_field_probe` | `turn_area_of_poloidal_magnetic_field_probe` | `magnetics/b_field_pol_probe/area` | eq-profiles→magnetics 153 | asserts more than the data supports |
| `area_of_toroidal_magnetic_field_probe` | `turn_area_of_toroidal_magnetic_field_probe` | `magnetics/b_field_phi_probe/area` | eq-profiles→magnetics 146 | asserts more than the data supports |
| `capacitance_of_ion_cyclotron_heating_antenna` | `capacitance_of_impedance_matching_element` | `ic_antennas/antenna/module/matching_element/capacitance` | eq-profiles→magnetics 122 | bound to the wrong object |
| `coolant_transit_time_of_plant_component_port` | `coolant_transit_time_of_calorimetry_component` | `calorimetry/group/component/transit_time` | brems→eq-boundary 15 | bound to the wrong object |
| `effective_turn_count_of_passive_loop` | `effective_turn_count_of_passive_loop_element` | `pf_passive/loop/element/turns_with_sign` | magnetics→spec-visible 169 | bound to the wrong object |
| `energy_confinement_enhancement_factor` | `ipb98y2_confinement_enhancement_factor` | `summary/global_quantities/h_98/value` | spec-visible→wall 225 | not self-descriptive |
| `faraday_angle` | `faraday_rotation_angle` | `equilibrium/time_slice/constraints/faraday_angle/reconstructed` <br> `equilibrium/time_slice/constraints/faraday_angle/measured` <br> `polarimeter/channel/faraday_angle` | audit 21, eq-boundary→profiles 60, magnetics→spec-visible 171 | not self-descriptive |
| `gap_at_outboard_midplane` | `radial_separation_of_inner_and_outer_separatrices_at_outboard_midplane` | `summary/boundary/distance_inner_outer_separatrices/value` | spec-visible→wall 206 | bound to the wrong object |
| `hard_xray_brightness` | `hard_xray_photon_radiance` | `hard_x_rays/channel/radiance` | audit 39 | minority spelling of a base the cohort already fixes |
| `hard_xray_emissivity` | `hard_xray_photon_emissivity` | `hard_x_rays/emissivity_profile_1d/emissivity` | eq-profiles→magnetics 115 | not self-descriptive |
| `height_of_poloidal_field_coil` | `height_of_conductor_cross_section` | `pf_active/coil/element/geometry/rectangle/height` | magnetics→spec-visible 165 | bound to the wrong object |
| `hot_neutral_temperature_at_plasma_boundary` | `hot_neutral_temperature` | `spectrometer_visible/channel/isotope_ratios/isotope/hot_neutrals_temperature` | magnetics→spec-visible 200 | asserts more than the data supports |
| `line_integrated_electron_number_density` | `line_integrated_electron_density` | `equilibrium/time_slice/constraints/n_e_line/reconstructed` <br> `equilibrium/time_slice/constraints/n_e_line/measured` <br> `interferometer/channel/n_e_line` | audit 22, eq-boundary→profiles 63, eq-profiles→magnetics 140 | minority spelling of a base the cohort already fixes |
| `mhd_energy` | `total_plasma_stored_energy` | `equilibrium/time_slice/global_quantities/energy_mhd` <br> `summary/global_quantities/energy_mhd` | eq-boundary→profiles 69, spec-visible→wall 223 | not self-descriptive |
| `minimum_safety_factor` | `minimum_absolute_safety_factor` | `equilibrium/time_slice/global_quantities/q_min/value` | eq-boundary→profiles 78 | not self-descriptive |
| `opacity_at_ece_channel_emission_position` | `optical_depth_at_ece_channel_emission_position` | `ece/channel/optical_depth` | brems→eq-boundary 43 | bound to the wrong object |
| `poloidal_angle_of_flux_surface` | `poloidal_orientation_angle_of_poloidal_magnetic_field_probe` | `magnetics/b_field_pol_probe/poloidal_angle` | magnetics→spec-visible 155 | bound to the wrong object |
| `power_of_soft_xray_detector` | `incident_power_of_soft_xray_detector` | `soft_x_rays/channel/power` | magnetics→spec-visible 187 | not self-descriptive |
| `pressure_of_ion_cyclotron_heating_antenna` | `pressure_amplitude_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/module/pressure/amplitude` | eq-profiles→magnetics 126 | not self-descriptive |
| `pulse_duration` | `confined_plasma_duration` | `summary/plasma_duration/value` | spec-visible→wall 249 | asserts more than the data supports |
| `radial_coordinate_at_inboard_midplane` | `radial_coordinate_of_flux_surface_at_inboard_midplane` | `equilibrium/time_slice/profiles_1d/r_inboard` | eq-boundary→profiles 100 | not self-descriptive |
| `radial_coordinate_at_outboard_midplane` | `radial_coordinate_of_flux_surface_at_outboard_midplane` | `equilibrium/time_slice/profiles_1d/r_outboard` | eq-boundary→profiles 101 | not self-descriptive |
| `radial_coordinate_of_x_point` | `radial_coordinate_of_primary_x_point` | `summary/boundary/x_point_main/r` | audit 73 | bound to the wrong object |
| `radial_derivative_of_poloidal_magnetic_flux` | `derivative_of_poloidal_magnetic_flux_with_respect_to_toroidal_flux_coordinate` | `equilibrium/time_slice/profiles_1d/dpsi_drho_tor` | audit 29 | not self-descriptive |
| `radial_outline_of_wall` | `radial_outline_of_plasma_facing_component` | `wall/description_2d/mobile/unit/outline/r` | spec-visible→wall 255 | minority spelling of a base the cohort already fixes |
| `radiated_power_over_core_region` | `radiated_power_inside_plasma_boundary` | `summary/global_quantities/power_radiated_inside_lcfs/value` | spec-visible→wall 229 | asserts more than the data supports |
| `radiative_temperature_at_magnetic_axis` | `radiative_temperature_at_innermost_ece_channel` | `ece/t_radiation_central` | brems→eq-boundary 49 | asserts more than the data supports |
| `ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density_and_neutral_density_of_isotope` | `ratio_of_neutral_density_of_isotope_to_neutral_density_of_other_isotopes` | `spectrometer_visible/channel/isotope_ratios/isotope/density_ratio` | magnetics→spec-visible 196 | asserts more than the data supports |
| `spectral_calibration_factor_at_line_of_sight` | `phase_to_line_integrated_electron_density_conversion_factor` | `interferometer/channel/wavelength/phase_to_n_e_line` | eq-profiles→magnetics 143 | not self-descriptive |
| `spectral_rate_of_spectrometer_channel` | `photoelectron_rate_of_spectrometer_channel` | `spectrometer_visible/channel/grating_spectrometer/intensity_spectrum` | magnetics→spec-visible 190 | not self-descriptive |
| `spectral_wavelength_of_optical_element` | `spectral_wavelength_of_spectrometer_channel` | `spectrometer_visible/channel/grating_spectrometer/wavelengths` | magnetics→spec-visible 194 | bound to the wrong object |
| `surface_temperature` | `apparent_surface_temperature` | `camera_ir/channel/camera/frame/apparent_temperature` | audit 6 | asserts more than the data supports |
| `temperature_of_soft_xray_detector` | `temperature_of_x_ray_detector` | `camera_x_rays/detector_temperature` | brems→eq-boundary 28 | asserts more than the data supports |
| `thermal_electron_pressure_at_post_sawtooth_crash` | `thermal_electron_pressure` | `core_profiles/profiles_1d/electrons/pressure_thermal` | brems→eq-boundary 33 | asserts more than the data supports |
| `toroidal_angle_of_antenna_strap` | `toroidal_angle_of_antenna_strap_outline` | `ic_antennas/antenna/module/strap/outline/phi` | eq-profiles→magnetics 127 | not self-descriptive |
| `toroidal_angle_of_poloidal_magnetic_field_probe` | `toroidal_orientation_angle_of_poloidal_magnetic_field_probe` | `magnetics/b_field_pol_probe/toroidal_angle` | magnetics→spec-visible 158 | not self-descriptive |
| `toroidal_angular_width_of_limiter_tile` | `toroidal_angular_centre_and_full_width_of_limiter_tile` | `wall/description_2d/limiter/unit/phi_extensions` | spec-visible→wall 254 | one name, two physically different quantities |
| `toroidal_coordinate_at_detector_pixel` | `toroidal_coordinate_of_detector_pixel` | `camera_x_rays/camera/pixel_position/phi` | brems→eq-boundary 25 | minority spelling of a base the cohort already fixes |
| `toroidal_vacuum_magnetic_field` | `toroidal_vacuum_magnetic_field_at_reference_major_radius` | `equilibrium/vacuum_toroidal_field/b0` | eq-profiles→magnetics 106 | asserts more than the data supports |
| `total_power_due_to_ion_cyclotron_heating` | `total_coupled_power_due_to_ion_cyclotron_heating` | `summary/heating_current_drive/power_ic/value` | audit 80 | asserts more than the data supports |
| `upper_photon_energy` | `upper_bound_photon_energy` | `hard_x_rays/channel/energy_band/upper_bound` <br> `hard_x_rays/emissivity_profile_1d/upper_bound` <br> `soft_x_rays/channel/energy_band/upper_bound` | eq-profiles→magnetics 110, eq-profiles→magnetics 120, magnetics→spec-visible 181 | minority spelling of a base the cohort already fixes |
| `vertical_coordinate_of_ece_channel` | `vertical_coordinate_of_measurement_position` | `ece/channel/position/z` | brems→eq-boundary 48 | bound to the wrong object |
| `vertical_coordinate_of_measurement_position` | `vertical_coordinate_of_aperture` | `camera_x_rays/aperture/centre/z` | brems→eq-boundary 18 | bound to the wrong object |
| `voltage_of_mass_spectrometer_channel` | `photomultiplier_voltage_of_mass_spectrometer_channel` | `spectrometer_mass/channel/photomultiplier_voltage` | audit 64 | not self-descriptive |
| `volume_of_flux_surface` | `volume_of_plasma_boundary` | `equilibrium/time_slice/global_quantities/volume` | eq-boundary→profiles 79 | bound to the wrong object |
| `wave_current_amplitude_of_antenna_strap` | `wave_current_amplitude_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/module/current/amplitude` | eq-profiles→magnetics 121 | bound to the wrong object |
| `wave_phase_of_ion_cyclotron_heating_antenna` | `voltage_phase_of_ion_cyclotron_heating_antenna` | `ic_antennas/antenna/module/voltage/phase` | eq-profiles→magnetics 132 | not self-descriptive |
| `wave_phase_of_wave_beam` | `fringe_jump_corrected_phase_of_interferometer_beam` | `interferometer/channel/wavelength/phase_corrected` | eq-profiles→magnetics 142 | not self-descriptive |
| `width_of_poloidal_field_coil` | `width_of_conductor_cross_section` | `pf_active/coil/element/geometry/rectangle/width` | magnetics→spec-visible 166 | bound to the wrong object |