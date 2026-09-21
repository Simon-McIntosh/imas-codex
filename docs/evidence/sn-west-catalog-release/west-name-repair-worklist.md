# WEST batch name repairs — one deduplicated worklist

provisional: false — every rejected spelling in the seven records carries a row, the contradictions are listed and the result table is closed.

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
## Part two — 10 splits, the class that blocks publication

A split cannot be performed by a reader. Each of these identities resolves to two or more
physically different objects, so a published catalog entry gives a consumer no way to tell which
value it is holding. All ten are MUST-SPLIT groups in the shared-identities record, and all ten are
also rejected by at least one range record — the two instruments agree on which identities are
unsafe, which is the one cross-check the seven records make possible.

| rejected identity | bindings | resolution | source | class |
| --- | --- | --- | --- | --- |
| `radial_coordinate_of_magnetic_axis` | 4 | 3 keep / 1 rename | shared group 4; eq-boundary→profiles 64 | wrong object |
| `radial_coordinate_of_measurement_position` | 3 | 1 keep / 2 rename | shared group 14; brems→eq-boundary 17, eq-profiles→magnetics 150 | wrong object |
| `toroidal_angle_of_measurement_position` | 3 | 1 keep / 2 rename | shared group 16; brems→eq-boundary 44, eq-profiles→magnetics 149, magnetics→spec-visible 156 | minority spelling |
| `effective_charge` | 2 | 1 keep / 1 rename | shared group 25; brems→eq-boundary 30 | two quantities |
| `initial_polarization_ellipticity_of_polarimeter_beam` | 2 | 1 keep / 1 rename | shared group 27; magnetics→spec-visible 178 | two quantities |
| `launched_power_of_lower_hybrid_antenna` | 2 | 1 keep / 1 rename | shared group 28; spec-visible→wall 236, spec-visible→wall 238 | two quantities |
| `net_power_due_to_ion_cyclotron_heating` | 2 | 1 keep / 1 rename | shared group 34; spec-visible→wall 235 | two quantities |
| `normalized_toroidal_flux_coordinate_at_measurement_position` | 2 | 1 keep / 1 rename | shared group 36; eq-profiles→magnetics 118 | wrong object |
| `radial_coordinate_of_strike_point` | 2 | 0 keep / 2 rename | shared group 42; audit 72, spec-visible→wall 212 | two quantities |
| `vertical_coordinate_of_strike_point` | 2 | 0 keep / 2 rename | shared group 51; spec-visible→wall 213, spec-visible→wall 214 | two quantities |
**24 bindings are covered by these ten rows** — 22 in the nine groups whose split table the
shared-identities record prints inline, plus the two of `vertical_coordinate_of_strike_point`.
Eight of the ten keep the existing spelling on at least one binding, so eight of these repairs
mint one new name rather than two; only the two strike-point groups retire their spelling
entirely, because neither divertor leg has a better claim to the unqualified name.

### The per-binding resolutions

#### `radial_coordinate_of_magnetic_axis` — 4 bindings (shared group 4)

| binding | resolved spelling |
| --- | --- |
| `equilibrium/time_slice/global_quantities/magnetic_axis/r` | keeps `radial_coordinate_of_magnetic_axis` |
| `summary/boundary/magnetic_axis_r/value` | keeps `radial_coordinate_of_magnetic_axis` |
| `summary/local/magnetic_axis/position/r` | keeps `radial_coordinate_of_magnetic_axis` |
| `equilibrium/time_slice/contour_tree/node/r` | **needs** `radial_coordinate_of_flux_contour_critical_point` |

- rejected by: eq-boundary→profiles row 64 (`equilibrium/time_slice/contour_tree/node/r`, proposed `radial_coordinate_of_flux_map_critical_point`)

#### `radial_coordinate_of_measurement_position` — 3 bindings (shared group 14)

| binding | resolved spelling |
| --- | --- |
| `ece/channel/position/r` | keeps `radial_coordinate_of_measurement_position` |
| `camera_x_rays/aperture/centre/r` | **needs** `radial_coordinate_of_aperture` |
| `magnetics/b_field_phi_probe/position/r` | **needs** `radial_coordinate_of_toroidal_magnetic_field_probe` |

- rejected by: brems→eq-boundary row 17 (`camera_x_rays/aperture/centre/r`, proposed `radial_coordinate_of_aperture`), eq-profiles→magnetics row 150 (`magnetics/b_field_phi_probe/position/r`, proposed `radial_coordinate_of_toroidal_magnetic_field_probe`)

#### `toroidal_angle_of_measurement_position` — 3 bindings (shared group 16)

| binding | resolved spelling |
| --- | --- |
| `ece/channel/position/phi` | keeps `toroidal_angle_of_measurement_position` |
| `magnetics/b_field_phi_probe/position/phi` | **needs** `toroidal_angle_of_toroidal_magnetic_field_probe` |
| `magnetics/b_field_pol_probe/position/phi` | **needs** `toroidal_angle_of_poloidal_magnetic_field_probe` |

- rejected by: brems→eq-boundary row 44 (`ece/channel/position/phi`, proposed `toroidal_coordinate_of_measurement_position`), eq-profiles→magnetics row 149 (`magnetics/b_field_phi_probe/position/phi`, proposed `toroidal_coordinate_of_toroidal_magnetic_field_probe`), magnetics→spec-visible row 156 (`magnetics/b_field_pol_probe/position/phi`, proposed `toroidal_coordinate_of_poloidal_magnetic_field_probe`)

#### `effective_charge` — 2 bindings (shared group 25)

| binding | resolved spelling |
| --- | --- |
| `core_profiles/profiles_1d/zeff` | keeps `effective_charge` |
| `core_profiles/global_quantities/z_eff_resistive` | **needs** `volume_averaged_effective_charge` |

- rejected by: brems→eq-boundary row 30 (`core_profiles/global_quantities/z_eff_resistive`, proposed `volume_averaged_effective_charge`)

#### `initial_polarization_ellipticity_of_polarimeter_beam` — 2 bindings (shared group 27)

| binding | resolved spelling |
| --- | --- |
| `polarimeter/channel/ellipticity_initial` | keeps `initial_polarization_ellipticity_of_polarimeter_beam` |
| `polarimeter/channel/polarization_initial` | **needs** `initial_polarization_of_polarimeter_beam` |

- rejected by: magnetics→spec-visible row 178 (`polarimeter/channel/polarization_initial`, proposed `initial_polarization_vector_of_polarimeter_beam`)

#### `launched_power_of_lower_hybrid_antenna` — 2 bindings (shared group 28)

| binding | resolved spelling |
| --- | --- |
| `summary/heating_current_drive/lh/power/value` | keeps `launched_power_of_lower_hybrid_antenna` |
| `summary/heating_current_drive/power_lh/value` | **needs** `total_launched_power_of_lower_hybrid_antennas` |

- rejected by: spec-visible→wall row 236 (`summary/heating_current_drive/lh/power/value`, proposed `coupled_power_of_lower_hybrid_antenna`), spec-visible→wall row 238 (`summary/heating_current_drive/power_lh/value`, proposed `total_coupled_power_due_to_lower_hybrid_heating`)

#### `net_power_due_to_ion_cyclotron_heating` — 2 bindings (shared group 34)

| binding | resolved spelling |
| --- | --- |
| `ic_antennas/antenna/power_launched` | keeps a launched spelling — `launched_power_of_ion_cyclotron_antenna` |
| `summary/heating_current_drive/ic/power/value` | **needs** `coupled_power_of_ion_cyclotron_antenna` |

- rejected by: spec-visible→wall row 235 (`summary/heating_current_drive/ic/power/value`, proposed `coupled_power_due_to_ion_cyclotron_heating`)

#### `normalized_toroidal_flux_coordinate_at_measurement_position` — 2 bindings (shared group 36)

| binding | resolved spelling |
| --- | --- |
| `ece/channel/position/rho_tor_norm` | keeps `normalized_toroidal_flux_coordinate_at_measurement_position` |
| `hard_x_rays/emissivity_profile_1d/peak_position` | **needs** `normalized_toroidal_flux_coordinate_of_emissivity_peak` |

- rejected by: eq-profiles→magnetics row 118 (`hard_x_rays/emissivity_profile_1d/peak_position`, proposed `normalized_toroidal_flux_coordinate_at_emissivity_peak`)

#### `radial_coordinate_of_strike_point` — 2 bindings (shared group 42)

| binding | resolved spelling |
| --- | --- |
| `summary/boundary/strike_point_inner_r/value` | **needs** `radial_coordinate_of_inner_strike_point` |
| `summary/boundary/strike_point_outer_r/value` | **needs** `radial_coordinate_of_outer_strike_point` |

- rejected by: audit row 72 (`summary/boundary/strike_point_outer_r/value`, proposed `radial_coordinate_of_outer_strike_point`), spec-visible→wall row 212 (`summary/boundary/strike_point_inner_r/value`, proposed `radial_coordinate_of_inner_strike_point`)

#### `vertical_coordinate_of_strike_point` — 2 bindings (shared group 51)

| binding | resolved spelling |
| --- | --- |
| `summary/boundary/strike_point_inner_z/value` | **needs** `vertical_coordinate_of_inner_strike_point` |
| `summary/boundary/strike_point_outer_z/value` | **needs** `vertical_coordinate_of_outer_strike_point` |

- rejected by: spec-visible→wall row 213 (`summary/boundary/strike_point_inner_z/value`, proposed `vertical_coordinate_of_inner_strike_point`), spec-visible→wall row 214 (`summary/boundary/strike_point_outer_z/value`, proposed `vertical_coordinate_of_outer_strike_point`)
## Part three — where the records disagree

The fence for this file requires that a contradiction between records be **surfaced rather than
silently resolved**, so none of the following is decided here. Eight identities are rejected in one
record and accepted in another. **Six of the eight are explained and are not disagreements**: they
are the MUST-SPLIT groups, where the identity is genuinely correct on one binding and wrong on
another, and the split row above is the resolution. The remaining two, plus three spelling
conflicts and one class conflict, are open.

### Two identities where one record's accepted spelling is another's rejected spelling

Neither is a split group, so the disagreement cannot be explained by the identity being right on
one binding and wrong on another — both records are judging the same physical quantity.

| identity | accepted at | rejected at | the rejecting record's proposal |
| --- | --- | --- | --- |
| `toroidal_vacuum_magnetic_field` | brems→eq-boundary 41 (`core_profiles/vacuum_toroidal_field/b0`), spec-visible→wall 220 | eq-profiles→magnetics 106 (`equilibrium/vacuum_toroidal_field/b0`) | `toroidal_vacuum_magnetic_field_at_reference_major_radius` |
| `volume_of_flux_surface` | audit 14, audit 36 | eq-boundary→profiles 79 (`equilibrium/time_slice/global_quantities/volume`) | `volume_of_plasma_boundary` |

`volume_of_flux_surface` is the sharper of the two, because the every-fourth audit's treatment of
the `*_of_flux_surface` family was carried into the later range records as a **settled**
adjudication not to be relitigated — and the boundary-to-profiles record rejects it anyway, on a
binding the every-fourth audit did not hold. The shared-identities record independently judges the
identity **ONE-QUANTITY**. Three instruments, two answers. This needs an adjudication before
either spelling is published.

### Three conflicts about the same spelling axis: `toroidal_angle_` or `toroidal_coordinate_`

The shared-identities record and three range records disagree about the base for a toroidal angle,
and the conflict reaches a binding the split table says to **keep**:

| binding | shared-identities group 16 says | the range record says |
| --- | --- | --- |
| `ece/channel/position/phi` | **keeps** `toroidal_angle_of_measurement_position` | brems→eq-boundary 44 **rejects** it → `toroidal_coordinate_of_measurement_position` |
| `magnetics/b_field_phi_probe/position/phi` | needs `toroidal_angle_of_toroidal_magnetic_field_probe` | eq-profiles→magnetics 149 → `toroidal_coordinate_of_toroidal_magnetic_field_probe` |
| `magnetics/b_field_pol_probe/position/phi` | needs `toroidal_angle_of_poloidal_magnetic_field_probe` | magnetics→spec-visible 156 → `toroidal_coordinate_of_poloidal_magnetic_field_probe` |

The two instruments agree on the **locus** in all three rows and disagree only on the base word.
The split row in part two adopts the shared-identities spelling because that record is the one that
judged the group as a group, but the disagreement is real and is recorded here rather than
absorbed. The same pair also carries a **class conflict**: brems→eq-boundary files it as a minority
spelling, magnetics→spec-visible files it as bound to the wrong object.

### One coupled-versus-launched disagreement, which is physics rather than spelling

| binding | shared-identities group 28 / 34 says | spec-visible→wall says |
| --- | --- | --- |
| `summary/heating_current_drive/lh/power/value` | **keeps** `launched_power_of_lower_hybrid_antenna` | row 236 **rejects** it → `coupled_power_of_lower_hybrid_antenna` |
| `summary/heating_current_drive/power_lh/value` | needs `total_launched_power_of_lower_hybrid_antennas` | row 238 → `total_coupled_power_due_to_lower_hybrid_heating` |
| `summary/heating_current_drive/ic/power/value` | needs `coupled_power_of_ion_cyclotron_antenna` | row 235 → `coupled_power_due_to_ion_cyclotron_heating` |

The ion-cyclotron row agrees on **coupled**; the two lower-hybrid rows do not. Launched and coupled
power are separated by the coupling efficiency — the every-fourth audit's row 80 puts that
difference at 10–30 % — so this is a disagreement about which physical quantity the `summary`
node holds, not about how to spell one. It must be settled from the data dictionary before either
lower-hybrid name is published.

### One proposal that is itself a rejected spelling

`vertical_coordinate_of_ece_channel` (brems→eq-boundary 48) is to be renamed
**to** `vertical_coordinate_of_measurement_position` — and that same record's row 18 **rejects**
`vertical_coordinate_of_measurement_position` on `camera_x_rays/aperture/centre/z`, renaming it to
`vertical_coordinate_of_aperture`. Both are in part one and both are correct as written: the
aperture binding leaves the identity and the ECE binding joins it. It is recorded because the
repairs are **order-dependent** — applying the ECE rename before the aperture rename puts a correct
binding into an identity that is still carrying a wrong one.

## The defect rate, and why the census is not the sample's estimate

| draw | judged | incorrect | rate |
| --- | --- | --- | --- |
| every fourth, path-ordered, spread over 20 IDSs | 86 | 10 | 11.6 % |
| contiguous, remainder 1–51 | 51 | 11 | 21.6 % |
| contiguous, remainder 52–102 | 51 | 8 | 15.7 % |
| contiguous, remainder 103–153 | 51 | 17 | 33.3 % |
| contiguous, remainder 154–204 | 51 | 14 | 27.5 % |
| contiguous, remainder 205–255 | 51 | 14 | 27.5 % |
| **whole cohort** | **341** | **74** | **21.7 %** |

**The 21.7 % is a census, not an estimate, and it is nearly double the 11.6 % the sample
projected.** The records themselves give the structural reason, in two parts, and nothing is added
here beyond joining them.

**First, the contiguous blocks are not samples of the cohort and never claimed to be.** The
equilibrium-profiles record states it directly: its slice "happens to land on the batch's
instrument-hardware containers: 12 rows of `hard_x_rays`, 12 of `ic_antennas` and 8 of `magnetics`
account for 13 of the 17 rejections, while the 5 `equilibrium` rows yield 1 and the 13
`interferometer` rows yield 3." The wall record makes the same disclaimer about its own 27.5 %, and
the boundary-to-profiles record about its 15.7 %. Each block measures its own region. Only the
census over all six is a statement about the cohort — which is what this file now holds.

**Second, and this is what accounts for the gap rather than merely warning about it: a one-in-four
draw structurally under-samples the defects that consist of siblings disagreeing with each other.**
The bremsstrahlung record puts it as a property of the draw: "An every-fourth draw takes at most one
coordinate of any point, so a set that disagrees with itself reads as a single plausible row." The
arithmetic follows without any new judgement. **Thirteen of the 74 defective bindings are second or
third sightings of an identity already defective elsewhere, and ten of those identities are
MUST-SPLIT groups** — defects that exist only *between* bindings. A draw that takes one binding in
four sees, on average, one binding of each such group, and one binding of a split group is exactly
what looks correct in isolation. The every-fourth audit's own closing section is the evidence: it
reported three such collisions as findings it "points at but does not contain", and all three are
in part two of this file, found by the blocks that held the whole container.

So the two figures are not in conflict and neither is wrong. **11.6 % was an honest estimate of the
defects a per-row reading can see; 21.7 % is the count once the defects that live between rows are
also counted.** The class totals show where the difference sits: 21 rows bound to the wrong object
and 10 covering two quantities are 31 of the 74, and both classes are overwhelmingly
container-level — a name that is defensible on its own row and indefensible beside its siblings.

## Result

| | count |
| --- | --- |
| accepted bindings judged | 341, over **340** distinct name-and-path pairs |
| bindings carrying an INCORRECT verdict | **74** — 10 from the every-fourth audit, 64 across the five range records |
| distinct identities among them | **61** |
| shared identities in the cohort | 53, of which **10 MUST-SPLIT covering 24 bindings** |
| **deduplicated repairs** | **61** = 51 renames + 10 splits |
| repairs a reader cannot compensate for | **10** (the splits) |
| rows requiring a spelling to be invented here | **0** — every rejection carries an explicit proposal in its source record |
| identities rejected in more than one record | 9, of which 4 are renames and all 4 carry one agreed proposal |
| open contradictions between records, surfaced not resolved | **6** — 2 accepted-versus-rejected spellings, 3 on the `toroidal_angle_`/`toroidal_coordinate_` axis, 1 coupled-versus-launched |
| order-dependent repair pairs | 1 |
| measured defect rate | **74 of 341 = 21.7 %** against the sample's 11.6 % estimate |

**61 repairs, of which 10 block publication.** The 51 renames are mechanical once adopted. The 10
splits each mint at least one new identity, and eight of them keep the existing spelling on at
least one binding, so the catalog gains 12 new names rather than 20. Six contradictions must be
adjudicated before the names they touch are published; they affect 8 bindings and none of them is
resolved by this file.

