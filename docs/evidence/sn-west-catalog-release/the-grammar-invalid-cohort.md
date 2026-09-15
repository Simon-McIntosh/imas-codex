# Grammar-invalid cohort after the bare-prefix classifier repair

Measured 2026-09-15 against the live Neo4j graph through the login-node-local tunnel. This is a bounded operational receipt, not a population rewrite.

## Outcome

- The positive control saw **5,125** `StandardName` nodes, and the exact predicate `refine_stop_reason = 'grammar_invalid'` returned **109**. The requested baseline therefore reproduces.
- Producing-source evidence is recoverable for **72/109** names: **8** have a live `PRODUCED_NAME` edge and **65** retain the singular historical `source_path` scalar (the two sets overlap). **37** have neither; the absence is shown row by row rather than silently dropped.
- Of the 109 names, **47 distinct DD sources** are currently unbound, `status='extracted'`, and below the five-attempt cap. A deterministic systematic sample selected indices 0, 3, 7, ..., 43 from those 47 paths sorted lexicographically.
- The required `sn retry --failed` preflight and live operation both refused the entire sample because those sources were already claimable rather than failed or attempt-capped: `eligible: 0 of 12 requested source(s)`, then `retried: 0 of 12 requested source(s)` and `refused: 12 source path(s) were missing or not failed/attempt-capped`. That refusal was correct and was not bypassed on resumption.
- The first exact focused run seeded only the 12 sample paths, then failed closed before the first source claim or model call because the required local compose endpoint returned HTTP 503. After anonymous catalog discovery returned HTTP 200 with `deepseek-v4-flash`, the same command and exact source list ran through the configured direct local endpoint with no paid fallback.
- All **12/12 sources were tried**. Two acquired `PRODUCED_NAME` edges to accepted names with `validation_status=valid`: `waves/coherent_wave/profiles_1d/current_phi_inside_n_phi` attached to `current_per_toroidal_mode_due_to_wave_driven_current_drive`, and `gyrokinetics_local/non_linear/fields_intensity_1d/b_field_parallel_perturbed_norm` attached to `flux_surface_averaged_time_averaged_normalized_perturbed_parallel_magnetic_field_magnitude`. Source-level recovery is therefore **2/12 (16.7%)**.
- Both successful targets predated this run, created on 2026-09-05 and 2026-09-03 respectively. The run reattached sources to grammar-valid identities rather than minting new nodes, so the complementary identity-level result is **0/12 newly composed**, consistent with the authoritative `SNRun.names_composed=0` receipt.
- `SNRun d2ab6f52-541f-4b85-baa9-dc0a71ab1e59` ended `degraded` with `stop_reason=time_limit_reached` after 663.03 seconds, at **$0.00 / $5.00 capped**. The remaining ten paths did not produce a binding before the 600-second fence and 60-second grace period. Several emitted explicit grammar failures and entered the single expanded-context retry; the rest were cancelled as slow calls.
- Scope proof: the stale run scope `12ff2fff-0ada-493a-b973-cffad99158fa` now appears on **0** sources. Sanctioned run scope `fdce33f1-bee1-41f6-b3d7-bc46b3649d10` appears on exactly the 12 sampled sources and **0** outside sources. The two successful sources have no claim token. The other ten remain extracted with current-run claim tokens after bounded cancellation; they are recorded as cleanup residue and were not mutated ad hoc.

Initial HTTP 503 transcript: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260915T112100248211-n-swcr-the-grammar-invalid-cohort-is-resized-after-the-classifier-repair/focused-sn-run.log`. The resumed run's frozen 98,146-byte transcript is `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260915T112100248211-n-swcr-the-grammar-invalid-cohort-is-resized-after-the-classifier-repair/focused-sn-run-resumed.log`, SHA-256 `46cff1b503e02e642984caa4993f8f73ce9fe379dcbb29850ff170cc59ea57e6`.

## Sample receipt

| DD source path | Historical grammar-invalid name and before stage | Source state before to after | After name stage |
|---|---|---|---|
| `camera_ir/fibre_bundle/geometry/surface` | `surface_area_of_optical_element` (exhausted) | `extracted`, attempts `0 to 2`; current-run claim retained | none |
| `cryostat/description_2d/cryostat/unit/annular/outline_outer/z` | `vertical_outline_of_cryostat` (exhausted) | `extracted`, attempts `0 to 2`; current-run claim retained | none |
| `gyrokinetics_local/linear/wavevector/eigenmode/fields/b_field_parallel_perturbed_norm` | `parallel_normalized_perturbed_magnetic_field_amplitude` (exhausted) | `extracted`, attempts `2 to 4`; current-run claim retained | none; expanded-context grammar retry did not finish |
| `gyrokinetics_local/non_linear/fields_intensity_1d/b_field_parallel_perturbed_norm` | `parallel_normalized_perturbed_magnetic_field` (exhausted) | `extracted` to `attached`, attempts `0 to 1`; claim cleared | `flux_surface_averaged_time_averaged_normalized_perturbed_parallel_magnetic_field_magnitude` (accepted, valid) |
| `neutron_diagnostic/detector/energy_band/lower_bound` | `lower_bound_energy_of_neutron_detector` (exhausted) | `extracted`, attempts `0 to 2`; current-run claim retained | none; grammar retry did not produce a binding |
| `plasma_transport/model/ggd/neutral/energy/v_parallel/values` | `parallel_neutral_species_energy_convection_velocity` (exhausted) | `extracted`, attempts `0 to 2`; current-run claim retained | none |
| `runaway_electrons/profiles_1d/e_field_critical` | `runaway_electron_critical_electric_field` (exhausted) | `extracted`, attempts `0 to 2`; current-run claim retained | none |
| `spectrometer_visible/channel/filter_spectrometer/photoelectric_voltage` | `voltage_of_spectrometer` (exhausted) | `extracted`, attempts `0 to 2`; current-run claim retained | none |
| `summary/gas_injection_prefill/tritium/value` | `tritium_prefill_count` (exhausted) | `extracted`, attempts `0 to 2`; current-run claim retained | none; grammar retry did not produce a binding |
| `summary/pedestal_fits/linear/pressure_electron/d_dpsi_norm_max/value` | `gradient_of_electron_pressure` (exhausted) | `extracted`, attempts `0 to 2`; current-run claim retained | none |
| `wall/description_ggd/ggd/energy_fluxes/current/emitted/values` | `energy_flux_due_to_eddy_current` (exhausted) | `extracted`, attempts `0 to 2`; current-run claim retained | none |
| `waves/coherent_wave/profiles_1d/current_phi_inside_n_phi` | `per_toroidal_mode_current_due_to_wave_driven_current_drive` (superseded) | `extracted` to `attached`, attempts `2 to 3`; claim cleared | `current_per_toroidal_mode_due_to_wave_driven_current_drive` (accepted, valid) |

### Guard refusals tied to sampled sources

The CLI emits one aggregate refusal, so these per-source rows bind that exact predicate to representative sampled state:

| Sampled source | State observed | Exact refusal |
|---|---|---|
| `camera_ir/fibre_bundle/geometry/surface` | `extracted`, attempts `0` | `missing or not failed/attempt-capped` |
| `gyrokinetics_local/linear/wavevector/eigenmode/fields/b_field_parallel_perturbed_norm` | `extracted`, attempts `2` | `missing or not failed/attempt-capped` |
| `waves/coherent_wave/profiles_1d/current_phi_inside_n_phi` | `extracted`, attempts `2` | `missing or not failed/attempt-capped` |

## Complete 109-name census

A live edge is marked `edge:`; a scalar-only historical producer is marked `historical:`. `none recorded` means neither instrument found a producer and is retained as an explicit result.

| Standard name | name_stage | validation | producing sources |
|---|---|---|---|
| `acceleration_of_passive_structure` | `exhausted` | `quarantined` | none recorded |
| `accumulated_total_coolant_absorbed_energy_of_calorimetry_component` | `exhausted` | `valid` | edge: `dd:calorimetry/group/component/energy_total/data` |
| `alpha_angle_of_poloidal_field_coil` | `exhausted` | `quarantined` | historical: `dd:pf_active/coil/element/geometry/oblique/alpha` |
| `alpha_parameter` | `exhausted` | `quarantined` | historical: `dd:summary/pedestal_fits/mtanh/stability/alpha_experimental/value` |
| `angle_of_antenna_strap` | `exhausted` | `quarantined` | historical: `dd:ic_antennas/antenna/module/strap/geometry/oblique/beta` |
| `core_density_of_pellet` | `exhausted` | `quarantined` | historical: `dd:spi/injector/pellet/core/species/density` |
| `critical_momentum_due_to_avalanche` | `exhausted` | `quarantined` | historical: `dd:runaway_electrons/profiles_1d/momentum_critical_avalanche` |
| `cumulative_ethylene_count_due_to_gas_injection` | `exhausted` | `quarantined` | none recorded |
| `deuterium_tritium_density_flux_surface_averaged_at_plasma_boundary` | `exhausted` | `quarantined` | historical: `dd:summary/local/separatrix_average/n_i/deuterium_tritium/value` |
| `effective_incident_neutral_coefficient_of_wall_material_due_to_sputtering` | `exhausted` | `quarantined` | none recorded |
| `electron_average_temperature_at_midplane` | `exhausted` | `quarantined` | historical: `dd:langmuir_probes/reciprocating/plunge/t_e_average` |
| `electron_energy_flux_limiter_coefficient` | `exhausted` | `quarantined` | none recorded |
| `electron_temperature_at_first_wall` | `exhausted` | `quarantined` | none recorded |
| `energy_density` | `accepted` | `quarantined` | edge: `derived:energy_density` |
| `energy_flux_at_wall_due_to_eddy_current` | `exhausted` | `quarantined` | historical: `dd:wall/description_ggd/ggd/energy_fluxes/current/incident/values` |
| `energy_flux_due_to_eddy_current` | `exhausted` | `quarantined` | historical: `dd:wall/description_ggd/ggd/energy_fluxes/current/emitted/values` |
| `energy_flux_due_to_radiation` | `exhausted` | `quarantined` | historical: `dd:wall/description_ggd/ggd/energy_fluxes/radiation/emitted/values` |
| `fast_electron_energy` | `exhausted` | `quarantined` | historical: `dd:summary/heating_current_drive/lh/energy_fast/value` |
| `flux_surface_averaged_bulk_electron_temperature_at_last_closed_flux_surface` | `superseded` | `quarantined` | none recorded |
| `flux_surface_normal_momentum_diffusion_coefficient` | `exhausted` | `quarantined` | none recorded |
| `gas_atomic_count_of_pellet_injector` | `exhausted` | `quarantined` | none recorded |
| `gradient_of_electron_pressure` | `exhausted` | `quarantined` | historical: `dd:summary/pedestal_fits/linear/pressure_electron/d_dpsi_norm_max/value` |
| `gradient_of_radial_electron_density` | `exhausted` | `quarantined` | historical: `dd:summary/pedestal_fits/mtanh/n_e/d_dpsi_norm/value` |
| `hard_xray_rate` | `exhausted` | `quarantined` | historical: `dd:hard_x_rays/emissivity_profile_1d/emissivity` |
| `incident_energy_flux_at_wall_due_to_radiation` | `exhausted` | `quarantined` | none recorded |
| `inverse_of_curvature_of_arc_of_circle_center` | `superseded` | `quarantined` | none recorded |
| `ion_charge_state_energy_flux_limiter_coefficient` | `exhausted` | `quarantined` | none recorded |
| `ion_power` | `exhausted` | `quarantined` | none recorded |
| `length_of_passive_structure` | `exhausted` | `quarantined` | historical: `dd:pf_passive/loop/element/geometry/oblique/length_beta` |
| `lower_bound_energy_of_neutron_detector` | `exhausted` | `quarantined` | historical: `dd:neutron_diagnostic/detector/energy_band/lower_bound` |
| `magnetic_field_magnitude` | `accepted` | `quarantined` | edge: `derived:magnetic_field_magnitude` |
| `magnetic_shear_at_sawtooth_inversion_radius` | `exhausted` | `quarantined` | historical: `dd:sawteeth/diagnostics/magnetic_shear_q1` |
| `maximum_power_at_inner_divertor_target` | `exhausted` | `quarantined` | historical: `dd:wall/global_quantities/power_density_inner_target_max` |
| `maximum_power_at_outer_divertor_target` | `exhausted` | `quarantined` | historical: `dd:wall/global_quantities/power_density_outer_target_max` |
| `momentum_due_to_hot_tail` | `exhausted` | `quarantined` | none recorded |
| `net_power` | `exhausted` | `quarantined` | none recorded |
| `neutral_beam_atomic_number` | `exhausted` | `quarantined` | historical: `dd:charge_exchange/channel/bes/z_ion` |
| `neutral_power_at_wall_due_to_recombination` | `exhausted` | `quarantined` | historical: `dd:wall/global_quantities/power_recombination_neutrals` |
| `neutral_species_fraction` | `exhausted` | `quarantined` | historical: `dd:pulse_schedule/density_control/valve/species/fraction` |
| `normalized_perturbed_pressure` | `accepted` | `quarantined` | edge: `derived:normalized_perturbed_pressure` |
| `normalized_poloidal_magnetic_flux_at_pedestal_top` | `exhausted` | `quarantined` | none recorded |
| `normalized_toroidal_hard_xray_peak_external_half_width` | `exhausted` | `quarantined` | none recorded |
| `normalized_toroidal_hard_xray_peak_upper_bound_width` | `exhausted` | `quarantined` | none recorded |
| `nuclear_power_density_at_midplane` | `exhausted` | `quarantined` | none recorded |
| `parallel_bulk_ion_velocity` | `exhausted` | `quarantined` | none recorded |
| `parallel_current_density_flux_surface_averaged_due_to_wave_driven_current_drive` | `exhausted` | `quarantined` | historical: `dd:plasma_sources/source/profiles_1d/j_parallel` |
| `parallel_flux_surface_averaged_electric_field_at_separatrix` | `superseded` | `quarantined` | none recorded |
| `parallel_incident_heat_flux_at_divertor_target` | `exhausted` | `quarantined` | none recorded |
| `parallel_neutral_momentum_diffusivity` | `exhausted` | `quarantined` | historical: `dd:plasma_transport/model/ggd/neutral/momentum/d_parallel/values` |
| `parallel_neutral_species_energy_convection_velocity` | `exhausted` | `quarantined` | historical: `dd:plasma_transport/model/ggd/neutral/energy/v_parallel/values` |
| `parallel_normalized_particle_perturbed_pressure` | `exhausted` | `quarantined` | historical: `dd:gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_particle/pressure_parallel` |
| `parallel_normalized_perturbed_magnetic_field` | `exhausted` | `quarantined` | historical: `dd:gyrokinetics_local/non_linear/fields_intensity_1d/b_field_parallel_perturbed_norm` |
| `parallel_normalized_perturbed_magnetic_field_amplitude` | `exhausted` | `quarantined` | historical: `dd:gyrokinetics_local/linear/wavevector/eigenmode/fields/b_field_parallel_perturbed_norm` |
| `parallel_normalized_perturbed_vector_potential_amplitude` | `exhausted` | `quarantined` | historical: `dd:gyrokinetics_local/non_linear/fields_intensity_3d/a_field_parallel_perturbed_norm` |
| `parallel_normalized_wave_vector` | `exhausted` | `quarantined` | historical: `dd:gyrokinetics_local/non_linear/binormal_wavevector_norm` |
| `parallel_per_toroidal_mode_current_density_due_to_wave_driven_current_drive` | `superseded` | `quarantined` | historical: `dd:waves/coherent_wave/profiles_1d/current_parallel_density_n_phi` |
| `parallel_per_toroidal_mode_electric_field` | `exhausted` | `quarantined` | historical: `dd:waves/coherent_wave/profiles_1d/e_field_n_phi/parallel/phase` |
| `parallel_wave_electric_field_amplitude` | `exhausted` | `quarantined` | historical: `dd:waves/coherent_wave/profiles_1d/e_field_n_phi/parallel/amplitude` |
| `per_toroidal_mode_current_due_to_wave_driven_current_drive` | `superseded` | `quarantined` | historical: `dd:waves/coherent_wave/profiles_1d/current_phi_inside_n_phi` |
| `per_toroidal_mode_right_hand_circularly_polarized_wave_electric_field` | `exhausted` | `quarantined` | historical: `dd:waves/coherent_wave/profiles_1d/e_field_n_phi/minus/phase` |
| `perpendicular_neutral_state_velocity_due_to_diamagnetic_drift` | `exhausted` | `quarantined` | historical: `dd:plasma_profiles/ggd/neutral/state/velocity_diamagnetic/diamagnetic` |
| `perpendicular_normalized_gyrocenter_perturbed_pressure` | `exhausted` | `quarantined` | historical: `dd:gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_gyrocenter/pressure_perpendicular` |
| `perpendicular_normalized_perturbed_pressure` | `accepted` | `quarantined` | edge: `derived:perpendicular_normalized_perturbed_pressure`; historical: `dd:gyrokinetics_local/linear/wavevector/eigenmode/moments_norm_particle/pressure_perpendicular` |
| `perturbed_vector_potential` | `exhausted` | `quarantined` | none recorded |
| `poloidal_ion_charge_state_momentum_diffusivity` | `exhausted` | `quarantined` | historical: `dd:core_transport/model/profiles_1d/ion/state/momentum/poloidal/d` |
| `poloidal_neutral_momentum_diffusivity` | `exhausted` | `quarantined` | historical: `dd:edge_transport/model/ggd/neutral/momentum/d_pol/values` |
| `power_of_neutral_beam_injector` | `exhausted` | `quarantined` | historical: `dd:summary/heating_current_drive/nbi/power_launched/value` |
| `power_over_core_region_due_to_impurity_radiation` | `exhausted` | `quarantined` | none recorded |
| `pressure_of_lower_hybrid_antenna` | `exhausted` | `quarantined` | historical: `dd:lh_antennas/antenna/pressure_tank` |
| `radial_angle_of_poloidal_field_coil` | `exhausted` | `quarantined` | historical: `dd:pf_active/coil/element/geometry/oblique/alpha` |
| `radiative_temperature` | `exhausted` | `quarantined` | historical: `dd:ece/channel/t_radiation_o` |
| `radius_of_plasma_filament` | `accepted` | `quarantined` | edge: `derived:radius_of_plasma_filament` |
| `radius_of_poloidal_field_coil` | `accepted` | `quarantined` | edge: `derived:radius_of_poloidal_field_coil` |
| `ratio_of_field_line_count_to_total_field_line_count` | `exhausted` | `quarantined` | none recorded |
| `runaway_electron_critical_electric_field` | `exhausted` | `quarantined` | historical: `dd:runaway_electrons/profiles_1d/e_field_critical` |
| `silane_prefill_count` | `exhausted` | `quarantined` | historical: `dd:summary/gas_injection_prefill/silane/value` |
| `size_of_camera` | `exhausted` | `quarantined` | none recorded |
| `spectral_wave_opacity_line_integrated_at_ece_channel_emission_position` | `exhausted` | `quarantined` | historical: `dd:ece/channel/optical_depth` |
| `spectral_width_of_filter` | `exhausted` | `quarantined` | historical: `dd:spectrometer_visible/channel/filter_spectrometer/filter/wavelength_width` |
| `surface_area_of_optical_element` | `exhausted` | `quarantined` | historical: `dd:camera_ir/fibre_bundle/geometry/surface` |
| `surface_temperature_of_plasma_facing_component` | `exhausted` | `quarantined` | none recorded |
| `thermal_electron_decay_length_over_scrape_off_layer` | `exhausted` | `quarantined` | historical: `dd:summary/scrape_off_layer/t_e_decay_length/value` |
| `thermal_energy_confinement_time` | `exhausted` | `quarantined` | none recorded |
| `thermal_energy_of_plant_component_port` | `exhausted` | `quarantined` | historical: `dd:calorimetry/group/component/energy_cumulated` |
| `thickness_of_passive_loop` | `exhausted` | `quarantined` | historical: `dd:pf_passive/loop/element/geometry/thick_line/thickness` |
| `tilt_angle_of_poloidal_field_coil` | `exhausted` | `quarantined` | none recorded |
| `time_derivative_of_electron_density` | `accepted` | `quarantined` | edge: `derived:time_derivative_of_electron_density` |
| `toroidal_co_passing_fast_electron_torque_density_due_to_collisional_transport` | `exhausted` | `quarantined` | none recorded |
| `toroidal_cumulative_inside_flux_surface_total_plasma_momentum_at_separatrix` | `exhausted` | `quarantined` | none recorded |
| `toroidal_volume_integrated_fast_electron_torque_density_due_to_collisions` | `superseded` | `quarantined` | none recorded |
| `total_energy_of_calorimetry_component` | `exhausted` | `quarantined` | historical: `dd:calorimetry/group/component/energy_total/data` |
| `total_neutron_power` | `exhausted` | `quarantined` | historical: `dd:summary/fusion/neutron_power_total/value` |
| `total_particle_count_accumulated_due_to_gas_injection` | `exhausted` | `quarantined` | historical: `dd:summary/gas_injection_accumulated/total/value` |
| `total_particle_flux` | `exhausted` | `quarantined` | none recorded |
| `total_power_of_neutral_beam_injector` | `exhausted` | `quarantined` | historical: `dd:summary/heating_current_drive/power_launched_nbi/value` |
| `total_suprathermal_electron_power_density_due_to_collisions` | `exhausted` | `quarantined` | none recorded |
| `tritium_prefill_count` | `exhausted` | `quarantined` | historical: `dd:summary/gas_injection_prefill/tritium/value` |
| `vertical_coordinate_of_halo_boundary` | `exhausted` | `quarantined` | historical: `dd:disruption/halo_currents/area/end_point/z` |
| `vertical_magnetic_field_at_wall` | `exhausted` | `quarantined` | historical: `dd:focs/b_field_z` |
| `vertical_outline_of_cryostat` | `exhausted` | `quarantined` | historical: `dd:cryostat/description_2d/cryostat/unit/annular/outline_outer/z` |
| `vertical_position_of_grating` | `exhausted` | `quarantined` | none recorded |
| `vertical_total_ion_momentum_diffusivity` | `exhausted` | `quarantined` | none recorded |
| `voltage_amplitude` | `exhausted` | `quarantined` | historical: `dd:ic_antennas/antenna/module/voltage/amplitude` |
| `voltage_of_diagnostic_antenna` | `exhausted` | `quarantined` | historical: `dd:ece/channel/voltage_t_radiation` |
| `voltage_of_spectrometer` | `exhausted` | `quarantined` | historical: `dd:spectrometer_visible/channel/filter_spectrometer/photoelectric_voltage` |
| `voltage_of_temperature_sensor` | `exhausted` | `quarantined` | historical: `dd:neutron_diagnostic/detector/temperature_sensor/amplitude` |
| `volume_averaged_electron_number_density_over_scrape_off_layer` | `exhausted` | `quarantined` | none recorded |
| `volume_averaged_runaway_electron_source_rate` | `exhausted` | `quarantined` | historical: `dd:runaway_electrons/global_quantities/volume_average/ddensity_dt_tritium` |
| `volume_integrated_net_plasma_particle_power_density` | `exhausted` | `quarantined` | none recorded |

## Reproduction and commands

1. Positive-control count: `MATCH (sn:StandardName) RETURN count(sn), sum(CASE WHEN sn.refine_stop_reason = 'grammar_invalid' THEN 1 ELSE 0 END)` produced `5125, 109`.
2. Retry preflight: `imas-codex sn retry --failed --dry-run --reason <repair evidence> <12 exact paths>` produced `eligible: 0 of 12`, `refused: 12`.
3. Live retry: the same exact cohort without `--dry-run` produced `retried: 0 of 12`, `refused: 12`.
4. Focused composition: `imas-codex sn run <12 --focus paths> --skip-global-maintenance --only compose --names-only -c 5 -t 10` exited 1 on required local endpoint HTTP 503 before any model call.
5. Resumption used the identical command after live catalog discovery returned `deepseek-v4-flash`. It tried all 12 paths, attached 2 to valid accepted names, spent $0.00, and ended at the time fence with 10 unresolved.

## Terminal state and follow-on

The requested measurement is complete at **2/12 source-level recoveries (16.7%)** and **0/12 newly minted identities**. The old stale scope was overwritten through the sanctioned focused route. Ten current-run claim tokens survived the bounded cancellation even though their sources read `status=extracted`; their cleanup belongs to the standard worker-failure/orphan-claim route, not this evidence-only write scope and not ad-hoc Cypher.
