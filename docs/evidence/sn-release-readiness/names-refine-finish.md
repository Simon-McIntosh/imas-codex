# Standard-name refinement cohort completion

> **COMPLETE — the exact 273-root cohort is settled.** The continuation used the ordinary names-axis review/refine pools on the existing deterministic scope. It did not rescore, restage, hand-score, direct-accept, or reword any unchanged root.

## Completion gate

- The sole cancelled claim, `energy_flux_at_control_surface`, was **912 seconds** old before continuation, exceeding the 600-second orphan-sweep threshold by **312 seconds**. It was not released by hand.
- Exact terminal check: **0** live claims or transient rows, **0** eligible name-review items, and **0** eligible name-refine items.
- The continuation run completed with `stop_reason=no_eligible_work`; run id `2de85884-f61d-4e5a-be57-db20521c2650`.
- Continuation spend: **USD 0.782349 / USD 30.00**, 7 calls, maximum call USD 0.477655.
- Complete refine-campaign spend: **USD 77.808743**, 1175 calls, **USD 0.289252 per entered name**, and **4.368 calls per entered name**. The prior interim was USD 77.026394 / USD 120 at USD 0.286343 and 4.342 calls per entered name; these final figures supersede it.

The generic `pool_pending_counts` display reported four refine rows after completion, but all four carry a non-null `review_quorum_shortfall`. The actual claim predicate requires that field to be null and reports zero eligible refine work. The count/display predicate omits that guard; this instrumentation mismatch is a fenced follow-on, not permission to refine or rescore the four roots.

## Final cohort figures

| Measure | Final |
|---|---:|
| Roots resolved from the completed drain | 273 |
| Roots legally entered into refine | 269 |
| Roots that minted one or more successors | 158 |
| Entered roots parked without a successor | 111 |
| Terminal successors accepted | 74 |
| Terminal successors below 0.85 | 84 |
| Live name claims or transient stages | 0 |
| Eligible pending review items | 0 |
| Eligible pending refine items | 0 |
| StandardName population | 4395 → 4666 (+271) |
| New `REFINED_FROM` edges | 271 |

The 111 entered roots that parked without minting a successor ended for explicit reasons: **57 `grammar_invalid`**, **46 `successor_collision`**, **5 `attempts_exhausted`**, and **3 `vocabulary_gap`**. Among the 158 roots that minted successors, terminal successors are **74 accepted** and **84 exhausted below 0.85**. The 84 terminal stop reasons are **36 `attempts_exhausted`**, **24 `grammar_invalid`**, **23 `successor_collision`**, and **1 `vocabulary_gap`**.

The four non-quorate roots remain deliberately unrescored and unrefined: `fast_neutral_density`, `hot_neutral_temperature`, `ion_power`, and `rotational_transform`. They remain at `reviewed` with their original score and `review_quorum_shortfall`; they are terminal for this no-rescore cohort because the refine claim path excludes non-quorate verdicts.

## WEST intersection — 20 identities, separately reported

Every successor set below is **FINAL**, not in flight. A row with no successor either parked at the root for the stated reason or is one of the preserved non-quorate roots.

| Root identity | Root terminal state | Successors | Terminal successor | Score | Terminal successor state / root reason | Set |
|---|---|---:|---|---:|---|---|
| `current_of_poloidal_field_coil` | `exhausted` | 0 | — | — | `grammar_invalid` | **FINAL** |
| `energy_confinement_time` | `superseded` | 1 | `thermal_energy_confinement_time` | 0.77500 | `grammar_invalid` | **FINAL** |
| `gap_of_antenna_strap` | `superseded` | 3 | `normal_distance_of_antenna_strap` | 0.63125 | `attempts_exhausted` | **FINAL** |
| `hot_neutral_temperature` | `reviewed` | 0 | — | — | `preserved non-quorate` | **FINAL** |
| `ion_atomic_number` | `superseded` | 1 | `ion_species_atomic_number` | 0.58750 | `successor_collision` | **FINAL** |
| `length_of_plasma_boundary` | `superseded` | 1 | `poloidal_length_of_flux_surface` | 0.97500 | `accepted` | **FINAL** |
| `lower_photon_energy` | `superseded` | 1 | `lower_bound_photon_energy` | 0.88125 | `accepted` | **FINAL** |
| `neutral_pressure` | `exhausted` | 0 | — | — | `successor_collision` | **FINAL** |
| `power_due_to_ohmic_dissipation` | `superseded` | 1 | `total_power_due_to_ohmic_dissipation` | 0.88750 | `accepted` | **FINAL** |
| `power_of_ion_cyclotron_heating_antenna` | `superseded` | 3 | `launched_power_of_ion_cyclotron_heating_antenna` | 0.65000 | `attempts_exhausted` | **FINAL** |
| `radiative_temperature` | `exhausted` | 0 | — | — | `grammar_invalid` | **FINAL** |
| `spectral_radiance_of_soft_xray_detector` | `superseded` | 1 | `incident_soft_xray_radiance` | 0.95625 | `accepted` | **FINAL** |
| `surface_temperature` | `superseded` | 1 | `surface_temperature_of_plasma_facing_component` | 0.81250 | `grammar_invalid` | **FINAL** |
| `thermal_energy_of_plant_component_port` | `exhausted` | 0 | — | — | `grammar_invalid` | **FINAL** |
| `thermal_power_of_plant_component_port` | `superseded` | 3 | `absorbed_coolant_power_of_plant_component_port` | 0.55000 | `attempts_exhausted` | **FINAL** |
| `toroidal_angle_of_magnetic_field_probe` | `superseded` | 1 | `toroidal_angle_of_poloidal_magnetic_field_probe` | 1.00000 | `accepted` | **FINAL** |
| `toroidal_width_of_antenna_strap` | `exhausted` | 0 | — | — | `grammar_invalid` | **FINAL** |
| `vertical_coordinate_of_plasma_boundary` | `superseded` | 3 | `vertical_coordinate_of_plasma_filament` | 0.66875 | `attempts_exhausted` | **FINAL** |
| `vertical_outline_of_wall_material` | `superseded` | 1 | `vertical_outline_of_plasma_facing_component` | 0.99375 | `accepted` | **FINAL** |
| `voltage_amplitude` | `exhausted` | 0 | — | — | `grammar_invalid` | **FINAL** |

## Antenna-strap cohort group

The 273-root cohort contains exactly three root identities whose ids contain `antenna_strap`. `vertical_coordinate_of_antenna_strap`, the separate documentation-gate false negative, is **not** one of the 273 refine roots and therefore has no successor from this campaign.

| Cohort member | Initial score | Successor score sequence | Group terminal result |
|---|---:|---|---|
| `angle_of_antenna_strap` | 0.60000 | none | `grammar_invalid` |
| `gap_of_antenna_strap` | 0.75000 | `normal_gap_at_wall` 0.51250 (superseded) → `normal_gap_of_antenna_strap` 0.76875 (superseded) → `normal_distance_of_antenna_strap` 0.63125 (exhausted) | `attempts_exhausted` |
| `toroidal_width_of_antenna_strap` | 0.58750 | none | `grammar_invalid` |

## Terminal state of all 273 roots

This table is the complete root census. `successor terminal` is the last node in the root lineage; the next section lists every successor, including intermediate superseded nodes.

| Root | Initial score | Root terminal state | Attempts | Root stop reason | Successors | Successor terminal | Terminal score | Terminal state / stop |
|---|---:|---|---:|---|---:|---|---:|---|
| `accumulated_lithium_count` | 0.84375 | `superseded` | 1 | `—` | 1 | `accumulated_lithium_count_due_to_gas_injection` | 1.00000 | `accepted` |
| `accumulated_nitrogen_count` | 0.83750 | `superseded` | 1 | `—` | 1 | `accumulated_nitrogen_count_due_to_gas_injection` | 1.00000 | `accepted` |
| `accumulated_propane_count` | 0.81250 | `superseded` | 1 | `—` | 1 | `accumulated_propane_count_due_to_gas_injection` | 1.00000 | `accepted` |
| `accumulated_silane_count` | 0.81250 | `superseded` | 1 | `—` | 1 | `accumulated_silane_count_due_to_gas_injection` | 1.00000 | `accepted` |
| `accumulated_xenon_count` | 0.83750 | `superseded` | 1 | `—` | 1 | `accumulated_xenon_count_due_to_gas_injection` | 1.00000 | `accepted` |
| `angle_of_antenna_strap` | 0.60000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `area_of_langmuir_probe` | 0.80625 | `superseded` | 1 | `—` | 3 | `front_surface_area_of_langmuir_probe` | 0.58750 | `attempts_exhausted` |
| `area_of_neutral_beam_injector` | 0.83750 | `superseded` | 1 | `—` | 1 | `beam_cross_sectional_area_of_aperture` | 0.71250 | `successor_collision` |
| `atomic_fraction_of_neutron_detector_converter` | 0.83750 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `back_surface_curvature_of_optical_element` | 0.55000 | `superseded` | 1 | `—` | 3 | `first_local_tangential_back_surface_radius_of_optical_element` | 0.67500 | `attempts_exhausted` |
| `beryllium_source_rate` | 0.73750 | `superseded` | 1 | `—` | 1 | `beryllium_source_rate_due_to_gas_injection` | 1.00000 | `accepted` |
| `bragg_crystal_width` | 0.48750 | `superseded` | 1 | `—` | 1 | `first_local_tangential_width_of_bragg_crystal` | 0.99375 | `accepted` |
| `co_passing_fast_electron_power_density_due_to_collisions` | 0.61250 | `superseded` | 2 | `transient_failure` | 2 | `co_passing_fast_electron_kinetic_power_density_due_to_collisions` | 0.48750 | `attempts_exhausted` |
| `conductivity` | 0.58750 | `superseded` | 1 | `—` | 1 | `plasma_electrical_conductivity` | 0.92500 | `accepted` |
| `coolant_temperature` | 0.80000 | `superseded` | 1 | `—` | 1 | `coolant_temperature_of_plant_component_port` | 0.93750 | `accepted` |
| `coolant_volume_of_breeder_blanket` | 0.75000 | `superseded` | 1 | `—` | 3 | `lithium_volume_of_breeder_blanket` | 0.63750 | `attempts_exhausted` |
| `core_density_of_pellet` | 0.78750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `counter_passing_density` | 0.56250 | `superseded` | 1 | `—` | 2 | `total_counter_passing_particle_density` | 1.00000 | `accepted` |
| `critical_electric_field` | 0.73125 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `critical_momentum_due_to_avalanche` | 0.67500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `current_due_to_ohmic_induction` | 0.76250 | `superseded` | 1 | `—` | 2 | `net_plasma_current_due_to_ohmic_current_drive` | 0.53750 | `successor_collision` |
| `current_of_correction_coil` | 0.72500 | `superseded` | 1 | `—` | 3 | `non_axisymmetric_current_of_conductor` | 0.51250 | `attempts_exhausted` |
| `current_of_poloidal_field_coil` | 0.76250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `curvature_of_optical_element` | 0.66250 | `superseded` | 1 | `—` | 3 | `inverse_of_tangential_curvature_of_optical_element` | 0.47500 | `attempts_exhausted` |
| `deposited_power` | 0.68750 | `superseded` | 1 | `—` | 1 | `volume_integrated_net_plasma_particle_power_density` | 0.61875 | `grammar_invalid` |
| `deuterium_count_due_to_gas_injection` | 0.73750 | `superseded` | 1 | `—` | 1 | `accumulated_deuterium_count_due_to_gas_injection` | 1.00000 | `accepted` |
| `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | 0.73750 | `superseded` | 1 | `—` | 3 | `deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.98125 | `accepted` |
| `deuterium_tritium_neutron_flux_due_to_beam_beam_fusion` | 0.68750 | `superseded` | 1 | `—` | 1 | `total_deuterium_tritium_neutron_source_rate_due_to_beam_beam_fusion` | 0.51250 | `successor_collision` |
| `diamagnetic_current_density` | 0.45000 | `exhausted` | 3 | `attempts_exhausted` | 0 | — | — | `—` |
| `diamagnetic_velocity_due_to_diamagnetic_drift` | 0.38750 | `superseded` | 1 | `—` | 2 | `binormal_ion_velocity_due_to_diamagnetic_drift` | 0.91250 | `accepted` |
| `effective_charge_at_plasma_boundary` | 0.67500 | `superseded` | 1 | `—` | 2 | `effective_charge_at_separatrix` | 1.00000 | `accepted` |
| `efficiency_of_neutron_detector` | 0.80000 | `superseded` | 1 | `—` | 1 | `spectral_efficiency_of_neutron_detector` | 0.98125 | `accepted` |
| `efficiency_of_spectrometer_channel` | 0.71250 | `superseded` | 1 | `—` | 2 | `viewing_efficiency_of_spectrometer_channel` | 0.88750 | `accepted` |
| `electron_average_temperature_at_midplane` | 0.57500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `electron_density_over_scrape_off_layer` | 0.58750 | `superseded` | 1 | `—` | 2 | `volume_averaged_electron_number_density_over_scrape_off_layer` | 0.58750 | `grammar_invalid` |
| `electron_diffusivity` | 0.77500 | `superseded` | 1 | `—` | 2 | `electron_particle_diffusivity` | 0.76250 | `successor_collision` |
| `electron_energy_diffusivity` | 0.67500 | `superseded` | 1 | `—` | 2 | `electron_heat_diffusivity` | 0.80625 | `successor_collision` |
| `electron_energy_flux_at_wall` | 0.82500 | `superseded` | 1 | `—` | 2 | `electron_kinetic_energy_flux_at_wall_due_to_surface_emission` | 1.00000 | `accepted` |
| `electron_particle_flux_at_wall` | 0.71875 | `superseded` | 1 | `—` | 1 | `electron_particle_flux_at_wall_due_to_surface_emission` | 0.96875 | `accepted` |
| `electron_pressure_at_plasma_boundary` | 0.78125 | `superseded` | 1 | `—` | 1 | `electron_pressure_at_separatrix` | 1.00000 | `accepted` |
| `electron_temperature_at_midplane` | 0.68750 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `electron_temperature_at_plasma_boundary` | 0.68750 | `superseded` | 1 | `—` | 3 | `electron_temperature_at_separatrix` | 0.67500 | `attempts_exhausted` |
| `electron_temperature_at_wall` | 0.73750 | `superseded` | 1 | `—` | 1 | `electron_temperature_at_first_wall` | 0.66250 | `grammar_invalid` |
| `electrostatic_potential_at_midplane` | 0.62500 | `superseded` | 1 | `—` | 3 | `plasma_electrostatic_potential_at_outboard_midplane` | 0.53750 | `attempts_exhausted` |
| `electrostatic_potential_at_wall` | 0.82500 | `superseded` | 1 | `—` | 3 | `plasma_electrostatic_potential_at_wall` | 0.61250 | `attempts_exhausted` |
| `energy_confinement_time` | 0.83750 | `superseded` | 1 | `—` | 1 | `thermal_energy_confinement_time` | 0.77500 | `grammar_invalid` |
| `energy_diffusivity` | 0.57500 | `superseded` | 1 | `—` | 2 | `energy_diffusion_coefficient_due_to_diffusion` | 0.72500 | `successor_collision` |
| `energy_flux` | 0.82500 | `superseded` | 1 | `—` | 3 | `energy_flux_at_control_surface` | 0.56250 | `attempts_exhausted` |
| `energy_flux_at_wall_due_to_eddy_current` | 0.60000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `energy_flux_at_wall_due_to_radiation` | 0.66250 | `superseded` | 1 | `—` | 1 | `incident_energy_flux_at_wall_due_to_radiation` | 0.55000 | `grammar_invalid` |
| `ethylene_count` | 0.60000 | `superseded` | 1 | `—` | 2 | `cumulative_ethylene_count_due_to_gas_injection` | 0.65000 | `grammar_invalid` |
| `extent_of_detector_pixel` | 0.77500 | `exhausted` | 1 | `vocabulary_gap` | 0 | — | — | `—` |
| `fast_electron_energy` | 0.83750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `fast_electron_power_density` | 0.80000 | `superseded` | 1 | `—` | 2 | `fast_electron_absorbed_wave_power_density` | 0.98125 | `accepted` |
| `fast_electron_power_density_due_to_collisions` | 0.70000 | `superseded` | 1 | `—` | 1 | `total_suprathermal_electron_power_density_due_to_collisions` | 0.76250 | `grammar_invalid` |
| `fast_electron_source_rate` | 0.60000 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `fast_electron_source_rate_due_to_compton_scattering` | 0.56250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `fast_electron_source_rate_due_to_dreicer` | 0.53750 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `fast_electron_source_rate_due_to_hot_tail` | 0.56250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `fast_neutral_density` | 0.30000 | `reviewed` | 0 | `preserved non-quorate` | 0 | — | — | `—` |
| `filter_window_width` | 0.57500 | `superseded` | 1 | `—` | 1 | `first_local_tangential_width_of_filter_window` | 1.00000 | `accepted` |
| `flux_due_to_recombination` | 0.52500 | `superseded` | 1 | `—` | 2 | `normal_particle_flux_at_wall_due_to_recombination` | 0.72500 | `successor_collision` |
| `flux_surface_averaged_electron_temperature_at_plasma_boundary` | 0.76250 | `superseded` | 1 | `—` | 1 | `flux_surface_averaged_bulk_electron_temperature_at_last_closed_flux_surface` | 0.80000 | `grammar_invalid` |
| `fraction_of_neutron_detector_converter` | 0.62500 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `frequency_of_electron_cyclotron_heating_antenna` | 0.71250 | `superseded` | 1 | `—` | 1 | `frequency_of_electron_cyclotron_beam` | 1.00000 | `accepted` |
| `gap_of_antenna_strap` | 0.75000 | `superseded` | 1 | `—` | 3 | `normal_distance_of_antenna_strap` | 0.63125 | `attempts_exhausted` |
| `gradient_of_electron_pressure` | 0.46250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `gradient_of_radial_electron_density` | 0.45000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `height_of_optical_element` | 0.65000 | `exhausted` | 1 | `vocabulary_gap` | 0 | — | — | `—` |
| `hot_neutral_temperature` | 0.30000 | `reviewed` | 0 | `preserved non-quorate` | 0 | — | — | `—` |
| `incident_power_of_breeder_blanket_module` | 0.73125 | `superseded` | 1 | `—` | 1 | `incident_neutron_power_of_breeder_blanket_module` | 0.98750 | `accepted` |
| `ion_atomic_number` | 0.67500 | `superseded` | 1 | `—` | 1 | `ion_species_atomic_number` | 0.58750 | `successor_collision` |
| `ion_charge_state_diamagnetic_velocity_due_to_diamagnetic_drift` | 0.68750 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `ion_diffusivity` | 0.70000 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `ion_momentum_diffusivity` | 0.56250 | `superseded` | 1 | `—` | 1 | `total_ion_momentum_diffusivity` | 0.48750 | `successor_collision` |
| `ion_particle_flux_at_wall` | 0.65000 | `superseded` | 1 | `—` | 3 | `ion_species_particle_flux_at_wall_due_to_surface_emission` | 0.72500 | `attempts_exhausted` |
| `ion_power` | 0.30000 | `reviewed` | 0 | `preserved non-quorate` | 0 | — | — | `—` |
| `ion_pressure` | 0.79375 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `ion_state_momentum_diffusivity` | 0.52500 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `ion_temperature_at_midplane` | 0.61250 | `superseded` | 1 | `—` | 3 | `ion_temperature_at_outboard_midplane_separatrix` | 0.66250 | `attempts_exhausted` |
| `launched_power_of_wave_beam` | 0.72500 | `superseded` | 1 | `—` | 3 | `net_forward_power_of_wave_beam` | 0.56250 | `attempts_exhausted` |
| `left_hand_circularly_polarized_electric_field` | 0.61250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `length_of_electron_cyclotron_beam` | 0.67500 | `superseded` | 1 | `—` | 1 | `distance_of_beam_tracing_ray` | 0.70625 | `successor_collision` |
| `length_of_iron_core_segment` | 0.58750 | `superseded` | 1 | `—` | 1 | `minor_length_of_iron_core_segment` | 0.91250 | `accepted` |
| `length_of_passive_structure` | 0.62500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `length_of_plasma_boundary` | 0.68125 | `superseded` | 1 | `—` | 1 | `poloidal_length_of_flux_surface` | 0.97500 | `accepted` |
| `lower_bound_energy_of_neutron_detector` | 0.82500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `lower_photon_energy` | 0.82500 | `superseded` | 1 | `—` | 1 | `lower_bound_photon_energy` | 0.88125 | `accepted` |
| `magnetic_shear_at_pedestal_top` | 0.84375 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `magnetic_shear_at_sawtooth_inversion_radius` | 0.58750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `major_radius` | 0.77500 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `mass_density` | 0.71250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `mass_of_wall_material` | 0.64375 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `maximum_power_at_inner_divertor_target` | 0.58750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `maximum_power_at_outer_divertor_target` | 0.57500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `minimum_magnetic_field_magnitude` | 0.70625 | `superseded` | 1 | `—` | 1 | `minimum_over_flux_surface_magnetic_field_magnitude` | 1.00000 | `accepted` |
| `mode_number` | 0.55000 | `superseded` | 1 | `—` | 2 | `perturbed_linear_mhd_mode_number` | 0.46875 | `successor_collision` |
| `momentum_convection_velocity` | 0.63750 | `superseded` | 1 | `—` | 3 | `flux_surface_normal_momentum_convection_velocity` | 0.81250 | `attempts_exhausted` |
| `momentum_due_to_hot_tail` | 0.58750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `net_power_of_plant_system` | 0.66250 | `superseded` | 1 | `—` | 1 | `net_power` | 0.70000 | `grammar_invalid` |
| `neutral_beam_atomic_number` | 0.57500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `neutral_beam_mass` | 0.77500 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `neutral_diffusivity` | 0.71250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `neutral_energy_diffusivity` | 0.82500 | `superseded` | 1 | `—` | 3 | `effective_neutral_energy_diffusion_coefficient` | 0.87500 | `accepted` |
| `neutral_particle_flux_at_wall` | 0.59375 | `superseded` | 1 | `—` | 3 | `neutral_species_kinetic_energy_flux_at_wall_due_to_surface_emission` | 0.65000 | `attempts_exhausted` |
| `neutral_power_at_wall_due_to_recombination` | 0.71250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `neutral_pressure` | 0.82500 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `neutron_rate_of_neutron_detector` | 0.83750 | `superseded` | 3 | `transient_failure` | 1 | `neutron_rate_of_detector` | 0.77500 | `attempts_exhausted` |
| `normalized_atomic_count_of_pellet` | 0.48125 | `superseded` | 1 | `—` | 3 | `molecular_gas_count_due_to_pellet_injection` | 0.62500 | `attempts_exhausted` |
| `normalized_count_at_detector_pixel` | 0.79375 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `normalized_perturbed_pressure` | 0.63750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `normalized_poloidal_magnetic_flux_at_pedestal_top` | 0.71250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `normalized_time` | 0.81250 | `superseded` | 1 | `—` | 2 | `normalized_gyrokinetic_time` | 0.96250 | `accepted` |
| `normalized_toroidal_flux_coordinate_at_magnetic_axis` | 0.66250 | `superseded` | 1 | `—` | 3 | `normalized_toroidal_flux_coordinate_at_ece_channel_emission_position` | 0.82500 | `attempts_exhausted` |
| `nuclear_power_density_of_breeder_blanket_module` | 0.84375 | `superseded` | 1 | `—` | 1 | `nuclear_power_density_at_midplane` | 0.65000 | `grammar_invalid` |
| `nuclear_power_of_limiter_tile` | 0.82500 | `superseded` | 1 | `—` | 3 | `nuclear_heating_power_of_limiter_tile` | 0.92500 | `accepted` |
| `optical_element_width` | 0.50000 | `superseded` | 1 | `—` | 1 | `first_local_tangential_width_of_reflector` | 1.00000 | `accepted` |
| `ordinary_mode_fraction_of_wave_beam` | 0.65000 | `superseded` | 1 | `—` | 1 | `ordinary_mode_fraction_of_electron_cyclotron_beam` | 0.98750 | `accepted` |
| `outer_atomic_count_of_pellet` | 0.61250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `oxygen_source_rate` | 0.81250 | `superseded` | 1 | `—` | 1 | `oxygen_source_rate_due_to_gas_injection` | 1.00000 | `accepted` |
| `parallel_energy_diffusivity` | 0.55000 | `superseded` | 1 | `—` | 2 | `parallel_thermal_neutral_energy_diffusivity` | 0.72500 | `successor_collision` |
| `parallel_flux_surface_averaged_electric_field_at_plasma_boundary` | 0.60000 | `superseded` | 1 | `—` | 2 | `parallel_flux_surface_averaged_electric_field_at_separatrix` | 0.67500 | `grammar_invalid` |
| `parallel_heat_flux_at_divertor_target` | 0.62500 | `superseded` | 1 | `—` | 1 | `parallel_incident_heat_flux_at_divertor_target` | 0.81250 | `grammar_invalid` |
| `parallel_ion_diffusivity` | 0.82500 | `superseded` | 1 | `—` | 1 | `parallel_ion_particle_diffusivity` | 0.90000 | `accepted` |
| `parallel_ion_velocity` | 0.70000 | `superseded` | 1 | `—` | 1 | `parallel_bulk_ion_velocity` | 0.66250 | `grammar_invalid` |
| `parallel_neutral_momentum_diffusivity` | 0.83750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `parallel_neutral_species_energy_convection_velocity` | 0.82500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `parallel_normalized_particle_perturbed_pressure` | 0.81250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `parallel_normalized_perturbed_current_density_bessel_1` | 0.84375 | `superseded` | 1 | `—` | 1 | `normalized_perturbed_parallel_gyrocenter_current_density_bessel_1` | 0.87500 | `accepted` |
| `parallel_normalized_perturbed_magnetic_field` | 0.80000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `parallel_normalized_perturbed_magnetic_field_amplitude` | 0.62500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `parallel_normalized_perturbed_vector_potential_amplitude` | 0.57500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `parallel_normalized_wave_vector` | 0.36250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `parallel_per_toroidal_mode_current_density_due_to_wave_driven_current_drive` | 0.75000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `parallel_per_toroidal_mode_electric_field` | 0.45000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `parallel_wave_electric_field_amplitude` | 0.82500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `particle_count` | 0.55000 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `particle_flux` | 0.83125 | `superseded` | 1 | `—` | 1 | `total_particle_flux` | 0.80000 | `grammar_invalid` |
| `particle_pressure` | 0.56250 | `superseded` | 1 | `—` | 2 | `total_plasma_pressure` | 0.91250 | `accepted` |
| `per_toroidal_mode_current_due_to_wave_driven_current_drive` | 0.82500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `perpendicular_normalized_gyrocenter_perturbed_pressure` | 0.78750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `perpendicular_normalized_perturbed_pressure` | 0.72500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `perpendicular_wave_vector_magnitude` | 0.76250 | `superseded` | 1 | `—` | 1 | `cross_field_wave_vector_magnitude` | 0.73750 | `successor_collision` |
| `plasma_frequency_at_measurement_position` | 0.63750 | `superseded` | 1 | `—` | 3 | `wave_critical_ordinary_mode_frequency` | 0.66250 | `attempts_exhausted` |
| `plasma_velocity_due_to_diamagnetic_drift` | 0.75000 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `poloidal_angle_at_plasma_boundary_gap_reference_point` | 0.62500 | `superseded` | 1 | `—` | 3 | `poloidal_angle_of_plasma_boundary_gap` | 0.93750 | `accepted` |
| `poloidal_center_of_mass_velocity` | 0.75000 | `superseded` | 1 | `—` | 3 | `poloidal_bulk_center_of_mass_velocity` | 0.89375 | `accepted` |
| `poloidal_current_density_due_to_viscosity` | 0.51250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `poloidal_diamagnetic_current_density` | 0.53750 | `superseded` | 1 | `—` | 1 | `poloidal_current_density_due_to_diamagnetic_drift` | 1.00000 | `accepted` |
| `poloidal_electron_energy_diffusion_coefficient` | 0.70000 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `poloidal_ion_charge_state_momentum_diffusivity` | 0.81250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `poloidal_ion_diffusivity` | 0.68750 | `superseded` | 1 | `—` | 2 | `poloidal_effective_ion_diffusion_coefficient` | 0.81250 | `successor_collision` |
| `poloidal_neutral_energy_flux` | 0.83750 | `superseded` | 1 | `—` | 1 | `poloidal_neutral_species_energy_flux` | 0.99375 | `accepted` |
| `poloidal_neutral_momentum_diffusivity` | 0.83750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `power_density` | 0.60000 | `superseded` | 1 | `—` | 3 | `net_plasma_power_density` | 0.72500 | `attempts_exhausted` |
| `power_due_to_impurity_radiation` | 0.58750 | `superseded` | 1 | `—` | 2 | `power_over_core_region_due_to_impurity_radiation` | 0.58750 | `grammar_invalid` |
| `power_due_to_ohmic_dissipation` | 0.78125 | `superseded` | 1 | `—` | 1 | `total_power_due_to_ohmic_dissipation` | 0.88750 | `accepted` |
| `power_of_beam_tracing_beam` | 0.60000 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `power_of_divertor` | 0.75000 | `superseded` | 1 | `—` | 3 | `deposited_power_at_divertor_target` | 0.51250 | `attempts_exhausted` |
| `power_of_divertor_due_to_fusion` | 0.73750 | `superseded` | 1 | `—` | 3 | `nuclear_heating_power_of_divertor` | 0.97500 | `accepted` |
| `power_of_divertor_due_to_radiation` | 0.53750 | `superseded` | 1 | `—` | 1 | `total_thermal_radiative_power_of_divertor_target` | 0.45000 | `successor_collision` |
| `power_of_ion_cyclotron_heating_antenna` | 0.62500 | `superseded` | 1 | `—` | 3 | `launched_power_of_ion_cyclotron_heating_antenna` | 0.65000 | `attempts_exhausted` |
| `power_of_neutral_beam_injector` | 0.63750 | `exhausted` | 2 | `grammar_invalid` | 0 | — | — | `—` |
| `pressure_of_lower_hybrid_antenna` | 0.83750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `radial_angle_of_poloidal_field_coil` | 0.52500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `radial_centroid_of_electron_cyclotron_launcher_mirror` | 0.43750 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `radial_coordinate_of_aperture` | 0.82500 | `superseded` | 1 | `—` | 1 | `radial_coordinate_of_diagnostic_aperture` | 0.98125 | `accepted` |
| `radial_coordinate_of_lower_hybrid_antenna_row` | 0.52500 | `superseded` | 1 | `—` | 3 | `radial_offset_of_lower_hybrid_antenna` | 0.76250 | `attempts_exhausted` |
| `radial_coordinate_of_shattering_position` | 0.56250 | `superseded` | 1 | `—` | 1 | `radial_coordinate_of_shatter_cone` | 0.97500 | `accepted` |
| `radial_derivative_of_elongation_of_flux_surface` | 0.77500 | `exhausted` | 3 | `attempts_exhausted` | 0 | — | — | `—` |
| `radial_electron_diffusion_coefficient` | 0.76250 | `superseded` | 1 | `—` | 2 | `electron_particle_diffusion_coefficient` | 0.68750 | `successor_collision` |
| `radial_energy_convection_velocity` | 0.57500 | `superseded` | 1 | `—` | 1 | `radial_effective_thermal_energy_velocity_due_to_convection` | 0.46250 | `successor_collision` |
| `radial_ion_charge_state_particle_flux` | 0.80000 | `superseded` | 1 | `—` | 1 | `flux_surface_normal_ion_charge_state_particle_flux` | 1.00000 | `accepted` |
| `radial_ion_convection_velocity` | 0.76250 | `superseded` | 1 | `—` | 1 | `radial_ion_particle_convection_velocity` | 1.00000 | `accepted` |
| `radial_ion_energy_convection_velocity` | 0.62500 | `exhausted` | 2 | `successor_collision` | 0 | — | — | `—` |
| `radial_ion_energy_diffusion_coefficient` | 0.81250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `radial_ion_state_energy_flux` | 0.82500 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `radial_momentum_diffusivity` | 0.75000 | `superseded` | 1 | `—` | 2 | `flux_surface_normal_momentum_diffusion_coefficient` | 0.63750 | `grammar_invalid` |
| `radial_neutral_energy_diffusion_coefficient` | 0.81250 | `superseded` | 1 | `—` | 3 | `flux_surface_normal_neutral_energy_diffusion_coefficient` | 0.77500 | `attempts_exhausted` |
| `radial_plasma_momentum_diffusion_coefficient` | 0.75000 | `superseded` | 1 | `—` | 3 | `flux_surface_normal_plasma_momentum_diffusivity` | 0.90000 | `accepted` |
| `radiated_power_density` | 0.78125 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `radiated_power_over_scrape_off_layer` | 0.66250 | `superseded` | 1 | `—` | 3 | `power_over_scrape_off_layer_due_to_radiation` | 0.58750 | `attempts_exhausted` |
| `radiative_temperature` | 0.76250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `radius_of_filter` | 0.63750 | `superseded` | 1 | `—` | 1 | `radius_of_filter_window` | 1.00000 | `accepted` |
| `radius_of_iron_core_segment` | 0.55000 | `superseded` | 1 | `—` | 2 | `inverse_of_curvature_of_arc_of_circle_center` | 0.46250 | `grammar_invalid` |
| `radius_of_plasma_filament` | 0.56875 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `radius_of_poloidal_field_coil` | 0.53750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `reference_wavelength_of_filter` | 0.63750 | `superseded` | 1 | `—` | 1 | `reference_wavelength_of_filter_window` | 0.66250 | `successor_collision` |
| `requested_voltage_of_spectrometer` | 0.63750 | `superseded` | 1 | `—` | 1 | `requested_voltage_of_spectrometer_channel` | 1.00000 | `accepted` |
| `rotational_transform` | 0.30000 | `reviewed` | 0 | `preserved non-quorate` | 0 | — | — | `—` |
| `runaway_electron_energy_density` | 0.82500 | `superseded` | 1 | `—` | 1 | `runaway_electron_kinetic_energy_density` | 0.97500 | `accepted` |
| `runaway_electron_source_rate` | 0.75000 | `superseded` | 1 | `—` | 3 | `tendency_of_runaway_electron_density` | 0.71250 | `attempts_exhausted` |
| `shattered_pellet_fragment_density` | 0.56250 | `superseded` | 1 | `—` | 1 | `shattered_pellet_species_number_density_of_pellet_fragment` | 0.97500 | `accepted` |
| `shattered_pellet_fragment_volume` | 0.83750 | `superseded` | 1 | `—` | 1 | `volume_of_pellet_fragment` | 1.00000 | `accepted` |
| `silane_prefill_count` | 0.77500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `source_rate_due_to_injection` | 0.64375 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `spectral_calibration_factor_of_spectrometer` | 0.63125 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `spectral_radiance_of_soft_xray_detector` | 0.62500 | `superseded` | 1 | `—` | 1 | `incident_soft_xray_radiance` | 0.95625 | `accepted` |
| `spectral_width_of_filter` | 0.80000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `spectral_width_of_spectrometer_channel` | 0.53750 | `superseded` | 1 | `—` | 3 | `root_mean_square_of_spectral_width_of_spectrometer_channel` | 0.58750 | `attempts_exhausted` |
| `surface_area_of_optical_element` | 0.60000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `surface_temperature` | 0.75000 | `superseded` | 1 | `—` | 1 | `surface_temperature_of_plasma_facing_component` | 0.81250 | `grammar_invalid` |
| `temperature_of_poloidal_field_coil` | 0.67500 | `superseded` | 1 | `—` | 2 | `temperature_of_coil_conductor` | 0.88750 | `accepted` |
| `thermal_electron_decay_length_over_scrape_off_layer` | 0.56250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `thermal_energy_of_plant_component_port` | 0.71250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `thermal_ion_power_density` | 0.80000 | `superseded` | 1 | `—` | 1 | `thermal_ion_absorbed_wave_power_density` | 0.95000 | `accepted` |
| `thermal_power_of_divertor` | 0.81250 | `superseded` | 1 | `—` | 1 | `heat_power_of_divertor` | 0.68750 | `successor_collision` |
| `thermal_power_of_plant_component_port` | 0.81250 | `superseded` | 1 | `—` | 3 | `absorbed_coolant_power_of_plant_component_port` | 0.55000 | `attempts_exhausted` |
| `thickness_of_breeder_blanket_module` | 0.51250 | `superseded` | 1 | `—` | 2 | `surface_thickness_of_breeder_blanket_module` | 0.58750 | `vocabulary_gap` |
| `thickness_of_cryostat` | 0.66250 | `superseded` | 1 | `—` | 3 | `surface_thickness_of_cryostat` | 0.59375 | `attempts_exhausted` |
| `thickness_of_passive_loop` | 0.81250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `thickness_of_plasma_filament` | 0.52500 | `superseded` | 1 | `—` | 3 | `normal_width_of_plasma_filament` | 0.56875 | `attempts_exhausted` |
| `time_derivative_of_flux_surface_averaged_metric` | 0.50000 | `superseded` | 1 | `—` | 1 | `tendency_of_derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface` | 0.98750 | `accepted` |
| `time_derivative_of_mode_width` | 0.61250 | `superseded` | 1 | `—` | 1 | `time_derivative_of_radial_width_of_neoclassical_tearing_mode` | 0.98750 | `accepted` |
| `toroidal_angle_of_magnetic_field_probe` | 0.68125 | `superseded` | 1 | `—` | 1 | `toroidal_angle_of_poloidal_magnetic_field_probe` | 1.00000 | `accepted` |
| `toroidal_co_passing_fast_electron_torque_density_due_to_collisions` | 0.67500 | `superseded` | 1 | `—` | 1 | `toroidal_co_passing_fast_electron_torque_density_due_to_collisional_transport` | 0.78750 | `grammar_invalid` |
| `toroidal_current_density` | 0.71875 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `toroidal_fast_electron_torque_density_due_to_collisions` | 0.75625 | `superseded` | 1 | `—` | 1 | `toroidal_fast_particle_torque_density_due_to_coulomb_collisions_with_electrons` | 0.85000 | `accepted` |
| `toroidal_fast_electron_torque_due_to_collisions` | 0.66250 | `superseded` | 1 | `—` | 1 | `toroidal_volume_integrated_fast_electron_torque_density_due_to_collisions` | 0.56250 | `grammar_invalid` |
| `toroidal_ion_momentum_diffusion_coefficient` | 0.82500 | `exhausted` | 3 | `attempts_exhausted` | 0 | — | — | `—` |
| `toroidal_ion_torque` | 0.76250 | `superseded` | 1 | `—` | 1 | `toroidal_explicit_ion_torque` | 0.91250 | `accepted` |
| `toroidal_normalized_wave_vector_of_beam_tracing_beam` | 0.41250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `toroidal_plasma_momentum_at_measurement_position` | 0.61250 | `superseded` | 1 | `—` | 2 | `toroidal_total_plasma_volumetric_angular_momentum_at_measurement_position` | 0.95000 | `accepted` |
| `toroidal_total_plasma_momentum_at_plasma_boundary` | 0.77500 | `superseded` | 1 | `—` | 1 | `toroidal_cumulative_inside_flux_surface_total_plasma_momentum_at_separatrix` | 0.66250 | `grammar_invalid` |
| `toroidal_trapped_fast_electron_torque_density_due_to_collisions` | 0.65000 | `superseded` | 1 | `—` | 1 | `toroidal_trapped_fast_electron_torque_density_due_to_collisional_transport` | 0.70000 | `successor_collision` |
| `toroidal_trapped_thermal_electron_torque_density_due_to_coulomb_collisions_with_electrons` | 0.79375 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `toroidal_width_of_antenna_strap` | 0.58750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `total_ion_density` | 0.58750 | `superseded` | 1 | `—` | 1 | `total_ion_number_density` | 0.92500 | `accepted` |
| `total_ion_power_density` | 0.78750 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `total_ion_pressure` | 0.81250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `total_neutral_density` | 0.81250 | `superseded` | 1 | `—` | 1 | `total_neutral_number_density` | 0.98125 | `accepted` |
| `total_neutron_power` | 0.75000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `total_particle_flux_of_divertor_due_to_recycling` | 0.80000 | `superseded` | 1 | `—` | 3 | `total_particle_flux_at_divertor_target_due_to_recycling` | 0.53750 | `attempts_exhausted` |
| `total_plasma_energy` | 0.83125 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `total_power_of_ion_cyclotron_heating_antenna` | 0.71250 | `superseded` | 1 | `—` | 3 | `total_launched_power_due_to_ion_cyclotron_heating` | 0.75000 | `attempts_exhausted` |
| `total_power_of_neutral_beam_injector` | 0.68750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `total_power_of_plant_system` | 0.63750 | `superseded` | 1 | `—` | 1 | `total_absorbed_power_of_plant_system` | 0.57500 | `successor_collision` |
| `total_thermal_particle_source_rate` | 0.69375 | `superseded` | 1 | `—` | 1 | `volume_integrated_particle_source_rate` | 0.89375 | `accepted` |
| `total_thermal_power_at_inlet` | 0.72500 | `superseded` | 1 | `—` | 3 | `total_incident_thermal_power` | 0.67500 | `attempts_exhausted` |
| `trapped_fast_particle_power_density_due_to_collisions` | 0.77500 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `tritium_prefill_count` | 0.67500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `tritium_tritium_neutron_source_rate_due_to_thermal_fusion` | 0.82500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `tungsten_density` | 0.66250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `upper_bound_wavelength_of_camera` | 0.70000 | `superseded` | 1 | `—` | 1 | `upper_bound_wavelength_of_visible_camera` | 0.97500 | `accepted` |
| `vertical_angle_of_poloidal_field_coil` | 0.50000 | `superseded` | 1 | `—` | 1 | `tilt_angle_of_poloidal_field_coil` | 0.83750 | `grammar_invalid` |
| `vertical_coordinate_of_active_limiter_point` | 0.50000 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `vertical_coordinate_of_bragg_crystal` | 0.66875 | `superseded` | 1 | `—` | 2 | `vertical_position_of_grating` | 0.58750 | `grammar_invalid` |
| `vertical_coordinate_of_closest_wall_point` | 0.63750 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `vertical_coordinate_of_coil_conductor` | 0.53750 | `exhausted` | 3 | `attempts_exhausted` | 0 | — | — | `—` |
| `vertical_coordinate_of_optical_element` | 0.67500 | `superseded` | 1 | `—` | 2 | `vertical_coordinate_of_fibre_bundle` | 0.97500 | `accepted` |
| `vertical_coordinate_of_plasma_boundary` | 0.72500 | `superseded` | 1 | `—` | 3 | `vertical_coordinate_of_plasma_filament` | 0.66875 | `attempts_exhausted` |
| `vertical_current_density` | 0.66250 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `vertical_front_surface_curvature_of_optical_element` | 0.71250 | `superseded` | 1 | `—` | 2 | `second_local_tangential_front_surface_radius_of_optical_element` | 0.93750 | `accepted` |
| `vertical_ion_momentum_diffusivity` | 0.81250 | `superseded` | 1 | `—` | 1 | `vertical_total_ion_momentum_diffusivity` | 0.70000 | `grammar_invalid` |
| `vertical_magnetic_field_at_wall` | 0.73750 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `vertical_outline_of_vacuum_vessel` | 0.58750 | `superseded` | 1 | `—` | 1 | `vertical_outline_of_wall` | 0.65000 | `successor_collision` |
| `vertical_outline_of_wall_material` | 0.63750 | `superseded` | 1 | `—` | 1 | `vertical_outline_of_plasma_facing_component` | 0.99375 | `accepted` |
| `voltage_amplitude` | 0.62500 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `voltage_of_diagnostic_antenna` | 0.62500 | `exhausted` | 2 | `grammar_invalid` | 0 | — | — | `—` |
| `voltage_of_reflectometer_antenna` | 0.60000 | `superseded` | 1 | `—` | 1 | `wave_voltage_amplitude` | 0.90000 | `accepted` |
| `voltage_of_spectrometer` | 0.60000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `voltage_of_temperature_sensor` | 0.75000 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `volume_averaged_runaway_electron_source_rate` | 0.61250 | `exhausted` | 1 | `grammar_invalid` | 0 | — | — | `—` |
| `wave_beam_energy` | 0.54375 | `superseded` | 1 | `—` | 1 | `thomson_scattering_laser_pulse_energy_at_outlet` | 0.87500 | `accepted` |
| `wave_beam_energy_at_launching_position` | 0.52500 | `superseded` | 1 | `—` | 2 | `launched_pulse_energy_of_thomson_scattering_laser` | 0.95625 | `accepted` |
| `wave_magnetic_field` | 0.52500 | `superseded` | 1 | `—` | 3 | `wave_magnetic_field_amplitude` | 0.50000 | `attempts_exhausted` |
| `wavelength` | 0.42500 | `exhausted` | 3 | `attempts_exhausted` | 0 | — | — | `—` |
| `width_of_aperture` | 0.58750 | `exhausted` | 1 | `successor_collision` | 0 | — | — | `—` |
| `width_of_hard_xray_detector` | 0.74375 | `superseded` | 1 | `—` | 1 | `first_local_tangential_width_of_hard_xray_detector` | 0.98750 | `accepted` |
| `width_of_neutron_detector` | 0.73750 | `superseded` | 1 | `—` | 1 | `first_local_tangential_width_of_diagnostic_aperture` | 0.96875 | `accepted` |
| `width_of_optical_element` | 0.52500 | `exhausted` | 1 | `vocabulary_gap` | 0 | — | — | `—` |
| `width_of_reflectometer_antenna` | 0.58750 | `superseded` | 1 | `—` | 1 | `first_local_tangential_width_of_reflectometer_antenna` | 0.56250 | `successor_collision` |
| `xenon_source_rate` | 0.78750 | `superseded` | 1 | `—` | 1 | `xenon_source_rate_due_to_gas_injection` | 1.00000 | `accepted` |

## Every successor and score delta

All **271** `REFINED_FROM` successor edges are listed. Intermediate successors correctly remain `superseded`; each root’s final successor is `accepted` or `exhausted`. No successor is drafted, reviewed, refining, or claimed.

| Root | Predecessor | Before | Successor | After | Δ | Current successor state |
|---|---|---:|---|---:|---:|---|
| `accumulated_lithium_count` | `accumulated_lithium_count` | 0.84375 | `accumulated_lithium_count_due_to_gas_injection` | 1.00000 | +0.15625 | `accepted` |
| `accumulated_nitrogen_count` | `accumulated_nitrogen_count` | 0.83750 | `accumulated_nitrogen_count_due_to_gas_injection` | 1.00000 | +0.16250 | `accepted` |
| `accumulated_propane_count` | `accumulated_propane_count` | 0.81250 | `accumulated_propane_count_due_to_gas_injection` | 1.00000 | +0.18750 | `accepted` |
| `accumulated_silane_count` | `accumulated_silane_count` | 0.81250 | `accumulated_silane_count_due_to_gas_injection` | 1.00000 | +0.18750 | `accepted` |
| `accumulated_xenon_count` | `accumulated_xenon_count` | 0.83750 | `accumulated_xenon_count_due_to_gas_injection` | 1.00000 | +0.16250 | `accepted` |
| `area_of_langmuir_probe` | `area_of_langmuir_probe` | 0.80625 | `surface_area_of_langmuir_probe` | 0.75000 | -0.05625 | `superseded` |
| `area_of_langmuir_probe` | `surface_area_of_langmuir_probe` | 0.75000 | `total_surface_area_of_langmuir_probe` | 0.75000 | +0.00000 | `superseded` |
| `area_of_langmuir_probe` | `total_surface_area_of_langmuir_probe` | 0.75000 | `front_surface_area_of_langmuir_probe` | 0.58750 | -0.16250 | `attempts_exhausted` |
| `area_of_neutral_beam_injector` | `area_of_neutral_beam_injector` | 0.83750 | `beam_cross_sectional_area_of_aperture` | 0.71250 | -0.12500 | `successor_collision` |
| `back_surface_curvature_of_optical_element` | `back_surface_curvature_of_optical_element` | 0.55000 | `first_local_tangential_back_surface_curvature_of_optical_element` | 0.80000 | +0.25000 | `superseded` |
| `back_surface_curvature_of_optical_element` | `first_local_tangential_back_surface_curvature_of_optical_element` | 0.80000 | `inverse_of_first_local_tangential_back_surface_curvature_of_optical_element` | 0.57500 | -0.22500 | `superseded` |
| `back_surface_curvature_of_optical_element` | `inverse_of_first_local_tangential_back_surface_curvature_of_optical_element` | 0.57500 | `first_local_tangential_back_surface_radius_of_optical_element` | 0.67500 | +0.10000 | `attempts_exhausted` |
| `beryllium_source_rate` | `beryllium_source_rate` | 0.73750 | `beryllium_source_rate_due_to_gas_injection` | 1.00000 | +0.26250 | `accepted` |
| `bragg_crystal_width` | `bragg_crystal_width` | 0.48750 | `first_local_tangential_width_of_bragg_crystal` | 0.99375 | +0.50625 | `accepted` |
| `co_passing_fast_electron_power_density_due_to_collisions` | `co_passing_fast_electron_power_density_due_to_collisions` | 0.61250 | `co_passing_fast_electron_kinetic_energy_power_density_due_to_collisions` | 0.41250 | -0.20000 | `superseded` |
| `co_passing_fast_electron_power_density_due_to_collisions` | `co_passing_fast_electron_kinetic_energy_power_density_due_to_collisions` | 0.41250 | `co_passing_fast_electron_kinetic_power_density_due_to_collisions` | 0.48750 | +0.07500 | `attempts_exhausted` |
| `conductivity` | `conductivity` | 0.58750 | `plasma_electrical_conductivity` | 0.92500 | +0.33750 | `accepted` |
| `coolant_temperature` | `coolant_temperature` | 0.80000 | `coolant_temperature_of_plant_component_port` | 0.93750 | +0.13750 | `accepted` |
| `coolant_volume_of_breeder_blanket` | `coolant_volume_of_breeder_blanket` | 0.75000 | `total_coolant_volume_of_breeder_blanket` | 0.62500 | -0.12500 | `superseded` |
| `coolant_volume_of_breeder_blanket` | `total_coolant_volume_of_breeder_blanket` | 0.62500 | `volume_of_breeder_blanket` | 0.64375 | +0.01875 | `superseded` |
| `coolant_volume_of_breeder_blanket` | `volume_of_breeder_blanket` | 0.64375 | `lithium_volume_of_breeder_blanket` | 0.63750 | -0.00625 | `attempts_exhausted` |
| `counter_passing_density` | `counter_passing_density` | 0.56250 | `total_counter_passing_gyrocenter_density` | 0.66250 | +0.10000 | `superseded` |
| `counter_passing_density` | `total_counter_passing_gyrocenter_density` | 0.66250 | `total_counter_passing_particle_density` | 1.00000 | +0.33750 | `accepted` |
| `current_due_to_ohmic_induction` | `current_due_to_ohmic_induction` | 0.76250 | `net_current_due_to_ohmic_induction` | 0.73750 | -0.02500 | `superseded` |
| `current_due_to_ohmic_induction` | `net_current_due_to_ohmic_induction` | 0.73750 | `net_plasma_current_due_to_ohmic_current_drive` | 0.53750 | -0.20000 | `successor_collision` |
| `current_of_correction_coil` | `current_of_correction_coil` | 0.72500 | `non_axisymmetric_current_of_coil_conductor` | 0.56250 | -0.16250 | `superseded` |
| `current_of_correction_coil` | `non_axisymmetric_current_of_coil_conductor` | 0.56250 | `current_of_coil_conductor` | 0.65000 | +0.08750 | `superseded` |
| `current_of_correction_coil` | `current_of_coil_conductor` | 0.65000 | `non_axisymmetric_current_of_conductor` | 0.51250 | -0.13750 | `attempts_exhausted` |
| `curvature_of_optical_element` | `curvature_of_optical_element` | 0.66250 | `inverse_of_tangential_surface_curvature_of_optical_element` | 0.51250 | -0.15000 | `superseded` |
| `curvature_of_optical_element` | `inverse_of_tangential_surface_curvature_of_optical_element` | 0.51250 | `tangential_surface_radius_of_optical_element` | 0.42500 | -0.08750 | `superseded` |
| `curvature_of_optical_element` | `tangential_surface_radius_of_optical_element` | 0.42500 | `inverse_of_tangential_curvature_of_optical_element` | 0.47500 | +0.05000 | `attempts_exhausted` |
| `deposited_power` | `deposited_power` | 0.68750 | `volume_integrated_net_plasma_particle_power_density` | 0.61875 | -0.06875 | `grammar_invalid` |
| `deuterium_count_due_to_gas_injection` | `deuterium_count_due_to_gas_injection` | 0.73750 | `accumulated_deuterium_count_due_to_gas_injection` | 1.00000 | +0.26250 | `accepted` |
| `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | 0.73750 | `total_deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.80000 | +0.06250 | `superseded` |
| `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | `total_deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.80000 | `volume_integrated_deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.77500 | -0.02500 | `superseded` |
| `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | `volume_integrated_deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.77500 | `deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.98125 | +0.20625 | `accepted` |
| `deuterium_tritium_neutron_flux_due_to_beam_beam_fusion` | `deuterium_tritium_neutron_flux_due_to_beam_beam_fusion` | 0.68750 | `total_deuterium_tritium_neutron_source_rate_due_to_beam_beam_fusion` | 0.51250 | -0.17500 | `successor_collision` |
| `diamagnetic_velocity_due_to_diamagnetic_drift` | `diamagnetic_velocity_due_to_diamagnetic_drift` | 0.38750 | `binormal_velocity_due_to_diamagnetic_drift` | 0.72500 | +0.33750 | `superseded` |
| `diamagnetic_velocity_due_to_diamagnetic_drift` | `binormal_velocity_due_to_diamagnetic_drift` | 0.72500 | `binormal_ion_velocity_due_to_diamagnetic_drift` | 0.91250 | +0.18750 | `accepted` |
| `effective_charge_at_plasma_boundary` | `effective_charge_at_plasma_boundary` | 0.67500 | `effective_charge_number_at_plasma_boundary` | 0.58750 | -0.08750 | `superseded` |
| `effective_charge_at_plasma_boundary` | `effective_charge_number_at_plasma_boundary` | 0.58750 | `effective_charge_at_separatrix` | 1.00000 | +0.41250 | `accepted` |
| `efficiency_of_neutron_detector` | `efficiency_of_neutron_detector` | 0.80000 | `spectral_efficiency_of_neutron_detector` | 0.98125 | +0.18125 | `accepted` |
| `efficiency_of_spectrometer_channel` | `efficiency_of_spectrometer_channel` | 0.71250 | `transmissivity_of_spectrometer_channel` | 0.57500 | -0.13750 | `superseded` |
| `efficiency_of_spectrometer_channel` | `transmissivity_of_spectrometer_channel` | 0.57500 | `viewing_efficiency_of_spectrometer_channel` | 0.88750 | +0.31250 | `accepted` |
| `electron_density_over_scrape_off_layer` | `electron_density_over_scrape_off_layer` | 0.58750 | `volume_averaged_electron_density_over_scrape_off_layer` | 0.56250 | -0.02500 | `superseded` |
| `electron_density_over_scrape_off_layer` | `volume_averaged_electron_density_over_scrape_off_layer` | 0.56250 | `volume_averaged_electron_number_density_over_scrape_off_layer` | 0.58750 | +0.02500 | `grammar_invalid` |
| `electron_diffusivity` | `electron_diffusivity` | 0.77500 | `electron_particle_diffusion_coefficient_due_to_diffusion` | 0.48750 | -0.28750 | `superseded` |
| `electron_diffusivity` | `electron_particle_diffusion_coefficient_due_to_diffusion` | 0.48750 | `electron_particle_diffusivity` | 0.76250 | +0.27500 | `successor_collision` |
| `electron_energy_diffusivity` | `electron_energy_diffusivity` | 0.67500 | `thermal_electron_energy_diffusivity` | 0.83750 | +0.16250 | `superseded` |
| `electron_energy_diffusivity` | `thermal_electron_energy_diffusivity` | 0.83750 | `electron_heat_diffusivity` | 0.80625 | -0.03125 | `successor_collision` |
| `electron_energy_flux_at_wall` | `electron_energy_flux_at_wall` | 0.82500 | `electron_energy_flux_at_wall_due_to_surface_emission` | 0.78125 | -0.04375 | `superseded` |
| `electron_energy_flux_at_wall` | `electron_energy_flux_at_wall_due_to_surface_emission` | 0.78125 | `electron_kinetic_energy_flux_at_wall_due_to_surface_emission` | 1.00000 | +0.21875 | `accepted` |
| `electron_particle_flux_at_wall` | `electron_particle_flux_at_wall` | 0.71875 | `electron_particle_flux_at_wall_due_to_surface_emission` | 0.96875 | +0.25000 | `accepted` |
| `electron_pressure_at_plasma_boundary` | `electron_pressure_at_plasma_boundary` | 0.78125 | `electron_pressure_at_separatrix` | 1.00000 | +0.21875 | `accepted` |
| `electron_temperature_at_plasma_boundary` | `electron_temperature_at_plasma_boundary` | 0.68750 | `bulk_electron_temperature_at_plasma_boundary` | 0.63750 | -0.05000 | `superseded` |
| `electron_temperature_at_plasma_boundary` | `bulk_electron_temperature_at_plasma_boundary` | 0.63750 | `thermal_electron_temperature_at_plasma_boundary` | 0.65000 | +0.01250 | `superseded` |
| `electron_temperature_at_plasma_boundary` | `thermal_electron_temperature_at_plasma_boundary` | 0.65000 | `electron_temperature_at_separatrix` | 0.67500 | +0.02500 | `attempts_exhausted` |
| `electron_temperature_at_wall` | `electron_temperature_at_wall` | 0.73750 | `electron_temperature_at_first_wall` | 0.66250 | -0.07500 | `grammar_invalid` |
| `electrostatic_potential_at_midplane` | `electrostatic_potential_at_midplane` | 0.62500 | `plasma_electrostatic_potential_at_midplane` | 0.52500 | -0.10000 | `superseded` |
| `electrostatic_potential_at_midplane` | `plasma_electrostatic_potential_at_midplane` | 0.52500 | `electrostatic_potential_at_outboard_midplane` | 0.61250 | +0.08750 | `superseded` |
| `electrostatic_potential_at_midplane` | `electrostatic_potential_at_outboard_midplane` | 0.61250 | `plasma_electrostatic_potential_at_outboard_midplane` | 0.53750 | -0.07500 | `attempts_exhausted` |
| `electrostatic_potential_at_wall` | `electrostatic_potential_at_wall` | 0.82500 | `electrostatic_potential_at_first_wall` | 0.65000 | -0.17500 | `superseded` |
| `electrostatic_potential_at_wall` | `electrostatic_potential_at_first_wall` | 0.65000 | `plasma_electrostatic_potential_at_first_wall` | 0.65000 | +0.00000 | `superseded` |
| `electrostatic_potential_at_wall` | `plasma_electrostatic_potential_at_first_wall` | 0.65000 | `plasma_electrostatic_potential_at_wall` | 0.61250 | -0.03750 | `attempts_exhausted` |
| `energy_confinement_time` | `energy_confinement_time` | 0.83750 | `thermal_energy_confinement_time` | 0.77500 | -0.06250 | `grammar_invalid` |
| `energy_diffusivity` | `energy_diffusivity` | 0.57500 | `thermal_energy_diffusivity` | 0.58750 | +0.01250 | `superseded` |
| `energy_diffusivity` | `thermal_energy_diffusivity` | 0.58750 | `energy_diffusion_coefficient_due_to_diffusion` | 0.72500 | +0.13750 | `successor_collision` |
| `energy_flux` | `energy_flux` | 0.82500 | `normal_energy_flux` | 0.65000 | -0.17500 | `superseded` |
| `energy_flux` | `normal_energy_flux` | 0.65000 | `normal_energy_flux_at_control_surface` | 0.55000 | -0.10000 | `superseded` |
| `energy_flux` | `normal_energy_flux_at_control_surface` | 0.55000 | `energy_flux_at_control_surface` | 0.56250 | +0.01250 | `attempts_exhausted` |
| `energy_flux_at_wall_due_to_radiation` | `energy_flux_at_wall_due_to_radiation` | 0.66250 | `incident_energy_flux_at_wall_due_to_radiation` | 0.55000 | -0.11250 | `grammar_invalid` |
| `ethylene_count` | `ethylene_count` | 0.60000 | `accumulated_ethylene_count_due_to_gas_injection` | 0.80000 | +0.20000 | `superseded` |
| `ethylene_count` | `accumulated_ethylene_count_due_to_gas_injection` | 0.80000 | `cumulative_ethylene_count_due_to_gas_injection` | 0.65000 | -0.15000 | `grammar_invalid` |
| `fast_electron_power_density` | `fast_electron_power_density` | 0.80000 | `flux_surface_averaged_fast_electron_absorbed_wave_power_density` | 0.83125 | +0.03125 | `superseded` |
| `fast_electron_power_density` | `flux_surface_averaged_fast_electron_absorbed_wave_power_density` | 0.83125 | `fast_electron_absorbed_wave_power_density` | 0.98125 | +0.15000 | `accepted` |
| `fast_electron_power_density_due_to_collisions` | `fast_electron_power_density_due_to_collisions` | 0.70000 | `total_suprathermal_electron_power_density_due_to_collisions` | 0.76250 | +0.06250 | `grammar_invalid` |
| `filter_window_width` | `filter_window_width` | 0.57500 | `first_local_tangential_width_of_filter_window` | 1.00000 | +0.42500 | `accepted` |
| `flux_due_to_recombination` | `flux_due_to_recombination` | 0.52500 | `normal_particle_flux_due_to_recombination` | 0.56250 | +0.03750 | `superseded` |
| `flux_due_to_recombination` | `normal_particle_flux_due_to_recombination` | 0.56250 | `normal_particle_flux_at_wall_due_to_recombination` | 0.72500 | +0.16250 | `successor_collision` |
| `flux_surface_averaged_electron_temperature_at_plasma_boundary` | `flux_surface_averaged_electron_temperature_at_plasma_boundary` | 0.76250 | `flux_surface_averaged_bulk_electron_temperature_at_last_closed_flux_surface` | 0.80000 | +0.03750 | `grammar_invalid` |
| `frequency_of_electron_cyclotron_heating_antenna` | `frequency_of_electron_cyclotron_heating_antenna` | 0.71250 | `frequency_of_electron_cyclotron_beam` | 1.00000 | +0.28750 | `accepted` |
| `gap_of_antenna_strap` | `gap_of_antenna_strap` | 0.75000 | `normal_gap_at_wall` | 0.51250 | -0.23750 | `superseded` |
| `gap_of_antenna_strap` | `normal_gap_at_wall` | 0.51250 | `normal_gap_of_antenna_strap` | 0.76875 | +0.25625 | `superseded` |
| `gap_of_antenna_strap` | `normal_gap_of_antenna_strap` | 0.76875 | `normal_distance_of_antenna_strap` | 0.63125 | -0.13750 | `attempts_exhausted` |
| `incident_power_of_breeder_blanket_module` | `incident_power_of_breeder_blanket_module` | 0.73125 | `incident_neutron_power_of_breeder_blanket_module` | 0.98750 | +0.25625 | `accepted` |
| `ion_atomic_number` | `ion_atomic_number` | 0.67500 | `ion_species_atomic_number` | 0.58750 | -0.08750 | `successor_collision` |
| `ion_momentum_diffusivity` | `ion_momentum_diffusivity` | 0.56250 | `total_ion_momentum_diffusivity` | 0.48750 | -0.07500 | `successor_collision` |
| `ion_particle_flux_at_wall` | `ion_particle_flux_at_wall` | 0.65000 | `ion_particle_flux_at_wall_due_to_surface_emission` | 0.71875 | +0.06875 | `superseded` |
| `ion_particle_flux_at_wall` | `ion_particle_flux_at_wall_due_to_surface_emission` | 0.71875 | `total_ion_particle_flux_at_wall_due_to_surface_emission` | 0.80000 | +0.08125 | `superseded` |
| `ion_particle_flux_at_wall` | `total_ion_particle_flux_at_wall_due_to_surface_emission` | 0.80000 | `ion_species_particle_flux_at_wall_due_to_surface_emission` | 0.72500 | -0.07500 | `attempts_exhausted` |
| `ion_temperature_at_midplane` | `ion_temperature_at_midplane` | 0.61250 | `scrape_off_layer_ion_temperature_at_outboard_midplane` | 0.52500 | -0.08750 | `superseded` |
| `ion_temperature_at_midplane` | `scrape_off_layer_ion_temperature_at_outboard_midplane` | 0.52500 | `ion_temperature_at_outboard_midplane` | 0.65000 | +0.12500 | `superseded` |
| `ion_temperature_at_midplane` | `ion_temperature_at_outboard_midplane` | 0.65000 | `ion_temperature_at_outboard_midplane_separatrix` | 0.66250 | +0.01250 | `attempts_exhausted` |
| `launched_power_of_wave_beam` | `launched_power_of_wave_beam` | 0.72500 | `net_launched_power_of_wave_beam` | 0.66250 | -0.06250 | `superseded` |
| `launched_power_of_wave_beam` | `net_launched_power_of_wave_beam` | 0.66250 | `difference_of_forward_power_of_wave_beam_and_reflected_power_of_wave_beam` | 0.48750 | -0.17500 | `superseded` |
| `launched_power_of_wave_beam` | `difference_of_forward_power_of_wave_beam_and_reflected_power_of_wave_beam` | 0.48750 | `net_forward_power_of_wave_beam` | 0.56250 | +0.07500 | `attempts_exhausted` |
| `length_of_electron_cyclotron_beam` | `length_of_electron_cyclotron_beam` | 0.67500 | `distance_of_beam_tracing_ray` | 0.70625 | +0.03125 | `successor_collision` |
| `length_of_iron_core_segment` | `length_of_iron_core_segment` | 0.58750 | `minor_length_of_iron_core_segment` | 0.91250 | +0.32500 | `accepted` |
| `length_of_plasma_boundary` | `length_of_plasma_boundary` | 0.68125 | `poloidal_length_of_flux_surface` | 0.97500 | +0.29375 | `accepted` |
| `lower_photon_energy` | `lower_photon_energy` | 0.82500 | `lower_bound_photon_energy` | 0.88125 | +0.05625 | `accepted` |
| `minimum_magnetic_field_magnitude` | `minimum_magnetic_field_magnitude` | 0.70625 | `minimum_over_flux_surface_magnetic_field_magnitude` | 1.00000 | +0.29375 | `accepted` |
| `mode_number` | `mode_number` | 0.55000 | `linear_mhd_mode_number` | 0.50000 | -0.05000 | `superseded` |
| `mode_number` | `linear_mhd_mode_number` | 0.50000 | `perturbed_linear_mhd_mode_number` | 0.46875 | -0.03125 | `successor_collision` |
| `momentum_convection_velocity` | `momentum_convection_velocity` | 0.63750 | `effective_momentum_velocity_due_to_convection` | 0.51250 | -0.12500 | `superseded` |
| `momentum_convection_velocity` | `effective_momentum_velocity_due_to_convection` | 0.51250 | `effective_momentum_convection_velocity` | 0.62500 | +0.11250 | `superseded` |
| `momentum_convection_velocity` | `effective_momentum_convection_velocity` | 0.62500 | `flux_surface_normal_momentum_convection_velocity` | 0.81250 | +0.18750 | `attempts_exhausted` |
| `net_power_of_plant_system` | `net_power_of_plant_system` | 0.66250 | `net_power` | 0.70000 | +0.03750 | `grammar_invalid` |
| `neutral_energy_diffusivity` | `neutral_energy_diffusivity` | 0.82500 | `effective_neutral_energy_diffusivity` | 0.67500 | -0.15000 | `superseded` |
| `neutral_energy_diffusivity` | `effective_neutral_energy_diffusivity` | 0.67500 | `effective_thermal_neutral_energy_diffusivity` | 0.80000 | +0.12500 | `superseded` |
| `neutral_energy_diffusivity` | `effective_thermal_neutral_energy_diffusivity` | 0.80000 | `effective_neutral_energy_diffusion_coefficient` | 0.87500 | +0.07500 | `accepted` |
| `neutral_particle_flux_at_wall` | `neutral_particle_flux_at_wall` | 0.59375 | `neutral_kinetic_energy_flux_at_wall_due_to_surface_emission` | 0.72500 | +0.13125 | `superseded` |
| `neutral_particle_flux_at_wall` | `neutral_kinetic_energy_flux_at_wall_due_to_surface_emission` | 0.72500 | `neutral_energy_flux_at_wall_due_to_surface_emission` | 0.72500 | +0.00000 | `superseded` |
| `neutral_particle_flux_at_wall` | `neutral_energy_flux_at_wall_due_to_surface_emission` | 0.72500 | `neutral_species_kinetic_energy_flux_at_wall_due_to_surface_emission` | 0.65000 | -0.07500 | `attempts_exhausted` |
| `neutron_rate_of_neutron_detector` | `neutron_rate_of_neutron_detector` | 0.83750 | `neutron_rate_of_detector` | 0.77500 | -0.06250 | `attempts_exhausted` |
| `normalized_atomic_count_of_pellet` | `normalized_atomic_count_of_pellet` | 0.48125 | `normalized_molecular_gas_count` | 0.47500 | -0.00625 | `superseded` |
| `normalized_atomic_count_of_pellet` | `normalized_molecular_gas_count` | 0.47500 | `normalized_molecular_gas_count_due_to_gas_injection` | 0.55000 | +0.07500 | `superseded` |
| `normalized_atomic_count_of_pellet` | `normalized_molecular_gas_count_due_to_gas_injection` | 0.55000 | `molecular_gas_count_due_to_pellet_injection` | 0.62500 | +0.07500 | `attempts_exhausted` |
| `normalized_time` | `normalized_time` | 0.81250 | `ratio_of_gyrokinetic_time_to_thermal_transit_time` | 0.67500 | -0.13750 | `superseded` |
| `normalized_time` | `ratio_of_gyrokinetic_time_to_thermal_transit_time` | 0.67500 | `normalized_gyrokinetic_time` | 0.96250 | +0.28750 | `accepted` |
| `normalized_toroidal_flux_coordinate_at_magnetic_axis` | `normalized_toroidal_flux_coordinate_at_magnetic_axis` | 0.66250 | `normalized_toroidal_flux_coordinate_of_magnetic_axis` | 0.53750 | -0.12500 | `superseded` |
| `normalized_toroidal_flux_coordinate_at_magnetic_axis` | `normalized_toroidal_flux_coordinate_of_magnetic_axis` | 0.53750 | `normalized_toroidal_magnetic_flux_at_magnetic_axis` | 0.46250 | -0.07500 | `superseded` |
| `normalized_toroidal_flux_coordinate_at_magnetic_axis` | `normalized_toroidal_magnetic_flux_at_magnetic_axis` | 0.46250 | `normalized_toroidal_flux_coordinate_at_ece_channel_emission_position` | 0.82500 | +0.36250 | `attempts_exhausted` |
| `nuclear_power_density_of_breeder_blanket_module` | `nuclear_power_density_of_breeder_blanket_module` | 0.84375 | `nuclear_power_density_at_midplane` | 0.65000 | -0.19375 | `grammar_invalid` |
| `nuclear_power_of_limiter_tile` | `nuclear_power_of_limiter_tile` | 0.82500 | `total_nuclear_heating_power_of_limiter_tile` | 0.82500 | +0.00000 | `superseded` |
| `nuclear_power_of_limiter_tile` | `total_nuclear_heating_power_of_limiter_tile` | 0.82500 | `total_deposited_nuclear_heating_power_of_limiter_tile` | 0.73750 | -0.08750 | `superseded` |
| `nuclear_power_of_limiter_tile` | `total_deposited_nuclear_heating_power_of_limiter_tile` | 0.73750 | `nuclear_heating_power_of_limiter_tile` | 0.92500 | +0.18750 | `accepted` |
| `optical_element_width` | `optical_element_width` | 0.50000 | `first_local_tangential_width_of_reflector` | 1.00000 | +0.50000 | `accepted` |
| `ordinary_mode_fraction_of_wave_beam` | `ordinary_mode_fraction_of_wave_beam` | 0.65000 | `ordinary_mode_fraction_of_electron_cyclotron_beam` | 0.98750 | +0.33750 | `accepted` |
| `oxygen_source_rate` | `oxygen_source_rate` | 0.81250 | `oxygen_source_rate_due_to_gas_injection` | 1.00000 | +0.18750 | `accepted` |
| `parallel_energy_diffusivity` | `parallel_energy_diffusivity` | 0.55000 | `parallel_neutral_energy_diffusivity` | 0.81250 | +0.26250 | `superseded` |
| `parallel_energy_diffusivity` | `parallel_neutral_energy_diffusivity` | 0.81250 | `parallel_thermal_neutral_energy_diffusivity` | 0.72500 | -0.08750 | `successor_collision` |
| `parallel_flux_surface_averaged_electric_field_at_plasma_boundary` | `parallel_flux_surface_averaged_electric_field_at_plasma_boundary` | 0.60000 | `flux_surface_averaged_field_aligned_electric_field_at_separatrix` | 0.82500 | +0.22500 | `superseded` |
| `parallel_flux_surface_averaged_electric_field_at_plasma_boundary` | `flux_surface_averaged_field_aligned_electric_field_at_separatrix` | 0.82500 | `parallel_flux_surface_averaged_electric_field_at_separatrix` | 0.67500 | -0.15000 | `grammar_invalid` |
| `parallel_heat_flux_at_divertor_target` | `parallel_heat_flux_at_divertor_target` | 0.62500 | `parallel_incident_heat_flux_at_divertor_target` | 0.81250 | +0.18750 | `grammar_invalid` |
| `parallel_ion_diffusivity` | `parallel_ion_diffusivity` | 0.82500 | `parallel_ion_particle_diffusivity` | 0.90000 | +0.07500 | `accepted` |
| `parallel_ion_velocity` | `parallel_ion_velocity` | 0.70000 | `parallel_bulk_ion_velocity` | 0.66250 | -0.03750 | `grammar_invalid` |
| `parallel_normalized_perturbed_current_density_bessel_1` | `parallel_normalized_perturbed_current_density_bessel_1` | 0.84375 | `normalized_perturbed_parallel_gyrocenter_current_density_bessel_1` | 0.87500 | +0.03125 | `accepted` |
| `particle_flux` | `particle_flux` | 0.83125 | `total_particle_flux` | 0.80000 | -0.03125 | `grammar_invalid` |
| `particle_pressure` | `particle_pressure` | 0.56250 | `total_kinetic_particle_pressure` | 0.68750 | +0.12500 | `superseded` |
| `particle_pressure` | `total_kinetic_particle_pressure` | 0.68750 | `total_plasma_pressure` | 0.91250 | +0.22500 | `accepted` |
| `perpendicular_wave_vector_magnitude` | `perpendicular_wave_vector_magnitude` | 0.76250 | `cross_field_wave_vector_magnitude` | 0.73750 | -0.02500 | `successor_collision` |
| `plasma_frequency_at_measurement_position` | `plasma_frequency_at_measurement_position` | 0.63750 | `electron_plasma_frequency_at_measurement_position` | 0.62500 | -0.01250 | `superseded` |
| `plasma_frequency_at_measurement_position` | `electron_plasma_frequency_at_measurement_position` | 0.62500 | `critical_ordinary_mode_frequency` | 0.73750 | +0.11250 | `superseded` |
| `plasma_frequency_at_measurement_position` | `critical_ordinary_mode_frequency` | 0.73750 | `wave_critical_ordinary_mode_frequency` | 0.66250 | -0.07500 | `attempts_exhausted` |
| `poloidal_angle_at_plasma_boundary_gap_reference_point` | `poloidal_angle_at_plasma_boundary_gap_reference_point` | 0.62500 | `poloidal_angle_of_plasma_boundary_gap_reference_point` | 0.56250 | -0.06250 | `superseded` |
| `poloidal_angle_at_plasma_boundary_gap_reference_point` | `poloidal_angle_of_plasma_boundary_gap_reference_point` | 0.56250 | `poloidal_coordinate_of_plasma_boundary_gap_reference_point` | 0.51250 | -0.05000 | `superseded` |
| `poloidal_angle_at_plasma_boundary_gap_reference_point` | `poloidal_coordinate_of_plasma_boundary_gap_reference_point` | 0.51250 | `poloidal_angle_of_plasma_boundary_gap` | 0.93750 | +0.42500 | `accepted` |
| `poloidal_center_of_mass_velocity` | `poloidal_center_of_mass_velocity` | 0.75000 | `poloidal_plasma_center_of_mass_velocity` | 0.83750 | +0.08750 | `superseded` |
| `poloidal_center_of_mass_velocity` | `poloidal_plasma_center_of_mass_velocity` | 0.83750 | `poloidal_bulk_plasma_center_of_mass_velocity` | 0.68750 | -0.15000 | `superseded` |
| `poloidal_center_of_mass_velocity` | `poloidal_bulk_plasma_center_of_mass_velocity` | 0.68750 | `poloidal_bulk_center_of_mass_velocity` | 0.89375 | +0.20625 | `accepted` |
| `poloidal_diamagnetic_current_density` | `poloidal_diamagnetic_current_density` | 0.53750 | `poloidal_current_density_due_to_diamagnetic_drift` | 1.00000 | +0.46250 | `accepted` |
| `poloidal_ion_diffusivity` | `poloidal_ion_diffusivity` | 0.68750 | `poloidal_total_ion_particle_diffusivity` | 0.67500 | -0.01250 | `superseded` |
| `poloidal_ion_diffusivity` | `poloidal_total_ion_particle_diffusivity` | 0.67500 | `poloidal_effective_ion_diffusion_coefficient` | 0.81250 | +0.13750 | `successor_collision` |
| `poloidal_neutral_energy_flux` | `poloidal_neutral_energy_flux` | 0.83750 | `poloidal_neutral_species_energy_flux` | 0.99375 | +0.15625 | `accepted` |
| `power_density` | `power_density` | 0.60000 | `net_power_density` | 0.71250 | +0.11250 | `superseded` |
| `power_density` | `net_power_density` | 0.71250 | `net_energy_power_density` | 0.59375 | -0.11875 | `superseded` |
| `power_density` | `net_energy_power_density` | 0.59375 | `net_plasma_power_density` | 0.72500 | +0.13125 | `attempts_exhausted` |
| `power_due_to_impurity_radiation` | `power_due_to_impurity_radiation` | 0.58750 | `radiated_power_over_core_region_due_to_impurity_radiation` | 0.68750 | +0.10000 | `superseded` |
| `power_due_to_impurity_radiation` | `radiated_power_over_core_region_due_to_impurity_radiation` | 0.68750 | `power_over_core_region_due_to_impurity_radiation` | 0.58750 | -0.10000 | `grammar_invalid` |
| `power_due_to_ohmic_dissipation` | `power_due_to_ohmic_dissipation` | 0.78125 | `total_power_due_to_ohmic_dissipation` | 0.88750 | +0.10625 | `accepted` |
| `power_of_divertor` | `power_of_divertor` | 0.75000 | `total_deposited_power_of_divertor` | 0.63750 | -0.11250 | `superseded` |
| `power_of_divertor` | `total_deposited_power_of_divertor` | 0.63750 | `total_deposited_power_at_divertor_target` | 0.52500 | -0.11250 | `superseded` |
| `power_of_divertor` | `total_deposited_power_at_divertor_target` | 0.52500 | `deposited_power_at_divertor_target` | 0.51250 | -0.01250 | `attempts_exhausted` |
| `power_of_divertor_due_to_fusion` | `power_of_divertor_due_to_fusion` | 0.73750 | `absorbed_power_of_divertor_due_to_fusion` | 0.56250 | -0.17500 | `superseded` |
| `power_of_divertor_due_to_fusion` | `absorbed_power_of_divertor_due_to_fusion` | 0.56250 | `power_of_divertor_due_to_fusion_reactions` | 0.72500 | +0.16250 | `superseded` |
| `power_of_divertor_due_to_fusion` | `power_of_divertor_due_to_fusion_reactions` | 0.72500 | `nuclear_heating_power_of_divertor` | 0.97500 | +0.25000 | `accepted` |
| `power_of_divertor_due_to_radiation` | `power_of_divertor_due_to_radiation` | 0.53750 | `total_thermal_radiative_power_of_divertor_target` | 0.45000 | -0.08750 | `successor_collision` |
| `power_of_ion_cyclotron_heating_antenna` | `power_of_ion_cyclotron_heating_antenna` | 0.62500 | `wave_power_of_ion_cyclotron_heating_antenna` | 0.70000 | +0.07500 | `superseded` |
| `power_of_ion_cyclotron_heating_antenna` | `wave_power_of_ion_cyclotron_heating_antenna` | 0.70000 | `wave_launched_power_of_ion_cyclotron_heating_antenna` | 0.58750 | -0.11250 | `superseded` |
| `power_of_ion_cyclotron_heating_antenna` | `wave_launched_power_of_ion_cyclotron_heating_antenna` | 0.58750 | `launched_power_of_ion_cyclotron_heating_antenna` | 0.65000 | +0.06250 | `attempts_exhausted` |
| `radial_coordinate_of_aperture` | `radial_coordinate_of_aperture` | 0.82500 | `radial_coordinate_of_diagnostic_aperture` | 0.98125 | +0.15625 | `accepted` |
| `radial_coordinate_of_lower_hybrid_antenna_row` | `radial_coordinate_of_lower_hybrid_antenna_row` | 0.52500 | `normal_distance_of_lower_hybrid_antenna` | 0.72500 | +0.20000 | `superseded` |
| `radial_coordinate_of_lower_hybrid_antenna_row` | `normal_distance_of_lower_hybrid_antenna` | 0.72500 | `radial_distance_of_lower_hybrid_antenna` | 0.67500 | -0.05000 | `superseded` |
| `radial_coordinate_of_lower_hybrid_antenna_row` | `radial_distance_of_lower_hybrid_antenna` | 0.67500 | `radial_offset_of_lower_hybrid_antenna` | 0.76250 | +0.08750 | `attempts_exhausted` |
| `radial_coordinate_of_shattering_position` | `radial_coordinate_of_shattering_position` | 0.56250 | `radial_coordinate_of_shatter_cone` | 0.97500 | +0.41250 | `accepted` |
| `radial_electron_diffusion_coefficient` | `radial_electron_diffusion_coefficient` | 0.76250 | `radial_electron_particle_diffusion_coefficient` | 0.66875 | -0.09375 | `superseded` |
| `radial_electron_diffusion_coefficient` | `radial_electron_particle_diffusion_coefficient` | 0.66875 | `electron_particle_diffusion_coefficient` | 0.68750 | +0.01875 | `successor_collision` |
| `radial_energy_convection_velocity` | `radial_energy_convection_velocity` | 0.57500 | `radial_effective_thermal_energy_velocity_due_to_convection` | 0.46250 | -0.11250 | `successor_collision` |
| `radial_ion_charge_state_particle_flux` | `radial_ion_charge_state_particle_flux` | 0.80000 | `flux_surface_normal_ion_charge_state_particle_flux` | 1.00000 | +0.20000 | `accepted` |
| `radial_ion_convection_velocity` | `radial_ion_convection_velocity` | 0.76250 | `radial_ion_particle_convection_velocity` | 1.00000 | +0.23750 | `accepted` |
| `radial_momentum_diffusivity` | `radial_momentum_diffusivity` | 0.75000 | `flux_surface_normal_momentum_diffusivity` | 0.56250 | -0.18750 | `superseded` |
| `radial_momentum_diffusivity` | `flux_surface_normal_momentum_diffusivity` | 0.56250 | `flux_surface_normal_momentum_diffusion_coefficient` | 0.63750 | +0.07500 | `grammar_invalid` |
| `radial_neutral_energy_diffusion_coefficient` | `radial_neutral_energy_diffusion_coefficient` | 0.81250 | `radial_total_neutral_heat_diffusivity_over_edge_region` | 0.58750 | -0.22500 | `superseded` |
| `radial_neutral_energy_diffusion_coefficient` | `radial_total_neutral_heat_diffusivity_over_edge_region` | 0.58750 | `radial_neutral_energy_diffusivity` | 0.73750 | +0.15000 | `superseded` |
| `radial_neutral_energy_diffusion_coefficient` | `radial_neutral_energy_diffusivity` | 0.73750 | `flux_surface_normal_neutral_energy_diffusion_coefficient` | 0.77500 | +0.03750 | `attempts_exhausted` |
| `radial_plasma_momentum_diffusion_coefficient` | `radial_plasma_momentum_diffusion_coefficient` | 0.75000 | `radial_plasma_momentum_diffusivity` | 0.82500 | +0.07500 | `superseded` |
| `radial_plasma_momentum_diffusion_coefficient` | `radial_plasma_momentum_diffusivity` | 0.82500 | `flux_surface_normal_bulk_plasma_momentum_diffusivity` | 0.48750 | -0.33750 | `superseded` |
| `radial_plasma_momentum_diffusion_coefficient` | `flux_surface_normal_bulk_plasma_momentum_diffusivity` | 0.48750 | `flux_surface_normal_plasma_momentum_diffusivity` | 0.90000 | +0.41250 | `accepted` |
| `radiated_power_over_scrape_off_layer` | `radiated_power_over_scrape_off_layer` | 0.66250 | `total_radiated_power_over_scrape_off_layer` | 0.73750 | +0.07500 | `superseded` |
| `radiated_power_over_scrape_off_layer` | `total_radiated_power_over_scrape_off_layer` | 0.73750 | `total_plasma_radiated_power_over_scrape_off_layer` | 0.63750 | -0.10000 | `superseded` |
| `radiated_power_over_scrape_off_layer` | `total_plasma_radiated_power_over_scrape_off_layer` | 0.63750 | `power_over_scrape_off_layer_due_to_radiation` | 0.58750 | -0.05000 | `attempts_exhausted` |
| `radius_of_filter` | `radius_of_filter` | 0.63750 | `radius_of_filter_window` | 1.00000 | +0.36250 | `accepted` |
| `radius_of_iron_core_segment` | `radius_of_iron_core_segment` | 0.55000 | `inverse_of_curvature_of_iron_core_segment` | 0.57500 | +0.02500 | `superseded` |
| `radius_of_iron_core_segment` | `inverse_of_curvature_of_iron_core_segment` | 0.57500 | `inverse_of_curvature_of_arc_of_circle_center` | 0.46250 | -0.11250 | `grammar_invalid` |
| `reference_wavelength_of_filter` | `reference_wavelength_of_filter` | 0.63750 | `reference_wavelength_of_filter_window` | 0.66250 | +0.02500 | `successor_collision` |
| `requested_voltage_of_spectrometer` | `requested_voltage_of_spectrometer` | 0.63750 | `requested_voltage_of_spectrometer_channel` | 1.00000 | +0.36250 | `accepted` |
| `runaway_electron_energy_density` | `runaway_electron_energy_density` | 0.82500 | `runaway_electron_kinetic_energy_density` | 0.97500 | +0.15000 | `accepted` |
| `runaway_electron_source_rate` | `runaway_electron_source_rate` | 0.75000 | `total_runaway_electron_source_rate` | 0.77500 | +0.02500 | `superseded` |
| `runaway_electron_source_rate` | `total_runaway_electron_source_rate` | 0.77500 | `net_runaway_electron_source_rate` | 0.78750 | +0.01250 | `superseded` |
| `runaway_electron_source_rate` | `net_runaway_electron_source_rate` | 0.78750 | `tendency_of_runaway_electron_density` | 0.71250 | -0.07500 | `attempts_exhausted` |
| `shattered_pellet_fragment_density` | `shattered_pellet_fragment_density` | 0.56250 | `shattered_pellet_species_number_density_of_pellet_fragment` | 0.97500 | +0.41250 | `accepted` |
| `shattered_pellet_fragment_volume` | `shattered_pellet_fragment_volume` | 0.83750 | `volume_of_pellet_fragment` | 1.00000 | +0.16250 | `accepted` |
| `spectral_radiance_of_soft_xray_detector` | `spectral_radiance_of_soft_xray_detector` | 0.62500 | `incident_soft_xray_radiance` | 0.95625 | +0.33125 | `accepted` |
| `spectral_width_of_spectrometer_channel` | `spectral_width_of_spectrometer_channel` | 0.53750 | `root_mean_square_of_variation_of_vacuum_wavelength_of_spectrometer_channel` | 0.56250 | +0.02500 | `superseded` |
| `spectral_width_of_spectrometer_channel` | `root_mean_square_of_variation_of_vacuum_wavelength_of_spectrometer_channel` | 0.56250 | `root_mean_square_of_difference_of_wavelength_of_spectrometer_channel_and_reference_wavelength_of_spectrometer_channel` | 0.45000 | -0.11250 | `superseded` |
| `spectral_width_of_spectrometer_channel` | `root_mean_square_of_difference_of_wavelength_of_spectrometer_channel_and_reference_wavelength_of_spectrometer_channel` | 0.45000 | `root_mean_square_of_spectral_width_of_spectrometer_channel` | 0.58750 | +0.13750 | `attempts_exhausted` |
| `surface_temperature` | `surface_temperature` | 0.75000 | `surface_temperature_of_plasma_facing_component` | 0.81250 | +0.06250 | `grammar_invalid` |
| `temperature_of_poloidal_field_coil` | `temperature_of_poloidal_field_coil` | 0.67500 | `bulk_temperature_of_poloidal_field_coil` | 0.79375 | +0.11875 | `superseded` |
| `temperature_of_poloidal_field_coil` | `bulk_temperature_of_poloidal_field_coil` | 0.79375 | `temperature_of_coil_conductor` | 0.88750 | +0.09375 | `accepted` |
| `thermal_ion_power_density` | `thermal_ion_power_density` | 0.80000 | `thermal_ion_absorbed_wave_power_density` | 0.95000 | +0.15000 | `accepted` |
| `thermal_power_of_divertor` | `thermal_power_of_divertor` | 0.81250 | `heat_power_of_divertor` | 0.68750 | -0.12500 | `successor_collision` |
| `thermal_power_of_plant_component_port` | `thermal_power_of_plant_component_port` | 0.81250 | `coolant_heating_power_of_plant_component_port` | 0.66250 | -0.15000 | `superseded` |
| `thermal_power_of_plant_component_port` | `coolant_heating_power_of_plant_component_port` | 0.66250 | `coolant_absorbed_power_of_plant_component_port` | 0.67500 | +0.01250 | `superseded` |
| `thermal_power_of_plant_component_port` | `coolant_absorbed_power_of_plant_component_port` | 0.67500 | `absorbed_coolant_power_of_plant_component_port` | 0.55000 | -0.12500 | `attempts_exhausted` |
| `thickness_of_breeder_blanket_module` | `thickness_of_breeder_blanket_module` | 0.51250 | `thickness_of_breeder_blanket` | 0.49375 | -0.01875 | `superseded` |
| `thickness_of_breeder_blanket_module` | `thickness_of_breeder_blanket` | 0.49375 | `surface_thickness_of_breeder_blanket_module` | 0.58750 | +0.09375 | `vocabulary_gap` |
| `thickness_of_cryostat` | `thickness_of_cryostat` | 0.66250 | `normal_thickness_of_cryostat` | 0.55000 | -0.11250 | `superseded` |
| `thickness_of_cryostat` | `normal_thickness_of_cryostat` | 0.55000 | `normal_distance_of_cryostat` | 0.55000 | +0.00000 | `superseded` |
| `thickness_of_cryostat` | `normal_distance_of_cryostat` | 0.55000 | `surface_thickness_of_cryostat` | 0.59375 | +0.04375 | `attempts_exhausted` |
| `thickness_of_plasma_filament` | `thickness_of_plasma_filament` | 0.52500 | `normal_thickness_of_plasma_filament` | 0.53750 | +0.01250 | `superseded` |
| `thickness_of_plasma_filament` | `normal_thickness_of_plasma_filament` | 0.53750 | `width_of_plasma_filament` | 0.53750 | +0.00000 | `superseded` |
| `thickness_of_plasma_filament` | `width_of_plasma_filament` | 0.53750 | `normal_width_of_plasma_filament` | 0.56875 | +0.03125 | `attempts_exhausted` |
| `time_derivative_of_flux_surface_averaged_metric` | `time_derivative_of_flux_surface_averaged_metric` | 0.50000 | `tendency_of_derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface` | 0.98750 | +0.48750 | `accepted` |
| `time_derivative_of_mode_width` | `time_derivative_of_mode_width` | 0.61250 | `time_derivative_of_radial_width_of_neoclassical_tearing_mode` | 0.98750 | +0.37500 | `accepted` |
| `toroidal_angle_of_magnetic_field_probe` | `toroidal_angle_of_magnetic_field_probe` | 0.68125 | `toroidal_angle_of_poloidal_magnetic_field_probe` | 1.00000 | +0.31875 | `accepted` |
| `toroidal_co_passing_fast_electron_torque_density_due_to_collisions` | `toroidal_co_passing_fast_electron_torque_density_due_to_collisions` | 0.67500 | `toroidal_co_passing_fast_electron_torque_density_due_to_collisional_transport` | 0.78750 | +0.11250 | `grammar_invalid` |
| `toroidal_fast_electron_torque_density_due_to_collisions` | `toroidal_fast_electron_torque_density_due_to_collisions` | 0.75625 | `toroidal_fast_particle_torque_density_due_to_coulomb_collisions_with_electrons` | 0.85000 | +0.09375 | `accepted` |
| `toroidal_fast_electron_torque_due_to_collisions` | `toroidal_fast_electron_torque_due_to_collisions` | 0.66250 | `toroidal_volume_integrated_fast_electron_torque_density_due_to_collisions` | 0.56250 | -0.10000 | `grammar_invalid` |
| `toroidal_ion_torque` | `toroidal_ion_torque` | 0.76250 | `toroidal_explicit_ion_torque` | 0.91250 | +0.15000 | `accepted` |
| `toroidal_plasma_momentum_at_measurement_position` | `toroidal_plasma_momentum_at_measurement_position` | 0.61250 | `toroidal_total_plasma_angular_momentum_at_measurement_position` | 0.72500 | +0.11250 | `superseded` |
| `toroidal_plasma_momentum_at_measurement_position` | `toroidal_total_plasma_angular_momentum_at_measurement_position` | 0.72500 | `toroidal_total_plasma_volumetric_angular_momentum_at_measurement_position` | 0.95000 | +0.22500 | `accepted` |
| `toroidal_total_plasma_momentum_at_plasma_boundary` | `toroidal_total_plasma_momentum_at_plasma_boundary` | 0.77500 | `toroidal_cumulative_inside_flux_surface_total_plasma_momentum_at_separatrix` | 0.66250 | -0.11250 | `grammar_invalid` |
| `toroidal_trapped_fast_electron_torque_density_due_to_collisions` | `toroidal_trapped_fast_electron_torque_density_due_to_collisions` | 0.65000 | `toroidal_trapped_fast_electron_torque_density_due_to_collisional_transport` | 0.70000 | +0.05000 | `successor_collision` |
| `total_ion_density` | `total_ion_density` | 0.58750 | `total_ion_number_density` | 0.92500 | +0.33750 | `accepted` |
| `total_neutral_density` | `total_neutral_density` | 0.81250 | `total_neutral_number_density` | 0.98125 | +0.16875 | `accepted` |
| `total_particle_flux_of_divertor_due_to_recycling` | `total_particle_flux_of_divertor_due_to_recycling` | 0.80000 | `total_particle_source_rate_at_divertor_target_due_to_recycling` | 0.58750 | -0.21250 | `superseded` |
| `total_particle_flux_of_divertor_due_to_recycling` | `total_particle_source_rate_at_divertor_target_due_to_recycling` | 0.58750 | `surface_integrated_total_particle_flux_at_divertor_target_due_to_recycling` | 0.53750 | -0.05000 | `superseded` |
| `total_particle_flux_of_divertor_due_to_recycling` | `surface_integrated_total_particle_flux_at_divertor_target_due_to_recycling` | 0.53750 | `total_particle_flux_at_divertor_target_due_to_recycling` | 0.53750 | +0.00000 | `attempts_exhausted` |
| `total_power_of_ion_cyclotron_heating_antenna` | `total_power_of_ion_cyclotron_heating_antenna` | 0.71250 | `total_launched_power_of_ion_cyclotron_heating_antenna` | 0.62500 | -0.08750 | `superseded` |
| `total_power_of_ion_cyclotron_heating_antenna` | `total_launched_power_of_ion_cyclotron_heating_antenna` | 0.62500 | `total_wave_launched_power_of_ion_cyclotron_heating_antenna` | 0.57500 | -0.05000 | `superseded` |
| `total_power_of_ion_cyclotron_heating_antenna` | `total_wave_launched_power_of_ion_cyclotron_heating_antenna` | 0.57500 | `total_launched_power_due_to_ion_cyclotron_heating` | 0.75000 | +0.17500 | `attempts_exhausted` |
| `total_power_of_plant_system` | `total_power_of_plant_system` | 0.63750 | `total_absorbed_power_of_plant_system` | 0.57500 | -0.06250 | `successor_collision` |
| `total_thermal_particle_source_rate` | `total_thermal_particle_source_rate` | 0.69375 | `volume_integrated_particle_source_rate` | 0.89375 | +0.20000 | `accepted` |
| `total_thermal_power_at_inlet` | `total_thermal_power_at_inlet` | 0.72500 | `total_heat_power_at_inlet` | 0.71250 | -0.01250 | `superseded` |
| `total_thermal_power_at_inlet` | `total_heat_power_at_inlet` | 0.71250 | `total_incident_thermal_power_at_inlet` | 0.66250 | -0.05000 | `superseded` |
| `total_thermal_power_at_inlet` | `total_incident_thermal_power_at_inlet` | 0.66250 | `total_incident_thermal_power` | 0.67500 | +0.01250 | `attempts_exhausted` |
| `upper_bound_wavelength_of_camera` | `upper_bound_wavelength_of_camera` | 0.70000 | `upper_bound_wavelength_of_visible_camera` | 0.97500 | +0.27500 | `accepted` |
| `vertical_angle_of_poloidal_field_coil` | `vertical_angle_of_poloidal_field_coil` | 0.50000 | `tilt_angle_of_poloidal_field_coil` | 0.83750 | +0.33750 | `grammar_invalid` |
| `vertical_coordinate_of_bragg_crystal` | `vertical_coordinate_of_bragg_crystal` | 0.66875 | `vertical_coordinate_of_grating` | 0.67500 | +0.00625 | `superseded` |
| `vertical_coordinate_of_bragg_crystal` | `vertical_coordinate_of_grating` | 0.67500 | `vertical_position_of_grating` | 0.58750 | -0.08750 | `grammar_invalid` |
| `vertical_coordinate_of_optical_element` | `vertical_coordinate_of_optical_element` | 0.67500 | `vertical_centroid_of_optical_element` | 0.58125 | -0.09375 | `superseded` |
| `vertical_coordinate_of_optical_element` | `vertical_centroid_of_optical_element` | 0.58125 | `vertical_coordinate_of_fibre_bundle` | 0.97500 | +0.39375 | `accepted` |
| `vertical_coordinate_of_plasma_boundary` | `vertical_coordinate_of_plasma_boundary` | 0.72500 | `vertical_outline_of_plasma_boundary` | 0.68750 | -0.03750 | `superseded` |
| `vertical_coordinate_of_plasma_boundary` | `vertical_outline_of_plasma_boundary` | 0.68750 | `vertical_outline_of_plasma_filament` | 0.68125 | -0.00625 | `superseded` |
| `vertical_coordinate_of_plasma_boundary` | `vertical_outline_of_plasma_filament` | 0.68125 | `vertical_coordinate_of_plasma_filament` | 0.66875 | -0.01250 | `attempts_exhausted` |
| `vertical_front_surface_curvature_of_optical_element` | `vertical_front_surface_curvature_of_optical_element` | 0.71250 | `inverse_of_second_local_tangential_front_surface_curvature_of_optical_element` | 0.60000 | -0.11250 | `superseded` |
| `vertical_front_surface_curvature_of_optical_element` | `inverse_of_second_local_tangential_front_surface_curvature_of_optical_element` | 0.60000 | `second_local_tangential_front_surface_radius_of_optical_element` | 0.93750 | +0.33750 | `accepted` |
| `vertical_ion_momentum_diffusivity` | `vertical_ion_momentum_diffusivity` | 0.81250 | `vertical_total_ion_momentum_diffusivity` | 0.70000 | -0.11250 | `grammar_invalid` |
| `vertical_outline_of_vacuum_vessel` | `vertical_outline_of_vacuum_vessel` | 0.58750 | `vertical_outline_of_wall` | 0.65000 | +0.06250 | `successor_collision` |
| `vertical_outline_of_wall_material` | `vertical_outline_of_wall_material` | 0.63750 | `vertical_outline_of_plasma_facing_component` | 0.99375 | +0.35625 | `accepted` |
| `voltage_of_reflectometer_antenna` | `voltage_of_reflectometer_antenna` | 0.60000 | `wave_voltage_amplitude` | 0.90000 | +0.30000 | `accepted` |
| `wave_beam_energy` | `wave_beam_energy` | 0.54375 | `thomson_scattering_laser_pulse_energy_at_outlet` | 0.87500 | +0.33125 | `accepted` |
| `wave_beam_energy_at_launching_position` | `wave_beam_energy_at_launching_position` | 0.52500 | `total_wave_beam_launched_pulse_energy_at_launching_position` | 0.55000 | +0.02500 | `superseded` |
| `wave_beam_energy_at_launching_position` | `total_wave_beam_launched_pulse_energy_at_launching_position` | 0.55000 | `launched_pulse_energy_of_thomson_scattering_laser` | 0.95625 | +0.40625 | `accepted` |
| `wave_magnetic_field` | `wave_magnetic_field` | 0.52500 | `perturbed_wave_magnetic_field` | 0.48750 | -0.03750 | `superseded` |
| `wave_magnetic_field` | `perturbed_wave_magnetic_field` | 0.48750 | `perturbed_magnetic_field_of_wave_beam` | 0.45000 | -0.03750 | `superseded` |
| `wave_magnetic_field` | `perturbed_magnetic_field_of_wave_beam` | 0.45000 | `wave_magnetic_field_amplitude` | 0.50000 | +0.05000 | `attempts_exhausted` |
| `width_of_hard_xray_detector` | `width_of_hard_xray_detector` | 0.74375 | `first_local_tangential_width_of_hard_xray_detector` | 0.98750 | +0.24375 | `accepted` |
| `width_of_neutron_detector` | `width_of_neutron_detector` | 0.73750 | `first_local_tangential_width_of_diagnostic_aperture` | 0.96875 | +0.23125 | `accepted` |
| `width_of_reflectometer_antenna` | `width_of_reflectometer_antenna` | 0.58750 | `first_local_tangential_width_of_reflectometer_antenna` | 0.56250 | -0.02500 | `successor_collision` |
| `xenon_source_rate` | `xenon_source_rate` | 0.78750 | `xenon_source_rate_due_to_gas_injection` | 1.00000 | +0.21250 | `accepted` |

## Receipts and logs

- Final machine-readable cohort receipt: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T161700554926-n-refinefinish/final-receipt.json`.
- Exact terminal check: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T161700554926-n-refinefinish/exact-terminal-check.json`.
- Preflight cancelled-claim age: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T161700554926-n-refinefinish/preflight-claim-age.log`.
- Continuation log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T161700554926-n-refinefinish/refine-continuation.log`.
- Prior interim receipt: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T151925667216-n-refineband/final-receipt.json`.
