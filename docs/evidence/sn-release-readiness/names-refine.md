warning: `VIRTUAL_ENV=/home/ITER/mcintos/Code/reckon/.venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
Using CPython 3.12.11
Creating virtual environment at: .venv
# Standard-name refinement campaign

> **BLOCKED — the executed campaign reached its 42-minute guard with four review items and one refine item pending.** The result below is an actual live campaign receipt, but it is not the requested completed pass and must not be promoted as one.

## Outcome

- Resolved drain cohort: **273** identities below 0.85 from `sn-review-restage-1b8c6c0ef31d2e5b831c`.
- Legally entered refine: **269**. **4** were preserved because their prior review had `review_quorum_shortfall='review carried no resolution method'`: `fast_neutral_density`, `hot_neutral_temperature`, `ion_power`, and `rotational_transform`. Refining would act on a non-quorate verdict; rescore is the normal recovery, but this campaign explicitly prohibited rescore.
- Roots producing at least one successor: **158**; roots parked before minting any successor: **111**. The latter were not falsely described as superseded: only 158/269 entered roots are superseded with a `REFINED_FROM` descendant.
- Terminal successors accepted by the quorum: **74**. Terminal successors below 0.85: **84**, comprising 82 exhausted/parked, one reviewed, and one drafted when the guard fired.
- StandardName count: **4395 → 4665** (**+270**). The graph contains **270** new `REFINED_FROM` edges, equal to the identity increase.
- Live claim residue after cancellation: **1**. This is an additional fail-closed blocker to continuation evidence.
- Actual spend: **$77.026394 / $120.00**; **$0.286343 per entered name**; **1168 calls**, or **4.342 calls per entered name**. Headroom was **$42.973606** and the largest call was **$0.674182**, so the cap held by far more than one call.
- Run: `7c35ff12-b777-4f02-a604-8b993e07c413`; stop reason `time_limit_reached`; graph status `degraded`; elapsed 2581.765 s. The command exited non-zero because the time guard expired, not because the cost cap fired.

The review-only reference was $0.0513/name and 2.20 calls/name. This refine campaign used $0.2863/entered name and 4.34 calls/entered name because each minted identity re-entered the review quorum and later rotations repeated both operations.

## Conflict-blocked passing identity

`electron_deposited_power` scored 0.95625 but is blocked by edit-cascade successor conflicts. Refine is **not** sanctioned: its name already passed the 0.85 quorum bar, and changing the name does not resolve the atomic subtree/cascade conflict. The sanctioned route is to resolve the edit-cascade conflict and let the ordinary edit/review acceptance path apply; it was left untouched.

## Stop reasons and integrity

Among the 158 roots that minted successors, terminal state was 74 accepted, 82 exhausted, one reviewed, and one drafted. Terminal stop reasons were 34 `attempts_exhausted`, 24 `grammar_invalid`, 23 `successor_collision`, and one `vocabulary_gap`; accepted and pending terminal nodes have no stop reason. An additional 111 entered roots parked without any successor, so the requested assertion that every entered predecessor was superseded is false and is reported as such.

Full machine-readable root state, all 270 lineage edges, run properties, and cost rows are in `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T151925667216-n-refineband/final-receipt.json`. The CLI log is `~/.local/share/imas-codex/logs/sn_run_dd.log`; the executed run is delimited by run id `7c35ff12-b777-4f02-a604-8b993e07c413`.

## Predecessor → successor score deltas

Every minted edge is listed. `pending` means the successor had not received a score when the time guard stopped the run.

| Root | Predecessor | Before | Successor | After | Δ | Terminal state |
|---|---|---:|---|---:|---:|---|
| `accumulated_lithium_count` | `accumulated_lithium_count` | 0.8438 | `accumulated_lithium_count_due_to_gas_injection` | 1.0000 | +0.1562 | `accepted` |
| `accumulated_nitrogen_count` | `accumulated_nitrogen_count` | 0.8375 | `accumulated_nitrogen_count_due_to_gas_injection` | 1.0000 | +0.1625 | `accepted` |
| `accumulated_propane_count` | `accumulated_propane_count` | 0.8125 | `accumulated_propane_count_due_to_gas_injection` | 1.0000 | +0.1875 | `accepted` |
| `accumulated_silane_count` | `accumulated_silane_count` | 0.8125 | `accumulated_silane_count_due_to_gas_injection` | 1.0000 | +0.1875 | `accepted` |
| `accumulated_xenon_count` | `accumulated_xenon_count` | 0.8375 | `accumulated_xenon_count_due_to_gas_injection` | 1.0000 | +0.1625 | `accepted` |
| `area_of_langmuir_probe` | `area_of_langmuir_probe` | 0.8063 | `surface_area_of_langmuir_probe` | 0.7500 | -0.0563 | `superseded` |
| `area_of_langmuir_probe` | `surface_area_of_langmuir_probe` | 0.7500 | `total_surface_area_of_langmuir_probe` | 0.7500 | +0.0000 | `superseded` |
| `area_of_langmuir_probe` | `total_surface_area_of_langmuir_probe` | 0.7500 | `front_surface_area_of_langmuir_probe` | 0.5875 | -0.1625 | `attempts_exhausted` |
| `area_of_neutral_beam_injector` | `area_of_neutral_beam_injector` | 0.8375 | `beam_cross_sectional_area_of_aperture` | 0.7125 | -0.1250 | `successor_collision` |
| `back_surface_curvature_of_optical_element` | `back_surface_curvature_of_optical_element` | 0.5500 | `first_local_tangential_back_surface_curvature_of_optical_element` | 0.8000 | +0.2500 | `superseded` |
| `back_surface_curvature_of_optical_element` | `first_local_tangential_back_surface_curvature_of_optical_element` | 0.8000 | `inverse_of_first_local_tangential_back_surface_curvature_of_optical_element` | 0.5750 | -0.2250 | `superseded` |
| `back_surface_curvature_of_optical_element` | `inverse_of_first_local_tangential_back_surface_curvature_of_optical_element` | 0.5750 | `first_local_tangential_back_surface_radius_of_optical_element` | 0.6750 | +0.1000 | `attempts_exhausted` |
| `beryllium_source_rate` | `beryllium_source_rate` | 0.7375 | `beryllium_source_rate_due_to_gas_injection` | 1.0000 | +0.2625 | `accepted` |
| `bragg_crystal_width` | `bragg_crystal_width` | 0.4875 | `first_local_tangential_width_of_bragg_crystal` | 0.9938 | +0.5063 | `accepted` |
| `co_passing_fast_electron_power_density_due_to_collisions` | `co_passing_fast_electron_power_density_due_to_collisions` | 0.6125 | `co_passing_fast_electron_kinetic_energy_power_density_due_to_collisions` | 0.4125 | -0.2000 | `superseded` |
| `co_passing_fast_electron_power_density_due_to_collisions` | `co_passing_fast_electron_kinetic_energy_power_density_due_to_collisions` | 0.4125 | `co_passing_fast_electron_kinetic_power_density_due_to_collisions` | 0.4875 | +0.0750 | `attempts_exhausted` |
| `conductivity` | `conductivity` | 0.5875 | `plasma_electrical_conductivity` | 0.9250 | +0.3375 | `accepted` |
| `coolant_temperature` | `coolant_temperature` | 0.8000 | `coolant_temperature_of_plant_component_port` | 0.9375 | +0.1375 | `accepted` |
| `coolant_volume_of_breeder_blanket` | `coolant_volume_of_breeder_blanket` | 0.7500 | `total_coolant_volume_of_breeder_blanket` | 0.6250 | -0.1250 | `superseded` |
| `coolant_volume_of_breeder_blanket` | `total_coolant_volume_of_breeder_blanket` | 0.6250 | `volume_of_breeder_blanket` | 0.6438 | +0.0188 | `superseded` |
| `coolant_volume_of_breeder_blanket` | `volume_of_breeder_blanket` | 0.6438 | `lithium_volume_of_breeder_blanket` | 0.6375 | -0.0063 | `attempts_exhausted` |
| `counter_passing_density` | `counter_passing_density` | 0.5625 | `total_counter_passing_gyrocenter_density` | 0.6625 | +0.1000 | `superseded` |
| `counter_passing_density` | `total_counter_passing_gyrocenter_density` | 0.6625 | `total_counter_passing_particle_density` | 1.0000 | +0.3375 | `accepted` |
| `current_due_to_ohmic_induction` | `current_due_to_ohmic_induction` | 0.7625 | `net_current_due_to_ohmic_induction` | 0.7375 | -0.0250 | `superseded` |
| `current_due_to_ohmic_induction` | `net_current_due_to_ohmic_induction` | 0.7375 | `net_plasma_current_due_to_ohmic_current_drive` | 0.5375 | -0.2000 | `successor_collision` |
| `current_of_correction_coil` | `current_of_correction_coil` | 0.7250 | `non_axisymmetric_current_of_coil_conductor` | 0.5625 | -0.1625 | `superseded` |
| `current_of_correction_coil` | `non_axisymmetric_current_of_coil_conductor` | 0.5625 | `current_of_coil_conductor` | 0.6500 | +0.0875 | `superseded` |
| `current_of_correction_coil` | `current_of_coil_conductor` | 0.6500 | `non_axisymmetric_current_of_conductor` | 0.5125 | -0.1375 | `attempts_exhausted` |
| `curvature_of_optical_element` | `curvature_of_optical_element` | 0.6625 | `inverse_of_tangential_surface_curvature_of_optical_element` | 0.5125 | -0.1500 | `superseded` |
| `curvature_of_optical_element` | `inverse_of_tangential_surface_curvature_of_optical_element` | 0.5125 | `tangential_surface_radius_of_optical_element` | 0.4250 | -0.0875 | `superseded` |
| `curvature_of_optical_element` | `tangential_surface_radius_of_optical_element` | 0.4250 | `inverse_of_tangential_curvature_of_optical_element` | 0.4750 | +0.0500 | `attempts_exhausted` |
| `deposited_power` | `deposited_power` | 0.6875 | `volume_integrated_net_plasma_particle_power_density` | 0.6187 | -0.0688 | `grammar_invalid` |
| `deuterium_count_due_to_gas_injection` | `deuterium_count_due_to_gas_injection` | 0.7375 | `accumulated_deuterium_count_due_to_gas_injection` | 1.0000 | +0.2625 | `accepted` |
| `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | 0.7375 | `total_deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.8000 | +0.0625 | `superseded` |
| `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | `total_deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.8000 | `volume_integrated_deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.7750 | -0.0250 | `superseded` |
| `deuterium_deuterium_neutron_flux_due_to_beam_beam_fusion` | `volume_integrated_deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.7750 | `deuterium_deuterium_neutron_source_rate_due_to_beam_beam_fusion` | 0.9812 | +0.2062 | `accepted` |
| `deuterium_tritium_neutron_flux_due_to_beam_beam_fusion` | `deuterium_tritium_neutron_flux_due_to_beam_beam_fusion` | 0.6875 | `total_deuterium_tritium_neutron_source_rate_due_to_beam_beam_fusion` | 0.5125 | -0.1750 | `successor_collision` |
| `diamagnetic_velocity_due_to_diamagnetic_drift` | `diamagnetic_velocity_due_to_diamagnetic_drift` | 0.3875 | `binormal_velocity_due_to_diamagnetic_drift` | 0.7250 | +0.3375 | `superseded` |
| `diamagnetic_velocity_due_to_diamagnetic_drift` | `binormal_velocity_due_to_diamagnetic_drift` | 0.7250 | `binormal_ion_velocity_due_to_diamagnetic_drift` | 0.9125 | +0.1875 | `accepted` |
| `effective_charge_at_plasma_boundary` | `effective_charge_at_plasma_boundary` | 0.6750 | `effective_charge_number_at_plasma_boundary` | 0.5875 | -0.0875 | `superseded` |
| `effective_charge_at_plasma_boundary` | `effective_charge_number_at_plasma_boundary` | 0.5875 | `effective_charge_at_separatrix` | 1.0000 | +0.4125 | `accepted` |
| `efficiency_of_neutron_detector` | `efficiency_of_neutron_detector` | 0.8000 | `spectral_efficiency_of_neutron_detector` | 0.9812 | +0.1812 | `accepted` |
| `efficiency_of_spectrometer_channel` | `efficiency_of_spectrometer_channel` | 0.7125 | `transmissivity_of_spectrometer_channel` | 0.5750 | -0.1375 | `superseded` |
| `efficiency_of_spectrometer_channel` | `transmissivity_of_spectrometer_channel` | 0.5750 | `viewing_efficiency_of_spectrometer_channel` | 0.8875 | +0.3125 | `accepted` |
| `electron_density_over_scrape_off_layer` | `electron_density_over_scrape_off_layer` | 0.5875 | `volume_averaged_electron_density_over_scrape_off_layer` | 0.5625 | -0.0250 | `superseded` |
| `electron_density_over_scrape_off_layer` | `volume_averaged_electron_density_over_scrape_off_layer` | 0.5625 | `volume_averaged_electron_number_density_over_scrape_off_layer` | 0.5875 | +0.0250 | `grammar_invalid` |
| `electron_diffusivity` | `electron_diffusivity` | 0.7750 | `electron_particle_diffusion_coefficient_due_to_diffusion` | 0.4875 | -0.2875 | `superseded` |
| `electron_diffusivity` | `electron_particle_diffusion_coefficient_due_to_diffusion` | 0.4875 | `electron_particle_diffusivity` | 0.7625 | +0.2750 | `successor_collision` |
| `electron_energy_diffusivity` | `electron_energy_diffusivity` | 0.6750 | `thermal_electron_energy_diffusivity` | 0.8375 | +0.1625 | `superseded` |
| `electron_energy_diffusivity` | `thermal_electron_energy_diffusivity` | 0.8375 | `electron_heat_diffusivity` | 0.8062 | -0.0313 | `successor_collision` |
| `electron_energy_flux_at_wall` | `electron_energy_flux_at_wall` | 0.8250 | `electron_energy_flux_at_wall_due_to_surface_emission` | 0.7812 | -0.0437 | `superseded` |
| `electron_energy_flux_at_wall` | `electron_energy_flux_at_wall_due_to_surface_emission` | 0.7812 | `electron_kinetic_energy_flux_at_wall_due_to_surface_emission` | 1.0000 | +0.2188 | `accepted` |
| `electron_particle_flux_at_wall` | `electron_particle_flux_at_wall` | 0.7188 | `electron_particle_flux_at_wall_due_to_surface_emission` | 0.9688 | +0.2500 | `accepted` |
| `electron_pressure_at_plasma_boundary` | `electron_pressure_at_plasma_boundary` | 0.7812 | `electron_pressure_at_separatrix` | 1.0000 | +0.2188 | `accepted` |
| `electron_temperature_at_plasma_boundary` | `electron_temperature_at_plasma_boundary` | 0.6875 | `bulk_electron_temperature_at_plasma_boundary` | 0.6375 | -0.0500 | `superseded` |
| `electron_temperature_at_plasma_boundary` | `bulk_electron_temperature_at_plasma_boundary` | 0.6375 | `thermal_electron_temperature_at_plasma_boundary` | 0.6500 | +0.0125 | `superseded` |
| `electron_temperature_at_plasma_boundary` | `thermal_electron_temperature_at_plasma_boundary` | 0.6500 | `electron_temperature_at_separatrix` | 0.6750 | +0.0250 | `attempts_exhausted` |
| `electron_temperature_at_wall` | `electron_temperature_at_wall` | 0.7375 | `electron_temperature_at_first_wall` | 0.6625 | -0.0750 | `grammar_invalid` |
| `electrostatic_potential_at_midplane` | `electrostatic_potential_at_midplane` | 0.6250 | `plasma_electrostatic_potential_at_midplane` | 0.5250 | -0.1000 | `superseded` |
| `electrostatic_potential_at_midplane` | `plasma_electrostatic_potential_at_midplane` | 0.5250 | `electrostatic_potential_at_outboard_midplane` | 0.6125 | +0.0875 | `superseded` |
| `electrostatic_potential_at_midplane` | `electrostatic_potential_at_outboard_midplane` | 0.6125 | `plasma_electrostatic_potential_at_outboard_midplane` | 0.5375 | -0.0750 | `attempts_exhausted` |
| `electrostatic_potential_at_wall` | `electrostatic_potential_at_wall` | 0.8250 | `electrostatic_potential_at_first_wall` | 0.6500 | -0.1750 | `superseded` |
| `electrostatic_potential_at_wall` | `electrostatic_potential_at_first_wall` | 0.6500 | `plasma_electrostatic_potential_at_first_wall` | 0.6500 | +0.0000 | `superseded` |
| `electrostatic_potential_at_wall` | `plasma_electrostatic_potential_at_first_wall` | 0.6500 | `plasma_electrostatic_potential_at_wall` | 0.6125 | -0.0375 | `attempts_exhausted` |
| `energy_confinement_time` | `energy_confinement_time` | 0.8375 | `thermal_energy_confinement_time` | 0.7750 | -0.0625 | `grammar_invalid` |
| `energy_diffusivity` | `energy_diffusivity` | 0.5750 | `thermal_energy_diffusivity` | 0.5875 | +0.0125 | `superseded` |
| `energy_diffusivity` | `thermal_energy_diffusivity` | 0.5875 | `energy_diffusion_coefficient_due_to_diffusion` | 0.7250 | +0.1375 | `successor_collision` |
| `energy_flux` | `energy_flux` | 0.8250 | `normal_energy_flux` | 0.6500 | -0.1750 | `superseded` |
| `energy_flux` | `normal_energy_flux` | 0.6500 | `normal_energy_flux_at_control_surface` | 0.5500 | -0.1000 | `superseded` |
| `energy_flux` | `normal_energy_flux_at_control_surface` | 0.5500 | `energy_flux_at_control_surface` | pending | pending | `drafted` |
| `energy_flux_at_wall_due_to_radiation` | `energy_flux_at_wall_due_to_radiation` | 0.6625 | `incident_energy_flux_at_wall_due_to_radiation` | 0.5500 | -0.1125 | `grammar_invalid` |
| `ethylene_count` | `ethylene_count` | 0.6000 | `accumulated_ethylene_count_due_to_gas_injection` | 0.8000 | +0.2000 | `superseded` |
| `ethylene_count` | `accumulated_ethylene_count_due_to_gas_injection` | 0.8000 | `cumulative_ethylene_count_due_to_gas_injection` | 0.6500 | -0.1500 | `grammar_invalid` |
| `fast_electron_power_density` | `fast_electron_power_density` | 0.8000 | `flux_surface_averaged_fast_electron_absorbed_wave_power_density` | 0.8313 | +0.0312 | `superseded` |
| `fast_electron_power_density` | `flux_surface_averaged_fast_electron_absorbed_wave_power_density` | 0.8313 | `fast_electron_absorbed_wave_power_density` | 0.9812 | +0.1500 | `accepted` |
| `fast_electron_power_density_due_to_collisions` | `fast_electron_power_density_due_to_collisions` | 0.7000 | `total_suprathermal_electron_power_density_due_to_collisions` | 0.7625 | +0.0625 | `grammar_invalid` |
| `filter_window_width` | `filter_window_width` | 0.5750 | `first_local_tangential_width_of_filter_window` | 1.0000 | +0.4250 | `accepted` |
| `flux_due_to_recombination` | `flux_due_to_recombination` | 0.5250 | `normal_particle_flux_due_to_recombination` | 0.5625 | +0.0375 | `superseded` |
| `flux_due_to_recombination` | `normal_particle_flux_due_to_recombination` | 0.5625 | `normal_particle_flux_at_wall_due_to_recombination` | 0.7250 | +0.1625 | `successor_collision` |
| `flux_surface_averaged_electron_temperature_at_plasma_boundary` | `flux_surface_averaged_electron_temperature_at_plasma_boundary` | 0.7625 | `flux_surface_averaged_bulk_electron_temperature_at_last_closed_flux_surface` | 0.8000 | +0.0375 | `grammar_invalid` |
| `frequency_of_electron_cyclotron_heating_antenna` | `frequency_of_electron_cyclotron_heating_antenna` | 0.7125 | `frequency_of_electron_cyclotron_beam` | 1.0000 | +0.2875 | `accepted` |
| `gap_of_antenna_strap` | `gap_of_antenna_strap` | 0.7500 | `normal_gap_at_wall` | 0.5125 | -0.2375 | `superseded` |
| `gap_of_antenna_strap` | `normal_gap_at_wall` | 0.5125 | `normal_gap_of_antenna_strap` | 0.7688 | +0.2563 | `superseded` |
| `gap_of_antenna_strap` | `normal_gap_of_antenna_strap` | 0.7688 | `normal_distance_of_antenna_strap` | 0.6313 | -0.1375 | `attempts_exhausted` |
| `incident_power_of_breeder_blanket_module` | `incident_power_of_breeder_blanket_module` | 0.7312 | `incident_neutron_power_of_breeder_blanket_module` | 0.9875 | +0.2563 | `accepted` |
| `ion_atomic_number` | `ion_atomic_number` | 0.6750 | `ion_species_atomic_number` | 0.5875 | -0.0875 | `successor_collision` |
| `ion_momentum_diffusivity` | `ion_momentum_diffusivity` | 0.5625 | `total_ion_momentum_diffusivity` | 0.4875 | -0.0750 | `successor_collision` |
| `ion_particle_flux_at_wall` | `ion_particle_flux_at_wall` | 0.6500 | `ion_particle_flux_at_wall_due_to_surface_emission` | 0.7188 | +0.0688 | `superseded` |
| `ion_particle_flux_at_wall` | `ion_particle_flux_at_wall_due_to_surface_emission` | 0.7188 | `total_ion_particle_flux_at_wall_due_to_surface_emission` | 0.8000 | +0.0813 | `superseded` |
| `ion_particle_flux_at_wall` | `total_ion_particle_flux_at_wall_due_to_surface_emission` | 0.8000 | `ion_species_particle_flux_at_wall_due_to_surface_emission` | 0.7250 | -0.0750 | `attempts_exhausted` |
| `ion_temperature_at_midplane` | `ion_temperature_at_midplane` | 0.6125 | `scrape_off_layer_ion_temperature_at_outboard_midplane` | 0.5250 | -0.0875 | `superseded` |
| `ion_temperature_at_midplane` | `scrape_off_layer_ion_temperature_at_outboard_midplane` | 0.5250 | `ion_temperature_at_outboard_midplane` | 0.6500 | +0.1250 | `superseded` |
| `ion_temperature_at_midplane` | `ion_temperature_at_outboard_midplane` | 0.6500 | `ion_temperature_at_outboard_midplane_separatrix` | 0.6625 | +0.0125 | `attempts_exhausted` |
| `launched_power_of_wave_beam` | `launched_power_of_wave_beam` | 0.7250 | `net_launched_power_of_wave_beam` | 0.6625 | -0.0625 | `superseded` |
| `launched_power_of_wave_beam` | `net_launched_power_of_wave_beam` | 0.6625 | `difference_of_forward_power_of_wave_beam_and_reflected_power_of_wave_beam` | 0.4875 | -0.1750 | `superseded` |
| `launched_power_of_wave_beam` | `difference_of_forward_power_of_wave_beam_and_reflected_power_of_wave_beam` | 0.4875 | `net_forward_power_of_wave_beam` | 0.5625 | +0.0750 | `attempts_exhausted` |
| `length_of_electron_cyclotron_beam` | `length_of_electron_cyclotron_beam` | 0.6750 | `distance_of_beam_tracing_ray` | 0.7063 | +0.0312 | `successor_collision` |
| `length_of_iron_core_segment` | `length_of_iron_core_segment` | 0.5875 | `minor_length_of_iron_core_segment` | 0.9125 | +0.3250 | `accepted` |
| `length_of_plasma_boundary` | `length_of_plasma_boundary` | 0.6812 | `poloidal_length_of_flux_surface` | 0.9750 | +0.2938 | `accepted` |
| `lower_photon_energy` | `lower_photon_energy` | 0.8250 | `lower_bound_photon_energy` | 0.8812 | +0.0563 | `accepted` |
| `minimum_magnetic_field_magnitude` | `minimum_magnetic_field_magnitude` | 0.7063 | `minimum_over_flux_surface_magnetic_field_magnitude` | 1.0000 | +0.2937 | `accepted` |
| `mode_number` | `mode_number` | 0.5500 | `linear_mhd_mode_number` | 0.5000 | -0.0500 | `superseded` |
| `mode_number` | `linear_mhd_mode_number` | 0.5000 | `perturbed_linear_mhd_mode_number` | 0.4688 | -0.0312 | `successor_collision` |
| `momentum_convection_velocity` | `momentum_convection_velocity` | 0.6375 | `effective_momentum_velocity_due_to_convection` | 0.5125 | -0.1250 | `superseded` |
| `momentum_convection_velocity` | `effective_momentum_velocity_due_to_convection` | 0.5125 | `effective_momentum_convection_velocity` | 0.6250 | +0.1125 | `superseded` |
| `momentum_convection_velocity` | `effective_momentum_convection_velocity` | 0.6250 | `flux_surface_normal_momentum_convection_velocity` | 0.8125 | +0.1875 | `attempts_exhausted` |
| `net_power_of_plant_system` | `net_power_of_plant_system` | 0.6625 | `net_power` | 0.7000 | +0.0375 | `grammar_invalid` |
| `neutral_energy_diffusivity` | `neutral_energy_diffusivity` | 0.8250 | `effective_neutral_energy_diffusivity` | 0.6750 | -0.1500 | `superseded` |
| `neutral_energy_diffusivity` | `effective_neutral_energy_diffusivity` | 0.6750 | `effective_thermal_neutral_energy_diffusivity` | 0.8000 | +0.1250 | `superseded` |
| `neutral_energy_diffusivity` | `effective_thermal_neutral_energy_diffusivity` | 0.8000 | `effective_neutral_energy_diffusion_coefficient` | 0.8750 | +0.0750 | `accepted` |
| `neutral_particle_flux_at_wall` | `neutral_particle_flux_at_wall` | 0.5938 | `neutral_kinetic_energy_flux_at_wall_due_to_surface_emission` | 0.7250 | +0.1312 | `superseded` |
| `neutral_particle_flux_at_wall` | `neutral_kinetic_energy_flux_at_wall_due_to_surface_emission` | 0.7250 | `neutral_energy_flux_at_wall_due_to_surface_emission` | 0.7250 | +0.0000 | `superseded` |
| `neutral_particle_flux_at_wall` | `neutral_energy_flux_at_wall_due_to_surface_emission` | 0.7250 | `neutral_species_kinetic_energy_flux_at_wall_due_to_surface_emission` | 0.6500 | -0.0750 | `attempts_exhausted` |
| `neutron_rate_of_neutron_detector` | `neutron_rate_of_neutron_detector` | 0.8375 | `neutron_rate_of_detector` | 0.7750 | -0.0625 | `attempts_exhausted` |
| `normalized_atomic_count_of_pellet` | `normalized_atomic_count_of_pellet` | 0.4813 | `normalized_molecular_gas_count` | 0.4750 | -0.0063 | `superseded` |
| `normalized_atomic_count_of_pellet` | `normalized_molecular_gas_count` | 0.4750 | `normalized_molecular_gas_count_due_to_gas_injection` | 0.5500 | +0.0750 | `superseded` |
| `normalized_atomic_count_of_pellet` | `normalized_molecular_gas_count_due_to_gas_injection` | 0.5500 | `molecular_gas_count_due_to_pellet_injection` | 0.6250 | +0.0750 | `attempts_exhausted` |
| `normalized_time` | `normalized_time` | 0.8125 | `ratio_of_gyrokinetic_time_to_thermal_transit_time` | 0.6750 | -0.1375 | `superseded` |
| `normalized_time` | `ratio_of_gyrokinetic_time_to_thermal_transit_time` | 0.6750 | `normalized_gyrokinetic_time` | 0.9625 | +0.2875 | `accepted` |
| `normalized_toroidal_flux_coordinate_at_magnetic_axis` | `normalized_toroidal_flux_coordinate_at_magnetic_axis` | 0.6625 | `normalized_toroidal_flux_coordinate_of_magnetic_axis` | 0.5375 | -0.1250 | `superseded` |
| `normalized_toroidal_flux_coordinate_at_magnetic_axis` | `normalized_toroidal_flux_coordinate_of_magnetic_axis` | 0.5375 | `normalized_toroidal_magnetic_flux_at_magnetic_axis` | 0.4625 | -0.0750 | `reviewed` |
| `nuclear_power_density_of_breeder_blanket_module` | `nuclear_power_density_of_breeder_blanket_module` | 0.8438 | `nuclear_power_density_at_midplane` | 0.6500 | -0.1937 | `grammar_invalid` |
| `nuclear_power_of_limiter_tile` | `nuclear_power_of_limiter_tile` | 0.8250 | `total_nuclear_heating_power_of_limiter_tile` | 0.8250 | +0.0000 | `superseded` |
| `nuclear_power_of_limiter_tile` | `total_nuclear_heating_power_of_limiter_tile` | 0.8250 | `total_deposited_nuclear_heating_power_of_limiter_tile` | 0.7375 | -0.0875 | `superseded` |
| `nuclear_power_of_limiter_tile` | `total_deposited_nuclear_heating_power_of_limiter_tile` | 0.7375 | `nuclear_heating_power_of_limiter_tile` | 0.9250 | +0.1875 | `accepted` |
| `optical_element_width` | `optical_element_width` | 0.5000 | `first_local_tangential_width_of_reflector` | 1.0000 | +0.5000 | `accepted` |
| `ordinary_mode_fraction_of_wave_beam` | `ordinary_mode_fraction_of_wave_beam` | 0.6500 | `ordinary_mode_fraction_of_electron_cyclotron_beam` | 0.9875 | +0.3375 | `accepted` |
| `oxygen_source_rate` | `oxygen_source_rate` | 0.8125 | `oxygen_source_rate_due_to_gas_injection` | 1.0000 | +0.1875 | `accepted` |
| `parallel_energy_diffusivity` | `parallel_energy_diffusivity` | 0.5500 | `parallel_neutral_energy_diffusivity` | 0.8125 | +0.2625 | `superseded` |
| `parallel_energy_diffusivity` | `parallel_neutral_energy_diffusivity` | 0.8125 | `parallel_thermal_neutral_energy_diffusivity` | 0.7250 | -0.0875 | `successor_collision` |
| `parallel_flux_surface_averaged_electric_field_at_plasma_boundary` | `parallel_flux_surface_averaged_electric_field_at_plasma_boundary` | 0.6000 | `flux_surface_averaged_field_aligned_electric_field_at_separatrix` | 0.8250 | +0.2250 | `superseded` |
| `parallel_flux_surface_averaged_electric_field_at_plasma_boundary` | `flux_surface_averaged_field_aligned_electric_field_at_separatrix` | 0.8250 | `parallel_flux_surface_averaged_electric_field_at_separatrix` | 0.6750 | -0.1500 | `grammar_invalid` |
| `parallel_heat_flux_at_divertor_target` | `parallel_heat_flux_at_divertor_target` | 0.6250 | `parallel_incident_heat_flux_at_divertor_target` | 0.8125 | +0.1875 | `grammar_invalid` |
| `parallel_ion_diffusivity` | `parallel_ion_diffusivity` | 0.8250 | `parallel_ion_particle_diffusivity` | 0.9000 | +0.0750 | `accepted` |
| `parallel_ion_velocity` | `parallel_ion_velocity` | 0.7000 | `parallel_bulk_ion_velocity` | 0.6625 | -0.0375 | `grammar_invalid` |
| `parallel_normalized_perturbed_current_density_bessel_1` | `parallel_normalized_perturbed_current_density_bessel_1` | 0.8438 | `normalized_perturbed_parallel_gyrocenter_current_density_bessel_1` | 0.8750 | +0.0312 | `accepted` |
| `particle_flux` | `particle_flux` | 0.8313 | `total_particle_flux` | 0.8000 | -0.0312 | `grammar_invalid` |
| `particle_pressure` | `particle_pressure` | 0.5625 | `total_kinetic_particle_pressure` | 0.6875 | +0.1250 | `superseded` |
| `particle_pressure` | `total_kinetic_particle_pressure` | 0.6875 | `total_plasma_pressure` | 0.9125 | +0.2250 | `accepted` |
| `perpendicular_wave_vector_magnitude` | `perpendicular_wave_vector_magnitude` | 0.7625 | `cross_field_wave_vector_magnitude` | 0.7375 | -0.0250 | `successor_collision` |
| `plasma_frequency_at_measurement_position` | `plasma_frequency_at_measurement_position` | 0.6375 | `electron_plasma_frequency_at_measurement_position` | 0.6250 | -0.0125 | `superseded` |
| `plasma_frequency_at_measurement_position` | `electron_plasma_frequency_at_measurement_position` | 0.6250 | `critical_ordinary_mode_frequency` | 0.7375 | +0.1125 | `superseded` |
| `plasma_frequency_at_measurement_position` | `critical_ordinary_mode_frequency` | 0.7375 | `wave_critical_ordinary_mode_frequency` | 0.6625 | -0.0750 | `attempts_exhausted` |
| `poloidal_angle_at_plasma_boundary_gap_reference_point` | `poloidal_angle_at_plasma_boundary_gap_reference_point` | 0.6250 | `poloidal_angle_of_plasma_boundary_gap_reference_point` | 0.5625 | -0.0625 | `superseded` |
| `poloidal_angle_at_plasma_boundary_gap_reference_point` | `poloidal_angle_of_plasma_boundary_gap_reference_point` | 0.5625 | `poloidal_coordinate_of_plasma_boundary_gap_reference_point` | 0.5125 | -0.0500 | `superseded` |
| `poloidal_angle_at_plasma_boundary_gap_reference_point` | `poloidal_coordinate_of_plasma_boundary_gap_reference_point` | 0.5125 | `poloidal_angle_of_plasma_boundary_gap` | 0.9375 | +0.4250 | `accepted` |
| `poloidal_center_of_mass_velocity` | `poloidal_center_of_mass_velocity` | 0.7500 | `poloidal_plasma_center_of_mass_velocity` | 0.8375 | +0.0875 | `superseded` |
| `poloidal_center_of_mass_velocity` | `poloidal_plasma_center_of_mass_velocity` | 0.8375 | `poloidal_bulk_plasma_center_of_mass_velocity` | 0.6875 | -0.1500 | `superseded` |
| `poloidal_center_of_mass_velocity` | `poloidal_bulk_plasma_center_of_mass_velocity` | 0.6875 | `poloidal_bulk_center_of_mass_velocity` | 0.8938 | +0.2063 | `accepted` |
| `poloidal_diamagnetic_current_density` | `poloidal_diamagnetic_current_density` | 0.5375 | `poloidal_current_density_due_to_diamagnetic_drift` | 1.0000 | +0.4625 | `accepted` |
| `poloidal_ion_diffusivity` | `poloidal_ion_diffusivity` | 0.6875 | `poloidal_total_ion_particle_diffusivity` | 0.6750 | -0.0125 | `superseded` |
| `poloidal_ion_diffusivity` | `poloidal_total_ion_particle_diffusivity` | 0.6750 | `poloidal_effective_ion_diffusion_coefficient` | 0.8125 | +0.1375 | `successor_collision` |
| `poloidal_neutral_energy_flux` | `poloidal_neutral_energy_flux` | 0.8375 | `poloidal_neutral_species_energy_flux` | 0.9938 | +0.1562 | `accepted` |
| `power_density` | `power_density` | 0.6000 | `net_power_density` | 0.7125 | +0.1125 | `superseded` |
| `power_density` | `net_power_density` | 0.7125 | `net_energy_power_density` | 0.5938 | -0.1188 | `superseded` |
| `power_density` | `net_energy_power_density` | 0.5938 | `net_plasma_power_density` | 0.7250 | +0.1312 | `attempts_exhausted` |
| `power_due_to_impurity_radiation` | `power_due_to_impurity_radiation` | 0.5875 | `radiated_power_over_core_region_due_to_impurity_radiation` | 0.6875 | +0.1000 | `superseded` |
| `power_due_to_impurity_radiation` | `radiated_power_over_core_region_due_to_impurity_radiation` | 0.6875 | `power_over_core_region_due_to_impurity_radiation` | 0.5875 | -0.1000 | `grammar_invalid` |
| `power_due_to_ohmic_dissipation` | `power_due_to_ohmic_dissipation` | 0.7812 | `total_power_due_to_ohmic_dissipation` | 0.8875 | +0.1062 | `accepted` |
| `power_of_divertor` | `power_of_divertor` | 0.7500 | `total_deposited_power_of_divertor` | 0.6375 | -0.1125 | `superseded` |
| `power_of_divertor` | `total_deposited_power_of_divertor` | 0.6375 | `total_deposited_power_at_divertor_target` | 0.5250 | -0.1125 | `superseded` |
| `power_of_divertor` | `total_deposited_power_at_divertor_target` | 0.5250 | `deposited_power_at_divertor_target` | 0.5125 | -0.0125 | `attempts_exhausted` |
| `power_of_divertor_due_to_fusion` | `power_of_divertor_due_to_fusion` | 0.7375 | `absorbed_power_of_divertor_due_to_fusion` | 0.5625 | -0.1750 | `superseded` |
| `power_of_divertor_due_to_fusion` | `absorbed_power_of_divertor_due_to_fusion` | 0.5625 | `power_of_divertor_due_to_fusion_reactions` | 0.7250 | +0.1625 | `superseded` |
| `power_of_divertor_due_to_fusion` | `power_of_divertor_due_to_fusion_reactions` | 0.7250 | `nuclear_heating_power_of_divertor` | 0.9750 | +0.2500 | `accepted` |
| `power_of_divertor_due_to_radiation` | `power_of_divertor_due_to_radiation` | 0.5375 | `total_thermal_radiative_power_of_divertor_target` | 0.4500 | -0.0875 | `successor_collision` |
| `power_of_ion_cyclotron_heating_antenna` | `power_of_ion_cyclotron_heating_antenna` | 0.6250 | `wave_power_of_ion_cyclotron_heating_antenna` | 0.7000 | +0.0750 | `superseded` |
| `power_of_ion_cyclotron_heating_antenna` | `wave_power_of_ion_cyclotron_heating_antenna` | 0.7000 | `wave_launched_power_of_ion_cyclotron_heating_antenna` | 0.5875 | -0.1125 | `superseded` |
| `power_of_ion_cyclotron_heating_antenna` | `wave_launched_power_of_ion_cyclotron_heating_antenna` | 0.5875 | `launched_power_of_ion_cyclotron_heating_antenna` | 0.6500 | +0.0625 | `attempts_exhausted` |
| `radial_coordinate_of_aperture` | `radial_coordinate_of_aperture` | 0.8250 | `radial_coordinate_of_diagnostic_aperture` | 0.9812 | +0.1562 | `accepted` |
| `radial_coordinate_of_lower_hybrid_antenna_row` | `radial_coordinate_of_lower_hybrid_antenna_row` | 0.5250 | `normal_distance_of_lower_hybrid_antenna` | 0.7250 | +0.2000 | `superseded` |
| `radial_coordinate_of_lower_hybrid_antenna_row` | `normal_distance_of_lower_hybrid_antenna` | 0.7250 | `radial_distance_of_lower_hybrid_antenna` | 0.6750 | -0.0500 | `superseded` |
| `radial_coordinate_of_lower_hybrid_antenna_row` | `radial_distance_of_lower_hybrid_antenna` | 0.6750 | `radial_offset_of_lower_hybrid_antenna` | 0.7625 | +0.0875 | `attempts_exhausted` |
| `radial_coordinate_of_shattering_position` | `radial_coordinate_of_shattering_position` | 0.5625 | `radial_coordinate_of_shatter_cone` | 0.9750 | +0.4125 | `accepted` |
| `radial_electron_diffusion_coefficient` | `radial_electron_diffusion_coefficient` | 0.7625 | `radial_electron_particle_diffusion_coefficient` | 0.6687 | -0.0938 | `superseded` |
| `radial_electron_diffusion_coefficient` | `radial_electron_particle_diffusion_coefficient` | 0.6687 | `electron_particle_diffusion_coefficient` | 0.6875 | +0.0188 | `successor_collision` |
| `radial_energy_convection_velocity` | `radial_energy_convection_velocity` | 0.5750 | `radial_effective_thermal_energy_velocity_due_to_convection` | 0.4625 | -0.1125 | `successor_collision` |
| `radial_ion_charge_state_particle_flux` | `radial_ion_charge_state_particle_flux` | 0.8000 | `flux_surface_normal_ion_charge_state_particle_flux` | 1.0000 | +0.2000 | `accepted` |
| `radial_ion_convection_velocity` | `radial_ion_convection_velocity` | 0.7625 | `radial_ion_particle_convection_velocity` | 1.0000 | +0.2375 | `accepted` |
| `radial_momentum_diffusivity` | `radial_momentum_diffusivity` | 0.7500 | `flux_surface_normal_momentum_diffusivity` | 0.5625 | -0.1875 | `superseded` |
| `radial_momentum_diffusivity` | `flux_surface_normal_momentum_diffusivity` | 0.5625 | `flux_surface_normal_momentum_diffusion_coefficient` | 0.6375 | +0.0750 | `grammar_invalid` |
| `radial_neutral_energy_diffusion_coefficient` | `radial_neutral_energy_diffusion_coefficient` | 0.8125 | `radial_total_neutral_heat_diffusivity_over_edge_region` | 0.5875 | -0.2250 | `superseded` |
| `radial_neutral_energy_diffusion_coefficient` | `radial_total_neutral_heat_diffusivity_over_edge_region` | 0.5875 | `radial_neutral_energy_diffusivity` | 0.7375 | +0.1500 | `superseded` |
| `radial_neutral_energy_diffusion_coefficient` | `radial_neutral_energy_diffusivity` | 0.7375 | `flux_surface_normal_neutral_energy_diffusion_coefficient` | 0.7750 | +0.0375 | `attempts_exhausted` |
| `radial_plasma_momentum_diffusion_coefficient` | `radial_plasma_momentum_diffusion_coefficient` | 0.7500 | `radial_plasma_momentum_diffusivity` | 0.8250 | +0.0750 | `superseded` |
| `radial_plasma_momentum_diffusion_coefficient` | `radial_plasma_momentum_diffusivity` | 0.8250 | `flux_surface_normal_bulk_plasma_momentum_diffusivity` | 0.4875 | -0.3375 | `superseded` |
| `radial_plasma_momentum_diffusion_coefficient` | `flux_surface_normal_bulk_plasma_momentum_diffusivity` | 0.4875 | `flux_surface_normal_plasma_momentum_diffusivity` | 0.9000 | +0.4125 | `accepted` |
| `radiated_power_over_scrape_off_layer` | `radiated_power_over_scrape_off_layer` | 0.6625 | `total_radiated_power_over_scrape_off_layer` | 0.7375 | +0.0750 | `superseded` |
| `radiated_power_over_scrape_off_layer` | `total_radiated_power_over_scrape_off_layer` | 0.7375 | `total_plasma_radiated_power_over_scrape_off_layer` | 0.6375 | -0.1000 | `superseded` |
| `radiated_power_over_scrape_off_layer` | `total_plasma_radiated_power_over_scrape_off_layer` | 0.6375 | `power_over_scrape_off_layer_due_to_radiation` | 0.5875 | -0.0500 | `attempts_exhausted` |
| `radius_of_filter` | `radius_of_filter` | 0.6375 | `radius_of_filter_window` | 1.0000 | +0.3625 | `accepted` |
| `radius_of_iron_core_segment` | `radius_of_iron_core_segment` | 0.5500 | `inverse_of_curvature_of_iron_core_segment` | 0.5750 | +0.0250 | `superseded` |
| `radius_of_iron_core_segment` | `inverse_of_curvature_of_iron_core_segment` | 0.5750 | `inverse_of_curvature_of_arc_of_circle_center` | 0.4625 | -0.1125 | `grammar_invalid` |
| `reference_wavelength_of_filter` | `reference_wavelength_of_filter` | 0.6375 | `reference_wavelength_of_filter_window` | 0.6625 | +0.0250 | `successor_collision` |
| `requested_voltage_of_spectrometer` | `requested_voltage_of_spectrometer` | 0.6375 | `requested_voltage_of_spectrometer_channel` | 1.0000 | +0.3625 | `accepted` |
| `runaway_electron_energy_density` | `runaway_electron_energy_density` | 0.8250 | `runaway_electron_kinetic_energy_density` | 0.9750 | +0.1500 | `accepted` |
| `runaway_electron_source_rate` | `runaway_electron_source_rate` | 0.7500 | `total_runaway_electron_source_rate` | 0.7750 | +0.0250 | `superseded` |
| `runaway_electron_source_rate` | `total_runaway_electron_source_rate` | 0.7750 | `net_runaway_electron_source_rate` | 0.7875 | +0.0125 | `superseded` |
| `runaway_electron_source_rate` | `net_runaway_electron_source_rate` | 0.7875 | `tendency_of_runaway_electron_density` | 0.7125 | -0.0750 | `attempts_exhausted` |
| `shattered_pellet_fragment_density` | `shattered_pellet_fragment_density` | 0.5625 | `shattered_pellet_species_number_density_of_pellet_fragment` | 0.9750 | +0.4125 | `accepted` |
| `shattered_pellet_fragment_volume` | `shattered_pellet_fragment_volume` | 0.8375 | `volume_of_pellet_fragment` | 1.0000 | +0.1625 | `accepted` |
| `spectral_radiance_of_soft_xray_detector` | `spectral_radiance_of_soft_xray_detector` | 0.6250 | `incident_soft_xray_radiance` | 0.9563 | +0.3313 | `accepted` |
| `spectral_width_of_spectrometer_channel` | `spectral_width_of_spectrometer_channel` | 0.5375 | `root_mean_square_of_variation_of_vacuum_wavelength_of_spectrometer_channel` | 0.5625 | +0.0250 | `superseded` |
| `spectral_width_of_spectrometer_channel` | `root_mean_square_of_variation_of_vacuum_wavelength_of_spectrometer_channel` | 0.5625 | `root_mean_square_of_difference_of_wavelength_of_spectrometer_channel_and_reference_wavelength_of_spectrometer_channel` | 0.4500 | -0.1125 | `superseded` |
| `spectral_width_of_spectrometer_channel` | `root_mean_square_of_difference_of_wavelength_of_spectrometer_channel_and_reference_wavelength_of_spectrometer_channel` | 0.4500 | `root_mean_square_of_spectral_width_of_spectrometer_channel` | 0.5875 | +0.1375 | `attempts_exhausted` |
| `surface_temperature` | `surface_temperature` | 0.7500 | `surface_temperature_of_plasma_facing_component` | 0.8125 | +0.0625 | `grammar_invalid` |
| `temperature_of_poloidal_field_coil` | `temperature_of_poloidal_field_coil` | 0.6750 | `bulk_temperature_of_poloidal_field_coil` | 0.7937 | +0.1187 | `superseded` |
| `temperature_of_poloidal_field_coil` | `bulk_temperature_of_poloidal_field_coil` | 0.7937 | `temperature_of_coil_conductor` | 0.8875 | +0.0938 | `accepted` |
| `thermal_ion_power_density` | `thermal_ion_power_density` | 0.8000 | `thermal_ion_absorbed_wave_power_density` | 0.9500 | +0.1500 | `accepted` |
| `thermal_power_of_divertor` | `thermal_power_of_divertor` | 0.8125 | `heat_power_of_divertor` | 0.6875 | -0.1250 | `successor_collision` |
| `thermal_power_of_plant_component_port` | `thermal_power_of_plant_component_port` | 0.8125 | `coolant_heating_power_of_plant_component_port` | 0.6625 | -0.1500 | `superseded` |
| `thermal_power_of_plant_component_port` | `coolant_heating_power_of_plant_component_port` | 0.6625 | `coolant_absorbed_power_of_plant_component_port` | 0.6750 | +0.0125 | `superseded` |
| `thermal_power_of_plant_component_port` | `coolant_absorbed_power_of_plant_component_port` | 0.6750 | `absorbed_coolant_power_of_plant_component_port` | 0.5500 | -0.1250 | `attempts_exhausted` |
| `thickness_of_breeder_blanket_module` | `thickness_of_breeder_blanket_module` | 0.5125 | `thickness_of_breeder_blanket` | 0.4938 | -0.0187 | `superseded` |
| `thickness_of_breeder_blanket_module` | `thickness_of_breeder_blanket` | 0.4938 | `surface_thickness_of_breeder_blanket_module` | 0.5875 | +0.0938 | `vocabulary_gap` |
| `thickness_of_cryostat` | `thickness_of_cryostat` | 0.6625 | `normal_thickness_of_cryostat` | 0.5500 | -0.1125 | `superseded` |
| `thickness_of_cryostat` | `normal_thickness_of_cryostat` | 0.5500 | `normal_distance_of_cryostat` | 0.5500 | +0.0000 | `superseded` |
| `thickness_of_cryostat` | `normal_distance_of_cryostat` | 0.5500 | `surface_thickness_of_cryostat` | 0.5938 | +0.0437 | `attempts_exhausted` |
| `thickness_of_plasma_filament` | `thickness_of_plasma_filament` | 0.5250 | `normal_thickness_of_plasma_filament` | 0.5375 | +0.0125 | `superseded` |
| `thickness_of_plasma_filament` | `normal_thickness_of_plasma_filament` | 0.5375 | `width_of_plasma_filament` | 0.5375 | +0.0000 | `superseded` |
| `thickness_of_plasma_filament` | `width_of_plasma_filament` | 0.5375 | `normal_width_of_plasma_filament` | 0.5687 | +0.0312 | `attempts_exhausted` |
| `time_derivative_of_flux_surface_averaged_metric` | `time_derivative_of_flux_surface_averaged_metric` | 0.5000 | `tendency_of_derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface` | 0.9875 | +0.4875 | `accepted` |
| `time_derivative_of_mode_width` | `time_derivative_of_mode_width` | 0.6125 | `time_derivative_of_radial_width_of_neoclassical_tearing_mode` | 0.9875 | +0.3750 | `accepted` |
| `toroidal_angle_of_magnetic_field_probe` | `toroidal_angle_of_magnetic_field_probe` | 0.6813 | `toroidal_angle_of_poloidal_magnetic_field_probe` | 1.0000 | +0.3187 | `accepted` |
| `toroidal_co_passing_fast_electron_torque_density_due_to_collisions` | `toroidal_co_passing_fast_electron_torque_density_due_to_collisions` | 0.6750 | `toroidal_co_passing_fast_electron_torque_density_due_to_collisional_transport` | 0.7875 | +0.1125 | `grammar_invalid` |
| `toroidal_fast_electron_torque_density_due_to_collisions` | `toroidal_fast_electron_torque_density_due_to_collisions` | 0.7562 | `toroidal_fast_particle_torque_density_due_to_coulomb_collisions_with_electrons` | 0.8500 | +0.0938 | `accepted` |
| `toroidal_fast_electron_torque_due_to_collisions` | `toroidal_fast_electron_torque_due_to_collisions` | 0.6625 | `toroidal_volume_integrated_fast_electron_torque_density_due_to_collisions` | 0.5625 | -0.1000 | `grammar_invalid` |
| `toroidal_ion_torque` | `toroidal_ion_torque` | 0.7625 | `toroidal_explicit_ion_torque` | 0.9125 | +0.1500 | `accepted` |
| `toroidal_plasma_momentum_at_measurement_position` | `toroidal_plasma_momentum_at_measurement_position` | 0.6125 | `toroidal_total_plasma_angular_momentum_at_measurement_position` | 0.7250 | +0.1125 | `superseded` |
| `toroidal_plasma_momentum_at_measurement_position` | `toroidal_total_plasma_angular_momentum_at_measurement_position` | 0.7250 | `toroidal_total_plasma_volumetric_angular_momentum_at_measurement_position` | 0.9500 | +0.2250 | `accepted` |
| `toroidal_total_plasma_momentum_at_plasma_boundary` | `toroidal_total_plasma_momentum_at_plasma_boundary` | 0.7750 | `toroidal_cumulative_inside_flux_surface_total_plasma_momentum_at_separatrix` | 0.6625 | -0.1125 | `grammar_invalid` |
| `toroidal_trapped_fast_electron_torque_density_due_to_collisions` | `toroidal_trapped_fast_electron_torque_density_due_to_collisions` | 0.6500 | `toroidal_trapped_fast_electron_torque_density_due_to_collisional_transport` | 0.7000 | +0.0500 | `successor_collision` |
| `total_ion_density` | `total_ion_density` | 0.5875 | `total_ion_number_density` | 0.9250 | +0.3375 | `accepted` |
| `total_neutral_density` | `total_neutral_density` | 0.8125 | `total_neutral_number_density` | 0.9812 | +0.1687 | `accepted` |
| `total_particle_flux_of_divertor_due_to_recycling` | `total_particle_flux_of_divertor_due_to_recycling` | 0.8000 | `total_particle_source_rate_at_divertor_target_due_to_recycling` | 0.5875 | -0.2125 | `superseded` |
| `total_particle_flux_of_divertor_due_to_recycling` | `total_particle_source_rate_at_divertor_target_due_to_recycling` | 0.5875 | `surface_integrated_total_particle_flux_at_divertor_target_due_to_recycling` | 0.5375 | -0.0500 | `superseded` |
| `total_particle_flux_of_divertor_due_to_recycling` | `surface_integrated_total_particle_flux_at_divertor_target_due_to_recycling` | 0.5375 | `total_particle_flux_at_divertor_target_due_to_recycling` | 0.5375 | +0.0000 | `attempts_exhausted` |
| `total_power_of_ion_cyclotron_heating_antenna` | `total_power_of_ion_cyclotron_heating_antenna` | 0.7125 | `total_launched_power_of_ion_cyclotron_heating_antenna` | 0.6250 | -0.0875 | `superseded` |
| `total_power_of_ion_cyclotron_heating_antenna` | `total_launched_power_of_ion_cyclotron_heating_antenna` | 0.6250 | `total_wave_launched_power_of_ion_cyclotron_heating_antenna` | 0.5750 | -0.0500 | `superseded` |
| `total_power_of_ion_cyclotron_heating_antenna` | `total_wave_launched_power_of_ion_cyclotron_heating_antenna` | 0.5750 | `total_launched_power_due_to_ion_cyclotron_heating` | 0.7500 | +0.1750 | `attempts_exhausted` |
| `total_power_of_plant_system` | `total_power_of_plant_system` | 0.6375 | `total_absorbed_power_of_plant_system` | 0.5750 | -0.0625 | `successor_collision` |
| `total_thermal_particle_source_rate` | `total_thermal_particle_source_rate` | 0.6937 | `volume_integrated_particle_source_rate` | 0.8938 | +0.2000 | `accepted` |
| `total_thermal_power_at_inlet` | `total_thermal_power_at_inlet` | 0.7250 | `total_heat_power_at_inlet` | 0.7125 | -0.0125 | `superseded` |
| `total_thermal_power_at_inlet` | `total_heat_power_at_inlet` | 0.7125 | `total_incident_thermal_power_at_inlet` | 0.6625 | -0.0500 | `superseded` |
| `total_thermal_power_at_inlet` | `total_incident_thermal_power_at_inlet` | 0.6625 | `total_incident_thermal_power` | 0.6750 | +0.0125 | `attempts_exhausted` |
| `upper_bound_wavelength_of_camera` | `upper_bound_wavelength_of_camera` | 0.7000 | `upper_bound_wavelength_of_visible_camera` | 0.9750 | +0.2750 | `accepted` |
| `vertical_angle_of_poloidal_field_coil` | `vertical_angle_of_poloidal_field_coil` | 0.5000 | `tilt_angle_of_poloidal_field_coil` | 0.8375 | +0.3375 | `grammar_invalid` |
| `vertical_coordinate_of_bragg_crystal` | `vertical_coordinate_of_bragg_crystal` | 0.6687 | `vertical_coordinate_of_grating` | 0.6750 | +0.0063 | `superseded` |
| `vertical_coordinate_of_bragg_crystal` | `vertical_coordinate_of_grating` | 0.6750 | `vertical_position_of_grating` | 0.5875 | -0.0875 | `grammar_invalid` |
| `vertical_coordinate_of_optical_element` | `vertical_coordinate_of_optical_element` | 0.6750 | `vertical_centroid_of_optical_element` | 0.5813 | -0.0938 | `superseded` |
| `vertical_coordinate_of_optical_element` | `vertical_centroid_of_optical_element` | 0.5813 | `vertical_coordinate_of_fibre_bundle` | 0.9750 | +0.3937 | `accepted` |
| `vertical_coordinate_of_plasma_boundary` | `vertical_coordinate_of_plasma_boundary` | 0.7250 | `vertical_outline_of_plasma_boundary` | 0.6875 | -0.0375 | `superseded` |
| `vertical_coordinate_of_plasma_boundary` | `vertical_outline_of_plasma_boundary` | 0.6875 | `vertical_outline_of_plasma_filament` | 0.6813 | -0.0062 | `superseded` |
| `vertical_coordinate_of_plasma_boundary` | `vertical_outline_of_plasma_filament` | 0.6813 | `vertical_coordinate_of_plasma_filament` | 0.6687 | -0.0125 | `attempts_exhausted` |
| `vertical_front_surface_curvature_of_optical_element` | `vertical_front_surface_curvature_of_optical_element` | 0.7125 | `inverse_of_second_local_tangential_front_surface_curvature_of_optical_element` | 0.6000 | -0.1125 | `superseded` |
| `vertical_front_surface_curvature_of_optical_element` | `inverse_of_second_local_tangential_front_surface_curvature_of_optical_element` | 0.6000 | `second_local_tangential_front_surface_radius_of_optical_element` | 0.9375 | +0.3375 | `accepted` |
| `vertical_ion_momentum_diffusivity` | `vertical_ion_momentum_diffusivity` | 0.8125 | `vertical_total_ion_momentum_diffusivity` | 0.7000 | -0.1125 | `grammar_invalid` |
| `vertical_outline_of_vacuum_vessel` | `vertical_outline_of_vacuum_vessel` | 0.5875 | `vertical_outline_of_wall` | 0.6500 | +0.0625 | `successor_collision` |
| `vertical_outline_of_wall_material` | `vertical_outline_of_wall_material` | 0.6375 | `vertical_outline_of_plasma_facing_component` | 0.9938 | +0.3563 | `accepted` |
| `voltage_of_reflectometer_antenna` | `voltage_of_reflectometer_antenna` | 0.6000 | `wave_voltage_amplitude` | 0.9000 | +0.3000 | `accepted` |
| `wave_beam_energy` | `wave_beam_energy` | 0.5437 | `thomson_scattering_laser_pulse_energy_at_outlet` | 0.8750 | +0.3313 | `accepted` |
| `wave_beam_energy_at_launching_position` | `wave_beam_energy_at_launching_position` | 0.5250 | `total_wave_beam_launched_pulse_energy_at_launching_position` | 0.5500 | +0.0250 | `superseded` |
| `wave_beam_energy_at_launching_position` | `total_wave_beam_launched_pulse_energy_at_launching_position` | 0.5500 | `launched_pulse_energy_of_thomson_scattering_laser` | 0.9563 | +0.4062 | `accepted` |
| `wave_magnetic_field` | `wave_magnetic_field` | 0.5250 | `perturbed_wave_magnetic_field` | 0.4875 | -0.0375 | `superseded` |
| `wave_magnetic_field` | `perturbed_wave_magnetic_field` | 0.4875 | `perturbed_magnetic_field_of_wave_beam` | 0.4500 | -0.0375 | `superseded` |
| `wave_magnetic_field` | `perturbed_magnetic_field_of_wave_beam` | 0.4500 | `wave_magnetic_field_amplitude` | 0.5000 | +0.0500 | `attempts_exhausted` |
| `width_of_hard_xray_detector` | `width_of_hard_xray_detector` | 0.7438 | `first_local_tangential_width_of_hard_xray_detector` | 0.9875 | +0.2438 | `accepted` |
| `width_of_neutron_detector` | `width_of_neutron_detector` | 0.7375 | `first_local_tangential_width_of_diagnostic_aperture` | 0.9688 | +0.2312 | `accepted` |
| `width_of_reflectometer_antenna` | `width_of_reflectometer_antenna` | 0.5875 | `first_local_tangential_width_of_reflectometer_antenna` | 0.5625 | -0.0250 | `successor_collision` |
| `xenon_source_rate` | `xenon_source_rate` | 0.7875 | `xenon_source_rate_due_to_gas_injection` | 1.0000 | +0.2125 | `accepted` |
