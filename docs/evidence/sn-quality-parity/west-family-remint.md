# WEST production family packet refresh and disposition

Snapshot: 2026-08-25, default live `codex` graph. The exact production packet was minted with the repository's `load_sources_file()` and `mint_sn_list()` path from `west_production_dd_paths.yaml`. All graph joins used `StandardName.id`.

## Outcome

The real direct-plus-immediate-family WEST packet is now fully dispositioned. The fresh mint remained cardinality-stable at **416 identities: 218 direct and 198 family-only**. Of those identities, **371 are ship-ready and every one has a refresh receipt**: 194 cleared ordinary documentation quorum in this run and 177 retain earlier ordinary-quorum refresh receipts. The other **45 are explicitly withheld** for their current name, validation, or documentation lifecycle; none is silently counted as shipped.

The documentation census, keyed on `StandardName.id`, moved from **35 non-accepted to 32 non-accepted**. Direct non-accepted documentation moved **7 → 4**; family-only non-accepted documentation remained **28 → 28** because those rows are gated predominantly by non-accepted names. The graph carried **4,656/4,656** `StandardName.id` values and **0/4,656** values for the undeclared `StandardName.name` property at reconciliation. A name-keyed query would therefore still return a plausible but false zero.

| Measure | Before | After |
|---|---:|---:|
| Production mint | 416 | 416 |
| Direct / family-only | 218 / 198 | 218 / 198 |
| Documentation accepted | 381 | 384 |
| Documentation not accepted | **35** | **32** |
| Direct documentation not accepted | 7 | 4 |
| Family-only documentation not accepted | 28 | 28 |
| Ship-ready and refreshed | — | **371** |
| Withheld with a named reason | — | **45** |

The after-state has 384 accepted documents, but only 371 identities ship: thirteen accepted documents belong to identities whose name lifecycle or validation state is not currently authoritative. Documentation acceptance is not used to waive those other lifecycle axes.

## Exact refresh execution

The prior bounded refresh receipts named 187 identities in union, 186 of which remain in the current mint. Fresh admission excluded those receipts and selected the remaining clean packet work: **191 accepted documents plus five accepted-name reviewed documents**, for **196 exact identities**.

Before any rewrite, `reset_standard_name_docs()` wrote and independently re-read **196/196 exact prior-text `DocsRevision` snapshots**. It then reset and scope-stamped the same 196 identities under scope `9909979b-c59e-44fc-bfdb-40381a0c1308`. No packet row had a live claim and no Standard Name run was active at admission.

The scoped production loop used `--docs-only --skip-global-maintenance --cost-limit 50 --min-score 0.85 --rotation-cap 3`. It completed as run `6a35a90e-34dc-4276-9da8-e837ddb566f9` with `stop_reason=no_eligible_work`:

| Pool | Operations |
|---|---:|
| Generate documentation | 196 |
| Review documentation | 233 |
| Refine documentation | 37 |

Of the 196 run identities, **194 finished accepted** and two remain fail-closed at `docs_stage=reviewed`: `spectral_wavelength_of_optical_element` and `square_of_magnetic_field_magnitude`. The log records **39 below-bar quorum outcomes across 36 distinct documents**, which routed into 37 ordinary refinement executions. Every one of the 194 promotions has a final aggregate score at or above 0.85 and at least two fresh attached `StandardNameReview` rows dated after reset. All 196 run identities have fresh review evidence. No direct-accept mutation was issued; promotion occurred only through `persist_reviewed_docs` in the ordinary review pool.

Representative refreshed packet rows include:

| Identity | Packet scope | WEST source binding | Final score | Disposition |
|---|---|---|---:|---|
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | direct | `equilibrium/time_slice/profiles_1d/j_parallel_wave` | 0.91875 | ship: refreshed in this run |
| `poloidal_magnetic_flux_at_magnetic_axis` | family-only | immediate family closure | 0.96875 | ship: refreshed in this run |
| `ratio_of_volume_averaged_tritium_density_to_volume_averaged_electron_density` | family-only | immediate family closure | 0.9625 | ship: refreshed after one ordinary refinement |
| `spectral_wavelength_of_optical_element` | direct | WEST diagnostic optics source | 0.35625 | withhold: documentation remains reviewed |

The direct bindings and family-only classifications in the exact ledger below are the production mint's current graph state; an empty source-binding list for a family-only row is expected because the row entered through immediate-family closure rather than the manifest's direct source join.

## Spend and authority

Actual run spend was **USD 34.268877 / USD 50.000000**, leaving **USD 15.731123** of this node's ceiling. The run's `SNRun.cost_spent` agrees with the sum of all **719** scoped `LLMCost.llm_cost` rows to floating-point precision.

The running authorised total moved from **USD 36.950842 to USD 71.219719 / USD 150.000000**, leaving **USD 78.780281**. The unused authority was not consumed because the packet reached `no_eligible_work`; the remaining rows are lifecycle-withheld, not capacity-deferred.

## Withheld identities

All 45 withheld identities are named in the exact ledger. Their reason is constructed from the current authoritative axes: non-accepted name lifecycle, non-valid validation state, or non-accepted documentation lifecycle. The two accepted-name/valid rows that remain documentation-reviewed are retained visibly and fail-closed. No withheld identity is represented as refreshed-and-shippable.

## Durable receipts

- Read-only exact mint, key coverage, admission, and before ledger: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T084423107172-n-westfamilyremint/preflight.json`
- Exact snapshot and reset receipt: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T084423107172-n-westfamilyremint/reset-receipt.json`
- Full production loop log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T084423107172-n-westfamilyremint/logs/sn-run-live.log`
- Independent after census, scoped spend reconciliation, fresh review proof, withhold reasons, and 416-row ledger: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T084423107172-n-westfamilyremint/post-run.json`

## Exact 416-identity disposition ledger

| Standard Name identity | Scope | Name stage | Docs stage | Validation | Docs score | Disposition |
|---|---|---|---|---|---:|---|
| `accumulated_deposited_energy_of_plasma_facing_component` | direct | accepted | accepted | valid | 0.90625 | ship: previously refreshed and accepted by ordinary quorum |
| `accumulated_total_particle_count_due_to_gas_injection` | direct | accepted | accepted | valid | 0.9312499999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `alfven_time` | family-only | accepted | accepted | valid | 0.8875 | ship: refreshed in this run and accepted by ordinary quorum |
| `area_of_diagnostic_aperture` | direct | accepted | accepted | valid | 0.94375 | ship: refreshed in this run and accepted by ordinary quorum |
| `area_of_poloidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `area_of_toroidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `argon_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `argon_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.94375 | ship: refreshed in this run and accepted by ordinary quorum |
| `atomic_count` | direct | accepted | accepted | valid | 0.90625 | ship: refreshed in this run and accepted by ordinary quorum |
| `atomic_mass` | direct | accepted | accepted | valid | 0.99375 | ship: previously refreshed and accepted by ordinary quorum |
| `beryllium_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `beryllium_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `beryllium_density_at_plasma_boundary` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `beta` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `boron_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `boron_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.99375 | ship: refreshed in this run and accepted by ordinary quorum |
| `boron_density_at_plasma_boundary` | family-only | accepted | accepted | valid | 0.99375 | ship: refreshed in this run and accepted by ordinary quorum |
| `breakdown_initial_time` | direct | accepted | accepted | valid | 0.975 | ship: previously refreshed and accepted by ordinary quorum |
| `breakdown_magnetic_field` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `capacitance_of_ion_cyclotron_heating_antenna` | direct | accepted | accepted | valid | 0.90625 | ship: refreshed in this run and accepted by ordinary quorum |
| `carbon_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.9437500000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `carbon_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `carbon_density_at_plasma_boundary` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `cold_neutral_fraction` | direct | accepted | accepted | valid | 0.96875 | ship: previously refreshed and accepted by ordinary quorum |
| `cold_neutral_temperature` | direct | accepted | accepted | valid | 0.89375 | ship: previously refreshed and accepted by ordinary quorum |
| `coolant_mass` | family-only | pending | null | null | — | withhold: name lifecycle pending; validation None; documentation lifecycle None |
| `coolant_temperature_at_inlet` | direct | accepted | accepted | valid | 0.9375 | ship: previously refreshed and accepted by ordinary quorum |
| `coolant_temperature_at_outlet` | direct | accepted | accepted | valid | 0.88125 | ship: previously refreshed and accepted by ordinary quorum |
| `coolant_transit_time_of_plant_component_port` | direct | accepted | accepted | valid | 0.93125 | ship: previously refreshed and accepted by ordinary quorum |
| `current_of_passive_loop` | direct | accepted | accepted | valid | 0.9312499999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `density_at_divertor_target` | family-only | accepted | accepted | valid | 0.9437500000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.9 | ship: refreshed in this run and accepted by ordinary quorum |
| `density_at_plasma_boundary` | family-only | accepted | accepted | valid | 0.9125 | ship: refreshed in this run and accepted by ordinary quorum |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | direct | accepted | accepted | valid | 0.8625 | ship: refreshed in this run and accepted by ordinary quorum |
| `derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_volume_of_flux_surface` | direct | accepted | accepted | valid | 0.9375 | ship: previously refreshed and accepted by ordinary quorum |
| `derivative_with_respect_to_toroidal_flux_coordinate_of_area_of_flux_surface` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface` | direct | accepted | accepted | valid | 0.96875 | ship: previously refreshed and accepted by ordinary quorum |
| `deuterium_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `deuterium_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `deuterium_density_at_plasma_boundary` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `deuterium_deuterium_neutron_flux` | family-only | drafted | pending | quarantined | 0.95 | withhold: name lifecycle drafted; validation quarantined; documentation lifecycle pending |
| `deuterium_tritium_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.93125 | ship: refreshed in this run and accepted by ordinary quorum |
| `deuterium_tritium_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.9624999999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `deuterium_tritium_density_at_plasma_boundary` | family-only | accepted | accepted | valid | 1.0 | ship: refreshed in this run and accepted by ordinary quorum |
| `difference_of_total_plasma_heating_power_and_time_derivative_of_plasma_stored_energy` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `difference_of_vacuum_poloidal_current_function_and_initial_vacuum_poloidal_current_function` | direct | accepted | accepted | valid | 0.9625 | ship: previously refreshed and accepted by ordinary quorum |
| `effective_charge` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `effective_turn_count_of_coil_conductor_element` | direct | accepted | accepted | valid | 0.9 | ship: previously refreshed and accepted by ordinary quorum |
| `effective_turn_count_of_passive_loop` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `electron_density` | family-only | drafted | pending | valid | — | withhold: name lifecycle drafted; documentation lifecycle pending |
| `electron_density_at_divertor_target` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `electron_density_at_magnetic_axis` | direct | accepted | accepted | valid | 0.9125 | ship: previously refreshed and accepted by ordinary quorum |
| `electron_density_at_plasma_boundary` | direct | accepted | accepted | valid | 1.0 | ship: previously refreshed and accepted by ordinary quorum |
| `electron_pressure` | family-only | accepted | accepted | valid | 0.875 | ship: refreshed in this run and accepted by ordinary quorum |
| `electron_temperature` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `electron_temperature_at_divertor_target` | direct | accepted | accepted | valid | 0.9875 | ship: previously refreshed and accepted by ordinary quorum |
| `electron_temperature_at_magnetic_axis` | direct | accepted | accepted | valid | 0.975 | ship: previously refreshed and accepted by ordinary quorum |
| `elongation_of_flux_surface` | direct | accepted | accepted | valid | 0.96875 | ship: previously refreshed and accepted by ordinary quorum |
| `elongation_of_plasma_boundary` | direct | accepted | accepted | valid | 0.96875 | ship: previously refreshed and accepted by ordinary quorum |
| `energy_confinement_enhancement_factor` | direct | accepted | accepted | valid | 0.9624999999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `energy_flux_at_divertor_target` | family-only | accepted | accepted | valid | 0.86875 | ship: refreshed in this run and accepted by ordinary quorum |
| `equilibrium_weight_of_flux_loop` | direct | accepted | accepted | valid | 0.8812500000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `equilibrium_weight_of_interferometer_beam` | direct | accepted | accepted | valid | 0.975 | ship: refreshed in this run and accepted by ordinary quorum |
| `equilibrium_weight_of_poloidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `etendue_of_hard_xray_detector` | direct | accepted | accepted | valid | 0.93125 | ship: previously refreshed and accepted by ordinary quorum |
| `etendue_of_spectrometer_channel` | direct | accepted | accepted | valid | 0.96875 | ship: previously refreshed and accepted by ordinary quorum |
| `external_magnetic_flux` | family-only | accepted | accepted | valid | 0.9875 | ship: refreshed in this run and accepted by ordinary quorum |
| `faraday_angle` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `fast_electron_density` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `fast_electron_pressure` | family-only | accepted | accepted | valid | 0.89375 | ship: refreshed in this run and accepted by ordinary quorum |
| `flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | family-only | accepted | accepted | valid | 0.99375 | ship: refreshed in this run and accepted by ordinary quorum |
| `flux_surface_averaged_electron_density_at_plasma_boundary` | family-only | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `flux_surface_averaged_inverse_of_major_radius` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `flux_surface_averaged_inverse_of_square_of_magnetic_field_magnitude` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `flux_surface_averaged_inverse_of_square_of_major_radius` | direct | accepted | accepted | valid | 0.975 | ship: previously refreshed and accepted by ordinary quorum |
| `flux_surface_averaged_magnetic_field_magnitude` | direct | accepted | accepted | valid | 0.9125 | ship: previously refreshed and accepted by ordinary quorum |
| `flux_surface_averaged_major_radius` | direct | accepted | accepted | valid | 0.99375 | ship: previously refreshed and accepted by ordinary quorum |
| `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_magnetic_field_magnitude` | direct | accepted | accepted | valid | 0.9437500000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_major_radius` | direct | accepted | accepted | valid | 0.9375 | ship: previously refreshed and accepted by ordinary quorum |
| `flux_surface_averaged_square_of_magnetic_field_magnitude` | direct | accepted | accepted | valid | 0.96875 | ship: previously refreshed and accepted by ordinary quorum |
| `flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude` | direct | accepted | accepted | valid | 0.88125 | ship: previously refreshed and accepted by ordinary quorum |
| `flux_surface_averaged_toroidal_flux_coordinate_gradient_magnitude` | direct | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `forward_power_of_ion_cyclotron_heating_antenna` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `forward_wave_phase_of_ion_cyclotron_heating_antenna` | direct | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `frequency_of_diagnostic_antenna` | direct | reviewed | accepted | valid | 0.91875 | withhold: name lifecycle reviewed |
| `frequency_of_ion_cyclotron_heating_antenna` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `gap_at_outboard_midplane` | direct | accepted | accepted | valid | 0.9875 | ship: previously refreshed and accepted by ordinary quorum |
| `gap_at_plasma_boundary` | direct | reviewed | accepted | valid | 0.9125 | withhold: name lifecycle reviewed |
| `gas_flow` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `greenwald_density` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `hard_xray_brightness` | direct | accepted | accepted | valid | 0.94375 | ship: refreshed in this run and accepted by ordinary quorum |
| `hard_xray_emissivity` | direct | accepted | accepted | valid | 0.93125 | ship: refreshed in this run and accepted by ordinary quorum |
| `height_of_poloidal_field_coil` | direct | accepted | accepted | valid | 0.9625 | ship: previously refreshed and accepted by ordinary quorum |
| `helium_3_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.9624999999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `helium_3_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `helium_3_density_at_plasma_boundary` | family-only | accepted | accepted | valid | 0.9875 | ship: refreshed in this run and accepted by ordinary quorum |
| `helium_4_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `helium_4_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `helium_4_density_at_plasma_boundary` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `hot_neutral_fraction` | direct | accepted | accepted | valid | 0.9875 | ship: previously refreshed and accepted by ordinary quorum |
| `hot_neutral_temperature` | direct | reviewed | accepted | valid | 0.9375 | withhold: name lifecycle reviewed |
| `hydrogen_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `hydrogen_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 1.0 | ship: refreshed in this run and accepted by ordinary quorum |
| `hydrogen_density_at_plasma_boundary` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `incident_soft_xray_radiance` | direct | accepted | accepted | valid | 0.9 | ship: previously refreshed and accepted by ordinary quorum |
| `initial_polarization_ellipticity_of_polarimeter_beam` | direct | accepted | accepted | valid | 0.93125 | ship: refreshed in this run and accepted by ordinary quorum |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | direct | drafted | pending | quarantined | — | withhold: name lifecycle drafted; validation quarantined; documentation lifecycle pending |
| `inverse_of_major_radius` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `inverse_of_square_of_magnetic_field_magnitude` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `inverse_of_square_of_major_radius` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `ion_atomic_mass` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `ion_current_of_mass_spectrometer_channel` | direct | accepted | accepted | valid | 0.93125 | ship: previously refreshed and accepted by ordinary quorum |
| `ion_density_at_plasma_boundary` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `ion_temperature_at_divertor_target` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `iron_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `iron_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `iron_density_at_plasma_boundary` | family-only | reviewed | accepted | valid | 0.90625 | withhold: name lifecycle reviewed |
| `krypton_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.9312499999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `krypton_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.9125000000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `krypton_density_at_plasma_boundary` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `launched_power_of_lower_hybrid_antenna` | direct | accepted | accepted | valid | 0.9375 | ship: previously refreshed and accepted by ordinary quorum |
| `length_of_interferometer_beam` | family-only | accepted | accepted | valid | 0.9624999999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `length_of_poloidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.875 | ship: previously refreshed and accepted by ordinary quorum |
| `length_of_toroidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.89375 | ship: previously refreshed and accepted by ordinary quorum |
| `line_averaged_effective_charge` | direct | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `line_averaged_electron_density` | direct | accepted | accepted | valid | 0.93125 | ship: previously refreshed and accepted by ordinary quorum |
| `line_averaged_electron_temperature` | family-only | accepted | accepted | valid | 0.94375 | ship: refreshed in this run and accepted by ordinary quorum |
| `line_averaged_hydrogen_density` | family-only | accepted | accepted | valid | 0.93125 | ship: refreshed in this run and accepted by ordinary quorum |
| `line_integrated_electron_number_density` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | direct | drafted | pending | quarantined | — | withhold: name lifecycle drafted; validation quarantined; documentation lifecycle pending |
| `lithium_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `lithium_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.96875 | ship: refreshed in this run and accepted by ordinary quorum |
| `lithium_density_at_plasma_boundary` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` | family-only | accepted | accepted | valid | 0.90625 | ship: refreshed in this run and accepted by ordinary quorum |
| `loop_voltage_at_plasma_boundary` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `lower_bound_photon_energy` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `lower_triangularity_of_flux_surface` | direct | accepted | accepted | valid | 0.9375 | ship: previously refreshed and accepted by ordinary quorum |
| `lower_triangularity_of_plasma_boundary` | direct | accepted | accepted | valid | 1.0 | ship: previously refreshed and accepted by ordinary quorum |
| `lower_wavelength_of_filter` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `magnetic_field` | family-only | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `magnetic_field_at_pedestal_top_low_field_side` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `magnetic_flux` | family-only | accepted | accepted | valid | 0.9437500000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `magnetic_flux_due_to_diamagnetic_drift` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `magnetic_shear_at_flux_surface` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `maximum_gas_flow` | family-only | accepted | accepted | valid | 0.9624999999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `maximum_magnetic_field` | family-only | accepted | accepted | valid | 0.9875 | ship: refreshed in this run and accepted by ordinary quorum |
| `maximum_magnetic_field_magnitude` | direct | accepted | accepted | valid | 0.89375 | ship: previously refreshed and accepted by ordinary quorum |
| `maximum_of_energy_flux_at_divertor_target` | direct | accepted | accepted | valid | 0.96875 | ship: previously refreshed and accepted by ordinary quorum |
| `mhd_energy` | direct | accepted | accepted | valid | 0.9375 | ship: previously refreshed and accepted by ordinary quorum |
| `minimum_gas_flow` | family-only | accepted | accepted | valid | 0.96875 | ship: refreshed in this run and accepted by ordinary quorum |
| `minimum_over_flux_surface_magnetic_field_magnitude` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `minimum_safety_factor` | direct | accepted | accepted | valid | 0.9437500000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `minor_radius_of_plasma_boundary` | direct | accepted | accepted | valid | 0.9624999999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `motional_stark_photon_radiance_at_spectral_line` | family-only | accepted | accepted | valid | 0.8999999999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `neon_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.91875 | ship: refreshed in this run and accepted by ordinary quorum |
| `neon_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.975 | ship: refreshed in this run and accepted by ordinary quorum |
| `neon_density_at_plasma_boundary` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `neutral_species_atomic_mass` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `neutral_temperature` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `neutron_flux` | direct | accepted | accepted | valid | 0.9 | ship: previously refreshed and accepted by ordinary quorum |
| `nitrogen_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `nitrogen_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `nitrogen_density_at_plasma_boundary` | family-only | accepted | accepted | valid | 1.0 | ship: refreshed in this run and accepted by ordinary quorum |
| `non_axisymmetric_magnetic_field` | family-only | accepted | accepted | valid | 0.8875 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_plasma_internal_inductance` | direct | accepted | accepted | valid | 0.93125 | ship: previously refreshed and accepted by ordinary quorum |
| `normalized_poloidal_flux_coordinate` | direct | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `normalized_poloidal_flux_coordinate_at_measurement_position` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_poloidal_flux_coordinate_at_minimum_safety_factor` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_poloidal_flux_coordinate_of_pedestal` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_poloidal_flux_coordinate_of_plasma_boundary` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_at_beam_tracing_point` | family-only | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_at_constraint_position` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_at_internal_transport_barrier` | family-only | accepted | accepted | valid | 0.8999999999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_at_measurement_position` | direct | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_at_minimum_safety_factor` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_at_pedestal_top` | family-only | accepted | accepted | valid | 0.85 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_at_pellet_path` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_at_plasma_boundary` | family-only | accepted | accepted | valid | 0.9125 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_at_sawtooth_inversion_radius` | family-only | accepted | accepted | valid | 0.9312499999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_at_sawtooth_mixing_radius` | family-only | accepted | accepted | valid | 0.9750000000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_toroidal_flux_coordinate_of_line_of_sight` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `normalized_toroidal_flux_coordinate_of_measurement_position` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `normalized_toroidal_flux_coordinate_of_neoclassical_tearing_mode_center` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `normalized_toroidal_plasma_beta` | direct | accepted | accepted | valid | 0.9312499999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `oxygen_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `oxygen_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.9875 | ship: refreshed in this run and accepted by ordinary quorum |
| `oxygen_density_at_plasma_boundary` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | direct | accepted | accepted | valid | 0.875 | ship: refreshed in this run and accepted by ordinary quorum |
| `parallel_magnetic_field` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `per_toroidal_and_poloidal_mode_number_launched_power_of_lower_hybrid_antenna` | family-only | accepted | accepted | valid | 0.8875 | ship: refreshed in this run and accepted by ordinary quorum |
| `per_toroidal_mode_launched_power_of_lower_hybrid_antenna` | family-only | accepted | accepted | valid | 0.90625 | ship: refreshed in this run and accepted by ordinary quorum |
| `perturbed_plasma_pressure` | family-only | accepted | accepted | valid | 0.9125 | ship: refreshed in this run and accepted by ordinary quorum |
| `perturbed_vacuum_magnetic_field` | family-only | accepted | accepted | valid | 0.94375 | ship: refreshed in this run and accepted by ordinary quorum |
| `phase_of_ion_cyclotron_heating_antenna` | family-only | accepted | accepted | quarantined | 0.98125 | withhold: validation quarantined |
| `photon_radiance_at_spectral_line` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `plasma_beta` | direct | accepted | accepted | valid | 0.975 | ship: previously refreshed and accepted by ordinary quorum |
| `plasma_current` | direct | accepted | accepted | valid | 0.9750000000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `plasma_magnetic_field` | family-only | accepted | accepted | valid | 0.88125 | ship: refreshed in this run and accepted by ordinary quorum |
| `plasma_pressure` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `plasma_pressure_imaginary_part` | family-only | accepted | accepted | valid | 1.0 | ship: refreshed in this run and accepted by ordinary quorum |
| `plasma_pressure_real_part` | family-only | accepted | accepted | valid | 0.9750000000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `polarization_ellipticity_of_polarimeter_beam` | family-only | accepted | accepted | valid | 0.9125000000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `poloidal_angle_of_flux_surface` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `poloidal_angle_of_measurement_position` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `poloidal_angle_of_toroidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.86875 | ship: previously refreshed and accepted by ordinary quorum |
| `poloidal_beta` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `poloidal_current_function` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `poloidal_length_of_flux_surface` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `poloidal_magnetic_field` | direct | accepted | accepted | valid | 0.8625 | ship: previously refreshed and accepted by ordinary quorum |
| `poloidal_magnetic_field_at_constraint_position` | direct | reviewed | accepted | valid | 0.90625 | withhold: name lifecycle reviewed |
| `poloidal_magnetic_flux` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `poloidal_magnetic_flux_at_flux_surface` | direct | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `poloidal_magnetic_flux_at_magnetic_axis` | direct | accepted | accepted | valid | 0.96875 | ship: refreshed in this run and accepted by ordinary quorum |
| `poloidal_magnetic_flux_at_measurement_position` | direct | accepted | accepted | valid | 0.9625 | ship: previously refreshed and accepted by ordinary quorum |
| `poloidal_magnetic_flux_at_plasma_boundary` | direct | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `poloidal_magnetic_flux_of_flux_loop` | direct | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `poloidal_plane_cross_sectional_area_of_flux_surface` | direct | accepted | accepted | valid | 0.99375 | ship: previously refreshed and accepted by ordinary quorum |
| `poloidal_plane_cross_sectional_area_of_plasma_boundary` | direct | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `poloidal_turn_count` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `power_at_wall_due_to_ohmic_dissipation` | family-only | accepted | accepted | valid | 0.9125000000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `power_due_to_ion_cyclotron_heating` | direct | accepted | accepted | valid | 0.96875 | ship: previously refreshed and accepted by ordinary quorum |
| `power_of_divertor_due_to_ohmic_dissipation` | family-only | accepted | accepted | valid | 0.8999999999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `power_of_soft_xray_detector` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `pressure_of_ion_cyclotron_heating_antenna` | direct | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `product_of_poloidal_current_function_and_derivative_with_respect_to_poloidal_magnetic_flux_coordinate_of_poloidal_current_function` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `pulse_duration` | direct | accepted | accepted | valid | 0.975 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate` | family-only | accepted | accepted | valid | 0.9750000000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_at_inboard_midplane` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate_at_outboard_midplane` | direct | accepted | accepted | valid | 0.9625 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate_of_antenna_strap` | family-only | accepted | accepted | valid | 0.8875 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_arc_of_circle_center` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `radial_coordinate_of_camera` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate_of_closest_wall_point` | family-only | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_coil_conductor_element` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_conductor_cross_section` | direct | accepted | accepted | valid | 0.9 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_control_surface` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_current_center` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_detector_pixel` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate_of_diagnostic_aperture` | family-only | accepted | accepted | valid | 0.91875 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_dr_dz_zero_point` | family-only | accepted | accepted | valid | 0.9437500000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_electron_cyclotron_launcher_mirror` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `radial_coordinate_of_ferritic_element` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_ferritic_element_centroid` | family-only | accepted | accepted | valid | 0.975 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_filter_window` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_flux_loop` | direct | accepted | accepted | valid | 0.9625 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate_of_geometric_axis` | direct | accepted | accepted | valid | 0.9125000000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate_of_line_of_sight` | direct | accepted | accepted | valid | 0.98125 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate_of_magnetic_axis` | direct | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_measurement_position` | direct | accepted | accepted | valid | 0.8875 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate_of_neutral_beam_injector` | family-only | accepted | accepted | valid | 0.9437500000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_pellet_path` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `radial_coordinate_of_poloidal_field_coil` | family-only | accepted | accepted | valid | 0.9750000000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_poloidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.875 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate_of_reflector` | family-only | drafted | pending | valid | — | withhold: name lifecycle drafted; documentation lifecycle pending |
| `radial_coordinate_of_rogowski_coil` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_shatter_cone` | family-only | accepted | accepted | valid | 0.9125000000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_shunt` | family-only | accepted | accepted | valid | 0.94375 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_spectrometer_channel` | family-only | accepted | accepted | valid | 0.96875 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_strike_point` | direct | accepted | accepted | valid | 0.93125 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_coordinate_of_thomson_scattering_laser` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_coordinate_of_x_point` | direct | accepted | accepted | valid | 0.975 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_derivative_of_poloidal_magnetic_flux` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_magnetic_field` | family-only | accepted | accepted | valid | 1.0 | ship: refreshed in this run and accepted by ordinary quorum |
| `radial_outline_of_antenna_strap` | direct | accepted | accepted | valid | 0.9375 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_outline_of_limiter_tile` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `radial_outline_of_plasma_boundary` | direct | accepted | accepted | quarantined | 0.9 | withhold: validation quarantined |
| `radial_outline_of_wall` | direct | accepted | accepted | quarantined | 0.95625 | withhold: validation quarantined |
| `radiated_power_over_core_region` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `radiative_temperature_at_magnetic_axis` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `ratio_of_coolant_mass_to_time` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `ratio_of_electron_temperature_at_magnetic_axis_to_volume_averaged_electron_temperature` | family-only | accepted | accepted | valid | 0.94375 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_ion_density_to_electron_density` | family-only | accepted | accepted | valid | 0.96875 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_ion_velocity_to_magnetic_field` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `ratio_of_iron_density_to_electron_density` | family-only | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_line_averaged_electron_density_to_greenwald_density` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `ratio_of_line_averaged_hydrogen_density_to_line_averaged_total_hydrogenic_density` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `ratio_of_neutral_density_of_isotope_to_difference_of_total_neutral_density_and_neutral_density_of_isotope` | direct | accepted | accepted | valid | 0.8812500000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_magnetic_field_magnitude` | family-only | accepted | accepted | valid | 0.96875 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_major_radius` | family-only | accepted | accepted | valid | 0.93125 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_total_ion_density_to_electron_density` | family-only | accepted | accepted | valid | 0.85 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_argon_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.96875 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_beryllium_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_boron_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_carbon_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_deuterium_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_deuterium_tritium_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.91875 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_helium_3_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.96875 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_helium_4_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.94375 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_krypton_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_neon_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_nitrogen_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.93125 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_oxygen_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.9125 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_tritium_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `ratio_of_volume_averaged_tungsten_density_to_volume_averaged_electron_density` | family-only | accepted | accepted | valid | 0.975 | ship: refreshed in this run and accepted by ordinary quorum |
| `reference_magnetic_field` | family-only | accepted | accepted | valid | 1.0 | ship: refreshed in this run and accepted by ordinary quorum |
| `reference_major_radius` | direct | accepted | accepted | valid | 0.90625 | ship: refreshed in this run and accepted by ordinary quorum |
| `reflected_phase_of_ion_cyclotron_heating_antenna` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `reflected_power_of_ion_cyclotron_heating_antenna` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `root_mean_square_of_wave_current_of_antenna_strap` | family-only | accepted | accepted | valid | 0.9875 | ship: refreshed in this run and accepted by ordinary quorum |
| `safety_factor` | direct | accepted | accepted | valid | 0.9624999999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `safety_factor_at_internal_transport_barrier` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `safety_factor_at_magnetic_axis` | direct | accepted | accepted | valid | 0.96875 | ship: previously refreshed and accepted by ordinary quorum |
| `safety_factor_at_normalized_poloidal_magnetic_flux_equal_to_0_95` | direct | accepted | accepted | valid | 0.9375 | ship: previously refreshed and accepted by ordinary quorum |
| `safety_factor_at_pedestal` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `safety_factor_at_pedestal_top` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `safety_factor_at_plasma_boundary` | family-only | accepted | accepted | valid | 0.9624999999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `spectral_bremsstrahlung_radiance` | direct | accepted | accepted | valid | 0.90625 | ship: previously refreshed and accepted by ordinary quorum |
| `spectral_calibration_factor_at_line_of_sight` | direct | accepted | accepted | valid | 0.89375 | ship: previously refreshed and accepted by ordinary quorum |
| `spectral_etendue_of_spectrometer_channel` | family-only | accepted | accepted | valid | 0.9125000000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `spectral_flux_of_spectrometer_channel` | direct | reviewed | accepted | valid | 0.90625 | withhold: name lifecycle reviewed |
| `spectral_radiance` | direct | accepted | accepted | valid | 0.9125 | ship: refreshed in this run and accepted by ordinary quorum |
| `spectral_signal_to_noise_ratio_of_spectrometer_channel` | direct | accepted | accepted | valid | 0.8500000000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `spectral_wavelength_of_optical_element` | direct | accepted | reviewed | valid | 0.7125 | withhold: documentation lifecycle reviewed |
| `square_of_magnetic_field_magnitude` | family-only | accepted | reviewed | valid | 0.6375 | withhold: documentation lifecycle reviewed |
| `square_of_toroidal_flux_coordinate_gradient_magnitude` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `surface_area_of_flux_surface` | direct | accepted | accepted | valid | 0.9375 | ship: previously refreshed and accepted by ordinary quorum |
| `temperature_at_divertor_target` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `temperature_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.9 | ship: refreshed in this run and accepted by ordinary quorum |
| `temperature_of_soft_xray_detector` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `tendency_of_derivative_with_respect_to_toroidal_flux_coordinate_of_volume_of_flux_surface` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `thermal_electron_density` | direct | accepted | accepted | valid | 0.99375 | ship: previously refreshed and accepted by ordinary quorum |
| `thermal_electron_pressure_at_post_sawtooth_crash` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `thermal_plasma_pressure` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `thickness_of_filter` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `time` | family-only | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `time_derivative_of_electron_density` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `toroidal_angle_of_antenna_strap` | direct | accepted | accepted | valid | 0.9437500000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_angle_of_measurement_position` | direct | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `toroidal_angle_of_poloidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.8875 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_angle_of_toroidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_angular_width_of_limiter_tile` | direct | accepted | accepted | valid | 0.9312499999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_beta` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `toroidal_coordinate_of_aperture` | direct | accepted | accepted | valid | 0.8687499999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `toroidal_coordinate_of_camera` | direct | accepted | accepted | valid | 0.9624999999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_coordinate_of_detector_pixel` | direct | accepted | accepted | quarantined | 0.96875 | withhold: validation quarantined |
| `toroidal_coordinate_of_line_of_sight` | direct | accepted | accepted | valid | 0.9125000000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_flux_coordinate` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_flux_coordinate_at_internal_transport_barrier` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `toroidal_flux_coordinate_at_pedestal_top` | family-only | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `toroidal_flux_coordinate_at_plasma_boundary` | family-only | accepted | accepted | valid | 0.91875 | ship: refreshed in this run and accepted by ordinary quorum |
| `toroidal_flux_coordinate_gradient_magnitude` | family-only | accepted | accepted | valid | 1.0 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_flux_coordinate_of_magnetic_axis` | family-only | accepted | accepted | valid | 0.9624999999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `toroidal_flux_coordinate_of_neoclassical_tearing_mode_center` | family-only | accepted | accepted | valid | 0.875 | ship: refreshed in this run and accepted by ordinary quorum |
| `toroidal_flux_surface_averaged_current_density` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_magnetic_field` | direct | accepted | accepted | valid | 0.9125 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_magnetic_field_at_magnetic_axis` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_magnetic_flux` | direct | accepted | accepted | valid | 0.9375 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_magnetic_flux_due_to_diamagnetic_drift` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `toroidal_vacuum_magnetic_field` | direct | accepted | accepted | valid | 0.8625 | ship: refreshed in this run and accepted by ordinary quorum |
| `total_electron_density` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `total_electron_pressure` | direct | accepted | accepted | valid | 0.88125 | ship: previously refreshed and accepted by ordinary quorum |
| `total_external_heating_power` | direct | accepted | accepted | valid | 0.90625 | ship: previously refreshed and accepted by ordinary quorum |
| `total_neutral_source_rate_due_to_gas_injection` | direct | accepted | accepted | valid | 0.93125 | ship: previously refreshed and accepted by ordinary quorum |
| `total_plasma_pressure` | family-only | accepted | accepted | valid | 0.95 | ship: refreshed in this run and accepted by ordinary quorum |
| `total_plasma_radiated_power` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `total_power_at_separatrix` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `total_power_due_to_ion_cyclotron_heating` | direct | accepted | accepted | valid | 0.93125 | ship: previously refreshed and accepted by ordinary quorum |
| `total_power_due_to_ohmic_dissipation` | direct | accepted | accepted | valid | 0.88125 | ship: previously refreshed and accepted by ordinary quorum |
| `triangularity_of_flux_surface` | family-only | accepted | accepted | valid | 0.9 | ship: refreshed in this run and accepted by ordinary quorum |
| `triangularity_of_plasma_boundary` | direct | accepted | accepted | valid | 0.8999999999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `tritium_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.975 | ship: refreshed in this run and accepted by ordinary quorum |
| `tritium_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 1.0 | ship: refreshed in this run and accepted by ordinary quorum |
| `tritium_density_at_plasma_boundary` | family-only | accepted | accepted | valid | 1.0 | ship: refreshed in this run and accepted by ordinary quorum |
| `tritium_tritium_neutron_flux` | family-only | drafted | pending | quarantined | — | withhold: name lifecycle drafted; validation quarantined; documentation lifecycle pending |
| `tungsten_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.975 | ship: refreshed in this run and accepted by ordinary quorum |
| `tungsten_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.9750000000000001 | ship: refreshed in this run and accepted by ordinary quorum |
| `tungsten_density_at_plasma_boundary` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `turn_count_of_correction_coil` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
| `turn_count_of_poloidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.9437500000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `turn_count_of_toroidal_field_coil` | family-only | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `turn_count_of_toroidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.93125 | ship: previously refreshed and accepted by ordinary quorum |
| `upper_photon_energy` | direct | accepted | accepted | valid | 0.9125000000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `upper_triangularity_of_flux_surface` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `upper_triangularity_of_plasma_boundary` | direct | accepted | accepted | valid | 0.9375 | ship: refreshed in this run and accepted by ordinary quorum |
| `upper_wavelength_of_filter` | direct | accepted | accepted | valid | 0.89375 | ship: previously refreshed and accepted by ordinary quorum |
| `vacuum_magnetic_field` | family-only | accepted | accepted | valid | 0.9 | ship: refreshed in this run and accepted by ordinary quorum |
| `vacuum_poloidal_current_function` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `variation_of_length_of_interferometer_beam` | direct | accepted | accepted | valid | 0.9625 | ship: refreshed in this run and accepted by ordinary quorum |
| `vertical_coordinate_of_camera` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_coordinate_of_conductor_cross_section` | direct | accepted | accepted | valid | 0.94375 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_coordinate_of_detector_pixel` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_coordinate_of_flux_loop` | direct | accepted | accepted | valid | 0.9125 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_coordinate_of_geometric_axis` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_coordinate_of_line_of_sight` | direct | accepted | accepted | quarantined | 0.8812500000000001 | withhold: validation quarantined |
| `vertical_coordinate_of_magnetic_axis` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_coordinate_of_measurement_position` | direct | accepted | accepted | valid | 0.9625 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_coordinate_of_poloidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.95625 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_coordinate_of_primary_x_point` | direct | accepted | accepted | valid | 0.9437500000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_coordinate_of_strike_point` | direct | accepted | accepted | valid | 0.9125000000000001 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_coordinate_of_toroidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_magnetic_field` | family-only | accepted | accepted | valid | 0.99375 | ship: refreshed in this run and accepted by ordinary quorum |
| `vertical_outline_of_antenna_strap` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_outline_of_limiter_tile` | direct | accepted | accepted | valid | 0.9624999999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `vertical_outline_of_plasma_facing_component` | direct | accepted | accepted | valid | 0.9312499999999999 | ship: previously refreshed and accepted by ordinary quorum |
| `voltage_of_mass_spectrometer_channel` | direct | accepted | accepted | valid | 0.95 | ship: previously refreshed and accepted by ordinary quorum |
| `voltage_of_poloidal_magnetic_field_probe` | direct | accepted | accepted | valid | 0.8875 | ship: previously refreshed and accepted by ordinary quorum |
| `volume_averaged_effective_charge` | family-only | accepted | accepted | valid | 0.9125 | ship: refreshed in this run and accepted by ordinary quorum |
| `volume_averaged_electron_density` | direct | accepted | accepted | valid | 0.975 | ship: previously refreshed and accepted by ordinary quorum |
| `volume_averaged_electron_temperature` | family-only | accepted | accepted | valid | 1.0 | ship: refreshed in this run and accepted by ordinary quorum |
| `volume_averaged_neutral_temperature` | family-only | accepted | accepted | valid | 0.95625 | ship: refreshed in this run and accepted by ordinary quorum |
| `volume_integrated_total_electron_density` | direct | accepted | accepted | quarantined | 0.90625 | withhold: validation quarantined |
| `volume_of_flux_surface` | direct | reviewed | accepted | valid | 0.9437500000000001 | withhold: name lifecycle reviewed |
| `volume_of_plasma_boundary` | direct | accepted | accepted | valid | 0.925 | ship: previously refreshed and accepted by ordinary quorum |
| `wave_current_of_antenna_strap` | family-only | accepted | accepted | valid | 0.8999999999999999 | ship: refreshed in this run and accepted by ordinary quorum |
| `wave_current_of_antenna_strap_amplitude` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `wave_phase_of_antenna_strap` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `wave_phase_of_ion_cyclotron_heating_antenna` | direct | accepted | accepted | valid | 0.8625 | ship: previously refreshed and accepted by ordinary quorum |
| `wave_phase_of_wave_beam` | direct | accepted | accepted | valid | 0.90625 | ship: previously refreshed and accepted by ordinary quorum |
| `wavelength_of_filter` | family-only | accepted | accepted | valid | 0.90625 | ship: refreshed in this run and accepted by ordinary quorum |
| `wavelength_of_spectral_line` | direct | accepted | accepted | valid | 0.91875 | ship: previously refreshed and accepted by ordinary quorum |
| `wavelength_of_wave_beam` | direct | accepted | accepted | valid | 0.9 | ship: previously refreshed and accepted by ordinary quorum |
| `width_of_poloidal_field_coil` | direct | accepted | accepted | valid | 0.9625 | ship: previously refreshed and accepted by ordinary quorum |
| `xenon_density_at_divertor_target` | family-only | accepted | accepted | valid | 0.925 | ship: refreshed in this run and accepted by ordinary quorum |
| `xenon_density_at_magnetic_axis` | family-only | accepted | accepted | valid | 0.98125 | ship: refreshed in this run and accepted by ordinary quorum |
| `xenon_density_at_plasma_boundary` | family-only | reviewed | pending | valid | — | withhold: name lifecycle reviewed; documentation lifecycle pending |
