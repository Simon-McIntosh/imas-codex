# Exhausted standard-name tail: deterministic recovery triage

Measured read-only against the live production `codex` graph on 2026-09-01. The population is exactly the set of `StandardName` rows whose `name_stage = 'exhausted'`. No model was called, no graph property or relationship was written, and no row is accepted or restaged by this report.

## Headline result

| Disposition | Count | Recovery meaning |
|---|---:|---|
| rescore-candidate | 32 | The same spelling has already earned at least one name-axis reviewer score at or above 0.85, and no deterministic grammar, collision, or vocabulary stop blocks a fresh quorum draw. Use `sn rescore`; never reword or hand-accept. |
| vocabulary-gap | 4 | The row carries `refine_stop_reason='vocabulary_gap'` or a source-linked gap whose live triage is `genuine` / editorial disposition is `add`. Route it to the ISN vocabulary process rather than another refinement attempt. |
| stay-parked | 239 | The available deterministic evidence does not justify a same-name redraw: the stop is grammar/collision based, or no reviewer of this exact spelling has reached 0.85. |
| **Population** | **275** | **32 + 4 + 239 = 275; residual 0.** |

The earlier planning estimate of roughly 349 is stale against this live graph. The exact parked population at measurement time is **275**.

## Deterministic rule and rank

The partition is mutually exclusive and evaluated in this order:

1. **vocabulary-gap** if `refine_stop_reason = 'vocabulary_gap'`, or if a currently producing source links to a `VocabGap` that is triaged `genuine` or has an active editorial `add` disposition.
2. **rescore-candidate** if the stop reason is absent or `attempts_exhausted` and the best score ever recorded by a name-axis reviewer for this exact identity is at least **0.85**. “Best” is the maximum of the current scalar `reviewer_score_name` and the attached `StandardNameReview.score` values. This is evidence that the unchanged spelling can clear the ordinary quorum; it is not acceptance authority.
3. **stay-parked** otherwise. In particular, `grammar_invalid`, `successor_collision`, and lifecycle-collision stops cannot be cured by redrawing the same name, and a lower-scoring spelling has no deterministic evidence for a fresh draw.

Rows are ranked by disposition (rescore first, vocabulary second, parked last), then best reviewer score descending, persisted lineage `chain_length` ascending, and exact identity ascending. Thus every placement and tie-break uses only `refine_stop_reason`, reviewer scores, `chain_length`, and source-linked vocabulary-gap evidence. A missing `chain_length` is the schema-defined pre-counter form and is ranked as zero; the coverage control below reports the nulls instead of hiding them.

## Firing and property controls

The query visibly fired: **275** rows matched `name_stage='exhausted'`, with **275/275** identifiers and **275/275** stage values. This is not a plausible zero produced by a misspelled property.

| Signal | Positive control over its owning label | Coverage inside the 275-row population |
|---|---:|---:|
| `StandardName.id` | 4670/4670 | 275/275 |
| `StandardName.name_stage` | 4670/4670 | 275/275 |
| `StandardName.reviewer_score_name` | 3862/4670 | 271/275 |
| `StandardName.chain_length` | 3099/4670 | 166/275 |
| `StandardName.refine_stop_reason` | 225/4670 | 213/275 |
| `StandardNameReview.score` | 26662/26662 | queried through the schema-authored `HAS_REVIEW` edge |
| `VocabGap.id/category/triage` | 545/545, 545/545, 541/545 | source-linked query; 7 immutable `VocabGapEvidence` nodes exist globally |

Null coverage is retained as evidence, not coalesced into a claim that the fields are universally populated. The disposition query uses the declared `StandardName.id`, `StandardName.reviewer_score_name`, `StandardName.chain_length`, `StandardName.refine_stop_reason`, `StandardNameReview.review_axis/score`, and `VocabGap.triage/editorial_disposition` properties and the schema-authored edge directions.

## Reconstruction and reproducibility

Committed query SHA-256: `2dad8167557938596301300f595a9ddeab04df287123c6de5d09b5bf8da3fb77`

Ordered-row SHA-256, pass 1: `d233528016c6f4925c5d24f0c2e5219a828d378e36881704094d7b43813b7b3f`  
Ordered-row SHA-256, pass 2: `d233528016c6f4925c5d24f0c2e5219a828d378e36881704094d7b43813b7b3f`

The hashes are over canonical JSON of every returned row (`sort_keys=True`, compact separators), after the committed `ORDER BY`. Two consecutive live passes returned the same **275** ordered rows and the identical hash. The query itself is committed below so the population, partition, ordering, and row hash can be reconstructed.

```cypher
MATCH (sn:StandardName)
WHERE sn.name_stage = 'exhausted'
CALL {
  WITH sn
  OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(rv:StandardNameReview)
  WHERE rv.review_axis = 'name'
  RETURN max(rv.score) AS historical_max_score, count(rv) AS review_count
}
CALL {
  WITH sn
  OPTIONAL MATCH (src:StandardNameSource)-[:PRODUCED_NAME]->(sn)
  OPTIONAL MATCH (src)-[:HAS_STANDARD_NAME_VOCAB_GAP]->(vg:VocabGap)
  WHERE vg.triage = 'genuine' OR vg.editorial_disposition = 'add'
  RETURN collect(DISTINCT vg.id) AS genuine_gap_ids
}
WITH sn,
     historical_max_score,
     review_count,
     genuine_gap_ids,
     CASE
       WHEN historical_max_score IS NULL THEN sn.reviewer_score_name
       WHEN sn.reviewer_score_name IS NULL THEN historical_max_score
       WHEN historical_max_score > sn.reviewer_score_name THEN historical_max_score
       ELSE sn.reviewer_score_name
     END AS best_reviewer_score,
     coalesce(sn.chain_length, 0) AS chain_length,
     sn.refine_stop_reason AS stop_reason
WITH sn,
     historical_max_score,
     review_count,
     genuine_gap_ids,
     best_reviewer_score,
     chain_length,
     stop_reason,
     CASE
       WHEN stop_reason = 'vocabulary_gap' OR size(genuine_gap_ids) > 0
         THEN 'vocabulary-gap'
       WHEN (stop_reason IS NULL OR stop_reason = 'attempts_exhausted')
         AND best_reviewer_score >= 0.85
         THEN 'rescore-candidate'
       ELSE 'stay-parked'
     END AS disposition
RETURN sn.id AS id,
       disposition,
       stop_reason,
       sn.reviewer_score_name AS current_reviewer_score,
       historical_max_score,
       best_reviewer_score,
       review_count,
       chain_length,
       genuine_gap_ids
ORDER BY CASE disposition
           WHEN 'rescore-candidate' THEN 0
           WHEN 'vocabulary-gap' THEN 1
           ELSE 2
         END,
         best_reviewer_score DESC,
         chain_length ASC,
         sn.id ASC
```

## Complete ranked population

### rescore-candidate (32)

| Rank | Identity | Stop reason | Current score | Historical max | Best score | Reviews | Chain | Vocabulary-gap evidence |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `fast_particle_pressure` | none recorded | 0.825 | 1 | 1 | 3 | 0 | none |
| 2 | `ion_diamagnetic_momentum_convection_velocity` | none recorded | 0.7125 | 1 | 1 | 3 | 0 | none |
| 3 | `magnetic_field_magnitude` | none recorded | 0.65 | 1 | 1 | 9 | 0 | none |
| 4 | `toroidal_momentum_convection_velocity` | none recorded | 0.65 | 1 | 1 | 10 | 0 | none |
| 5 | `toroidal_neutral_state_momentum_source` | none recorded | 0.8 | 1 | 1 | 5 | 1 | none |
| 6 | `trapped_thermal_ion_charge_state_torque_density_due_to_collisions` | `attempts_exhausted` | 0.7 | 1 | 1 | 12 | 1 | none |
| 7 | `electron_temperature_at_separatrix` | `attempts_exhausted` | 0.675 | 1 | 1 | 3 | 3 | none |
| 8 | `launched_power_of_ion_cyclotron_heating_antenna` | `attempts_exhausted` | 0.65 | 1 | 1 | 3 | 3 | none |
| 9 | `deuterium_tritium_flux` | none recorded | 0.8 | 0.9875 | 0.9875 | 10 | 0 | none |
| 10 | `particle_convection_velocity` | none recorded | 0.55 | 0.9875 | 0.9875 | 7 | 0 | none |
| 11 | `normalized_toroidal_flux_coordinate_at_ece_channel_emission_position` | `attempts_exhausted` | 0.825 | 0.9875 | 0.9875 | 3 | 3 | none |
| 12 | `wave_curvature_of_wave_beam` | none recorded | 0.7875 | 0.975 | 0.975 | 3 | 0 | none |
| 13 | `diamagnetic_current_density` | `attempts_exhausted` | 0.45 | 0.9625 | 0.9625 | 3 | 0 | none |
| 14 | `radial_coordinate_of_launching_position` | none recorded | 0.7 | 0.95 | 0.95 | 5 | 0 | none |
| 15 | `ion_charge_state_torque_density` | `attempts_exhausted` | 0.5875 | 0.95 | 0.95 | 12 | 1 | none |
| 16 | `parallel_momentum_flux` | none recorded | 0.66875 | 0.9375 | 0.9375 | 4 | 0 | none |
| 17 | `radial_coordinate_of_flux_surface` | none recorded | 0.7 | 0.9375 | 0.9375 | 6 | 0 | none |
| 18 | `toroidal_current_density_due_to_distribution_function_driven` | none recorded | 0.5875 | 0.9375 | 0.9375 | 3 | 0 | none |
| 19 | `total_ion_energy_diffusion_coefficient` | none recorded | 0.75 | 0.9375 | 0.9375 | 3 | 0 | none |
| 20 | `flux_surface_normal_neutral_energy_diffusion_coefficient` | `attempts_exhausted` | 0.775 | 0.925 | 0.925 | 3 | 3 | none |
| 21 | `gap_at_plasma_boundary` | `attempts_exhausted` | 0.65 | 0.9125 | 0.9125 | 6 | 0 | none |
| 22 | `toroidal_ion_momentum_diffusion_coefficient` | `attempts_exhausted` | 0.825 | 0.9125 | 0.9125 | 3 | 0 | none |
| 23 | `ion_species_particle_flux_at_wall_due_to_surface_emission` | `attempts_exhausted` | 0.725 | 0.9125 | 0.9125 | 3 | 3 | none |
| 24 | `flux_surface_normal_momentum_convection_velocity` | `attempts_exhausted` | 0.8125 | 0.875 | 0.875 | 3 | 3 | none |
| 25 | `ion_state_average_charge_number` | none recorded | 0.84375 | 0.8625 | 0.8625 | 8 | 0 | none |
| 26 | `gradient_of_normalized_pressure_at_flux_surface` | none recorded | 0.7 | 0.8625 | 0.8625 | 3 | 3 | none |
| 27 | `plasma_power_at_wall` | none recorded | 0.6875 | 0.85 | 0.85 | 4 | 0 | none |
| 28 | `wave_curvature_of_beam_tracing_beam` | none recorded | 0.6875 | 0.85 | 0.85 | 3 | 0 | none |
| 29 | `fast_ion_charge_state_torque_due_to_collisions` | none recorded | 0.79375 | 0.85 | 0.85 | 2 | 1 | none |
| 30 | `toroidal_angle_of_secondary_x_point` | none recorded | 0.825 | 0.85 | 0.85 | 3 | 1 | none |
| 31 | `first_local_tangential_back_surface_radius_of_optical_element` | `attempts_exhausted` | 0.675 | 0.85 | 0.85 | 3 | 3 | none |
| 32 | `total_launched_power_due_to_ion_cyclotron_heating` | `attempts_exhausted` | 0.75 | 0.85 | 0.85 | 3 | 3 | none |

### vocabulary-gap (4)

| Rank | Identity | Stop reason | Current score | Historical max | Best score | Reviews | Chain | Vocabulary-gap evidence |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `extent_of_detector_pixel` | `vocabulary_gap` | 0.775 | 0.8375 | 0.8375 | 3 | 0 | none |
| 2 | `height_of_optical_element` | `vocabulary_gap` | 0.65 | 0.75 | 0.75 | 3 | 0 | none |
| 3 | `width_of_optical_element` | `vocabulary_gap` | 0.525 | 0.5875 | 0.5875 | 3 | 0 | none |
| 4 | `surface_thickness_of_breeder_blanket_module` | `vocabulary_gap` | 0.5875 | 0.5875 | 0.5875 | 2 | 2 | none |

### stay-parked (239)

| Rank | Identity | Stop reason | Current score | Historical max | Best score | Reviews | Chain | Vocabulary-gap evidence |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `count_at_detector_pixel` | none recorded | — | — | — | 0 | 0 | none |
| 2 | `launched_power_of_electron_cyclotron_launcher` | none recorded | — | — | — | 0 | 0 | none |
| 3 | `length_of_antenna_strap` | none recorded | — | — | — | 0 | 0 | none |
| 4 | `voltage_of_neutron_detector` | none recorded | — | — | — | 0 | 0 | none |
| 5 | `current_of_poloidal_field_coil` | `grammar_invalid` | 0.7625 | 1 | 1 | 3 | 0 | none |
| 6 | `lower_bound_energy_of_neutron_detector` | `grammar_invalid` | 0.825 | 1 | 1 | 3 | 0 | none |
| 7 | `parallel_neutral_momentum_diffusivity` | `grammar_invalid` | 0.8375 | 1 | 1 | 3 | 0 | none |
| 8 | `parallel_normalized_particle_perturbed_pressure` | `grammar_invalid` | 0.8125 | 1 | 1 | 3 | 0 | none |
| 9 | `parallel_wave_electric_field_amplitude` | `grammar_invalid` | 0.825 | 1 | 1 | 3 | 0 | none |
| 10 | `perpendicular_normalized_gyrocenter_perturbed_pressure` | `grammar_invalid` | 0.7875 | 1 | 1 | 3 | 0 | none |
| 11 | `poloidal_ion_charge_state_momentum_diffusivity` | `grammar_invalid` | 0.8125 | 1 | 1 | 3 | 0 | none |
| 12 | `poloidal_neutral_momentum_diffusivity` | `grammar_invalid` | 0.8375 | 1 | 1 | 3 | 0 | none |
| 13 | `radial_coordinate_of_pellet_path` | `successor_collision` | 0.8125 | 1 | 1 | 5 | 0 | none |
| 14 | `thickness_of_passive_loop` | `grammar_invalid` | 0.8125 | 1 | 1 | 3 | 0 | none |
| 15 | `time_derivative_of_electron_density` | `grammar_invalid` | 0.65 | 1 | 1 | 5 | 0 | none |
| 16 | `tritium_tritium_neutron_source_rate_due_to_thermal_fusion` | `grammar_invalid` | 0.825 | 1 | 1 | 3 | 0 | none |
| 17 | `tungsten_density` | `successor_collision` | 0.6625 | 1 | 1 | 3 | 0 | none |
| 18 | `first_local_tangential_width_of_reflectometer_antenna` | `successor_collision` | 0.5625 | 1 | 1 | 3 | 1 | none |
| 19 | `spectral_width_of_filter` | `grammar_invalid` | 0.8 | 0.9875 | 0.9875 | 3 | 0 | none |
| 20 | `normalized_poloidal_magnetic_flux_at_pedestal_top` | `grammar_invalid` | 0.7125 | 0.975 | 0.975 | 3 | 0 | none |
| 21 | `poloidal_turn_count` | `successor_collision` | 0.6125 | 0.975 | 0.975 | 5 | 0 | none |
| 22 | `toroidal_width_of_antenna_strap` | `grammar_invalid` | 0.5875 | 0.975 | 0.975 | 3 | 0 | none |
| 23 | `tilt_angle_of_poloidal_field_coil` | `grammar_invalid` | 0.8375 | 0.9625 | 0.9625 | 3 | 1 | none |
| 24 | `core_density_of_pellet` | `grammar_invalid` | 0.7875 | 0.95 | 0.95 | 3 | 0 | none |
| 25 | `fast_electron_energy` | `grammar_invalid` | 0.8375 | 0.95 | 0.95 | 3 | 0 | none |
| 26 | `ion_charge_state_diamagnetic_velocity_due_to_diamagnetic_drift` | `successor_collision` | 0.6875 | 0.95 | 0.95 | 3 | 0 | none |
| 27 | `neutral_diffusivity` | `successor_collision` | 0.7125 | 0.95 | 0.95 | 3 | 0 | none |
| 28 | `parallel_neutral_species_energy_convection_velocity` | `grammar_invalid` | 0.825 | 0.95 | 0.95 | 3 | 0 | none |
| 29 | `radiative_temperature` | `grammar_invalid` | 0.7625 | 0.95 | 0.95 | 3 | 0 | none |
| 30 | `thermal_energy_confinement_time` | `grammar_invalid` | 0.775 | 0.95 | 0.95 | 3 | 1 | none |
| 31 | `total_ion_momentum_diffusivity` | `successor_collision` | 0.4875 | 0.9375 | 0.9375 | 3 | 1 | none |
| 32 | `neutral_pressure` | `successor_collision` | 0.825 | 0.925 | 0.925 | 3 | 0 | none |
| 33 | `flux_surface_averaged_bulk_electron_temperature_at_last_closed_flux_surface` | `grammar_invalid` | 0.8 | 0.925 | 0.925 | 3 | 1 | none |
| 34 | `heat_power_of_divertor` | `successor_collision` | 0.6875 | 0.925 | 0.925 | 3 | 1 | none |
| 35 | `total_particle_flux` | `grammar_invalid` | 0.8 | 0.925 | 0.925 | 3 | 1 | none |
| 36 | `vertical_outline_of_wall` | `successor_collision` | 0.65 | 0.925 | 0.925 | 3 | 1 | none |
| 37 | `atomic_fraction_of_neutron_detector_converter` | `successor_collision` | 0.8375 | 0.9125 | 0.9125 | 3 | 0 | none |
| 38 | `pressure_of_lower_hybrid_antenna` | `grammar_invalid` | 0.8375 | 0.9125 | 0.9125 | 3 | 0 | none |
| 39 | `width_of_antenna_strap` | `successor_collision` | 0.8375 | 0.9125 | 0.9125 | 3 | 0 | none |
| 40 | `parallel_incident_heat_flux_at_divertor_target` | `grammar_invalid` | 0.8125 | 0.9125 | 0.9125 | 3 | 1 | none |
| 41 | `surface_temperature_of_plasma_facing_component` | `grammar_invalid` | 0.8125 | 0.9125 | 0.9125 | 3 | 1 | none |
| 42 | `total_suprathermal_electron_power_density_due_to_collisions` | `grammar_invalid` | 0.7625 | 0.9125 | 0.9125 | 3 | 1 | none |
| 43 | `reference_wavelength_of_filter_window` | `successor_collision` | 0.6625 | 0.9 | 0.9 | 3 | 1 | none |
| 44 | `parallel_flux_surface_averaged_electric_field_at_separatrix` | `grammar_invalid` | 0.675 | 0.875 | 0.875 | 3 | 2 | none |
| 45 | `power_over_core_region_due_to_impurity_radiation` | `grammar_invalid` | 0.5875 | 0.875 | 0.875 | 3 | 2 | none |
| 46 | `total_power_due_to_fusion` | `successor_collision` | 0.84375 | 0.8625 | 0.8625 | 2 | 0 | none |
| 47 | `beam_cross_sectional_area_of_aperture` | `successor_collision` | 0.7125 | 0.8625 | 0.8625 | 3 | 1 | none |
| 48 | `vertical_total_ion_momentum_diffusivity` | `grammar_invalid` | 0.7 | 0.8625 | 0.8625 | 3 | 1 | none |
| 49 | `magnetic_shear_at_pedestal_top` | `successor_collision` | 0.84375 | 0.85 | 0.85 | 2 | 0 | none |
| 50 | `parallel_per_toroidal_mode_current_density_due_to_wave_driven_current_drive` | `grammar_invalid` | 0.75 | 0.85 | 0.85 | 3 | 0 | none |
| 51 | `radial_ion_state_energy_flux` | `successor_collision` | 0.825 | 0.85 | 0.85 | 3 | 0 | none |
| 52 | `total_plasma_energy` | `successor_collision` | 0.83125 | 0.85 | 0.85 | 2 | 0 | none |
| 53 | `voltage_of_temperature_sensor` | `grammar_invalid` | 0.75 | 0.85 | 0.85 | 3 | 0 | none |
| 54 | `measured_voltage_of_spectrometer_channel` | `successor_collision` | 0.81875 | 0.85 | 0.85 | 2 | 2 | none |
| 55 | `ion_pressure` | `successor_collision` | 0.79375 | 0.8375 | 0.8375 | 2 | 0 | none |
| 56 | `parallel_per_toroidal_mode_electric_field` | `grammar_invalid` | 0.45 | 0.8375 | 0.8375 | 3 | 0 | none |
| 57 | `plasma_velocity_due_to_diamagnetic_drift` | `successor_collision` | 0.75 | 0.8375 | 0.8375 | 3 | 0 | none |
| 58 | `radial_derivative_of_elongation_of_flux_surface` | `attempts_exhausted` | 0.775 | 0.8375 | 0.8375 | 3 | 0 | none |
| 59 | `radial_ion_energy_diffusion_coefficient` | `successor_collision` | 0.8125 | 0.8375 | 0.8375 | 2 | 0 | none |
| 60 | `silane_prefill_count` | `grammar_invalid` | 0.775 | 0.8375 | 0.8375 | 3 | 0 | none |
| 61 | `toroidal_trapped_thermal_electron_torque_density_due_to_coulomb_collisions_with_electrons` | `successor_collision` | 0.79375 | 0.8375 | 0.8375 | 2 | 0 | none |
| 62 | `total_diamagnetic_current_density` | none recorded | 0.81875 | 0.8375 | 0.8375 | 2 | 0 | none |
| 63 | `total_ion_pressure` | `successor_collision` | 0.8125 | 0.8375 | 0.8375 | 2 | 0 | none |
| 64 | `net_power` | `grammar_invalid` | 0.7 | 0.8375 | 0.8375 | 3 | 1 | none |
| 65 | `parallel_bulk_ion_velocity` | `grammar_invalid` | 0.6625 | 0.8375 | 0.8375 | 3 | 1 | none |
| 66 | `parallel_thermal_neutral_energy_diffusivity` | `successor_collision` | 0.725 | 0.8375 | 0.8375 | 3 | 2 | none |
| 67 | `power_due_to_fusion_reactions` | none recorded | 0.8125 | 0.8375 | 0.8375 | 6 | 3 | none |
| 68 | `radial_offset_of_lower_hybrid_antenna` | `attempts_exhausted` | 0.7625 | 0.8375 | 0.8375 | 3 | 3 | none |
| 69 | `per_toroidal_mode_current_due_to_wave_driven_current_drive` | `grammar_invalid` | 0.825 | 0.825 | 0.825 | 2 | 0 | none |
| 70 | `total_ion_energy_diffusivity` | none recorded | 0.81875 | 0.825 | 0.825 | 2 | 0 | none |
| 71 | `total_neutron_power` | `grammar_invalid` | 0.75 | 0.825 | 0.825 | 3 | 0 | none |
| 72 | `neutral_internal_state_torque_density` | none recorded | 0.8 | 0.825 | 0.825 | 2 | 1 | none |
| 73 | `peak_incident_heat_flux_at_limiter` | none recorded | 0.5625 | 0.825 | 0.825 | 6 | 1 | none |
| 74 | `cumulative_ethylene_count_due_to_gas_injection` | `grammar_invalid` | 0.65 | 0.825 | 0.825 | 3 | 2 | none |
| 75 | `electron_heat_diffusivity` | `successor_collision` | 0.80625 | 0.825 | 0.825 | 2 | 2 | none |
| 76 | `energy_flux_at_control_surface` | `attempts_exhausted` | 0.5625 | 0.825 | 0.825 | 3 | 3 | none |
| 77 | `tendency_of_runaway_electron_density` | `attempts_exhausted` | 0.7125 | 0.825 | 0.825 | 3 | 3 | none |
| 78 | `ion_average_temperature_at_plasma_boundary` | none recorded | 0.80625 | 0.8125 | 0.8125 | 2 | 0 | none |
| 79 | `neutral_beam_mass` | `successor_collision` | 0.775 | 0.8125 | 0.8125 | 2 | 0 | none |
| 80 | `normalized_count_at_detector_pixel` | `successor_collision` | 0.79375 | 0.8125 | 0.8125 | 2 | 0 | none |
| 81 | `parallel_normalized_perturbed_magnetic_field` | `grammar_invalid` | 0.8 | 0.8125 | 0.8125 | 2 | 0 | none |
| 82 | `poloidal_electron_energy_diffusion_coefficient` | `successor_collision` | 0.7 | 0.8125 | 0.8125 | 3 | 0 | none |
| 83 | `total_ion_power_density` | `successor_collision` | 0.7875 | 0.8125 | 0.8125 | 2 | 0 | none |
| 84 | `trapped_fast_particle_power_density_due_to_collisions` | `successor_collision` | 0.775 | 0.8125 | 0.8125 | 2 | 0 | none |
| 85 | `voltage_of_spectrometer` | `grammar_invalid` | 0.6 | 0.8125 | 0.8125 | 3 | 0 | none |
| 86 | `ratio_of_plasma_vorticity_to_major_radius` | none recorded | 0.8125 | 0.8125 | 0.8125 | 1 | 1 | none |
| 87 | `poloidal_effective_ion_diffusion_coefficient` | `successor_collision` | 0.8125 | 0.8125 | 0.8125 | 2 | 2 | none |
| 88 | `neutral_species_kinetic_energy_flux_at_wall_due_to_surface_emission` | `attempts_exhausted` | 0.65 | 0.8125 | 0.8125 | 3 | 3 | none |
| 89 | `total_incident_thermal_power` | `attempts_exhausted` | 0.675 | 0.8125 | 0.8125 | 3 | 3 | none |
| 90 | `major_radius` | `successor_collision` | 0.775 | 0.8 | 0.8 | 2 | 0 | none |
| 91 | `radial_ion_momentum` | `successor_collision` | 0.575 | 0.8 | 0.8 | 10 | 0 | none |
| 92 | `tritium_prefill_count` | `grammar_invalid` | 0.675 | 0.8 | 0.8 | 3 | 0 | none |
| 93 | `cross_field_wave_vector_magnitude` | `successor_collision` | 0.7375 | 0.8 | 0.8 | 2 | 1 | none |
| 94 | `neutral_internal_state_diffusivity` | none recorded | 0.78125 | 0.8 | 0.8 | 2 | 1 | none |
| 95 | `spun_wavelength_of_fiber_optic_current_sensor` | none recorded | 0.7125 | 0.8 | 0.8 | 3 | 1 | none |
| 96 | `toroidal_co_passing_fast_electron_torque_density_due_to_collisional_transport` | `grammar_invalid` | 0.7875 | 0.8 | 0.8 | 3 | 1 | none |
| 97 | `toroidal_trapped_fast_electron_torque_density_due_to_collisional_transport` | `successor_collision` | 0.7 | 0.8 | 0.8 | 3 | 1 | none |
| 98 | `vertical_total_force_of_poloidal_field_coil` | none recorded | 0.5875 | 0.8 | 0.8 | 3 | 1 | none |
| 99 | `absorbed_coolant_power_of_plant_component_port` | `attempts_exhausted` | 0.55 | 0.8 | 0.8 | 3 | 3 | none |
| 100 | `net_plasma_power_density` | `attempts_exhausted` | 0.725 | 0.8 | 0.8 | 3 | 3 | none |
| 101 | `power_over_scrape_off_layer_due_to_radiation` | `attempts_exhausted` | 0.5875 | 0.8 | 0.8 | 3 | 3 | none |
| 102 | `radiated_power_density` | `successor_collision` | 0.78125 | 0.7875 | 0.7875 | 2 | 0 | none |
| 103 | `toroidal_angle_of_along_pellet_path` | none recorded | 0.725 | 0.7875 | 0.7875 | 3 | 0 | none |
| 104 | `neutral_internal_state_momentum_flux` | none recorded | 0.75 | 0.7875 | 0.7875 | 2 | 1 | none |
| 105 | `neutron_rate_of_detector` | `attempts_exhausted` | 0.775 | 0.7875 | 0.7875 | 2 | 1 | none |
| 106 | `nuclear_power_density_at_midplane` | `grammar_invalid` | 0.65 | 0.7875 | 0.7875 | 3 | 1 | none |
| 107 | `ratio_of_magnetic_field_to_current_of_poloidal_field_coil` | none recorded | 0.675 | 0.7875 | 0.7875 | 3 | 1 | none |
| 108 | `electron_particle_diffusion_coefficient` | `successor_collision` | 0.6875 | 0.7875 | 0.7875 | 3 | 2 | none |
| 109 | `normal_particle_flux_at_wall_due_to_recombination` | `successor_collision` | 0.725 | 0.7875 | 0.7875 | 3 | 2 | none |
| 110 | `perpendicular_normalized_perturbed_pressure` | `grammar_invalid` | 0.725 | 0.775 | 0.775 | 3 | 0 | none |
| 111 | `safety_factor_at_pedestal` | `successor_collision` | 0.575 | 0.775 | 0.775 | 5 | 0 | none |
| 112 | `thermal_electron_decay_length_over_scrape_off_layer` | `grammar_invalid` | 0.5625 | 0.775 | 0.775 | 3 | 0 | none |
| 113 | `toroidal_momentum_flux` | `attempts_exhausted` | 0.75 | 0.775 | 0.775 | 2 | 0 | none |
| 114 | `toroidal_particle_current` | none recorded | 0.75625 | 0.775 | 0.775 | 2 | 0 | none |
| 115 | `total_power_of_neutral_beam_injector` | `grammar_invalid` | 0.6875 | 0.775 | 0.775 | 3 | 0 | none |
| 116 | `tilt_angle_of_antenna_strap` | none recorded | 0.7375 | 0.775 | 0.775 | 2 | 1 | none |
| 117 | `toroidal_cumulative_inside_flux_surface_total_plasma_momentum_at_separatrix` | `grammar_invalid` | 0.6625 | 0.775 | 0.775 | 3 | 1 | none |
| 118 | `electron_particle_diffusivity` | `successor_collision` | 0.7625 | 0.775 | 0.775 | 2 | 2 | none |
| 119 | `energy_diffusion_coefficient_due_to_diffusion` | `successor_collision` | 0.725 | 0.775 | 0.775 | 3 | 2 | none |
| 120 | `molecular_gas_count_due_to_pellet_injection` | `attempts_exhausted` | 0.625 | 0.775 | 0.775 | 3 | 3 | none |
| 121 | `ion_energy_convection_velocity` | none recorded | 0.6125 | 0.7625 | 0.7625 | 3 | 0 | none |
| 122 | `ion_state_momentum_diffusivity` | `successor_collision` | 0.525 | 0.7625 | 0.7625 | 3 | 0 | none |
| 123 | `mutual_inductance` | none recorded | 0.68125 | 0.7625 | 0.7625 | 7 | 0 | none |
| 124 | `normalized_perpendicular_gyroaveraged_perturbed_energy` | none recorded | 0.75625 | 0.7625 | 0.7625 | 2 | 0 | none |
| 125 | `voltage_amplitude` | `grammar_invalid` | 0.625 | 0.7625 | 0.7625 | 3 | 0 | none |
| 126 | `distance_of_beam_tracing_ray` | `successor_collision` | 0.70625 | 0.7625 | 0.7625 | 2 | 1 | none |
| 127 | `flux_surface_normal_momentum_diffusion_coefficient` | `grammar_invalid` | 0.6375 | 0.7625 | 0.7625 | 3 | 2 | none |
| 128 | `net_forward_power_of_wave_beam` | `attempts_exhausted` | 0.5625 | 0.7625 | 0.7625 | 3 | 3 | none |
| 129 | `plasma_electrostatic_potential_at_outboard_midplane` | `attempts_exhausted` | 0.5375 | 0.7625 | 0.7625 | 3 | 3 | none |
| 130 | `critical_electric_field` | `successor_collision` | 0.73125 | 0.75 | 0.75 | 2 | 0 | none |
| 131 | `electron_convection_velocity` | none recorded | 0.7125 | 0.75 | 0.75 | 4 | 0 | none |
| 132 | `mass_density` | `successor_collision` | 0.7125 | 0.75 | 0.75 | 2 | 0 | none |
| 133 | `size_of_camera` | `grammar_invalid` | 0.725 | 0.75 | 0.75 | 2 | 0 | none |
| 134 | `toroidal_total_plasma_angular_momentum` | none recorded | 0.6625 | 0.75 | 0.75 | 7 | 0 | none |
| 135 | `vertical_magnetic_field_at_wall` | `grammar_invalid` | 0.7375 | 0.75 | 0.75 | 2 | 0 | none |
| 136 | `total_deuterium_tritium_neutron_source_rate_due_to_beam_beam_fusion` | `successor_collision` | 0.5125 | 0.75 | 0.75 | 3 | 1 | none |
| 137 | `flux_surface_normal_contravariant_flux_surface_averaged_metric` | none recorded | 0.75 | 0.75 | 0.75 | 2 | 2 | none |
| 138 | `front_surface_area_of_langmuir_probe` | `attempts_exhausted` | 0.5875 | 0.75 | 0.75 | 3 | 3 | none |
| 139 | `wave_critical_ordinary_mode_frequency` | `attempts_exhausted` | 0.6625 | 0.75 | 0.75 | 3 | 3 | none |
| 140 | `neutral_power_at_wall_due_to_recombination` | `grammar_invalid` | 0.7125 | 0.7375 | 0.7375 | 3 | 0 | none |
| 141 | `normalized_perturbed_pressure` | `grammar_invalid` | 0.6375 | 0.7375 | 0.7375 | 3 | 0 | none |
| 142 | `parallel_normalized_perturbed_magnetic_field_amplitude` | `grammar_invalid` | 0.625 | 0.7375 | 0.7375 | 3 | 0 | none |
| 143 | `particle_count` | `successor_collision` | 0.55 | 0.7375 | 0.7375 | 3 | 0 | none |
| 144 | `ion_temperature_at_outboard_midplane_separatrix` | `attempts_exhausted` | 0.6625 | 0.7375 | 0.7375 | 3 | 3 | none |
| 145 | `critical_momentum_due_to_avalanche` | `grammar_invalid` | 0.675 | 0.725 | 0.725 | 3 | 0 | none |
| 146 | `ion_diamagnetic_momentum_damping_rate` | none recorded | 0.6125 | 0.725 | 0.725 | 11 | 0 | none |
| 147 | `power_of_neutral_beam_injector` | `grammar_invalid` | 0.6375 | 0.725 | 0.725 | 3 | 0 | none |
| 148 | `radial_ion_energy_convection_velocity` | `successor_collision` | 0.625 | 0.725 | 0.725 | 3 | 0 | none |
| 149 | `toroidal_current_density` | `successor_collision` | 0.71875 | 0.725 | 0.725 | 2 | 0 | none |
| 150 | `electron_temperature_at_first_wall` | `grammar_invalid` | 0.6625 | 0.725 | 0.725 | 3 | 1 | none |
| 151 | `total_thermal_radiative_power_of_divertor_target` | `successor_collision` | 0.45 | 0.725 | 0.725 | 3 | 1 | none |
| 152 | `fast_electron_source_rate_due_to_hot_tail` | `successor_collision` | 0.5625 | 0.7125 | 0.7125 | 3 | 0 | none |
| 153 | `ion_convection_velocity` | none recorded | 0.6875 | 0.7125 | 0.7125 | 3 | 0 | none |
| 154 | `ion_diffusivity` | `successor_collision` | 0.7 | 0.7125 | 0.7125 | 2 | 0 | none |
| 155 | `parallel_normalized_perturbed_vector_potential_amplitude` | `grammar_invalid` | 0.575 | 0.7125 | 0.7125 | 3 | 0 | none |
| 156 | `thermal_energy_of_plant_component_port` | `grammar_invalid` | 0.7125 | 0.7125 | 0.7125 | 3 | 0 | none |
| 157 | `plasma_breakdown_time` | none recorded | 0.65625 | 0.7125 | 0.7125 | 2 | 3 | none |
| 158 | `electron_temperature_at_midplane` | `successor_collision` | 0.6875 | 0.7 | 0.7 | 3 | 0 | none |
| 159 | `magnetic_shear_at_sawtooth_inversion_radius` | `grammar_invalid` | 0.5875 | 0.7 | 0.7 | 3 | 0 | none |
| 160 | `momentum_coefficient_due_to_diamagnetic_drift` | none recorded | 0.5625 | 0.7 | 0.7 | 3 | 0 | none |
| 161 | `vertical_coordinate_of_closest_wall_point` | `successor_collision` | 0.6375 | 0.7 | 0.7 | 3 | 0 | none |
| 162 | `vertical_current_density` | `successor_collision` | 0.6625 | 0.7 | 0.7 | 3 | 0 | none |
| 163 | `fast_electron_source_rate` | `successor_collision` | 0.6 | 0.6875 | 0.6875 | 3 | 0 | none |
| 164 | `co_passing_fast_electron_kinetic_power_density_due_to_collisions` | `attempts_exhausted` | 0.4875 | 0.6875 | 0.6875 | 3 | 2 | none |
| 165 | `peak_wave_current_of_antenna_strap_amplitude` | `successor_collision` | 0.6875 | 0.6875 | 0.6875 | 3 | 2 | none |
| 166 | `plasma_electrostatic_potential_at_wall` | `attempts_exhausted` | 0.6125 | 0.6875 | 0.6875 | 3 | 3 | none |
| 167 | `vertical_coordinate_of_plasma_filament` | `attempts_exhausted` | 0.66875 | 0.6875 | 0.6875 | 2 | 3 | none |
| 168 | `poloidal_convection_velocity` | none recorded | 0.65 | 0.675 | 0.675 | 2 | 0 | none |
| 169 | `radial_derivative_of_toroidal_particle_velocity` | none recorded | 0.4625 | 0.675 | 0.675 | 5 | 0 | none |
| 170 | `surface_area_of_optical_element` | `grammar_invalid` | 0.6 | 0.675 | 0.675 | 3 | 0 | none |
| 171 | `vertical_coordinate_of_coil_conductor` | `attempts_exhausted` | 0.5375 | 0.675 | 0.675 | 3 | 0 | none |
| 172 | `ratio_of_critical_alpha_parameter_to_alpha_parameter_at_pedestal` | none recorded | 0.675 | 0.675 | 0.675 | 2 | 2 | none |
| 173 | `lithium_volume_of_breeder_blanket` | `attempts_exhausted` | 0.6375 | 0.675 | 0.675 | 3 | 3 | none |
| 174 | `fast_electron_source_rate_due_to_compton_scattering` | `successor_collision` | 0.5625 | 0.6625 | 0.6625 | 3 | 0 | none |
| 175 | `left_hand_circularly_polarized_electric_field` | `successor_collision` | 0.6125 | 0.6625 | 0.6625 | 3 | 0 | none |
| 176 | `mass_of_wall_material` | `successor_collision` | 0.64375 | 0.6625 | 0.6625 | 2 | 0 | none |
| 177 | `radial_centroid_of_electron_cyclotron_launcher_mirror` | `successor_collision` | 0.4375 | 0.6625 | 0.6625 | 3 | 0 | none |
| 178 | `radial_neutral_state_momentum_convection_velocity` | `successor_collision` | 0.65 | 0.6625 | 0.6625 | 2 | 0 | none |
| 179 | `source_rate_due_to_injection` | `successor_collision` | 0.64375 | 0.6625 | 0.6625 | 2 | 0 | none |
| 180 | `toroidal_volume_integrated_fast_electron_torque_density_due_to_collisions` | `grammar_invalid` | 0.5625 | 0.6625 | 0.6625 | 3 | 1 | none |
| 181 | `total_absorbed_power_of_plant_system` | `successor_collision` | 0.575 | 0.6625 | 0.6625 | 3 | 1 | none |
| 182 | `net_plasma_current_due_to_ohmic_current_drive` | `successor_collision` | 0.5375 | 0.6625 | 0.6625 | 3 | 2 | none |
| 183 | `volume_averaged_electron_number_density_over_scrape_off_layer` | `grammar_invalid` | 0.5875 | 0.6625 | 0.6625 | 3 | 2 | none |
| 184 | `length_of_passive_structure` | `grammar_invalid` | 0.625 | 0.65 | 0.65 | 3 | 0 | none |
| 185 | `maximum_power_at_inner_divertor_target` | `grammar_invalid` | 0.5875 | 0.65 | 0.65 | 3 | 0 | none |
| 186 | `normalized_neutron_flux` | none recorded | 0.575 | 0.65 | 0.65 | 3 | 0 | none |
| 187 | `outer_atomic_count_of_pellet` | `successor_collision` | 0.6125 | 0.65 | 0.65 | 3 | 0 | none |
| 188 | `poloidal_current_density_due_to_viscosity` | `successor_collision` | 0.5125 | 0.65 | 0.65 | 3 | 0 | none |
| 189 | `power_of_lower_hybrid_antenna` | `successor_collision` | 0.6125 | 0.65 | 0.65 | 4 | 0 | none |
| 190 | `voltage_of_diagnostic_antenna` | `grammar_invalid` | 0.625 | 0.65 | 0.65 | 3 | 0 | none |
| 191 | `wavelength` | `attempts_exhausted` | 0.425 | 0.65 | 0.65 | 3 | 0 | none |
| 192 | `incident_energy_flux_at_wall_due_to_radiation` | `grammar_invalid` | 0.55 | 0.65 | 0.65 | 3 | 1 | none |
| 193 | `vertical_position_of_grating` | `grammar_invalid` | 0.5875 | 0.65 | 0.65 | 3 | 2 | none |
| 194 | `non_axisymmetric_current_of_conductor` | `attempts_exhausted` | 0.5125 | 0.65 | 0.65 | 3 | 3 | none |
| 195 | `normal_distance_of_antenna_strap` | `attempts_exhausted` | 0.63125 | 0.65 | 0.65 | 2 | 3 | none |
| 196 | `angle_of_plasma_boundary_gap` | none recorded | 0.5875 | 0.6375 | 0.6375 | 3 | 0 | none |
| 197 | `energy_flux_at_wall_due_to_eddy_current` | `grammar_invalid` | 0.6 | 0.6375 | 0.6375 | 2 | 0 | none |
| 198 | `spectral_calibration_factor_of_spectrometer` | `successor_collision` | 0.63125 | 0.6375 | 0.6375 | 2 | 0 | none |
| 199 | `volume_averaged_runaway_electron_source_rate` | `grammar_invalid` | 0.6125 | 0.6375 | 0.6375 | 2 | 0 | none |
| 200 | `ion_species_atomic_number` | `successor_collision` | 0.5875 | 0.6375 | 0.6375 | 3 | 1 | none |
| 201 | `volume_integrated_net_plasma_particle_power_density` | `grammar_invalid` | 0.61875 | 0.6375 | 0.6375 | 2 | 1 | none |
| 202 | `total_particle_flux_at_divertor_target_due_to_recycling` | `attempts_exhausted` | 0.5375 | 0.6375 | 0.6375 | 3 | 3 | none |
| 203 | `angle_of_antenna_strap` | `grammar_invalid` | 0.6 | 0.625 | 0.625 | 2 | 0 | none |
| 204 | `fraction_of_neutron_detector_converter` | `successor_collision` | 0.625 | 0.625 | 0.625 | 2 | 0 | none |
| 205 | `maximum_power_at_outer_divertor_target` | `grammar_invalid` | 0.575 | 0.625 | 0.625 | 3 | 0 | none |
| 206 | `momentum_due_to_hot_tail` | `grammar_invalid` | 0.5875 | 0.625 | 0.625 | 3 | 0 | none |
| 207 | `power_of_beam_tracing_beam` | `successor_collision` | 0.6 | 0.625 | 0.625 | 2 | 0 | none |
| 208 | `radial_angle_of_poloidal_field_coil` | `grammar_invalid` | 0.525 | 0.625 | 0.625 | 3 | 0 | none |
| 209 | `vertical_coordinate_of_active_limiter_point` | `successor_collision` | 0.5 | 0.625 | 0.625 | 3 | 0 | none |
| 210 | `per_toroidal_mode_flux_surface_average_total_absorbed_power_density` | none recorded | 0.625 | 0.625 | 0.625 | 1 | 1 | none |
| 211 | `fast_electron_source_rate_due_to_dreicer` | `successor_collision` | 0.5375 | 0.6125 | 0.6125 | 3 | 0 | none |
| 212 | `gradient_of_electron_pressure` | `grammar_invalid` | 0.4625 | 0.6125 | 0.6125 | 3 | 0 | none |
| 213 | `width_of_aperture` | `successor_collision` | 0.5875 | 0.6125 | 0.6125 | 2 | 0 | none |
| 214 | `neutral_beam_atomic_number` | `grammar_invalid` | 0.575 | 0.6 | 0.6 | 3 | 0 | none |
| 215 | `neutral_convection_velocity` | none recorded | 0.58125 | 0.6 | 0.6 | 2 | 0 | none |
| 216 | `outer_hard_xray_half_width` | `grammar_invalid` | 0.58125 | 0.6 | 0.6 | 2 | 0 | none |
| 217 | `surface_thickness_of_cryostat` | `attempts_exhausted` | 0.59375 | 0.6 | 0.6 | 2 | 3 | none |
| 218 | `electron_average_temperature_at_midplane` | `grammar_invalid` | 0.575 | 0.5875 | 0.5875 | 2 | 0 | none |
| 219 | `inertial_current_density_due_to_diamagnetic_drift` | none recorded | 0.5875 | 0.5875 | 0.5875 | 1 | 1 | none |
| 220 | `deposited_power_at_divertor_target` | `attempts_exhausted` | 0.5125 | 0.5875 | 0.5875 | 3 | 3 | none |
| 221 | `root_mean_square_of_spectral_width_of_spectrometer_channel` | `attempts_exhausted` | 0.5875 | 0.5875 | 0.5875 | 2 | 3 | none |
| 222 | `wave_magnetic_field_amplitude` | `attempts_exhausted` | 0.5 | 0.5875 | 0.5875 | 3 | 3 | none |
| 223 | `flux_surface_averaged_velocity` | none recorded | 0.575 | 0.575 | 0.575 | 3 | 0 | none |
| 224 | `radius_of_plasma_filament` | `grammar_invalid` | 0.56875 | 0.575 | 0.575 | 2 | 0 | none |
| 225 | `radius_of_poloidal_field_coil` | `grammar_invalid` | 0.5375 | 0.575 | 0.575 | 3 | 0 | none |
| 226 | `normal_width_of_plasma_filament` | `attempts_exhausted` | 0.56875 | 0.575 | 0.575 | 2 | 3 | none |
| 227 | `total_size_of_camera` | none recorded | 0.525 | 0.5625 | 0.5625 | 2 | 1 | none |
| 228 | `ratio_of_charge_of_conductor_to_voltage_of_conductor` | `successor_collision` | 0.4125 | 0.5625 | 0.5625 | 3 | 2 | none |
| 229 | `inverse_of_tangential_curvature_of_optical_element` | `attempts_exhausted` | 0.475 | 0.5625 | 0.5625 | 3 | 3 | none |
| 230 | `gradient_of_radial_electron_density` | `grammar_invalid` | 0.45 | 0.55 | 0.55 | 3 | 0 | none |
| 231 | `radial_effective_thermal_energy_velocity_due_to_convection` | `successor_collision` | 0.4625 | 0.55 | 0.55 | 3 | 1 | none |
| 232 | `parallel_normalized_wave_vector` | `grammar_invalid` | 0.3625 | 0.525 | 0.525 | 3 | 0 | none |
| 233 | `inverse_of_curvature_of_arc_of_circle_center` | `grammar_invalid` | 0.4625 | 0.525 | 0.525 | 3 | 2 | none |
| 234 | `perturbed_linear_mhd_mode_number` | `successor_collision` | 0.46875 | 0.4875 | 0.4875 | 2 | 2 | none |
| 235 | `toroidal_normalized_wave_vector_of_beam_tracing_beam` | `successor_collision` | 0.4125 | 0.4375 | 0.4375 | 2 | 0 | none |
| 236 | `normalized_toroidal_hard_xray_peak_lower_bound_width` | none recorded | 0.3 | 0.325 | 0.325 | 2 | 3 | none |
| 237 | `reference_beta` | none recorded | 0.3 | 0.3 | 0.3 | 2 | 0 | none |
| 238 | `toroidal_lithium_velocity_at_separatrix` | none recorded | 0 | 0 | 0 | 2 | 0 | none |
| 239 | `ion_charge_state_torque_density_due_to_collisions` | none recorded | 0 | 0 | 0 | 1 | 1 | none |

## Boundary of authority

This artifact is ranking evidence only. It does not authorize a graph edit, acceptance, reword, fold, source migration, or vocabulary addition. A rescore candidate must go through the sanctioned same-name `sn rescore` path and ordinary quorum review; a vocabulary-gap row belongs to ISN governance; every stay-parked row remains unchanged until new deterministic evidence or operator steering arrives.

