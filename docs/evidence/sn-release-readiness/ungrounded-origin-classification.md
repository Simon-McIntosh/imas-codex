# Origin mechanism of every unsourced chain-cap identity with no source-bearing ancestor

## Verdict

Every one of the **92** chain-cap identities that carry no producing source and no
source-bearing ancestor at any depth is assigned exactly one origin mechanism, and the
per-mechanism counts are asserted here to sum to the cohort:

| Origin mechanism | Identities | Discriminator, read on the identity's chain root |
|---|---:|---|
| Generated root with no producing source | **55** | root carries `generated_at`, carries no `imported_at`, and carries a generation model |
| Bulk catalog-import fold root | **37** | root carries `created_at = 2026-07-04T21:20:38.632Z` together with `imported_at = 2026-07-04T21:21:17.079Z` |
| **Total** | **92** | equals the cohort exactly |

The two discriminators are **disjoint over the cohort**: zero identities carry a root of
both kinds, and **zero** carry a root of neither. The residual — identities fitting no
mechanism, which this node owes listed individually rather than forced into a bucket — is
**zero**. One identity has a root carrying both a derived origin and the generated-root
fingerprint; it is named under *the single multi-mechanism identity* below, so the
assignment of exactly one mechanism per identity is not achieved by hiding an overlap.

Distinct roots behind these counts number **54** and **33**. A root can anchor more than
one chain-capped descendant, so root counts are stated separately from identity counts
throughout, never substituted for them.

## Why the mechanism is read from the chain root

The composer requires a claimed source, and every one of these 92 identities was minted
by `persist_refined_name`: all **92** carry `refine_reason`, and their creation stamp is
the refine on-create stamp (`created_at == generated_at`). Refinement therefore
*propagated* the absence rather than creating it. An identity is unsourced because the
**root** of its `REFINED_FROM` chain was minted without a source, and that is where the
mechanism is legible. The root walk selects nodes with no outgoing `REFINED_FROM`.

## Authority, method, and the cohort

The question is the plan's open one: the composer requires a claimed source, so how did
identities with no DD binding anywhere come to exist. Measured live on **2026-09-17**
through `imas_codex.graph.client.GraphClient` against the active graph. Every statement
is a **read**; no node, edge or property was written and no write path was reached.

The cohort is re-measured in this node rather than carried:

| Measure | Live value |
|---|---:|
| `chain_length >= 3` identities | 174 |
| of those, carrying no producing source | 102 |
| unsourced with no sourced ancestor at any depth (the cohort) | **92** |

The producer predicate fires where it must: **2,789** over the whole label, **72** over
the cohort's sourced half, and `PRODUCED_NAME` holds **5,493** edge rows. The absence in
92 is therefore an absence the predicate can see through, not an inability to match.

The unsourced test is an `EXISTS` subquery rather than an `OPTIONAL MATCH`, so each
identity contributes exactly one row and these are identity counts, not row counts.

## The candidate mechanisms, each adjudicated with its own control

The four mechanisms proposed for this question were each tested as a predicate, with the
same predicate counted over the whole label as a positive control. A cohort count of zero
is meaningful only beside a control that fires; where the control is also zero the
mechanism has no live footprint and no identity can be assigned to it.

| Candidate mechanism | Predicate | In cohort | Over the label |
|---|---|---:|---:|
| Dedup fold | `consolidated_at IS NOT NULL` | 0 | **0** |
| Derived-parent path | `origin = 'derived'` on the identity | 0 | **428** |
| Derived-parent path | description carries the deterministic placeholder | 0 | **1** |
| Derived-parent path | `parent_enriched_at IS NOT NULL` | 0 | **183** |
| Refine successor | `refine_reason IS NOT NULL` | **92** | **1,648** |
| Refine successor | a `REFINED_FROM` predecessor exists | 91 | **1,752** |
| Migration | `edit_mode IS NOT NULL` | 31 | **1,099** |
| Migration | `edit_mode = 'rename'` | 21 | **493** |
| Migration | `source_paths` non-empty | 6 | **2,510** |

Read out:

- **The dedup fold is not separable in this graph.** Its discriminator fires **0** times
  over the whole label, so a zero in the cohort is a zero-basis predicate, not a measured
  absence. No identity is assigned to it, and that is stated rather than left as a silent
  omission: an attribution to a mechanism with no live footprint would be unfalsifiable.
- **The derived-parent path mints roots, but none of the cohorts owns it as its origin.**
  `origin = 'derived'` fires 428 times over the label, so the mechanism is real and
  live — and no identity in the cohort is itself derived. It does appear in the ancestry
  of exactly one identity, handled below.
- **The refine successor route is true for all 92 and therefore cannot discriminate
  among them.** `refine_reason` fires on every identity, which is the propagation route
  the previous section describes. It is the mechanism by which each identity came to
  exist; it is not the mechanism that made the chain ungrounded, and it cannot separate
  one identity from another.
- **Migration is present in the cohort without being the origin.** 31 identities carry
  their own `edit_mode` (21 of them `rename`) and 6 carry a non-empty scalar
  `source_paths`. These are identity-level mint routes, disclosed in the next section;
  they are not the mint of the chain root and do not explain the absence.

## The two mechanisms that do explain the cohort

**Generated root with no producing source — 55 identities, 54 distinct roots.** Each of
these roots carries `generated_at`, carries no `imported_at`, and carries a generation
model; **zero** of the 54 lack a model, so the model half of the fingerprint holds
throughout. Their `origin` is null or `pipeline`. Their creation dates run from
2026-06-17 to 2026-09-04 (newest root `created_at` 2026-09-04T13:36:34.102Z, section E of
`logs/classify.log`), so this is not solely a pre-ledger class: it also contains
roots minted well after the source-gated writer existed, which is why the class is named
by the privilege it exercised — a root written with no producer edge — rather than as a
historical artifact.

**Bulk catalog-import fold root — 37 identities, 33 distinct roots.** Every root in this
bucket carries the identical bulk-write stamp `created_at = 2026-07-04T21:20:38.632Z`
with `imported_at = 2026-07-04T21:21:17.079Z` and no `generated_at` and no generation
model. One bulk import event on 2026-07-04 explains all 33 roots; that route is removed
from current main, so it cannot be minting now.

**The summation holds, and was verified twice.** The classification pass, in which each
identity's mechanism is derived from its own root set in Python, returns 55 and 37. An
independent pair of Cypher counts over the same two root predicates also returns 55 and
37, with an overlap of 0 and a residual of 0. Two agreements, in different instruments.

## The single multi-mechanism identity

`normalized_toroidal_plasma_beta` has **two** roots: `normalized_beta`, which carries the
generated-root fingerprint, and `beta`, which carries `origin = 'derived'`. Its root set
therefore satisfies both the generated-root and the derived-root predicate. It is
assigned to the generated-root mechanism — the stated priority, root creation order —
and the derived root is named here rather than dropped. It is also the only identity in
the cohort with a derived root at all, which is the precise sense in which the
derived-parent path is present in the cohort's ancestry but never the whole explanation.

## The two singleton rows

- `toroidal_diamagnetic_magnetic_flux_at_flux_surface` is the one identity with no
  `REFINED_FROM` predecessor at all. Its root is itself, it carries the generated-root
  fingerprint, and its stage is `superseded`, so it is not release-relevant.
- `normalized_toroidal_plasma_beta` is the one identity with a derived root, above.

## A second axis, disclosed and not used for the assignment

The assignment above is made on the **root axis** — what minted the head of each chain.
The cohort also carries identity-level provenance, and it is reported here so the two
axes cannot be conflated:

| Identity-level property | In cohort | Over the whole label |
|---|---:|---:|
| `refine_reason` set | **92** (all) | 1,648 |
| `edit_mode` set | 31 | 1,099 |
| `edit_mode = 'rename'` | 21 | 493 |
| `source_paths` non-empty, no producer | 6 | 2,510 |
| no `REFINED_FROM` predecessor at all | 1 | — |

`refine_reason` is set on **all 92**, so it is true of every cohort identity and cannot
separate one from another; what it establishes is that every one of the 92 was itself a
**refine product**, which is the premise that moves the question to the root. Many of the
31 `edit_mode` rows carry `edit_mode = 'rename'` with a reason reading *canonical renderer
migration; semantic …*, so a rename pass also touched a third of the cohort without
creating any of it.

## Reproduction

Every figure above came from a read-only script run from this worktree against the live
graph. Scripts and logs live in this run's directory
(`~/.config/reckon/crew/runs/r-20260917T183209868161-n-the-ungrounded-identities-are-classified-by-how-they-arose/`):

| Script | Log | Exit | Contents |
|---|---|---|---|
| `classify.py` | `logs/classify.log` | **0** | property vocabulary; cohort re-measurement and controls; the 92 identities; mechanism controls over the label; roots per identity; per-identity assignment with the sum assertion; identity-level refine mints |
| `verify.py` | `logs/verify.log` | **0** | the 55 / 37 split re-derived in pure Cypher, independently of the Python classification |
| `candidates.py` | `logs/candidates.log` | **0** | each candidate mechanism measured as cohort count beside its whole-label control |
| `lists.py` | `logs/lists.log` | **0** | the per-bucket identity lists, and the derived-root, no-predecessor and `edit_mode` rows |

The cohort predicate, verbatim, is the one `unsourced-cohort-partition.md` fixes:

```cypher
MATCH (sn:StandardName)
WHERE sn.chain_length >= 3
  AND NOT EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }
  AND NOT EXISTS { MATCH (sn)-[:REFINED_FROM*1..10]->(a:StandardName)
                   WHERE a <> sn
                     AND EXISTS { MATCH (:StandardNameSource)-[:PRODUCED_NAME]->(a) } }
RETURN count(sn)
```

The two root predicates that make the assignment:

```cypher
// A. generated root with no producing source
root.generated_at IS NOT NULL AND root.imported_at IS NULL

// B. bulk catalog-import fold root
toString(root.created_at) STARTS WITH '2026-07-04T21:20:38.632'
```

where `root` is reached from each cohort identity by

```cypher
MATCH (sn)-[:REFINED_FROM*0..10]->(root:StandardName)
WHERE NOT EXISTS { MATCH (root)-[:REFINED_FROM]->(:StandardName) }
```

**No graph mutation was performed.** Every statement in all four scripts is a `MATCH`
… `RETURN`; no `CREATE`, `MERGE`, `SET`, `DELETE` or `REMOVE` appears in any of them.

## Appendix — the identities, by assigned mechanism

Generated identity lists from `logs/lists.log` (exit 0), reproduced here so the classification can be audited without the run directory. Buckets A and B are disjoint and exhaustive over the cohort: 55 + 37 = 92.

### A — generated root with no producing source (55)

- `accumulated_coolant_absorbed_energy_of_plasma_facing_component`
- `beat_length`
- `convected_heat_flux_coefficient`
- `coolant_absorbed_energy_accumulated_of_plasma_facing_component`
- `cumulative_inside_flux_surface_ion_charge_state_source_rate`
- `cumulative_inside_flux_surface_ion_heating_power`
- `deposited_energy_accumulated_of_plasma_facing_component`
- `derivative_of_electron_density_at_pedestal_maximum_with_respect_to_normalized_poloidal_flux_coordinate`
- `derivative_of_normalized_effective_particle_energy_with_respect_to_poloidal_angle`
- `diameter_of_fibre_bundle`
- `difference_of_radial_coordinate_and_radial_coordinate_of_outboard_midplane_separatrix`
- `doppler_beat_frequency`
- `electron_temperature_peaking_factor`
- `flux_surface_normal_surface_integrated_net_energy_flux_at_plasma_boundary`
- `inner_hard_xray_half_width_of_emissivity_peak`
- `left_hand_circularly_polarized_wave_fraction`
- `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel`
- `mean_ion_state_ionisation_potential`
- `net_coefficient_due_to_neoclassical_tearing_mode`
- `neutral_beam_particle_fraction_of_beamlet_group`
- `neutral_hydrogenic_isotope_fraction`
- `neutron_flux_at_line_of_sight`
- `normalized_toroidal_plasma_beta`
- `parallel_gyrocenter_momentum_flux_normalized_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_vector_potential`
- `parallel_total_pressure_over_edge_region`
- `particle_probability`
- `perpendicular_gyrocenter_heat_perturbed_flux_normalized_of_gyrokinetic_eigenmode`
- `perpendicular_normalized_gyrocenter_heat_perturbed_flux_of_gyrokinetic_eigenmode`
- `plasma_pulse_duration`
- `poloidal_accumulated_magnetic_flux_due_to_resistive_dissipation`
- `poloidal_cross_sectional_area_of_plasma_boundary`
- `poloidal_magnetic_flux_perturbed_at_measurement_position_due_to_wave_particle_interaction`
- `poloidal_parity_of_gyrokinetic_eigenmode`
- `poloidal_suprathermal_electron_angle_perturbed_at_measurement_position`
- `radiated_power_at_wall_due_to_surface_emission`
- `ratio_of_ion_average_temperature_to_volume_averaged_ion_average_temperature`
- `ratio_of_neutral_species_gas_count_to_total_gas_count`
- `root_mean_square_of_fluctuating_floating_electrostatic_potential`
- `spectral_bremsstrahlung_radiance`
- `spectral_bremsstrahlung_rate`
- `steady_state_total_plasma_absorbed_power`
- `thermal_plasma_field_aligned_power_over_halo_region_due_to_conductive_losses`
- `toroidal_diamagnetic_magnetic_flux_at_flux_surface`
- `toroidal_net_plasma_torque_of_neoclassical_tearing_mode`
- `toroidal_offset_at_measurement_position`
- `toroidal_sonic_mach_number`
- `toroidal_total_momentum`
- `total_co_passing_pressure`
- `total_current_due_to_fusion_born_alpha`
- `total_launched_wave_power_of_electron_cyclotron_launcher`
- `total_neutral_particle_source_rate_at_wall_due_to_convection`
- `velocity_of_pellet_magnitude`
- `volume_averaged_linear_thermal_electron_decay_time_due_to_disruption`
- `volume_averaged_lithium_fraction`
- `wall_gap_of_antenna_strap`

### B — bulk catalog-import fold root (37)

- `absorbed_coolant_power_of_plant_component_port`
- `curvature_inverse_of_arc_of_circle_center`
- `deposited_power_at_divertor_target`
- `energy_flux_at_control_surface`
- `ethylene_count_cumulative_due_to_gas_injection`
- `flux_surface_averaged_parallel_electric_field_at_separatrix`
- `flux_surface_normal_momentum_convection_velocity`
- `flux_surface_normal_neutral_energy_diffusion_coefficient`
- `front_surface_area_of_langmuir_probe`
- `inverse_of_tangential_curvature_of_optical_element`
- `ion_temperature_at_outboard_midplane_separatrix`
- `ion_upper_bound_charge_number`
- `lithium_volume_of_breeder_blanket`
- `molecular_gas_count_due_to_pellet_injection`
- `net_forward_power_of_wave_beam`
- `net_plasma_power_density`
- `neutral_species_kinetic_energy_flux_at_wall_due_to_surface_emission`
- `non_axisymmetric_current_of_conductor`
- `normal_distance_of_antenna_strap`
- `normal_width_of_plasma_filament`
- `parallel_electric_field_flux_surface_averaged_at_separatrix`
- `plasma_electrostatic_potential_at_outboard_midplane`
- `plasma_electrostatic_potential_at_wall`
- `power_over_scrape_off_layer_due_to_radiation`
- `radial_offset_of_lower_hybrid_antenna`
- `root_mean_square_of_spectral_width_of_spectrometer_channel`
- `root_mean_square_spectral_width_of_spectrometer_channel`
- `spectral_width_root_mean_square_of_spectrometer_channel`
- `tangential_curvature_inverse_of_optical_element`
- `tendency_of_runaway_electron_density`
- `total_electron_power_density`
- `total_incident_thermal_power`
- `total_launched_power_due_to_ion_cyclotron_heating`
- `total_particle_flux_at_divertor_target_due_to_recycling`
- `volume_integrated_toroidal_fast_electron_torque_density_due_to_collisions`
- `wave_critical_ordinary_mode_frequency`
- `wave_magnetic_field_amplitude`

### Also carrying their own `edit_mode` (31)

Disclosed for the second axis; not the assignment. A row here is in exactly one of A or B above.

- `beat_length` — `hint`
- `coolant_absorbed_energy_accumulated_of_plasma_facing_component` — `rename`
- `curvature_inverse_of_arc_of_circle_center` — `rename`
- `deposited_energy_accumulated_of_plasma_facing_component` — `rename`
- `derivative_of_electron_density_at_pedestal_maximum_with_respect_to_normalized_poloidal_flux_coordinate` — `rename`
- `derivative_of_normalized_effective_particle_energy_with_respect_to_poloidal_angle` — `rename`
- `electron_temperature_peaking_factor` — `hint`
- `ethylene_count_cumulative_due_to_gas_injection` — `rename`
- `flux_surface_averaged_parallel_electric_field_at_separatrix` — `rename`
- `inner_hard_xray_half_width_of_emissivity_peak` — `rename`
- `ion_upper_bound_charge_number` — `hint`
- `mean_ion_state_ionisation_potential` — `hint`
- `normalized_toroidal_plasma_beta` — `hint`
- `parallel_electric_field_flux_surface_averaged_at_separatrix` — `rename`
- `parallel_gyrocenter_momentum_flux_normalized_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_vector_potential` — `rename`
- `perpendicular_gyrocenter_heat_perturbed_flux_normalized_of_gyrokinetic_eigenmode` — `rename`
- `perpendicular_normalized_gyrocenter_heat_perturbed_flux_of_gyrokinetic_eigenmode` — `hint`
- `poloidal_cross_sectional_area_of_plasma_boundary` — `rename`
- `poloidal_magnetic_flux_perturbed_at_measurement_position_due_to_wave_particle_interaction` — `rename`
- `poloidal_suprathermal_electron_angle_perturbed_at_measurement_position` — `rename`
- `ratio_of_neutral_species_gas_count_to_total_gas_count` — `rename`
- `root_mean_square_spectral_width_of_spectrometer_channel` — `rename`
- `spectral_bremsstrahlung_radiance` — `hint`
- `spectral_bremsstrahlung_rate` — `rename`
- `spectral_width_root_mean_square_of_spectrometer_channel` — `rename`
- `tangential_curvature_inverse_of_optical_element` — `rename`
- `thermal_plasma_field_aligned_power_over_halo_region_due_to_conductive_losses` — `hint`
- `toroidal_offset_at_measurement_position` — `rename`
- `toroidal_sonic_mach_number` — `hint`
- `toroidal_total_momentum` — `hint`
- `volume_integrated_toroidal_fast_electron_torque_density_due_to_collisions` — `rename`

