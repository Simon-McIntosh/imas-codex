<meta name="docs-project" content="imas-codex">
<meta name="reckon-type" content="evidence">
<meta name="plan-slug" content="paid-identity-deletion-protection">
<meta name="plan-status" content="active">
<meta name="plan-title" content="Is the derived origin label truthful &mdash; per-name evidence">
<meta name="plan-evidence-for" content="paid-identity-deletion-protection">

# Is the `origin='derived'` label truthful?

§7a stage 1 reclassified the *leaders*; stage 2 is "repoint every reader at the
source binding" and stage 4 drops the field. Both assume the label is a
computable summary of the topology — that "derived" means "structural parent
scaffolding with no direct Data Dictionary source of its own". This evidence
tests that assumption name by name, and reports what the readers actually
branch on.

All figures are read-only measurements against the live graph (`codex`) at
`bolt://98dci4-gpu-0002:7687`, taken 2026-09-18 from worktree base
`7a095a91ed86596ebf4d3781abfc8add742275a6`. No graph write of any kind was
performed.

## Verdict

| Question | Answer |
|---|---|
| How many names are live at `origin='derived'`? | **428** — identical to the figure audit's re-measured 428 (`plan-figure-audit.md` row 6); that figure is current, not drifted |
| How many have a direct Data Dictionary producing source? | **1** — `spectral_signal_to_noise_ratio_of_spectrometer_channel` |
| Therefore how many are *not* structural scaffolding? | **1 of 428**, and it is a falsified label, not a scaffolding name |
| Do they exist only as abstractions over children? | **421 of 428** have live children; **7 are leaves** with none |
| Both legs of the plan's criterion hold? | **420 of 428 (98.1%)** |
| Is the label derivable from the absence of a producer edge? | **No.** Necessary (427/428) but not sufficient — 2,415 counterexamples, precision 0.15 |
| Is it recoverable from the `derived`-typed producer edge? | **Not exactly.** 121 names disagree (6 marked without such an edge, 115 carrying one unmarked); precision 0.79 |
| Is it recoverable from the scalar `StandardName.source_types`? | **No.** Zero names carry `'derived'` in that property; 251 of the 428 have it empty |
| Does anything still branch on it? | **Yes — 52 non-deletion read lines across 34 decision paths**, plus the 2 deletion reads §7a retires (`derived-origin-readers.md`) |

## The population, against the figure audit's 428

| Leg | Count | Instrument |
|---|---|---|
| live names total | 5,130 | `all_names_total` |
| `origin='derived'` | 428 | `derived_total` |
| — `name_stage` accepted / superseded / pending / exhausted | 401 / 24 / 2 / 1 | `derived_by_stage` |
| — `docs_stage='accepted'` with non-empty documentation | 412 | `derived_docs_accepted_nonempty` |
| — with any `PRODUCED_NAME` producer | 422 | `derived_with_any_producer` |
| — with a `derived`-typed producer | 422 | `derived_producer_source_types` |
| — with a **`dd`-typed** producer | **1** | `derived_with_dd_producer` |
| — with no producer at all | 6 | `derived_without_any_producer` |
| — with live children | 421 (956 distinct children) | `derived_children_total` |
| — childless | 7 | `derived_childless` |
| — carrying an outgoing `HAS_STRUCTURAL_AUTHORITY` edge | 309 (0 incoming) | `derived_structural_authority_out/_in` |

The audit's four §7 figures all reproduce: 428 alive, 401 accepted / 0
approved, 412 docs-accepted non-empty, and the paid-work population. This node
adds the leg the audit did not measure — the **DD producer**, which is the
leg the plan's own definition turns on.

## Truthfulness, name by name

The plan's criterion for a genuine structural parent is two-legged: *no direct
DD source of its own*, **and** *exists only as an abstraction over children*.
Measured across all 428:

- **Leg 1 fails for 1 name.** `spectral_signal_to_noise_ratio_of_spectrometer_channel`
  is `accepted`/`accepted`, carries documentation and one child, and is produced
  by **two** sources:
  - `dd:spectrometer_visible/channel/isotope_ratios/signal_to_noise`, status
    `attached`, batch `focus`
  - `derived:spectral_signal_to_noise_ratio_of_spectrometer_channel`, status
    `composed`, batch `derived_parent`

  It is a DD-realised identity wearing the scaffolding label. A DD source
  attached a realisation to a name that the derived-parent machinery had
  already classified as an abstraction — neither writer re-read the other's
  claim.
- **Leg 2 fails for 7 names** (no live children at all): `angle_of_optical_element`,
  `diffusion_coefficient_due_to_diffusion`, `factor_of_spectrometer_channel`,
  `flux_at_wall_due_to_eddy_current`, `ion_diffusivity`, `permeability_of_ferritic_element`,
  and `momentum_flux_due_to_perturbed_parallel_magnetic_field` (the last also
  carries no documentation).
- **420 of 428 satisfy both legs** and hold the label honestly.

Two further mismatches sit *inside* the 420 and matter to §7a's reader
repoint, because they are the readers' own subject matter:

- **25 of 428 are not live identities at all**: 24 `superseded` and 1
  `exhausted`. The delete path keys on the label, and 6 of the 6 producer-less
  names are in that set or adjacent to it.
- **251 of 428 carry an empty `source_types` property**, and 12 carry `'dd'`
  in that property with no dd producer — the property and the edge disagree in
  both directions.

### The published zero no longer reproduces under its own predicate

The figure audit records `S7_derived_with_direct_dd_producer = 0` and the plan's
remaining-effort table derives "§7 reclassification remainder | 0" from it.
Measured today:

| Predicate | Derived names matching | All names matching |
|---|---|---|
| `-[:PRODUCED_NAME]->` from any `source_type='dd'` source (**status-agnostic**) | **1** | 2,288 |
| … from a `dd` source with `status='extracted'` (the schema's example predicate) | **0** | 5 |

The recorded zero is reproducible — under a predicate that admits **five names
in the entire graph |graph|**. Only 5 of 5,130 names have an `extracted` dd
producer; the 1 falsified name's producer sits at `attached`, and 2,178 dd
sources sit at `attached` against 1,250 at `extracted`, 2,589 at `composed` and
1,820 at `stale`. So "the reclassification remainder is 0" was a true
statement about a nearly-empty set, standing in for the fact that was needed —
the status-agnostic count, which is 1. Any predicate here must state whether it
filters on `status`, because that single clause moves the answer from 0 to 1.

## Per-name report — all 428 live names at `origin='derived'`

One row per live name, in id order. `direct dd producer` is the status-agnostic test (`(src:StandardNameSource {source_type: 'dd'})-[:PRODUCED_NAME]->(sn)`); `producer source types` lists the `source_type` of every producing `StandardNameSource` node, or `none`; `children` counts live `HAS_PARENT` children; the verdict column applies the plan's two-legged criterion (no direct DD source *and* an abstraction over children). The aggregate of this table is the population table above.

| # | name | direct `dd` producer | producer source types | children | name stage | docs stage | verdict |
|---|---|---|---|---|---|---|---|
| 1 | `absorbed_power_of_beam_tracing_beam` | no | derived | 3 | accepted | accepted | scaffolding |
| 2 | `absorbed_wave_power` | no | derived | 3 | accepted | accepted | scaffolding |
| 3 | `alfven_frequency` | no | derived | 2 | accepted | accepted | scaffolding |
| 4 | `angle_of_optical_element` | no | derived | 0 | accepted | accepted | scaffolding-shaped, leaf (no children) |
| 5 | `area_of_flux_surface` | no | derived | 3 | superseded | accepted | scaffolding |
| 6 | `argon_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 7 | `average_charge_number` | no | derived | 1 | accepted | accepted | scaffolding |
| 8 | `average_external_magnetic_flux` | no | derived | 1 | accepted | accepted | scaffolding |
| 9 | `average_temperature_over_scrape_off_layer` | no | derived | 1 | accepted | accepted | scaffolding |
| 10 | `back_surface_neutron_flux` | no | derived | 1 | accepted | accepted | scaffolding |
| 11 | `beryllium_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 12 | `beta` | no | none | 3 | accepted | accepted | scaffolding |
| 13 | `boron_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 14 | `bulk_center_of_mass_velocity` | no | derived | 1 | accepted | accepted | scaffolding |
| 15 | `bulk_neutral_internal_state_velocity` | no | derived | 1 | accepted | accepted | scaffolding |
| 16 | `carbon_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 17 | `center_of_mass_velocity` | no | derived | 5 | accepted | accepted | scaffolding |
| 18 | `co_passing_fast_current_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 19 | `co_passing_fast_ion_torque_density_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 20 | `co_passing_thermal_ion_torque_density_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 21 | `co_passing_torque_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 22 | `co_passing_torque_density_due_to_j_cross_b_force` | no | derived | 1 | accepted | accepted | scaffolding |
| 23 | `conductivity` | no | derived | 5 | superseded | accepted | scaffolding |
| 24 | `count_at_detector_pixel` | no | derived | 1 | accepted | accepted | scaffolding |
| 25 | `counter_passing_current_density` | no | derived | 1 | accepted | reviewed | scaffolding |
| 26 | `counter_passing_fast_current_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 27 | `counter_passing_fast_electron_torque_density_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 28 | `counter_passing_fast_ion_torque_density_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 29 | `counter_passing_thermal_electron_torque_density_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 30 | `counter_passing_thermal_ion_torque_density_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 31 | `counter_passing_torque_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 32 | `counter_passing_torque_density_due_to_j_cross_b_force` | no | derived | 1 | accepted | accepted | scaffolding |
| 33 | `cumulative_inside_flux_surface_torque` | no | derived | 1 | accepted | accepted | scaffolding |
| 34 | `current_density` | no | derived | 12 | accepted | accepted | scaffolding |
| 35 | `current_density_due_to_anomalous_transport` | no | derived | 5 | accepted | accepted | scaffolding |
| 36 | `current_density_due_to_bootstrap_current_drive` | no | derived | 1 | accepted | accepted | scaffolding |
| 37 | `current_density_due_to_distribution_function_driven` | no | derived | 1 | accepted | accepted | scaffolding |
| 38 | `current_density_due_to_fast_ion` | no | derived | 1 | accepted | reviewed | scaffolding |
| 39 | `current_density_due_to_heat_viscosity` | no | derived | 6 | accepted | accepted | scaffolding |
| 40 | `current_density_due_to_ion_inertia` | no | derived | 5 | accepted | accepted | scaffolding |
| 41 | `current_density_due_to_ion_neutral_friction` | no | derived | 6 | accepted | accepted | scaffolding |
| 42 | `current_density_due_to_non_inductive_current_drive` | no | derived | 1 | accepted | accepted | scaffolding |
| 43 | `current_density_due_to_parallel_viscosity` | no | derived | 6 | accepted | accepted | scaffolding |
| 44 | `current_density_due_to_perpendicular_viscosity` | no | derived | 6 | accepted | accepted | scaffolding |
| 45 | `current_density_due_to_wave_driven_current_drive` | no | derived | 2 | accepted | accepted | scaffolding |
| 46 | `current_density_per_toroidal_mode_due_to_wave_driven_current_drive` | no | derived | 1 | accepted | accepted | scaffolding |
| 47 | `current_due_to_wave_driven_current_drive` | no | derived | 2 | accepted | accepted | scaffolding |
| 48 | `current_of_antenna_strap` | no | derived | 1 | accepted | accepted | scaffolding |
| 49 | `current_weighted_average_external_magnetic_flux` | no | derived | 1 | accepted | accepted | scaffolding |
| 50 | `curvature_of_arc_of_circle_center` | no | derived | 1 | accepted | accepted | scaffolding |
| 51 | `decay_length_over_scrape_off_layer` | no | derived | 2 | accepted | accepted | scaffolding |
| 52 | `density_at_divertor_target` | no | derived | 19 | accepted | accepted | scaffolding |
| 53 | `density_at_internal_transport_barrier` | no | derived | 19 | accepted | accepted | scaffolding |
| 54 | `density_at_limiter` | no | derived | 19 | accepted | accepted | scaffolding |
| 55 | `density_at_magnetic_axis` | no | derived | 19 | accepted | accepted | scaffolding |
| 56 | `density_at_pedestal_maximum` | no | derived | 1 | accepted | accepted | scaffolding |
| 57 | `density_at_pedestal_top` | no | derived | 19 | accepted | accepted | scaffolding |
| 58 | `density_at_plasma_boundary` | no | derived | 19 | accepted | accepted | scaffolding |
| 59 | `deuterium_density` | no | derived | 3 | accepted | accepted | scaffolding |
| 60 | `deuterium_tritium_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 61 | `diamagnetic_current_density` | no | derived | 3 | accepted | accepted | scaffolding |
| 62 | `diffusion_coefficient_due_to_diffusion` | no | derived | 0 | accepted | accepted | scaffolding-shaped, leaf (no children) |
| 63 | `edge_plasma_momentum_diffusivity` | no | derived | 1 | accepted | accepted | scaffolding |
| 64 | `effective_ion_state_momentum_diffusivity` | no | derived | 1 | accepted | accepted | scaffolding |
| 65 | `effective_neutral_internal_state_velocity_due_to_convection` | no | derived | 1 | accepted | accepted | scaffolding |
| 66 | `effective_particle_energy` | no | derived | 1 | accepted | accepted | scaffolding |
| 67 | `effective_thermal_ion_charge_state_energy_velocity_due_to_convection` | no | derived | 1 | accepted | accepted | scaffolding |
| 68 | `efficiency_of_plant_system` | no | derived | 2 | accepted | accepted | scaffolding |
| 69 | `electric_field` | no | derived | 6 | accepted | accepted | scaffolding |
| 70 | `electron_density_at_pedestal_maximum` | no | derived | 1 | accepted | accepted | scaffolding |
| 71 | `electron_number_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 72 | `electron_particle_convection_velocity` | no | derived | 1 | accepted | accepted | scaffolding |
| 73 | `electron_power_at_inside_flux_surface` | no | derived | 2 | accepted | accepted | scaffolding |
| 74 | `electron_power_density_due_to_collisions` | no | derived | 2 | accepted | accepted | scaffolding |
| 75 | `electron_pressure` | no | derived | 2 | accepted | accepted | scaffolding |
| 76 | `electron_torque_density_due_to_collisions` | no | derived | 2 | accepted | accepted | scaffolding |
| 77 | `electron_torque_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 78 | `electron_velocity` | no | derived | 5 | accepted | accepted | scaffolding |
| 79 | `electrostatic_potential_amplitude` | no | derived | 1 | accepted | accepted | scaffolding |
| 80 | `electrostatic_potential_imaginary_part` | no | derived | 1 | accepted | accepted | scaffolding |
| 81 | `electrostatic_potential_real_part` | no | derived | 1 | accepted | accepted | scaffolding |
| 82 | `energy_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 83 | `energy_diffusion_coefficient` | no | derived | 4 | accepted | accepted | scaffolding |
| 84 | `energy_flux_at_limiter` | no | derived | 1 | accepted | accepted | scaffolding |
| 85 | `energy_flux_at_wall_due_to_recombination` | no | derived | 3 | accepted | accepted | scaffolding |
| 86 | `energy_flux_at_wall_due_to_surface_emission` | no | derived | 1 | accepted | accepted | scaffolding |
| 87 | `energy_flux_limiter_coefficient` | no | derived | 2 | accepted | accepted | scaffolding |
| 88 | `energy_of_detector_pixel` | no | derived | 2 | accepted | accepted | scaffolding |
| 89 | `energy_velocity_due_to_convection` | no | derived | 1 | accepted | accepted | scaffolding |
| 90 | `explicit_ion_torque` | no | derived | 1 | accepted | accepted | scaffolding |
| 91 | `external_magnetic_flux` | no | derived | 1 | accepted | accepted | scaffolding |
| 92 | `factor_of_spectrometer_channel` | no | derived | 0 | accepted | accepted | scaffolding-shaped, leaf (no children) |
| 93 | `fast_current_density` | no | derived | 3 | accepted | accepted | scaffolding |
| 94 | `fast_electron_pressure` | no | derived | 2 | accepted | accepted | scaffolding |
| 95 | `fast_electron_torque_density_due_to_collisions` | no | derived | 2 | accepted | accepted | scaffolding |
| 96 | `fast_ion_charge_state_torque_density_due_to_collisions` | no | derived | 4 | accepted | accepted | scaffolding |
| 97 | `fast_ion_pressure` | no | derived | 3 | accepted | accepted | scaffolding |
| 98 | `fast_ion_state_pressure` | no | derived | 1 | superseded | accepted | scaffolding |
| 99 | `fast_ion_torque_density_due_to_collisions` | no | derived | 4 | accepted | accepted | scaffolding |
| 100 | `fast_neutral_pressure` | no | derived | 2 | accepted | accepted | scaffolding |
| 101 | `fast_neutral_state_pressure` | no | derived | 1 | superseded | accepted | scaffolding |
| 102 | `fast_particle_torque_density_due_to_coulomb_collisions_with_electrons` | no | derived | 1 | accepted | accepted | scaffolding |
| 103 | `fast_particle_torque_due_to_j_cross_b_force` | no | derived | 1 | accepted | accepted | scaffolding |
| 104 | `fast_torque_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 105 | `field_line_average_temperature_over_scrape_off_layer` | no | derived | 2 | accepted | accepted | scaffolding |
| 106 | `fluctuating_ion_current_density` | no | derived | 1 | accepted | reviewed | scaffolding |
| 107 | `flux_at_divertor_target` | no | derived | 1 | accepted | accepted | scaffolding |
| 108 | `flux_at_first_wall` | no | derived | 2 | accepted | accepted | scaffolding |
| 109 | `flux_at_limiter` | no | derived | 1 | accepted | accepted | scaffolding |
| 110 | `flux_at_wall_due_to_eddy_current` | no | derived | 0 | accepted | accepted | scaffolding-shaped, leaf (no children) |
| 111 | `flux_at_wall_due_to_pumping` | no | derived | 1 | accepted | accepted | scaffolding |
| 112 | `flux_at_wall_due_to_recombination` | no | derived | 1 | accepted | accepted | scaffolding |
| 113 | `flux_at_wall_due_to_surface_emission` | no | derived | 2 | accepted | accepted | scaffolding |
| 114 | `flux_due_to_e_cross_b_drift` | no | derived | 1 | accepted | accepted | scaffolding |
| 115 | `flux_due_to_eddy_current` | no | derived | 2 | accepted | accepted | scaffolding |
| 116 | `flux_due_to_fusion` | no | none | 1 | superseded | reviewed | scaffolding |
| 117 | `flux_due_to_pumping` | no | derived | 1 | accepted | accepted | scaffolding |
| 118 | `flux_due_to_recombination` | no | derived | 1 | superseded | accepted | scaffolding |
| 119 | `flux_due_to_surface_emission` | no | derived | 1 | accepted | accepted | scaffolding |
| 120 | `flux_limiter_coefficient` | no | derived | 6 | accepted | accepted | scaffolding |
| 121 | `greenwald_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 122 | `gyrocenter_frequency` | no | derived | 2 | accepted | accepted | scaffolding |
| 123 | `half_width_at_emissivity_peak` | no | derived | 1 | accepted | accepted | scaffolding |
| 124 | `halo_current` | no | derived | 2 | accepted | accepted | scaffolding |
| 125 | `hard_xray_half_width_at_emissivity_peak` | no | derived | 2 | accepted | accepted | scaffolding |
| 126 | `heat_convection_velocity` | no | derived | 1 | accepted | accepted | scaffolding |
| 127 | `heat_flux` | no | derived | 1 | accepted | accepted | scaffolding |
| 128 | `helium_3_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 129 | `helium_4_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 130 | `hydrogen_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 131 | `hydrogenic_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 132 | `implicit_ion_momentum_source_rate` | no | derived | 1 | accepted | accepted | scaffolding |
| 133 | `inner_squareness_of_flux_surface` | no | derived | 2 | accepted | accepted | scaffolding |
| 134 | `inner_squareness_of_plasma_boundary` | no | derived | 2 | accepted | accepted | scaffolding |
| 135 | `inverse_of_major_radius` | no | derived | 1 | accepted | accepted | scaffolding |
| 136 | `inverse_of_square_of_magnetic_field_magnitude` | no | derived | 1 | accepted | accepted | scaffolding |
| 137 | `inverse_of_square_of_major_radius` | no | derived | 1 | accepted | accepted | scaffolding |
| 138 | `ion_absorbed_wave_power` | no | derived | 2 | accepted | accepted | scaffolding |
| 139 | `ion_charge_state_average_charge_number` | no | derived | 1 | accepted | accepted | scaffolding |
| 140 | `ion_charge_state_diffusivity` | no | derived | 2 | accepted | accepted | scaffolding |
| 141 | `ion_charge_state_energy_velocity_due_to_convection` | no | derived | 2 | accepted | reviewed | scaffolding |
| 142 | `ion_charge_state_momentum_diffusivity` | no | derived | 4 | accepted | accepted | scaffolding |
| 143 | `ion_charge_state_momentum_flux` | no | derived | 5 | accepted | accepted | scaffolding |
| 144 | `ion_charge_state_particle_flux` | no | derived | 3 | accepted | accepted | scaffolding |
| 145 | `ion_charge_state_power` | no | derived | 1 | accepted | accepted | scaffolding |
| 146 | `ion_charge_state_power_at_inside_flux_surface` | no | derived | 2 | accepted | accepted | scaffolding |
| 147 | `ion_current_density` | no | derived | 3 | accepted | accepted | scaffolding |
| 148 | `ion_density` | no | derived | 5 | accepted | accepted | scaffolding |
| 149 | `ion_density_at_plasma_boundary` | no | derived | 2 | accepted | accepted | scaffolding |
| 150 | `ion_diffusivity` | no | derived | 0 | exhausted | accepted | scaffolding-shaped, leaf (no children) |
| 151 | `ion_energy` | no | derived | 1 | accepted | accepted | scaffolding |
| 152 | `ion_heat_convection_velocity` | no | derived | 1 | accepted | accepted | scaffolding |
| 153 | `ion_momentum_convection_velocity` | no | derived | 3 | accepted | accepted | scaffolding |
| 154 | `ion_momentum_damping_rate` | no | derived | 4 | accepted | accepted | scaffolding |
| 155 | `ion_momentum_diffusion_coefficient` | no | derived | 3 | accepted | accepted | scaffolding |
| 156 | `ion_momentum_flux` | no | derived | 4 | accepted | accepted | scaffolding |
| 157 | `ion_momentum_flux_limiter_coefficient` | no | derived | 3 | accepted | accepted | scaffolding |
| 158 | `ion_momentum_source_rate` | no | derived | 1 | accepted | accepted | scaffolding |
| 159 | `ion_particle_convection_velocity` | no | derived | 1 | accepted | accepted | scaffolding |
| 160 | `ion_power_at_inside_flux_surface` | no | derived | 2 | accepted | accepted | scaffolding |
| 161 | `ion_power_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 162 | `ion_power_density_due_to_collisions` | no | derived | 2 | accepted | accepted | scaffolding |
| 163 | `ion_power_due_to_collisions` | no | derived | 2 | accepted | accepted | scaffolding |
| 164 | `ion_pressure` | no | none | 2 | accepted | accepted | scaffolding |
| 165 | `ion_rotation_frequency` | no | derived | 1 | accepted | accepted | scaffolding |
| 166 | `ion_state_energy_convection_velocity` | no | derived | 1 | superseded | accepted | scaffolding |
| 167 | `ion_state_momentum` | no | derived | 1 | superseded | accepted | scaffolding |
| 168 | `ion_state_momentum_convection_velocity` | no | derived | 2 | superseded | accepted | scaffolding |
| 169 | `ion_state_momentum_diffusion_coefficient` | no | derived | 1 | superseded | accepted | scaffolding |
| 170 | `ion_state_momentum_diffusivity` | no | derived | 1 | accepted | accepted | scaffolding |
| 171 | `ion_state_momentum_flux_limiter_coefficient` | no | derived | 2 | superseded | accepted | scaffolding |
| 172 | `ion_torque` | no | derived | 1 | accepted | accepted | scaffolding |
| 173 | `ion_torque_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 174 | `ion_torque_density_due_to_collisions` | no | derived | 2 | accepted | accepted | scaffolding |
| 175 | `ion_torque_density_due_to_thermalization` | no | derived | 1 | accepted | accepted | scaffolding |
| 176 | `ion_torque_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 177 | `iron_density` | no | derived | 3 | accepted | accepted | scaffolding |
| 178 | `kinetic_energy_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 179 | `kinetic_energy_flux_at_wall_due_to_surface_emission` | no | derived | 3 | accepted | accepted | scaffolding |
| 180 | `krypton_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 181 | `left_hand_circularly_polarized_wave_electric_field` | no | derived | 3 | accepted | accepted | scaffolding |
| 182 | `length_of_antenna_strap` | no | derived | 2 | accepted | accepted | scaffolding |
| 183 | `length_of_interferometer_beam` | no | derived | 1 | accepted | accepted | scaffolding |
| 184 | `length_of_passive_loop_element` | no | derived | 2 | accepted | accepted | scaffolding |
| 185 | `linear_mhd_mode_reference_phase` | no | derived | 1 | accepted | accepted | scaffolding |
| 186 | `linear_neutral_internal_state_momentum_flux` | no | derived | 1 | accepted | accepted | scaffolding |
| 187 | `logarithm_of_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 188 | `mach_number` | no | derived | 1 | accepted | accepted | scaffolding |
| 189 | `magnetic_field` | no | derived | 16 | accepted | accepted | scaffolding |
| 190 | `magnetic_field_at_pedestal_top_high_field_side` | no | derived | 1 | accepted | accepted | scaffolding |
| 191 | `magnetic_field_at_pedestal_top_low_field_side` | no | derived | 1 | accepted | accepted | scaffolding |
| 192 | `magnetic_field_magnitude` | no | derived | 6 | accepted | accepted | scaffolding |
| 193 | `magnetic_flux` | no | derived | 3 | accepted | accepted | scaffolding |
| 194 | `magnetic_flux_due_to_diamagnetic_drift` | no | derived | 1 | accepted | accepted | scaffolding |
| 195 | `magnetic_vector_potential` | no | derived | 6 | accepted | accepted | scaffolding |
| 196 | `major_radius` | no | derived | 9 | accepted | accepted | scaffolding |
| 197 | `maximum_magnetic_field` | no | derived | 1 | accepted | accepted | scaffolding |
| 198 | `mhd_mode_number` | no | derived | 1 | accepted | accepted | scaffolding |
| 199 | `mhd_mode_reference_phase` | no | derived | 1 | accepted | accepted | scaffolding |
| 200 | `mode_reference_phase` | no | derived | 1 | accepted | accepted | scaffolding |
| 201 | `momentum` | no | derived | 4 | accepted | accepted | scaffolding |
| 202 | `momentum_damping_rate` | no | derived | 2 | accepted | accepted | scaffolding |
| 203 | `momentum_diffusion_coefficient` | no | derived | 8 | accepted | accepted | scaffolding |
| 204 | `momentum_diffusivity` | no | derived | 8 | accepted | accepted | scaffolding |
| 205 | `momentum_flux` | no | derived | 9 | accepted | accepted | scaffolding |
| 206 | `momentum_flux_due_to_e_cross_b_drift` | no | derived | 1 | accepted | accepted | scaffolding |
| 207 | `momentum_flux_due_to_perturbed_parallel_magnetic_field` | no | derived | 0 | accepted | pending | scaffolding-shaped, leaf (no children) |
| 208 | `momentum_flux_due_to_perturbed_parallel_vector_potential` | no | derived | 1 | accepted | accepted | scaffolding |
| 209 | `momentum_flux_limiter_coefficient` | no | derived | 8 | accepted | accepted | scaffolding |
| 210 | `momentum_flux_normalized_due_to_e_cross_b_drift` | no | derived | 1 | accepted | accepted | scaffolding |
| 211 | `momentum_flux_normalized_due_to_perturbed_parallel_vector_potential` | no | derived | 2 | accepted | accepted | scaffolding |
| 212 | `momentum_source` | no | derived | 7 | accepted | accepted | scaffolding |
| 213 | `momentum_source_rate` | no | derived | 1 | accepted | accepted | scaffolding |
| 214 | `neon_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 215 | `net_ion_momentum_source` | no | derived | 1 | accepted | accepted | scaffolding |
| 216 | `neutral_density` | no | derived | 3 | accepted | accepted | scaffolding |
| 217 | `neutral_fraction` | no | derived | 2 | superseded | pending | scaffolding |
| 218 | `neutral_internal_state_momentum_velocity_due_to_convection` | no | derived | 1 | accepted | accepted | scaffolding |
| 219 | `neutral_internal_state_velocity_due_to_convection` | no | derived | 1 | accepted | accepted | scaffolding |
| 220 | `neutral_momentum_convection_velocity` | no | derived | 1 | accepted | accepted | scaffolding |
| 221 | `neutral_momentum_diffusion_coefficient` | no | derived | 2 | accepted | accepted | scaffolding |
| 222 | `neutral_momentum_diffusivity` | no | derived | 2 | accepted | accepted | scaffolding |
| 223 | `neutral_momentum_flux` | no | derived | 5 | accepted | accepted | scaffolding |
| 224 | `neutral_momentum_flux_limiter_coefficient` | no | derived | 4 | accepted | accepted | scaffolding |
| 225 | `neutral_particle_convection_velocity` | no | derived | 2 | accepted | reviewed | scaffolding |
| 226 | `neutral_species_center_of_mass_velocity` | no | derived | 1 | accepted | reviewed | scaffolding |
| 227 | `neutral_species_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 228 | `neutral_species_energy_flux` | no | derived | 1 | accepted | accepted | scaffolding |
| 229 | `neutral_species_internal_state_velocity_due_to_e_cross_b_drift` | no | derived | 1 | accepted | accepted | scaffolding |
| 230 | `neutral_species_momentum_convection_velocity` | no | derived | 2 | accepted | accepted | scaffolding |
| 231 | `neutral_species_particle_diffusivity` | no | derived | 1 | accepted | accepted | scaffolding |
| 232 | `neutral_state_density` | no | derived | 1 | superseded | accepted | scaffolding |
| 233 | `neutral_state_energy_convection_velocity` | no | derived | 2 | superseded | accepted | scaffolding |
| 234 | `neutral_state_momentum_convection_velocity` | no | derived | 2 | superseded | accepted | scaffolding |
| 235 | `neutral_state_momentum_diffusion_coefficient` | no | derived | 2 | superseded | accepted | scaffolding |
| 236 | `neutral_state_momentum_diffusivity` | no | derived | 1 | superseded | accepted | scaffolding |
| 237 | `neutral_state_momentum_flux` | no | derived | 3 | superseded | accepted | scaffolding |
| 238 | `neutral_state_momentum_flux_limiter_coefficient` | no | derived | 4 | superseded | accepted | scaffolding |
| 239 | `neutral_state_particle_diffusivity` | no | derived | 1 | accepted | accepted | scaffolding |
| 240 | `neutral_torque_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 241 | `neutral_velocity` | no | derived | 7 | accepted | accepted | scaffolding |
| 242 | `neutral_velocity_due_to_diamagnetic_drift` | no | derived | 1 | accepted | accepted | scaffolding |
| 243 | `neutron_flux` | no | none | 2 | superseded | accepted | scaffolding |
| 244 | `neutron_flux_at_first_wall` | no | derived | 1 | accepted | accepted | scaffolding |
| 245 | `neutron_source_rate_due_to_beam_beam_fusion` | no | derived | 1 | accepted | accepted | scaffolding |
| 246 | `neutron_source_rate_due_to_thermal_fusion` | no | derived | 2 | accepted | accepted | scaffolding |
| 247 | `nitrogen_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 248 | `non_axisymmetric_magnetic_field` | no | derived | 4 | accepted | accepted | scaffolding |
| 249 | `normalized_effective_particle_energy` | no | derived | 1 | accepted | accepted | scaffolding |
| 250 | `normalized_gyrocenter_perturbed_current_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 251 | `normalized_particle_perturbed_current_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 252 | `normalized_particle_perturbed_energy` | no | derived | 1 | accepted | accepted | scaffolding |
| 253 | `normalized_particle_perturbed_pressure` | no | derived | 1 | accepted | accepted | scaffolding |
| 254 | `normalized_perturbed_parallel_magnetic_field_magnitude` | no | derived | 1 | accepted | accepted | scaffolding |
| 255 | `normalized_perturbed_pressure` | no | derived | 1 | accepted | accepted | scaffolding |
| 256 | `opacity` | no | derived | 2 | accepted | accepted | scaffolding |
| 257 | `outer_squareness_of_flux_surface` | no | derived | 2 | accepted | accepted | scaffolding |
| 258 | `outer_squareness_of_plasma_boundary` | no | derived | 2 | accepted | accepted | scaffolding |
| 259 | `outline` | no | derived | 2 | pending | None | scaffolding |
| 260 | `oxygen_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 261 | `parallel_current_density_at_constraint_position` | no | derived | 1 | accepted | accepted | scaffolding |
| 262 | `parallel_electric_field_at_separatrix` | no | derived | 1 | accepted | accepted | scaffolding |
| 263 | `parallel_wave_electric_field` | no | derived | 2 | accepted | accepted | scaffolding |
| 264 | `particle_count` | no | derived | 2 | accepted | accepted | scaffolding |
| 265 | `particle_diffusivity` | no | derived | 4 | accepted | accepted | scaffolding |
| 266 | `particle_energy` | no | derived | 2 | accepted | accepted | scaffolding |
| 267 | `particle_flux_at_wall_due_to_pumping` | no | none | 2 | superseded | drafted | scaffolding |
| 268 | `particle_flux_at_wall_due_to_surface_emission` | no | derived | 3 | accepted | accepted | scaffolding |
| 269 | `particle_flux_limiter_coefficient` | no | derived | 6 | accepted | accepted | scaffolding |
| 270 | `particle_source_rate` | no | derived | 4 | superseded | pending | scaffolding |
| 271 | `particle_temperature` | no | derived | 1 | pending | None | scaffolding |
| 272 | `particle_torque_density_due_to_coulomb_collisions_with_electrons` | no | derived | 1 | accepted | accepted | scaffolding |
| 273 | `particle_torque_due_to_j_cross_b_force` | no | derived | 1 | accepted | accepted | scaffolding |
| 274 | `permeability_of_ferritic_element` | no | derived | 0 | accepted | accepted | scaffolding-shaped, leaf (no children) |
| 275 | `perpendicular_normalized_perturbed_pressure` | no | derived | 1 | accepted | accepted | scaffolding |
| 276 | `perturbed_electrostatic_potential_amplitude` | no | derived | 1 | accepted | accepted | scaffolding |
| 277 | `perturbed_plasma_magnetic_field` | no | derived | 3 | accepted | accepted | scaffolding |
| 278 | `perturbed_plasma_magnetic_vector_potential` | no | derived | 3 | accepted | accepted | scaffolding |
| 279 | `perturbed_plasma_velocity` | no | derived | 3 | accepted | accepted | scaffolding |
| 280 | `perturbed_pressure_bessel_1` | no | derived | 1 | accepted | accepted | scaffolding |
| 281 | `perturbed_vacuum_magnetic_field` | no | derived | 3 | accepted | accepted | scaffolding |
| 282 | `perturbed_vacuum_magnetic_vector_potential` | no | derived | 1 | accepted | accepted | scaffolding |
| 283 | `pfirsch_schlueter_current_density` | no | derived | 5 | accepted | accepted | scaffolding |
| 284 | `pfirsch_schlueter_current_density_due_to_diamagnetic_drift` | no | derived | 1 | accepted | accepted | scaffolding |
| 285 | `phase_of_ion_cyclotron_heating_antenna` | no | none | 2 | superseded | accepted | scaffolding |
| 286 | `plasma_heating_power` | no | derived | 1 | accepted | accepted | scaffolding |
| 287 | `plasma_internal_energy` | no | derived | 1 | accepted | accepted | scaffolding |
| 288 | `plasma_magnetic_field` | no | derived | 1 | accepted | accepted | scaffolding |
| 289 | `plasma_magnetic_vector_potential` | no | derived | 1 | accepted | accepted | scaffolding |
| 290 | `plasma_mass_density` | no | derived | 3 | accepted | accepted | scaffolding |
| 291 | `plasma_mass_density_imaginary_part` | no | derived | 1 | accepted | accepted | scaffolding |
| 292 | `plasma_mass_density_real_part` | no | derived | 1 | accepted | accepted | scaffolding |
| 293 | `plasma_momentum` | no | derived | 1 | accepted | accepted | scaffolding |
| 294 | `plasma_momentum_diffusivity` | no | derived | 3 | accepted | accepted | scaffolding |
| 295 | `plasma_momentum_flux` | no | derived | 1 | accepted | accepted | scaffolding |
| 296 | `plasma_pressure_imaginary_part` | no | derived | 1 | accepted | accepted | scaffolding |
| 297 | `plasma_pressure_real_part` | no | derived | 1 | accepted | accepted | scaffolding |
| 298 | `plasma_temperature` | no | derived | 2 | accepted | accepted | scaffolding |
| 299 | `plasma_temperature_imaginary_part` | no | derived | 1 | accepted | accepted | scaffolding |
| 300 | `plasma_temperature_real_part` | no | derived | 1 | accepted | accepted | scaffolding |
| 301 | `plasma_velocity` | no | derived | 5 | accepted | accepted | scaffolding |
| 302 | `poloidal_perturbed_plasma_magnetic_field` | no | derived | 2 | accepted | accepted | scaffolding |
| 303 | `poloidal_perturbed_plasma_magnetic_vector_potential` | no | derived | 1 | accepted | accepted | scaffolding |
| 304 | `poloidal_perturbed_plasma_velocity` | no | derived | 2 | accepted | accepted | scaffolding |
| 305 | `poloidal_perturbed_vacuum_magnetic_field` | no | derived | 2 | accepted | accepted | scaffolding |
| 306 | `power_at_divertor_target_due_to_recombination` | no | derived | 2 | accepted | accepted | scaffolding |
| 307 | `power_at_inside_flux_surface` | no | derived | 3 | accepted | accepted | scaffolding |
| 308 | `power_density_due_to_collisions` | no | derived | 3 | accepted | accepted | scaffolding |
| 309 | `power_density_due_to_fusion` | no | derived | 2 | accepted | accepted | scaffolding |
| 310 | `power_due_to_collisions` | no | derived | 2 | accepted | accepted | scaffolding |
| 311 | `power_due_to_conduction` | no | derived | 3 | accepted | accepted | scaffolding |
| 312 | `power_due_to_convection` | no | derived | 3 | accepted | accepted | scaffolding |
| 313 | `power_due_to_recombination` | no | derived | 2 | accepted | accepted | scaffolding |
| 314 | `power_of_beam_tracing_beam` | no | derived | 2 | accepted | accepted | scaffolding |
| 315 | `power_of_breeder_blanket_module` | no | derived | 1 | accepted | accepted | scaffolding |
| 316 | `power_of_divertor_due_to_recombination` | no | derived | 2 | accepted | accepted | scaffolding |
| 317 | `power_of_lower_hybrid_antenna` | no | derived | 3 | accepted | accepted | scaffolding |
| 318 | `pressure_at_pedestal_top` | no | derived | 1 | accepted | accepted | scaffolding |
| 319 | `pressure_bessel_1` | no | derived | 1 | accepted | accepted | scaffolding |
| 320 | `pressure_over_scrape_off_layer` | no | derived | 2 | accepted | accepted | scaffolding |
| 321 | `radial_perturbed_plasma_magnetic_field` | no | derived | 2 | accepted | accepted | scaffolding |
| 322 | `radial_perturbed_plasma_magnetic_vector_potential` | no | derived | 1 | accepted | accepted | scaffolding |
| 323 | `radial_perturbed_plasma_velocity` | no | derived | 2 | accepted | accepted | scaffolding |
| 324 | `radial_perturbed_vacuum_magnetic_field` | no | derived | 2 | accepted | accepted | scaffolding |
| 325 | `radial_perturbed_vacuum_magnetic_vector_potential` | no | derived | 2 | accepted | accepted | scaffolding |
| 326 | `radiated_power_of_breeder_blanket_module` | no | derived | 2 | accepted | accepted | scaffolding |
| 327 | `radius_of_ferritic_element` | no | derived | 2 | superseded | pending | scaffolding |
| 328 | `radius_of_passive_loop` | no | derived | 2 | accepted | accepted | scaffolding |
| 329 | `radius_of_plasma_filament` | no | derived | 2 | accepted | accepted | scaffolding |
| 330 | `radius_of_poloidal_field_coil` | no | derived | 2 | accepted | accepted | scaffolding |
| 331 | `ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_magnetic_field_magnitude` | no | derived | 1 | accepted | accepted | scaffolding |
| 332 | `ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_major_radius` | no | derived | 1 | accepted | accepted | scaffolding |
| 333 | `reference_magnetic_field` | no | derived | 1 | accepted | accepted | scaffolding |
| 334 | `reference_phase` | no | derived | 1 | accepted | accepted | scaffolding |
| 335 | `refractive_index` | no | derived | 2 | accepted | accepted | scaffolding |
| 336 | `right_hand_circularly_polarized_wave_electric_field_amplitude` | no | derived | 1 | accepted | accepted | scaffolding |
| 337 | `rotation_frequency_due_to_e_cross_b_drift` | no | derived | 1 | accepted | accepted | scaffolding |
| 338 | `runaway_electron_convection_velocity` | no | derived | 1 | accepted | accepted | scaffolding |
| 339 | `runaway_electron_current_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 340 | `runaway_electron_diffusivity` | no | derived | 1 | accepted | accepted | scaffolding |
| 341 | `runaway_electron_particle_flux` | no | derived | 1 | accepted | accepted | scaffolding |
| 342 | `source_due_to_diamagnetic_drift` | no | derived | 1 | accepted | accepted | scaffolding |
| 343 | `source_rate_due_to_beam_beam_fusion` | no | derived | 1 | accepted | accepted | scaffolding |
| 344 | `source_rate_due_to_gas_injection` | no | derived | 11 | accepted | accepted | scaffolding |
| 345 | `source_rate_due_to_injection` | no | derived | 2 | accepted | accepted | scaffolding |
| 346 | `source_rate_due_to_thermal_fusion` | no | derived | 1 | accepted | accepted | scaffolding |
| 347 | `spectral_signal_to_noise_ratio_of_spectrometer_channel` | yes | dd, derived | 1 | accepted | accepted | **not scaffolding — DD-produced** |
| 348 | `square_of_magnetic_field_magnitude` | no | derived | 3 | accepted | reviewed | scaffolding |
| 349 | `square_of_major_radius` | no | derived | 2 | accepted | accepted | scaffolding |
| 350 | `square_of_toroidal_flux_coordinate_gradient_magnitude` | no | derived | 3 | accepted | accepted | scaffolding |
| 351 | `squareness_of_flux_surface` | no | derived | 2 | accepted | accepted | scaffolding |
| 352 | `squareness_of_plasma_boundary` | no | derived | 2 | accepted | accepted | scaffolding |
| 353 | `straight_field_line_angle` | no | derived | 1 | accepted | accepted | scaffolding |
| 354 | `stray_breakdown_magnetic_field` | no | derived | 1 | accepted | accepted | scaffolding |
| 355 | `suprathermal_neutral_internal_state_pressure` | no | derived | 1 | accepted | accepted | scaffolding |
| 356 | `temperature_at_divertor_target` | no | derived | 2 | accepted | accepted | scaffolding |
| 357 | `temperature_at_magnetic_axis` | no | derived | 2 | accepted | accepted | scaffolding |
| 358 | `temperature_over_scrape_off_layer` | no | derived | 1 | accepted | accepted | scaffolding |
| 359 | `thermal_electron_torque_density_due_to_collisions` | no | derived | 4 | accepted | accepted | scaffolding |
| 360 | `thermal_electron_torque_due_to_collisions` | no | derived | 1 | accepted | reviewed | scaffolding |
| 361 | `thermal_ion_charge_state_energy_velocity_due_to_convection` | no | derived | 1 | accepted | accepted | scaffolding |
| 362 | `thermal_ion_charge_state_torque_density_due_to_collisions` | no | derived | 4 | accepted | accepted | scaffolding |
| 363 | `thermal_ion_energy` | no | derived | 2 | accepted | accepted | scaffolding |
| 364 | `thermal_ion_energy_diffusion_coefficient` | no | derived | 1 | accepted | accepted | scaffolding |
| 365 | `thermal_ion_torque_density_due_to_collisions` | no | derived | 4 | accepted | accepted | scaffolding |
| 366 | `thermal_ion_torque_density_due_to_thermalization` | no | derived | 1 | accepted | accepted | scaffolding |
| 367 | `thermal_ion_torque_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 368 | `thermal_plasma_internal_energy` | no | derived | 1 | accepted | accepted | scaffolding |
| 369 | `time_averaged_normalized_perturbed_parallel_magnetic_field_magnitude` | no | derived | 1 | accepted | accepted | scaffolding |
| 370 | `time_derivative_of_electron_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 371 | `toroidal_current_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 372 | `toroidal_flux_coordinate_gradient` | no | derived | 1 | accepted | accepted | scaffolding |
| 373 | `toroidal_flux_coordinate_gradient_magnitude` | no | derived | 2 | accepted | accepted | scaffolding |
| 374 | `toroidal_ion_velocity_at_plasma_boundary` | no | derived | 1 | accepted | accepted | scaffolding |
| 375 | `toroidal_perturbed_plasma_magnetic_field` | no | derived | 2 | accepted | accepted | scaffolding |
| 376 | `toroidal_perturbed_plasma_magnetic_vector_potential` | no | derived | 1 | accepted | accepted | scaffolding |
| 377 | `toroidal_perturbed_plasma_velocity` | no | derived | 2 | accepted | accepted | scaffolding |
| 378 | `toroidal_perturbed_vacuum_magnetic_field` | no | derived | 2 | accepted | accepted | scaffolding |
| 379 | `torque_density` | no | derived | 7 | accepted | accepted | scaffolding |
| 380 | `torque_density_due_to_collisions` | no | derived | 3 | accepted | accepted | scaffolding |
| 381 | `torque_density_due_to_coulomb_collisions_with_electrons` | no | derived | 1 | accepted | accepted | scaffolding |
| 382 | `torque_density_due_to_j_cross_b_force` | no | derived | 4 | accepted | accepted | scaffolding |
| 383 | `torque_due_to_collisions` | no | derived | 4 | accepted | accepted | scaffolding |
| 384 | `torque_due_to_j_cross_b_force` | no | derived | 1 | accepted | accepted | scaffolding |
| 385 | `torque_due_to_neutral_beam_shinethrough` | no | derived | 1 | accepted | accepted | scaffolding |
| 386 | `total_current_density` | no | derived | 5 | accepted | accepted | scaffolding |
| 387 | `total_fast_ion_pressure` | no | derived | 2 | accepted | accepted | scaffolding |
| 388 | `total_hydrogenic_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 389 | `total_ion_energy_convection_velocity` | no | derived | 1 | accepted | accepted | scaffolding |
| 390 | `total_ion_energy_flux` | no | derived | 3 | accepted | accepted | scaffolding |
| 391 | `total_ion_temperature` | no | derived | 2 | accepted | accepted | scaffolding |
| 392 | `total_momentum_convection_velocity` | no | derived | 2 | accepted | accepted | scaffolding |
| 393 | `total_momentum_diffusion_coefficient` | no | derived | 1 | accepted | accepted | scaffolding |
| 394 | `total_momentum_diffusivity` | no | derived | 1 | accepted | accepted | scaffolding |
| 395 | `total_momentum_flux` | no | derived | 2 | accepted | accepted | scaffolding |
| 396 | `total_momentum_source` | no | derived | 1 | accepted | accepted | scaffolding |
| 397 | `total_plasma_momentum` | no | derived | 1 | accepted | accepted | scaffolding |
| 398 | `total_runaway_electron_current` | no | derived | 1 | accepted | accepted | scaffolding |
| 399 | `total_thermal_ion_energy_diffusion_coefficient` | no | derived | 1 | accepted | accepted | scaffolding |
| 400 | `total_thermal_plasma_internal_energy` | no | derived | 1 | accepted | accepted | scaffolding |
| 401 | `total_xenon_fraction` | no | derived | 1 | accepted | accepted | scaffolding |
| 402 | `trapped_current_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 403 | `trapped_fast_current_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 404 | `trapped_fast_ion_torque_density_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 405 | `trapped_thermal_electron_torque_density_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 406 | `trapped_thermal_ion_torque_density_due_to_collisions` | no | derived | 1 | accepted | accepted | scaffolding |
| 407 | `trapped_torque_density` | no | derived | 1 | accepted | accepted | scaffolding |
| 408 | `trapped_torque_density_due_to_j_cross_b_force` | no | derived | 1 | accepted | accepted | scaffolding |
| 409 | `triangularity_of_flux_surface` | no | derived | 2 | accepted | accepted | scaffolding |
| 410 | `tritium_density` | no | derived | 3 | accepted | accepted | scaffolding |
| 411 | `tritium_velocity` | no | derived | 1 | accepted | accepted | scaffolding |
| 412 | `tungsten_density` | no | derived | 2 | accepted | accepted | scaffolding |
| 413 | `vacuum_magnetic_field` | no | derived | 2 | accepted | accepted | scaffolding |
| 414 | `vacuum_magnetic_vector_potential` | no | derived | 4 | accepted | accepted | scaffolding |
| 415 | `vector_potential` | no | derived | 2 | accepted | accepted | scaffolding |
| 416 | `velocity_due_to_convection` | no | derived | 2 | accepted | accepted | scaffolding |
| 417 | `velocity_due_to_e_cross_b_drift` | no | derived | 1 | accepted | accepted | scaffolding |
| 418 | `voltage_of_ion_cyclotron_heating_antenna` | no | derived | 1 | accepted | accepted | scaffolding |
| 419 | `volumetric_force` | no | derived | 3 | accepted | accepted | scaffolding |
| 420 | `wave_current_of_antenna_strap` | no | derived | 2 | accepted | accepted | scaffolding |
| 421 | `wave_electric_field` | no | derived | 8 | accepted | accepted | scaffolding |
| 422 | `wave_mode_number` | no | derived | 1 | accepted | accepted | scaffolding |
| 423 | `wave_power` | no | derived | 1 | accepted | accepted | scaffolding |
| 424 | `wave_voltage` | no | derived | 1 | accepted | accepted | scaffolding |
| 425 | `wavelength_of_visible_camera` | no | derived | 2 | accepted | accepted | scaffolding |
| 426 | `weight_of_interferometer_beam` | no | derived | 1 | accepted | accepted | scaffolding |
| 427 | `width_of_spectrometer_channel` | no | derived | 1 | accepted | accepted | scaffolding |
| 428 | `xenon_density` | no | derived | 2 | accepted | accepted | scaffolding |

## Is the label derivable from the absence of a producer edge?

The plan asks whether the field can be replaced by the topology — whether
"derived" is what the absence of a `PRODUCED_NAME` edge to a `dd` source
*means*. Both directions were measured.

**Direction 1 — every derived name as a proportion of the names with no dd
producer (precision):**

```cypher
// no dd-typed producer, and not marked derived  →  the counterexamples
MATCH (sn:StandardName)
WHERE coalesce(sn.origin, '<null>') <> 'derived'
  AND NOT (sn)<-[:PRODUCED_NAME]-(:StandardNameSource {source_type: 'dd'})
RETURN coalesce(sn.origin, '<null>') AS origin, count(sn) AS n
```

| Result | n |
|---|---|
| `origin` null, no dd producer | 1,425 |
| `origin='pipeline'`, no dd producer | 990 |
| **counterexamples** | **2,415** |

Of the 2,842 names with no dd producer (5,130 − 2,288), only 427 carry the
label. **Precision 0.15.** The absence is not the label.

**Direction 2 — every name with no dd producer, as a proportion of the derived
names (recall):**

```cypher
MATCH (sn:StandardName {origin: 'derived'})
WHERE NOT (sn)<-[:PRODUCED_NAME]-(:StandardNameSource {source_type: 'dd'})
RETURN count(sn) AS n
```

**427 of 428** — and the single miss is the falsified name above. So the
absence is *necessary* (recall 0.998) but nowhere near *sufficient*: the field
must not be dropped on the assumption that it is recomputable from the edge: it
is not, and 2,415 names would be relabelled if the rule were adopted.

**The two substitutes that were tested as an alternative, and both fail:**

| Candidate substitute | Derived names it covers | Names it would newly call derived | Disagreement |
|---|---|---|---|
| `(\:StandardNameSource {source_type:'derived'})-[:PRODUCED_NAME]->(sn)` | 422 / 428 | 115 | **121** (precision 0.79) — the plan recorded 341 (18 + 323); the derived-parent reconcile work has since closed two thirds of it |
| `'derived' IN sn.source_types` | **0 / 428** | 0 | 115-and-6 both unaddressed; the property never holds this value anywhere in the graph |

```cypher
// the property, which nothing populates with 'derived'
MATCH (sn:StandardName {origin: 'derived'})
WHERE 'derived' IN coalesce(sn.source_types, [])
RETURN count(sn) AS n     // 0
```

So: the *edge* is the only instrument that carries the meaning, it is one step
short of exact, and the scalar binding the plan calls "the source binding" does
not exist for this value. **§7a stage 2 is not a refactor.** Repointing the 52
read lines at the `derived`-typed producer edge changes behaviour for 121
names — including the two whose posture is safety-relevant
(`attachment_audit.py:2446`, `graph_ops.py:26713`) — and leaves 6 derived names
with no source of any kind to point at.

## Reader branches on the label (non-deletion)

Every code path that branches on the label for a decision *other than deletion*,
from the committed census `derived-origin-readers.md` (census at HEAD
`4c9d1dabb4ab0afb679ccf15b85eb552e585169c`, 2026-09-18): **52 distinct read
lines across 34 decision paths**. The census's own two-clause form is kept;
its line anchors are re-derived at HEAD and must not be taken from the
2026-09-08 revision. The 2 deletion reads (`graph_ops.py:3810,4160`) are §7a's
retirement target and are out of scope here.

| File:lines | Decision the label drives |
|---|---|
| `cli/sn.py:398,430,441,485,496` | `_compute_pool_progress`: derived parent is docs-eligible without a name score; excluded from name-review and refinement counts |
| `standard_names/export.py:1099` | `_run_gate_c`: label bypasses the ordinary name-axis score gate — **sets catalog eligibility** |
| `standard_names/edit.py:745` | `_stranded_rename_refusal`: refuses a manual rename as deterministic from children |
| `standard_names/parents.py:239` | `is_single_child_shadow`: a derived child cannot prove its sole parent a redundant shadow |
| `standard_names/parents.py:400,425` | `_replay_described_parent_authorities`: structural-authority backfill cohort |
| `standard_names/review/pipeline.py:621,631` | `_fetch_review_derived_children`: structural-peel explanation enters review prompts |
| `standard_names/workers.py:4835` | `validate_name_candidate`: appends the derived-parent structural audit |
| `standard_names/workers.py:8436` | `process_review_name_batch`: routes low-similarity derived rows to docs refinement |
| `standard_names/workers.py:9123` | `_enrich_for_docs_gen`: grounds documentation generation on live child names and suppresses placeholder child prose |
| `standard_names/workers.py:9966` | `_load_docs_review_parent_children`: child grounding for docs review |
| `standard_names/workers.py:9973` | enrich-parents claim winner recheck (`origin: 'derived'` inline) |
| `standard_names/workers.py:10616` | `process_refine_docs_batch`: live child context in docs refinement |
| `standard_names/attachment_audit.py:2446` | `detach_one_attachment`: **can strip a DD realization's last-source protection** |
| `standard_names/graph_ops.py:1120` | `_MANIFEST_DRAIN_PLAN_QUERY`: propagates a bounded source drain scope to derived ancestors |
| `standard_names/graph_ops.py:2864` | `_is_single_child_shadow` (batch-aware) |
| `standard_names/graph_ops.py:3703,3753` | derived-parent candidate admission (seedable + legacy paths) |
| `standard_names/graph_ops.py:4731` | `normalize_derived_parent_lifecycle` unit-gap selector |
| `standard_names/graph_ops.py:7722` | `persist_generated_name_winners`: source drain to derived ancestors |
| `standard_names/graph_ops.py:14323,14392` | `reconcile_reviewable_name_stage`: excludes derived from ordinary name review |
| `standard_names/graph_ops.py:16744,18034` | descriptionless-repair and `REFINE_NAME_ELIGIBILITY_WHERE` exclusions |
| `standard_names/graph_ops.py:17518,24968,25268,25970` | review/refinement eligibility and claim selectors: ineligible for name review, admitted to paid docs work without a score |
| `standard_names/graph_ops.py:18552,18706,18832,18836` | refined-name persistence: derived identity not reusable as a refinement target |
| `standard_names/graph_ops.py:19521,19858` | generated supersession: derived predecessors excluded from automatic source migration |
| `standard_names/graph_ops.py:22476,22549,22694` | exhausted-orphan supersession: derived rows refused |
| `standard_names/graph_ops.py:25985,25991` | `pool_pending_counts`: docs-eligible without a score, excluded from backlogs |
| `standard_names/graph_ops.py:26146` | `_verify_enrich_parents_claim_winners` |
| `standard_names/graph_ops.py:26195` | `claim_enrich_parents_batch`: paid description enrichment cohort |
| `standard_names/graph_ops.py:26481,26629,26713,26764` | structural-authority persistence and acceptance — **26713 can structurally accept a DD-produced identity** |
| `standard_names/graph_ops.py:27010` | `classify_orphan_parent_source_candidates`: structural-admission validation applies only to derived-labelled provenance-orphan parents |
| `standard_names/signed_manifest.py:371` | unit-gap structural signing query (`unit: '1', origin: 'derived'`) |

Every one of these must read the topology — the producer edge, the
`StandardNameSource` binding — before the field can be dropped without
changing behaviour silently. Two of them change **safety posture** when the
label is false, and this node measured exactly one name for which it is:
`spectral_signal_to_noise_ratio_of_spectrometer_channel` is, today, a name that
`attachment_audit.py:2446` may treat as structurally anchored and
`graph_ops.py:26713` may structurally accept, while it is in fact DD-produced.

<figure>
  <svg viewBox="0 0 780 300" width="100%" role="img"
       aria-label="Top: 428 names carry origin derived. Middle: the two candidate substitute instruments drawn to the same scale as the 5,130-name surface, showing that the absence of a dd producer has precision 0.15 and the derived-typed producer edge 0.79. Bottom: the source_types property never holds the value.">
    <g font-family="system-ui, -apple-system, sans-serif" font-size="11" fill="#1a1a1c">
      <text x="24" y="32">carry origin='derived' — 428</text>
      <rect x="24" y="36" width="58.4" height="16" fill="#4a6fa5"/>
      <rect x="24" y="36" width="3" height="16" fill="#b3261e"/>
      <text x="90" y="48" fill="#54545a">428 of 5,130 live names</text>
      <text x="24" y="66" font-size="10" fill="#b3261e">one of these carries a direct dd producer — the label is false for it</text>

      <text x="24" y="90">have NO dd-typed producer — 2,842</text>
      <rect x="24" y="94" width="58.3" height="16" fill="#4a6fa5"/>
      <rect x="82.3" y="94" width="329.5" height="16" fill="#c9c9cf"/>
      <text x="420" y="106" fill="#b3261e">precision 0.15 — only 427 are labelled derived</text>
      <text x="24" y="124" font-size="10" fill="#54545a">grey 2,415: origin null 1,425 · pipeline 990 — the label is not derivable from this absence</text>

      <text x="24" y="148">carry a derived-typed producer — 537</text>
      <rect x="24" y="152" width="57.6" height="16" fill="#4a6fa5"/>
      <rect x="81.6" y="152" width="15.7" height="16" fill="#c9c9cf"/>
      <text x="105" y="164" fill="#2e7d32">precision 0.79 — 422 derived, 115 not</text>
      <text x="24" y="182" font-size="10" fill="#54545a">and 6 derived names carry no producer edge of any kind for a reader to fall back to</text>

      <text x="24" y="206">carry 'derived' in the source_types property — 0</text>
      <rect x="24" y="210" width="8" height="16" fill="#b3261e"/>
      <text x="40" y="222" fill="#b3261e">the scalar binding cannot stand in for the field</text>

      <text x="24" y="252" font-size="10" fill="#54545a">Bars share one scale: 5,130 live names = 700 px. The 1 falsified name is drawn at a minimum 3 px, not to scale.</text>
      <text x="24" y="268" font-size="10" fill="#54545a">Blue — names the instrument describes correctly. Grey — names it would newly classify. Red — a name the label gets wrong.</text>
    </g>
  </svg>
  <figcaption>The three candidate sources of "derived", drawn against the same
  5,130-name surface. Only the edge carries the meaning, and it is 121 names
  short of exact.</figcaption>
</figure>

## Method, controls, and the placement exception

Each measurement is a bounded read: a named row set (the 428 or an
explicitly-scoped superset), a count-only projection, and no heavy local
compute alongside. The longest query took 0.06 s. Logs and exact query text:
`/tmp/pidp_derived_truthfulness.py`, `/tmp/pidp_derived_anomaly.py`,
`/tmp/pidp_dd_status.py` with output JSON beside them.

**Placement.** This is live-graph work, which the repo's AGENTS.md excepts from
compute-node placement: `GraphClient()` resolves a loopback tunnel endpoint
inside a SLURM step and fails before reaching the graph. It ran on the login
node under the exception's bounds, and is disclosed here for exactly the reason
the exception requires.

**Controls, because a zero is a claim about the instrument.** Four zeros appear
above and each is bracketed against something known present:

| The zero | Its control |
|---|---|
| 0 derived names with an `extracted` dd producer | the same predicate without the status clause returns **1**, and the status census shows 2,178 dd sources at `attached` — the instrument can see dd producers |
| 0 names with `'derived' IN source_types` | the same predicate for the `dd` value returns 1,531 names, so the property is readable and is populated for other values |
| 0 incoming `HAS_STRUCTURAL_AUTHORITY` on derived names | the outgoing direction returns **309**, so the edge exists and the derived names hold it |
| 1 derived name with a dd producer | the 2,288-name / 4,954-edge dd population is non-empty, and the one name is listed with both of its producer nodes |

The `dd`-status census needed to attribute the published zero is
`pidp_dd_status.py`'s own output, not an inference: `extracted` is a 1,250-source
slice of the dd population, and `attached` — where the falsified name's producer
sits — is 2,178.

## Follow-ons (outside this node's fence)

1. **§7a stage 2 is a behaviour change for 121 names, not a refactor.** The
   plan should absorb that before ordering stage 3, and the 6 producer-less and
   7 childless derived names are the ones no edge can classify.
2. **§7a stage 1's recorded zero holds only under a status filter.** The plan's
   "reclassification remainder: 0" should state its predicate or be restated as 1.
3. **`spectral_signal_to_noise_ratio_of_spectrometer_channel` needs a
   disposition** — it is simultaneously a DD realisation and a derived parent,
   and its `source_types` property is empty despite two producers. This is a §8
   source-binding case, and it is one name.
4. **The `dd` producer's batch (`focus`) and the derived parent's
   (`derived_parent`) never re-read each other's claim on the name.** Where a
   DD realisation attaches to a name another pass has classified as an
   abstraction, nothing reconciles the two.
5. `plan-figure-audit.md:148` ("§7 reclassification remainder | 0") is the
   figure this node's second verdict table corrects.