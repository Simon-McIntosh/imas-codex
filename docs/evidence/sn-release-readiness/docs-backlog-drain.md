# Accepted-name documentation backlog drain

## Outcome

The documentation drain reduced the fresh global census of accepted Standard Names without accepted documentation from **257 exact `StandardName.id` identities to 56**, while spending **USD 57.099010 of the hard USD 80.00 ceiling**. The ceiling therefore retained USD 22.900990. This is substantial progress, but it is **not a publishable-state pass**: the 56 remaining identities are still excluded by the documentation gate, so catalog release remains on hold.

| Measure | Before | After |
|---|---:|---:|
| Accepted names | 2,335 | 2,326 |
| Accepted names without `docs_stage = 'accepted'` | **257** | **56** |
| Distinct backlog `StandardName.id` values | **257** | **56** |
| `pending` | 131 | 13 |
| `drafted` | 64 | 0 |
| `reviewed` | 50 | 35 |
| `exhausted` | 12 | 8 |

The before census proved property coverage before trusting the result: 4,666 `StandardName` candidates, 4,666 with `id`, 4,666 with `name_stage`, and 4,662 with `docs_stage`; all 2,335 accepted rows carried an `id`. The after census repeated the proof: 4,656 candidates, 4,656 with `id`, 4,656 with `name_stage`, and 4,652 with `docs_stage`; all 2,326 accepted rows carried an `id`. Both backlog counts are row counts and `count(DISTINCT s.id)` counts, and they agree exactly.

## Bar in force

This campaign used the live production documentation path, not a reduced generation-only proxy. The current prompts require a definition, a governing equation where one is meaningful, prose definitions for symbols, semantic scope and exclusions, essential relationships, a sign convention only when the quantity is COCOS-sensitive, and resolvable inline Standard Name cross-references. The hard maximum-word gate has been removed; the minimum-word floor remains. Promotion still requires the documentation quorum to accept the document at the configured 0.85 score threshold.

The linked quality record establishes that this production-enriched path is suitable for the drain: its standing production-faithful artifact-free means were 0.9621 on the holdout and 0.9434 off-holdout, against 0.8876 for the catalog reference. Those figures qualify the generation path; they do not override the per-identity quorum or convert a terminal failure into acceptance. The current deterministic instrument retains six checks—defining equation, symbol definitions, resolving relationship link, conditional sign convention, link hygiene, and minimum word count—while dimensional consistency and sign diagnostics retain their stated evaluability qualifications.

## Exact scope and paid execution

The mutation scope came from a fresh census matching `(:StandardName)` rows with `name_stage = 'accepted'` and `coalesce(docs_stage, 'pending') <> 'accepted'`, ordered and addressed exclusively by `s.id`. An exact-name dry run admitted all 257 identities and performed no writes. The 12 initially exhausted documents were then passed through the sanctioned documentation-rescore staging primitive, preserving their text and review history while returning the same identities to the review/refine loop. No name was hand-accepted.

The paid command used `sn run --docs-only`, one `--name` selector for every identity in the frozen 257-row census, `--min-score 0.85`, `--cost-limit 80`, `--time 35`, and `--skip-global-maintenance`. Its durable `SNRun` is `755ebdb4-54b6-4dde-815d-4179743d91ee`. It ran from 2026-08-25 02:29:27Z to 03:01:23Z, then received a graceful interrupt after the scoped actionable pools had drained to terminal results. The durable run is accordingly marked `interrupted`, not falsely relabeled complete; its spend is exact and the post-census found zero claimed backlog identities.

| Pool | Calls/events | USD |
|---|---:|---:|
| Documentation generation | 131 | 0.624397 |
| Documentation review | 616 | 44.258644 |
| Documentation refinement | 72 | 12.215969 |
| **Total** | **819** | **57.099010** |

The run generated 131 documents, recorded 274 review actions, and performed 72 documentation refinements. One final review could not be accepted from a single completed seat when the other seat was unavailable; it was left in governed non-accepted state rather than weakening the quorum.

## Identity reconciliation and concurrent churn

The global 257-to-56 snapshots are the release-gate measure, but they must not be presented as a purely campaign-attributable delta because the graph changed concurrently. Of the 257 frozen starting identities, **253 still existed after the run**: **210 had accepted documentation** and **43 remained** (`reviewed` 35, `exhausted` 8). Four starting identities were no longer present under the same exact identity:

- `mode_width`
- `perturbed_current_density`
- `perturbed_magnetic_field`
- `thermal_electron_torque_density_due_to_coulomb_collisions_with_electrons`

Conversely, 13 identities in the after backlog were not members of the frozen starting census. All 13 were `pending`:

- `bulk_center_of_mass_velocity`
- `explicit_ion_torque`
- `fast_particle_torque_density_due_to_coulomb_collisions_with_electrons`
- `ion_particle_convection_velocity`
- `kinetic_energy_density`
- `neutral_species_energy_flux`
- `neutron_source_rate_due_to_beam_beam_fusion`
- `parallel_neutral_momentum_diffusion_coefficient`
- `particle_torque_density_due_to_coulomb_collisions_with_electrons`
- `source_rate_due_to_beam_beam_fusion`
- `toroidal_vacuum_magnetic_field`
- `wave_voltage`
- `wavelength_of_visible_camera`

Thus the defensible campaign conversion is 210 of 253 still-visible starting identities, while 257 to 56 is the separately valid before/after release-gate snapshot. The total `StandardName` population changed 4,666 to 4,656 and the accepted population changed 2,335 to 2,326 during the window; those facts are why the two measures are kept distinct.

## Remainder and dispositions

The final 56-row remainder is exactly **13 pending, 35 reviewed, 0 drafted, and 8 exhausted**. The 13 pending rows are the concurrently appearing identities listed above. Of the 35 reviewed rows, 34 record the terminal quorum shortfall “blind seats disagreed and the escalator seat did not resolve them”; `wetted_area_of_divertor` is the remaining reviewed identity, at score 0.600000 and documentation chain length 1. None is silently credited as accepted.

The initial 12 exhausted identities were all explicitly dispositioned by the sanctioned restage. Five cleared the gate on fresh review—`change_in_ion_state_mean_ionisation_potential` (0.85625), `halo_current` (0.91875), `initial_polarization_ellipticity_of_polarimeter_beam` (0.93750), `runaway_electron_critical_momentum_due_to_hot_tail` (0.90000), and `temperature_at_sensor_attachment_point` (0.93125). Seven returned to exhausted. `perturbed_vector_potential`, which entered from another starting stage, also exhausted, giving the final eight below.

Every final exhausted identity is at the documentation refinement cap (`docs_chain_length = 3`) and therefore cannot clear the gate through another unattended loop. Their disposition is **release HOLD**: do not rescore repeatedly and do not hand-accept. Each requires a reasoned, steered documentation correction through the governed edit/review path, or an explicit release-authority decision that changes the gate.

| Exhausted `StandardName.id` | Final score | Disposition |
|---|---:|---|
| `neutron_flux_due_to_fusion` | 0.52500 | HOLD; already a named documentation exhaustion in the release record; steer a content correction and re-review. |
| `parallel_normalized_gyrocenter_momentum_flux_of_gyrokinetic_eigenmode_due_to_perturbed_parallel_vector_potential` | 0.45000 | HOLD; steer a content correction and re-review. |
| `perturbed_gyrocenter_pressure` | 0.75000 | HOLD; steer a content correction and re-review. |
| `perturbed_vector_potential` | 0.58750 | HOLD; newly exhausted in this campaign; steer a content correction and re-review. |
| `poloidal_perturbed_magnetic_flux_at_measurement_position_due_to_wave_particle_interaction` | 0.61875 | HOLD; steer a content correction and re-review. |
| `saturated_permeability_of_ferritic_element` | 0.76250 | HOLD; steer a content correction and re-review. |
| `total_neutral_particle_flux_at_wall_due_to_surface_emission` | 0.50000 | HOLD; steer a content correction and re-review. |
| `total_thermal_plasma_internal_energy` | 0.81875 | HOLD; closest exhausted row to the bar, but still below it and capped; steer a content correction and re-review. |

## Evidence and follow-on work

Durable machine-readable evidence is stored with the worker receipt:

- `before-census.json`: complete 257-identity starting census and coverage proof.
- `exhausted-restage.json`: dry-run and apply receipt for every one of the 12 initially exhausted identities.
- `after-census.json`: complete 56-identity final census, exact starting-cohort reconciliation, run accounting, pool accounting, and exhausted dispositions.

The next concrete action is a governed recovery tranche: first disposition the 34 quorum-shortfall rows without weakening the quorum, then steer corrections for `wetted_area_of_divertor` and the eight exhausted identities, and separately admit the 13 new pending identities from a new exact-ID census. A run warning also exposed an out-of-scope persistence defect: one documentation review produced a free-form DD-gap evidence-rule string that was not a valid `DDGapEvidenceRule` enum value. It did not alter this census or spend result, but its producer/schema mismatch should be repaired before relying on that evidence field.
