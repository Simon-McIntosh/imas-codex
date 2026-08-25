# WEST documentation refresh: second tranche

## Outcome

The second WEST documentation tranche completed through the ordinary
documentation generation, quorum-review, and refinement pools with
`stop_reason=no_eligible_work`. A fresh source-bound census selected exactly
18 unfinished `StandardName.id` identities after excluding all 129 identities
that the first resumed tranche had already accepted. Fourteen of the 18
finished `docs_stage=accepted`; four pre-existing reviewed documents remained
fail-closed because the ordinary refinement eligibility predicate did not
claim them.

| Measure | Recorded result |
|---|---:|
| Exact WEST source identities | **355** |
| Distinct current `StandardName.id` identities bound to those sources | **234** |
| First-tranche accepted identities excluded | **129/129** |
| First-tranche exclusion overlap in the selected cohort | **0** |
| Selected unfinished documentation identities | **18** |
| Capacity-deferred otherwise-eligible identities | **0** |
| Selected identities accepted after the run | **14/18** |
| Distinct refreshed documents failing at least one quorum | **5** |
| Below-bar quorum outcomes | **6** |
| Documentation refinement executions | **8** |
| WEST docs-accepted before → after | **202 → 216** |
| Actual spend / hard tranche ceiling | **USD 4.592470 / USD 60.000000** |
| Running plan spend / authorised ceiling | **USD 36.950842 / USD 150.000000** |

The run made 57 provider calls recorded as `LLMCost` rows. Their summed
`llm_cost` is USD 4.5924700000000005, equal to `SNRun.cost_spent` to
floating-point precision. The hard tranche ceiling retained **USD 55.407530**
headroom, and the plan authority retains **USD 113.049158**.

## Fresh exact-identity census and admission

The census started from the 355 exact `StandardNameSource.id` rows in the
frozen WEST manifest, followed each live `PRODUCED_NAME` edge, and then
deduplicated only on `StandardName.id`. It did not reuse the first tranche's
candidate report as the current cohort. At 2026-08-25T00:55:22.276Z the graph
contained 4,666 `StandardName` nodes, all 4,666 carrying `id`, and exactly zero
carrying the undeclared `name` property. A `StandardName.name` census would
therefore have returned a clean zero and was never used.

The source-bound current cohort contained 234 distinct identities with the
following pre-run documentation states: 202 accepted, 20 pending, 6 reviewed,
5 drafted, and 1 exhausted. The exact receipt from the first resumed tranche
listed 129 accepted identities; all 129 still resolved exactly and still held
accepted documentation. They were subtracted by identity before eligibility
was evaluated, and the selected overlap was independently asserted to be zero.

Eligibility required an accepted, validation-valid, non-terminal name; a
pending, drafted, or reviewed documentation stage; and no live worker or drain
claim. This left 18 identities: 7 pending, 5 drafted, and 6 reviewed. Sizing
used the first tranche's measured USD 22.690535 / 130 selected identities =
**USD 0.1745425769 per selected identity**. At that empirical rate the USD 60
ceiling could admit 343 identities, so all 18 were admitted with a rate-based
expectation of USD 3.141766 and zero capacity deferrals. Runtime leases, rather
than that empirical expectation, enforced the hard USD 60 stop.

The exact scope was `ac0dcdfc-e172-403e-8142-094bc3d41209`. The invocation was
equivalent to:

```text
imas-codex sn run --scope-run-id ac0dcdfc-e172-403e-8142-094bc3d41209 \
  --docs-only --skip-global-maintenance --cost-limit 60 --time 20 \
  --min-score 0.85 --rotation-cap 3
```

It completed as `SNRun` `8be5e846-6d73-4b86-b3ff-850e24572b59` in 554.46 s.
The pools performed 7 generations for USD 0.0546, 20 review operations for
USD 3.0583, and 8 refinements for USD 1.4795. No tail invocation was needed.

## Quorum and refinement evidence

Five distinct documents landed below the 0.85 quorum bar during this tranche:

- `lower_bound_photon_energy` — one below-bar outcome, one refinement, accepted
  at 0.91875.
- `normalized_poloidal_flux_coordinate_of_plasma_boundary` — one below-bar
  outcome at its final rotation, one refinement, accepted at 0.92500.
- `poloidal_plane_cross_sectional_area_of_flux_surface` — one below-bar outcome,
  one refinement, accepted at 0.99375.
- `voltage_of_poloidal_magnetic_field_probe` — two below-bar outcomes, two
  refinements, accepted at 0.88750.
- `width_of_poloidal_field_coil` — one below-bar outcome, one refinement,
  accepted at 0.96250.

Thus the required count is **5 refreshed documents failing quorum**, represented
by **6 below-bar review outcomes**. All five ultimately cleared the same ordinary
quorum; none was hand-accepted. The total of eight refinements is larger because
`flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude`
and `length_of_toroidal_magnetic_field_probe` entered the tranche already
reviewed below the bar, were refined once each, and then accepted at 0.88125 and
0.89375 respectively. Their earlier failures are not counted as failures first
observed during this tranche.

Four other pre-existing reviewed documents received no operation and stayed
reviewed: `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface`,
`parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive`,
`radial_coordinate_of_conductor_cross_section`, and
`spectral_wavelength_of_optical_element`. Keeping them visible is the correct
fail-closed result; the scoped loop did not invent missing legacy resolution
authority or accept them around the quorum.

## WEST docs-accepted census before and after

The exact 234-identity, source-bound census remained cardinality-stable during
the run.

| Observation | Accepted | Pending | Drafted | Reviewed | Exhausted | Total |
|---|---:|---:|---:|---:|---:|---:|
| Before | **202** | 20 | 5 | 6 | 1 | 234 |
| After | **216** | 13 | 0 | 4 | 1 | 234 |

The gain is exactly the 14 selected identities that accepted. All selected
claim fields were clear at reconciliation. Global key coverage remained
4,666/4,666 on `StandardName.id` and 0/4,666 on `StandardName.name` at
2026-08-25T01:08:45.351Z.

Representative accepted identities show the documentation binding and the
current quorum result:

| Standard Name identity | Description | WEST DD source path binding | Final docs score |
|---|---|---|---:|
| `flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude` | Flux-surface average of the squared magnitude of the normalized toroidal-flux-radius gradient; the averaged contravariant radial metric coefficient. | `equilibrium/time_slice/profiles_1d/gm3` | 0.88125 |
| `incident_soft_xray_radiance` | Incident soft-X-ray power radiance per projected detector area and solid angle, integrated over the selected energy band. | `soft_x_rays/channel/brightness` | 0.90000 |
| `length_of_toroidal_magnetic_field_probe` | Non-negative axial extent of the probe coil along its local normal sensing axis, distinct from magnetic-flux sensitivity. | `magnetics/b_field_phi_probe/length` | 0.89375 |
| `lower_bound_photon_energy` | Lower boundary of an X-ray photon-energy band, specifying the minimum energy included in the detection or emission band. | `hard_x_rays/emissivity_profile_1d/lower_bound`; `hard_x_rays/channel/energy_band/lower_bound`; `camera_x_rays/energy_threshold_lower`; `soft_x_rays/channel/energy_band/lower_bound` | 0.91875 |

## Spend ledger

The running pre-tranche ledger recorded by the live authority is reproduced
rather than inferred:

| Prior measured operation | Actual USD |
|---|---:|
| Paired-arm documentation replay | 0.311539 |
| WEST mid-loop documentation completion | 3.932799 |
| First bounded accepted-document refresh | 5.120219 |
| Read-only prior-text supplementary quorum | 0.303280 |
| First resumed accepted-document tranche | 22.690535 |
| **Running total before this tranche** | **32.358372** |
| This second tranche | **4.592470** |
| **Running total after this tranche** | **36.950842 / 150.000000** |

The paired-arm replay is retained because the live plan's own USD 9.67
pre-first-tranche running figure included it; removing it here would silently
change the ledger basis. The tranche itself used only 7.65% of its USD 60 hard
ceiling. Its realised cost was USD 1.450704 above the first-tranche-rate
expectation because five newly refreshed documents failed quorum and one of
them required two refinement cycles; that is measured path mix, not a ceiling
breach.

## Exact identities still deferred

The post-run source-bound WEST census contains **18 exact identities whose
documentation is not accepted**. Four are accepted-name rows retained at
`docs_stage=reviewed`; one is documentation-exhausted; the remaining thirteen
are pending behind a non-accepted or quarantined name lifecycle. They are
listed in full so no count substitutes for identity:

| Standard Name identity | Name stage | Docs stage | Retained gate |
|---|---|---|---|
| `absorbed_coolant_power_of_plant_component_port` | exhausted | pending | Name lifecycle gates documentation. |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | accepted | reviewed | Ordinary refinement eligibility retained it fail-closed at score 0.59375. |
| `initial_polarization_ellipticity_of_polarimeter_beam` | accepted | exhausted | Documentation is terminal at score 0.70625 and needs a sanctioned recovery operation. |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | drafted | pending | Quarantined name lifecycle gates documentation. |
| `ion_species_atomic_number` | exhausted | pending | Name lifecycle gates documentation. |
| `launched_power_of_ion_cyclotron_heating_antenna` | exhausted | pending | Name lifecycle gates documentation. |
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | drafted | pending | Quarantined name lifecycle gates documentation. |
| `lower_bound_hard_xray_peak_width` | superseded | pending | Superseded, quarantined name lifecycle gates documentation. |
| `normal_distance_of_antenna_strap` | exhausted | pending | Name lifecycle gates documentation. |
| `normalized_poloidal_flux_coordinate` | reviewed | pending | Name quorum must accept before documentation can run. |
| `normalized_toroidal_hard_xray_peak_lower_bound_width` | exhausted | pending | Name lifecycle gates documentation. |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | accepted | reviewed | Ordinary refinement eligibility retained it fail-closed at score 0.76875. |
| `plasma_breakdown_time` | exhausted | pending | Name lifecycle gates documentation. |
| `radial_coordinate_of_conductor_cross_section` | accepted | reviewed | Ordinary refinement eligibility retained it fail-closed at score 0.83125. |
| `spectral_wavelength_of_optical_element` | accepted | reviewed | Ordinary refinement eligibility retained it fail-closed at score 0.35625. |
| `surface_temperature_of_plasma_facing_component` | exhausted | pending | Name lifecycle gates documentation. |
| `thermal_energy_confinement_time` | exhausted | pending | Name lifecycle gates documentation. |
| `vertical_coordinate_of_plasma_filament` | exhausted | pending | Name lifecycle gates documentation. |

## Durable receipts and validation

- Fresh exact-identity census and empirical sizing:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T004902923589-n-westtranchetwo/preflight.json`
- Exact scope binding and counter baseline:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T004902923589-n-westtranchetwo/scope-receipt.json`
- Complete pipeline log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T004902923589-n-westtranchetwo/sn-run-live.log`
- Independent post-run graph and spend reconciliation:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T004902923589-n-westtranchetwo/post-run.json`

The pipeline command exited 0. The independent reconciliation asserted exact
one-row resolution for all 234 census identities, exact `id` coverage and zero
`name` coverage, one scoped `SNRun`, equality between `SNRun.cost_spent` and
summed `LLMCost.llm_cost`, spend below USD 60, and no selected identity outside
the first-tranche exclusion.
