# WEST documentation tranche completion

## Outcome

The bounded WEST documentation loop completed with `stop_reason=no_eligible_work`. The source census contained 36 documentation-mid-loop identities. Exact-name preflight admitted 32 of them and rejected four whose name lifecycle was already terminal. The run then exhausted all work claimable through the ordinary documentation generation, quorum-review, and refinement pools without global maintenance or accepted-document regeneration.

| Measure | Recorded result |
|---|---:|
| Source mid-loop identities | 36 |
| Exact-name identities admitted | 32 |
| Terminal name-lifecycle exclusions | 4 |
| Documentation pool operations | 48 |
| Provider calls | 75 |
| Projected admission cost | $75.733766 |
| Authorized cap | $150.000000 |
| Actual spend | $3.932799 |
| Accepted on first review | 18 |
| Reviewed below the bar during this run | 4 |
| Entered documentation refinement | 4 |
| Accepted after refinement | 4 |
| Final `docs_stage=accepted` | 22 |
| Final non-terminal documentation identities | 13 |
| Final terminal documentation identities | 1 |

The 22 acceptances comprise 18 first-review acceptances and four acceptances after refinement. Every identity that entered refinement subsequently passed quorum. The six identities left at `docs_stage=reviewed` were already reviewed before this run; none is a newly landed below-bar result. They remained fail-closed because prior review resolution data did not make them eligible for refinement.

## Scope and admission

The scope was derived from `west-docs-eligible.json`: every resolved WEST identity marked documentation-eligible after the recorded concurrent-refinement exclusion, with already accepted documentation omitted. The exact key was `StandardName.id`. Four of the 36 source identities were removed by exact-name preflight because their name lifecycle was `superseded` or `exhausted`, leaving 32 names passed to `sn run --name`. The live command also used `--docs-only` and `--skip-global-maintenance`; it therefore did not refresh accepted documentation or mutate unrelated global state.

The priced admission was recorded before any model call. It observed 18 identities immediately eligible for document generation, four immediately eligible for review, and no identity immediately eligible for refinement. Six pre-existing reviewed documents were ineligible for refinement because of prior quorum shortfall or absent legacy resolution metadata. Three exact-scope identities had non-terminal but non-accepted name stages and were consequently gated from document generation.

| Admission component | Projected calls or cycles | Projected USD |
|---|---:|---:|
| Document generation | 18.0 | $0.214408 |
| Quorum review | 28.6 | $74.832435 |
| Document refinement | 6.6 | $0.686923 |
| **Total** | — | **$75.733766** |

The projection used exact production-enriched prompt rendering for all 18 pending generations, exact rendered expected reviewer exposure averaged over the four drafted documents, and the current historical refinement cost per identity applied to a prior-smoothed conditional refinement flow. It deliberately priced conditional reviewer exposure conservatively. Admission consumed zero provider calls and projected $75.733766, 50.49% of the $150 cap. The executed loop spent $3.932799, 2.62% of the cap and 5.19% of the projected amount. The difference is expected: the projection reserved conditional review and refinement exposure, while the actual run paid only the paths reached.

## Executed quorum loop

The live run was recorded as `260c881c-8735-49f3-bad1-7b6bddd3c6db`, completed in 482.546 seconds, and stopped because no eligible work remained. Graph `LLMCost` rows are the provider-call and spend authority.

| Pool | Pipeline operations | Provider calls | Actual USD |
|---|---:|---:|---:|
| Generate documentation | 18 | 18 | $0.098076 |
| Review documentation | 26 | 53 | $3.197988 |
| Refine documentation | 4 | 4 | $0.636735 |
| **Total** | **48** | **75** | **$3.932799** |

The 26 review operations made 53 provider calls: two base quorum seats per operation plus one conditional disagreement-resolution call. Eighteen identities passed their first review. Four first reviews landed below the acceptance bar; all four entered refinement, and all four passed their subsequent review. The cost recorded on `SNRun` equals the sum of its 75 `LLMCost` rows.

## Cardinality and stage census

The `StandardName` cardinality was read from the graph immediately before priced admission and after the completed run. It did not change during the operation.

| Observation | UTC timestamp | `StandardName` total | Documentation stages across the 36-source cohort |
|---|---|---:|---|
| Before | 2026-08-23T16:49:16.447Z | 4,666 | pending 25; drafted 4; reviewed 6; exhausted 1 |
| After | 2026-08-23T17:00:08.284Z | 4,666 | accepted 22; pending 7; reviewed 6; exhausted 1 |

The earlier source census recorded a graph total of 4,665. The live pre-run read was therefore required and is the baseline used here; the total remained exactly 4,666 across this executed loop.

## Per-identity lifecycle

| Standard name identity | Name stage at preflight | Documentation stage before → after | Executed outcome or retained gate |
|---|---|---|---|
| `area_of_diagnostic_aperture` | accepted | drafted → accepted | Reviewed below bar; refined; accepted |
| `capacitance` | superseded | pending → pending | Excluded by terminal name lifecycle |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | accepted | reviewed → reviewed | Unchanged; prior quorum shortfall blocks refinement |
| `equilibrium_weight_of_interferometer_beam` | accepted | drafted → accepted | Accepted on first review |
| `flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude` | accepted | reviewed → reviewed | Unchanged; legacy resolution metadata blocks refinement |
| `frequency_of_diagnostic_antenna` | accepted | pending → accepted | Generated; accepted on first review |
| `gap_at_plasma_boundary` | accepted | pending → accepted | Generated; accepted on first review |
| `hard_xray_brightness` | accepted | pending → accepted | Generated; accepted on first review |
| `hard_xray_emissivity` | accepted | pending → accepted | Reviewed below bar; refined; accepted |
| `initial_polarization_ellipticity_of_polarimeter_beam` | accepted | exhausted → exhausted | Unchanged terminal documentation stage |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | drafted | pending → pending | Documentation gated by non-accepted name stage |
| `length_of_toroidal_magnetic_field_probe` | accepted | reviewed → reviewed | Unchanged; legacy resolution metadata blocks refinement |
| `line_averaged_effective_charge` | accepted | pending → accepted | Generated; accepted on first review |
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | drafted | pending → pending | Documentation gated by non-accepted name stage |
| `lower_bound_hard_xray_peak_width` | superseded | pending → pending | Excluded by terminal name lifecycle |
| `normalized_poloidal_flux_coordinate` | reviewed | pending → pending | Documentation gated by non-accepted name stage |
| `normalized_toroidal_flux_coordinate_at_measurement_position` | accepted | pending → accepted | Generated; accepted on first review |
| `normalized_toroidal_hard_xray_peak_lower_bound_width` | exhausted | pending → pending | Excluded by terminal name lifecycle |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | accepted | reviewed → reviewed | Unchanged; prior quorum shortfall blocks refinement |
| `plasma_breakdown_time` | exhausted | pending → pending | Excluded by terminal name lifecycle |
| `poloidal_magnetic_field_at_constraint_position` | accepted | pending → accepted | Reviewed below bar; refined; accepted |
| `poloidal_magnetic_flux_at_flux_surface` | accepted | pending → accepted | Generated; accepted on first review |
| `poloidal_magnetic_flux_at_magnetic_axis` | accepted | pending → accepted | Generated; accepted on first review |
| `poloidal_magnetic_flux_at_plasma_boundary` | accepted | pending → accepted | Generated; accepted on first review |
| `poloidal_magnetic_flux_of_flux_loop` | accepted | drafted → accepted | Accepted on first review |
| `pressure_of_ion_cyclotron_heating_antenna` | accepted | pending → accepted | Generated; accepted on first review |
| `radial_coordinate_of_conductor_cross_section` | accepted | reviewed → reviewed | Unchanged; prior quorum shortfall blocks refinement |
| `radial_coordinate_of_magnetic_axis` | accepted | pending → accepted | Reviewed below bar; refined; accepted |
| `reference_major_radius` | accepted | pending → accepted | Generated; accepted on first review |
| `spectral_flux_of_spectrometer_channel` | accepted | pending → accepted | Generated; accepted on first review |
| `spectral_radiance` | accepted | pending → accepted | Generated; accepted on first review |
| `spectral_wavelength_of_optical_element` | accepted | reviewed → reviewed | Unchanged; prior quorum shortfall blocks refinement |
| `toroidal_angle_of_measurement_position` | accepted | drafted → accepted | Accepted on first review |
| `toroidal_vacuum_magnetic_field` | accepted | pending → accepted | Generated; accepted on first review |
| `upper_triangularity_of_plasma_boundary` | accepted | pending → accepted | Generated; accepted on first review |
| `volume_of_flux_surface` | accepted | pending → accepted | Generated; accepted on first review |

## Retained residuals

Thirteen identities remain non-terminal on the documentation axis:

- Six remain `reviewed`: four carry a prior quorum shortfall (`derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface`, `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive`, `radial_coordinate_of_conductor_cross_section`, and `spectral_wavelength_of_optical_element`), while two lack legacy resolution metadata (`flux_surface_averaged_square_of_toroidal_flux_coordinate_gradient_magnitude` and `length_of_toroidal_magnetic_field_probe`). The refinement claim remained fail-closed for all six.
- Seven remain `pending`: four were excluded by terminal name lifecycle (`capacitance`, `lower_bound_hard_xray_peak_width`, `normalized_toroidal_hard_xray_peak_lower_bound_width`, and `plasma_breakdown_time`), and three are gated by a non-accepted name stage (`inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`, `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position`, and `normalized_poloidal_flux_coordinate`).

One additional identity, `initial_polarization_ellipticity_of_polarimeter_beam`, remains terminal at `docs_stage=exhausted`. Recovering name-axis identities or repairing legacy review-resolution state is outside this bounded documentation run; no identity was hand-accepted or edited around the quorum.

## Recorded evidence

Every number above is reproduced from these on-disk records:

- `docs/evidence/sn-quality-parity/west-docs-eligible.json` — landed source census, exact identities, initial documentation stages, and concurrent-refinement exclusion.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T164040045364-n-westcomplete/dry-run-priced-admission.json` — pre-call UTC graph count, resolved exact scope, zero-call priced admission, projected component costs, cap, and admission verdict.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T164040045364-n-westcomplete/stream.jsonl` — durable command stream containing the exact-name dry-run confirmation and the executed CLI loop, including pool progress, completion status, run identifier, elapsed time, and headline spend.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T164040045364-n-westcomplete/post-run-snapshot.json` — authoritative `SNRun` and `LLMCost` aggregates, post-run UTC graph count, and all 36 per-identity before/after stages and operation deltas.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T164040045364-n-westcomplete/evidence-check.log` — independent reconciliation assertions over the admission and post-run records.
