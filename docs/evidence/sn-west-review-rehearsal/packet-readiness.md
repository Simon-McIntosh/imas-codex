# WEST reviewer packet readiness — live graph census

## Verdict

**The WEST reviewer packet is not a stable, releasable batch today.** At the
closing read-only census on **2026-08-25 07:20:29 UTC**, the real review mint
contained **416 `StandardName.id` identities**. Of those, **379 were
name-accepted or approved**, **381 were docs-accepted**, and **368 were
simultaneously name-accepted or approved, docs-accepted, and validation-valid**.
The real RC export eligibility and catalog-model path partitioned the cohort
into **369 survivor candidates and 47 identity-bearing exclusions**, but the
actual exporter then failed its hierarchy ordering step and produced **no
complete catalog manifest or export report**. Separately, **178/416 identities
(42.8%) had stage-transition evidence in the preceding twelve hours**, so the
cohort also fails a zero-churn stability interpretation even before the export
failure is considered.

This is a current-graph result, not a reuse of the historic campaign reports.
The source tree was commit `bbd9b6718e7343506dc76874b4e4ae99b3306077`.

## Cohort authority and census

The source authority was
`imas_codex/standard_names/manifests/west_production_dd_paths.yaml`, validated
through `load_sources_file()`. The review identity set was then minted through
the production `mint_sn_list()` path: exact `StandardNameSource.id` lookup,
authoritative `PRODUCED_NAME` traversal, exclusion of `superseded` and
`exhausted` targets, deduplication on `StandardName.id`, and the production
one-hop parent/sibling/child family closure.

The identifier preflight covered **4,656/4,656 `StandardName` nodes on `id`**
and **0/4,656 on the undeclared `name` property**. It also covered
**9,668/9,668 `StandardNameSource` nodes on `id`**. The census never joined on
either of the plausible-but-wrong keys.

| Census layer | Result |
|---|---:|
| WEST manifest source identities | **355** |
| Manifest source rows found in the graph | **355/355** |
| Sources with a live `PRODUCED_NAME` target | **328/355** |
| Sources without a live target | **27/355**: 17 `extracted`, 10 `skipped` |
| Distinct directly source-bound live `StandardName.id` identities | **218** |
| Immediate-family identities added by the real mint | **198** |
| Exact reviewer cohort | **416** |

The 27 unmatched sources are therefore not hidden by the larger family-closed
packet count:

- `equilibrium/time_slice/boundary/outline/z`
- `equilibrium/time_slice/constraints/faraday_angle/weight`
- `equilibrium/time_slice/contour_tree/node/z`
- `equilibrium/time_slice/convergence/iterations_n`
- `equilibrium/time_slice/time`
- `core_profiles/profiles_1d/time`
- `spectrometer_visible/channel/isotope_ratios/isotope/element/z_n`
- `pf_active/coil/current`
- `ic_antennas/antenna/module/strap/distance_to_conductor`
- `ic_antennas/antenna/module/strap/width_phi`
- `ic_antennas/antenna/module/voltage/amplitude`
- `ic_antennas/antenna/power_launched`
- `calorimetry/group/component/energy_cumulated`
- `calorimetry/group/component/power`
- `hard_x_rays/emissivity_profile_1d/half_width_external`
- `hard_x_rays/emissivity_profile_1d/time`
- `camera_x_rays/camera/camera_dimensions`
- `camera_x_rays/detector_humidity`
- `camera_x_rays/detector_humidity/time`
- `camera_x_rays/detector_temperature/time`
- `camera_x_rays/frame/time`
- `camera_ir/channel/camera/frame/apparent_temperature`
- `barometry/gauge/pressure`
- `ece/channel/t_radiation`
- `summary/disruption/time/value`
- `summary/global_quantities/tau_energy/value`
- `summary/global_quantities/tau_energy_98/value`

### Current lifecycle state

| Axis | Current cohort distribution |
|---|---|
| Name stage | **379 accepted**, 30 reviewed, 6 drafted, 1 pending |
| Documentation stage | **381 accepted**, 29 pending, 5 reviewed, 1 null |
| Validation | **405 valid**, 10 quarantined, 1 null |
| Direct WEST bindings only | 218 identities: **209 name-accepted**, **211 docs-accepted**, **200 accepted/accepted/valid** |
| Family closure only | 198 identities: **170 name-accepted**, **170 docs-accepted**, **168 accepted/accepted/valid** |

The larger docs-accepted count than name-accepted count is real. Some identities
have retained accepted documentation while their name axis has been reopened or
held at review.

## Twelve-hour stability window

The closing window was **2026-08-24 19:20:23 UTC through 2026-08-25 07:20:23
UTC**. An identity counted as changed when the graph carried stage-transition
evidence in that window: a recent name/docs review record, a recent docs
revision, or a recent stage-associated node timestamp (`generated_at`,
`reviewed_name_at`, `docs_generated_at`, `reviewed_docs_at`, parent enrichment,
catalog approval, contest/quarantine, refine stop, or quorum-shortfall time).
Generic internal-change rows without stage evidence were deliberately excluded;
in particular, 30 identities carrying only the recent invariant-sign-document
repair event were not called stage churn.

The resulting churn was **178 distinct cohort identities**. The final cohort
cardinality and lifecycle distributions were unchanged between the opening and
closing censuses, but cardinality stability is not stage stability: 178 recent
transitions means the packet is still a moving target.

## Real export-path result

The executed path was `run_export(force=True, final=False,
review_batch=<the 416 minted ids>)`, matching the exporter call made by the
review-release orchestrator. A second read-only projection ran the same
population fetch, eligibility classifier, score gate, strict grammar gate, and
ISN entry validator so the identity ledger remained available even though the
full writer later raised.

### Identity partition before catalog ordering

| Outcome | Count | Meaning |
|---|---:|---|
| Survived eligibility, score, grammar, and ISN entry validation | **369** | Candidate identities that reached catalog grouping |
| Withheld with exactly one ledger reason | **47** | Identity-bearing exclusions below |
| Accounted total | **416/416** | No duplicate, overlap, outside-population, or unattributed identity |

The 369 survivor candidates comprised **363 accepted names and 6 reviewed
catalog-edit names**. The reviewed identities that the real review-batch path
still admitted were:

- `frequency_of_diagnostic_antenna` — name score 0.575
- `gap_at_plasma_boundary` — name score 0.750
- `iron_density_at_plasma_boundary` — name score 1.000
- `poloidal_magnetic_field_at_constraint_position` — name score 0.675
- `spectral_flux_of_spectrometer_channel` — name score 0.8375
- `volume_of_flux_surface` — name score 1.000

That behavior follows the current export implementation: an exact review batch
admits its named identities into the population, and `origin=catalog_edit`
skips the ordinary name-score exclusion. It must not be confused with a claim
that all 369 identities are currently name-accepted.

### Exclusion ledger — 47 withheld identities

| Reason | Count | Exact identities |
|---|---:|---|
| `documentation_not_accepted` | **18** | `beryllium_density_at_plasma_boundary`; `carbon_density_at_plasma_boundary`; `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface`; `deuterium_density_at_plasma_boundary`; `electron_density`; `helium_4_density_at_plasma_boundary`; `hydrogen_density_at_plasma_boundary`; `lithium_density_at_plasma_boundary`; `neon_density_at_plasma_boundary`; `normalized_poloidal_flux_coordinate`; `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive`; `poloidal_magnetic_flux`; `radial_coordinate_of_conductor_cross_section`; `radial_coordinate_of_reflector`; `spectral_wavelength_of_optical_element`; `square_of_magnetic_field_magnitude`; `tungsten_density_at_plasma_boundary`; `xenon_density_at_plasma_boundary` |
| `name_review_quorum_shortfall` | **14** | `beta`; `hot_neutral_temperature`; `normalized_toroidal_flux_coordinate_of_line_of_sight`; `normalized_toroidal_flux_coordinate_of_measurement_position`; `poloidal_turn_count`; `radial_coordinate_of_arc_of_circle_center`; `radial_coordinate_of_electron_cyclotron_launcher_mirror`; `radial_coordinate_of_pellet_path`; `radial_coordinate_of_shunt`; `ratio_of_ion_velocity_to_magnetic_field`; `safety_factor_at_pedestal`; `time_derivative_of_electron_density`; `toroidal_beta`; `turn_count_of_correction_coil` |
| `invalid_validation_status` | **11** | `coolant_mass` (`null`); `deuterium_deuterium_neutron_flux`; `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`; `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position`; `phase_of_ion_cyclotron_heating_antenna`; `radial_outline_of_plasma_boundary`; `radial_outline_of_wall`; `toroidal_coordinate_of_detector_pixel`; `tritium_tritium_neutron_flux`; `vertical_coordinate_of_line_of_sight`; `volume_integrated_total_electron_density` (the last ten are quarantined) |
| `invalid_catalog_entry` | **2** | `breakdown_initial_time`: metadata entry rejects the graph unit `s`; `capacitance_of_ion_cyclotron_heating_antenna`: documentation fails the required positive-sign-convention form |
| `resolution_unrecorded` | **2** | `equilibrium_weight_of_interferometer_beam`; `radial_coordinate_of_filter_window` — docs reviews exist but no winning group records a resolution method |

### Hard writer failure and advisories

The full exporter did not complete. It wrote 14 partial domain files, then
`order_entries_by_hierarchy()` raised `OrderingError` on this two-identity
cycle:

- `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel`
- `spectral_signal_to_noise_ratio_of_spectrometer_channel`

No `catalog.yml` and no `.export_report.json` were written. Therefore **369 is
the number that survived the real filters, not the size of a usable emitted
packet; the completed releasable artifact count is zero**.

Before the ordering failure, the RC path also recorded these non-withholding
quality findings:

- the graph-test gate failed but was downgraded to advisory by RC semantics;
- the cross-field gate found 295 dangling documentation links and remained
  advisory for RC;
- post-validation link cleanup pruned **296** dangling internal links from the
  would-be output;
- the divergence gate recorded **207** advisory entries.

These do not alter the 369/47 exclusion arithmetic, but they reinforce that the
current packet should not be frozen for reviewer attention.

## Representative current rows

The examples below are direct WEST bindings, not family-only additions. The
score column reports **name/docs** reviewer scores from the current graph. The
opening sentence is taken from `StandardName.documentation`, not from the
shorter description field.

| Standard Name identity | Documentation opening sentence | WEST DD source path | Unit | Reviewer score (name/docs) | Export disposition |
|---|---|---|---|---:|---|
| `accumulated_deposited_energy_of_plasma_facing_component` | Accumulated deposited energy is the total heat energy transferred to a plasma-facing component during a complete plasma discharge and its post-discharge return toward thermal equilibrium. | `calorimetry/group/component/energy_total/data` | `J` | 0.925 / 0.90625 | survivor |
| `area_of_poloidal_magnetic_field_probe` | The quantity is the geometric cross-sectional area enclosed by one winding turn of a poloidal magnetic-field probe coil. | `magnetics/b_field_pol_probe/area` | `m^2` | 1.000 / 0.925 | survivor |
| `breakdown_initial_time` | This quantity identifies the instant at which plasma breakdown begins: plasma is initiated and discharge current starts to flow. | `summary/time_breakdown/value` | `s` | 0.925 / 0.975 | withheld: invalid catalog entry |
| `capacitance_of_ion_cyclotron_heating_antenna` | Capacitance of an ion cyclotron heating antenna is the non-negative charge-to-voltage proportionality of a specified lumped impedance-matching element in the antenna RF circuit. | `ic_antennas/antenna/module/matching_element/capacitance` | `F` | 0.96875 / 0.8875 | withheld: invalid catalog entry |
| `cold_neutral_fraction` | The cold neutral fraction is a dimensionless ratio bounded in $[0, 1]$ expressing the relative abundance of the cold, low-energy recycled component of a hydrogen-isotope neutral population. | `spectrometer_visible/channel/isotope_ratios/isotope/cold_neutrals_fraction` | `1` | 1.000 / 0.96875 | survivor |
| `coolant_temperature_at_inlet` | The coolant inlet temperature is the absolute thermodynamic temperature of the coolant or working fluid at the entrance to a cooling loop, plant component, or cooling module. | `calorimetry/group/component/temperature_in` | `K` | 1.000 / 0.9375 | survivor |
| `current_of_passive_loop` | This quantity is the conventional electric current circulating in a single passive conducting loop. | `pf_passive/loop/current` | `A` | 1.000 / 0.93125 | survivor |
| `electron_density_at_divertor_target` | Number density of the electron species at the divertor target is the local particle count per volume at the sheath entrance immediately upstream of the target-facing plasma boundary. | `summary/local/divertor_target/n_e/value` | `m^-3` | 1.000 / 0.950 | survivor |
| `elongation_of_plasma_boundary` | Plasma-boundary elongation is the ratio of the vertical half-height to the minor radius of the outermost closed plasma cross-section, describing its vertical shaping relative to a circular cross-section. | `equilibrium/time_slice/boundary/elongation` | `1` | 1.000 / 0.96875 | survivor |
| `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_magnetic_field_magnitude` | This coefficient describes the flux-surface-averaged geometric weighting of a toroidal-flux radial coordinate by the inverse square of the local magnetic-field magnitude. | `equilibrium/time_slice/profiles_1d/gm6` | `T^-2` | 0.96875 / 0.94375 | survivor |
| `hard_xray_emissivity` | Band-integrated hard-X-ray photon emissivity is the local hard-X-ray bremsstrahlung photon source density in position and emission direction over a specified photon-energy interval. | `hard_x_rays/emissivity_profile_1d/emissivity` | `m^-3.s^-1.sr^-1` | 0.95625 / 0.950 | survivor |
| `incident_soft_xray_radiance` | Incident soft-X-ray power radiance is the energy-integrated radiant power arriving at a receiving surface from the viewing direction, expressed per projected area and per solid angle. | `soft_x_rays/channel/brightness` | `W.m^-2.sr^-1` | 0.95625 / 0.900 | survivor |
| `line_integrated_electron_number_density` | The quantity is the column density of free electrons encountered along a specified electromagnetic propagation path. | `equilibrium/time_slice/constraints/n_e_line/reconstructed` | `m^-2` | 0.8125 / 0.925 | survivor |
| `maximum_of_energy_flux_at_divertor_target` | Peak energy flux at a divertor target is the largest local rate of energy deposition per unit target-surface area on one divertor-target surface. | `summary/local/divertor_target/power_flux_peak/value` | `W.m^-2` | 0.9375 / 0.96875 | survivor |

## Evidence and verification

- Full opening census, per-identity properties, bindings, units, and recent
  transition evidence:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070724966717-n-westrcstatus/live-census.json`
- Closing census and exact 12-hour cohort/churn identities:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070724966717-n-westrcstatus/final-census.json`
- Identity-bearing 369/47 export-filter partition:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070724966717-n-westrcstatus/export-filter-census.json`
- Full real-export log, including the catalog validation refusals, link
  pruning, and hierarchy-cycle traceback:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070724966717-n-westrcstatus/real-export.log`
- Incomplete staging tree retained as failure evidence:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070724966717-n-westrcstatus/export-staging/`
- Focused export-accounting/release-hold/additive-export tests:
  **7 passed, 2 deselected, 0 failed** in 7.90 s. Log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070724966717-n-westrcstatus/packet-readiness-tests.log`

## Conditions for a fresh readiness check

The next census should run only after the two-name ordering cycle is removed or
governed, the 47 withheld identities are repaired or explicitly dispositioned,
the 27 unmatched sources have reviewed refusal/binding outcomes, and the
twelve-hour stage-churn count reaches the intended freeze criterion. The real
export must then complete with a closing identity ledger, `catalog.yml`, and
`.export_report.json`; a filter-only 369 count is not release completion.
