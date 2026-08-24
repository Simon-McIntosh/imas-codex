# WEST documentation resumed refresh: first tranche

## Outcome

The first resumed WEST documentation tranche refreshed 130 exact
`StandardName.id` identities. Of those, 129 finished documentation review at
`docs_stage=accepted`; all 130 received new documentation text and all 130 had
an append-safe `DocsRevision` snapshot of their prior accepted text before the
reset. Fifteen identities failed at least one documentation quorum and entered
the ordinary refinement path, accounting for 19 refinement executions. Fourteen
of those identities ultimately passed; one remains drafted after two refinements
and an incomplete final quorum.

Actual provider spend was **USD 22.690535 against the hard USD 40.000000
ceiling** (56.73% used; USD 17.309465 headroom). The main run spent USD
22.504644. Its idle shutdown found one long-running review still in flight, so
a same-scope tail run was admitted with a USD 17.49 ceiling; that kept the
combined possible maximum at USD 39.994644 and spent USD 0.185891. The sum of
461 `LLMCost.llm_cost` rows is USD 22.690535000000008, agreeing with the two
`SNRun.cost_spent` values to floating-point precision.

The exact WEST original-identity census moved from **215 docs-accepted before
to 214 docs-accepted after**. This one-row decrease is the deliberately visible
drafted tail, not missing graph data or a silent acceptance. The full
after-state over 231 original identities is: 214 accepted, 3 drafted, 6
reviewed, 7 pending, and 1 exhausted.

## Exact-identity census and admission

The fresh preflight queried `StandardName.id`, never `StandardName.name`.
Coverage was 4,666 `StandardName` candidates, 4,666 with `id`, and zero with a
`name` property. All 231 frozen WEST original identities resolved exactly once.
The live census found 132 original refresh identities not covered by the prior
40-identity tranche: 130 clean accepted-name/accepted-doc identities and two
already-drafted identities. None of the 130 clean candidates overlapped the 20
concurrent name-refinement originals.

Admission was sized from the prior tranche's measured USD 5.120219 / 40 = USD
0.128005475 per identity. The resulting USD 40 rate capacity was 312 identities,
so all 130 clean candidates were admitted, with a rate-based expectation of USD
16.64071175 and **zero otherwise-clean identities deferred by capacity**. A
zero-call full rendering separately reported USD 447.769285 of conservative
provider-policy exposure (USD 1.544706 generation, USD 442.050448 review, USD
4.174132 refinement). That reservation is not an actual-spend forecast and was
not allowed to replace the plan-mandated empirical sizing basis; runtime budget
leases enforced the hard ceiling.

The realized USD 0.174542577 per selected identity exceeded the prior tranche
rate because this tranche contained 15 quorum failures and 19 paid refinement
executions. That measured rate is the appropriate empirical input for sizing a
later comparable tranche; it must remain distinct from worst-case provider
reservation.

## Quorum and refinement evidence

The 15 refreshed identities that failed quorum and therefore routed through
the ordinary refinement path were:

- `frequency_of_ion_cyclotron_heating_antenna` — 2 refinements, accepted at 0.95625
- `normalized_poloidal_flux_coordinate_of_plasma_boundary` — 2 refinements, drafted after the final quorum remained incomplete
- `normalized_toroidal_plasma_beta` — 1 refinement, accepted at 0.93125
- `plasma_beta` — 1 refinement, accepted at 0.975
- `plasma_pressure` — 1 refinement, accepted at 0.95625
- `radial_outline_of_antenna_strap` — 1 refinement, accepted at 0.9375
- `radiative_temperature_at_magnetic_axis` — 1 refinement, accepted at 0.925
- `reflected_phase_of_ion_cyclotron_heating_antenna` — 1 refinement, accepted at 0.94375
- `spectral_signal_to_noise_ratio_of_spectrometer_channel` — 3 refinements, accepted at 0.85
- `toroidal_angle_of_toroidal_magnetic_field_probe` — 1 refinement, accepted at 0.925
- `toroidal_flux_surface_averaged_current_density` — 1 refinement, accepted at 0.91875
- `toroidal_magnetic_field_at_magnetic_axis` — 1 refinement, accepted at 0.95625
- `toroidal_magnetic_flux` — 1 refinement, accepted at 0.9375
- `total_plasma_radiated_power` — 1 refinement, accepted at 0.95
- `vertical_coordinate_of_poloidal_magnetic_field_probe` — 1 refinement, accepted at 0.95625

The remaining drafted identity was not hand-accepted. Its two complete quorums
scored 0.550 and 0.637, correctly routing it back through refinement. The tail
retry completed two reviewer calls but its third reviewer exceeded the
three-minute run plus 60-second grace, so the pipeline retained the refined
document as drafted. This is ordinary recoverable pipeline state.

One earlier structured-output failure for `ratio_of_coolant_mass_to_time` was
also fail-closed; a later complete quorum accepted it at 0.94375. One transient
Neo4j deadlock was retried successfully by the standard claim machinery.

## Representative refreshed documents

- `flux_surface_averaged_ratio_of_square_of_toroidal_flux_coordinate_gradient_magnitude_to_square_of_major_radius`
  is bound to `equilibrium/time_slice/profiles_1d/gm2`. Its refreshed document
  defines the flux-surface average of the squared toroidal-flux-coordinate
  gradient weighted by inverse squared major radius, gives the equation, and
  distinguishes the weighted metric from the unweighted gradient-magnitude
  average. Documentation quorum score: 0.9375.
- `flux_surface_averaged_square_of_magnetic_field_magnitude` is bound to
  `equilibrium/time_slice/profiles_1d/gm5`. Its refreshed document defines the
  inverse-gradient-weighted surface average of total equilibrium magnetic-field
  strength squared and distinguishes averaging before and after squaring.
  Documentation quorum score: 0.96875.
- `forward_power_of_ion_cyclotron_heating_antenna` is bound to both
  `ic_antennas/antenna/module/power_forward` and
  `ic_antennas/antenna/power_forward`. Its refreshed document identifies the
  forward-traveling RF component at the feed-line reference plane and separates
  it from reflected and net launched power. Documentation quorum score: 0.94375.

## Exact identities deferred

No clean accepted-name/accepted-doc identity was deferred by the USD 40
capacity calculation. The exact documentation tail deferred to a later scoped
run is:

- `normalized_poloidal_flux_coordinate_of_plasma_boundary` — selected here;
  drafted after two refinements and an incomplete final quorum
- `voltage_of_poloidal_magnetic_field_probe` — already drafted at fresh
  preflight, so not reset as accepted documentation
- `width_of_poloidal_field_coil` — already drafted at fresh preflight, so not
  reset as accepted documentation

The following 20 exact original identities remained excluded for the concurrent
name-refinement lane. Thirteen have a currently visible successor, shown after
the arrow; all 13 successor documents are pending. Seven have no successor yet
and therefore retain only the original identity here.

- `current_of_poloidal_field_coil` → no successor
- `energy_confinement_time` → `thermal_energy_confinement_time`
- `gap_of_antenna_strap` → `normal_gap_at_wall`
- `hot_neutral_temperature` → no successor
- `ion_atomic_number` → `ion_species_atomic_number`
- `length_of_plasma_boundary` → `poloidal_length_of_flux_surface`
- `lower_photon_energy` → `lower_bound_photon_energy`
- `neutral_pressure` → no successor
- `power_due_to_ohmic_dissipation` → `total_power_due_to_ohmic_dissipation`
- `power_of_ion_cyclotron_heating_antenna` → `wave_power_of_ion_cyclotron_heating_antenna`
- `radiative_temperature` → no successor
- `spectral_radiance_of_soft_xray_detector` → `incident_soft_xray_radiance`
- `surface_temperature` → `surface_temperature_of_plasma_facing_component`
- `thermal_energy_of_plant_component_port` → no successor
- `thermal_power_of_plant_component_port` → `coolant_heating_power_of_plant_component_port`
- `toroidal_angle_of_magnetic_field_probe` → `toroidal_angle_of_poloidal_magnetic_field_probe`
- `toroidal_width_of_antenna_strap` → no successor
- `vertical_coordinate_of_plasma_boundary` → `vertical_outline_of_plasma_boundary`
- `vertical_outline_of_wall_material` → `vertical_outline_of_plasma_facing_component`
- `voltage_amplitude` → no successor

## Runs, safeguards, and artifacts

Both invocations used the same exact scope
`9d530e18-7531-4657-b1d2-7efe53ba2bcb`, `--docs-only`, and
`--skip-global-maintenance`. The main run `958b494d-1c89-4f44-b07f-b411929962d5`
recorded 130 generations, 148 completed review batches, 19 refinements, USD
22.504644 spend, and `interrupted` after the backlog watchdog reached zero. The
tail run `dead9ab2-38e7-4ae6-a273-4863b1205424` recorded USD 0.185891 and
`time_limit_reached`; it made no unsafe single-review acceptance. The combined
actual spend is the authoritative tranche spend stated above.

The reset created an exact prior-text `DocsRevision` for 130/130 selected
identities before setting their documentation stage to pending. Post-run hash
comparison shows refreshed text for 130/130 identities. The focused snapshot
regression suite passed 20 tests.

Evidence receipts and logs:

- fresh census and zero-call sizing: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T205922249982-n-westrefreshtranche/preflight.json`
- reset and snapshot receipt: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T205922249982-n-westrefreshtranche/reset-receipt.json`
- exact post-run reconciliation: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T205922249982-n-westrefreshtranche/post-run.json`
- main run log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T205922249982-n-westrefreshtranche/sn-run-live.log`
- tail run log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T205922249982-n-westrefreshtranche/sn-run-tail.log`
- snapshot test log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T205922249982-n-westrefreshtranche/snapshot-regression-tests.log`

One auxiliary evidence-write defect did not alter the accepted pipeline outcome:
after `power_due_to_ion_cyclotron_heating` passed at 0.96875, DD-gap evidence
persistence supplied a prose sentence where `DDGapEvidenceRule` requires an enum
token. That schema/producer mismatch is outside this tranche's write scope and
needs a separate repair; the documentation itself remained accepted.
