NEEDS-HELP: the authorized recovery was only partially completed before the node time fence expired; two governed edits hit safety refusals, the edit batch selector proved broader than the five ruled names, and the required docs drains remain unrun.

tried: Applied the route ruling where the live CLI admitted it: staged and freshly reviewed all 13 reviewed names; read and classified all four frozen export-ledger records; staged the opacity hint and two accepted-name replacements through `sn edit`; attempted the other two ruled corrections; and stopped the edit review when `--edits` claimed work beyond the five-name cohort. The completed 13-name `SNRun` spent exactly $1.396644. The interrupted review receipt records $0.000000 but is explicitly inexact.

options: (1) add a name scope to the three open edit reviews, repair the line-of-sight provenance conflict, select a grammar-valid electron-content identity, then run the two docs drains; (2) first fix `sn run --edits` so it cannot claim unrelated open edits, then resume; or (3) leave the three edits staged and execute only the independently scoped docs drains in a fresh node.

leaning: Option 2, followed by option 1. The broad selector is the safety defect that made shared review unsafe. The two rejected edits need different governed inputs, not bypasses.

cost-if-wrong: Another quorum may spend and still reject candidates; changing the line-of-sight binding without resolving its conflict could detach valid provenance; guessing the electron-content identity could change its physics. No direct acceptance or graph-text mutation was used, so current state remains auditable.

# WEST withheld-name recovery

## Outcome

The sanctioned routes produced one newly fully accepted original identity and
two valid governed successor drafts, but did not finish the recovery:

- `normalized_poloidal_flux_coordinate` re-earned name and docs acceptance
  through `SNRun 283db39c-a930-473a-9de3-5dfbb361efc5`.
- The other 12 reviewed names received fresh name quorums. Ten scored
  0.96875-1.000 but stayed reviewed because accepted descendants blocked the
  acceptance cascade. `beta` and `hot_neutral_temperature` scored 0.300 at
  the semantic gate.
- `phase_of_ion_cyclotron_heating_antenna` and
  `toroidal_coordinate_of_detector_pixel` were superseded through governed
  `sn edit --rename`; their canonical successors are valid drafts. The CLI
  requires rename instead of hint when the old identity is name-accepted.
- The opacity correction is staged as the requested governed hint.
- The line-of-sight rename was refused because one DD source is bound to two
  names. The electron-density replacement was refused because
  `total_electron_density_inside_flux_surface` does not round-trip through
  ISN 0.8.2.
- The docs-pending regression and independently reviewed docs target were not
  drained before the time fence.

The fully accepted triplet count is **4 of 28**, up from **3 of 28**. Exactly
**1 of 28** became newly fully accepted. “Fully accepted” means
`accepted / accepted / valid`; accepted but quarantined identities do not
count.

The authoritative completed-run spend is **$1.396644**. Interrupted edit review
`SNRun d435f0df-d0fa-4b10-b0b3-d779ca6cd817` records **$0.000000** but has
`cost_is_exact=false`; an exact node total therefore cannot be proven beyond
the completed amount. Combined command ceilings were $23.00 ($13 + $10), below
the $30.00 authorization. Persisted `LLMCost` rows total $1.396644.

## Command receipts

All project commands used the shared root environment, `PYTHONPATH=$PWD`, and
`uv run --no-sync`. Stage-only commands create stamps but no `SNRun` and
spend $0.

| Command or evidence read | Exit | SNRun or stamp | Recorded spend | Result |
|---|---:|---|---:|---|
| Read live/worker plans, prior WEST report, scoped guidance, CLI help, and frozen `.export_report.json` | 0 | none | $0.000000 | Route, ledger, lifecycle, and flag contracts established. |
| GraphClient baseline census, 28 exact ids | 0 | none | $0.000000 | 28/28 matched; 4,688/4,688 positive controls; approved 0, contested 0. |
| `sn rescore breakdown_initial_time --dry-run` | 2 | none | $0.000000 | Accepted name refused; later superseded by the ledger ruling. |
| `sn rescore beryllium_density_at_plasma_boundary --cost-limit 1` | 143 | `962d8cd0-f782-4725-813d-48e04eae06de`; `sn-rescore-20260902T140720Z` | $0.000000 exact | Staged, then teardown hung; terminated before an LLM call. Receipt remains started. |
| `sn rescore beta --stage-only` | 0 | `sn-rescore-20260902T141320Z` | $0.000000 | Staged. |
| `sn rescore carbon_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T141329Z` | $0.000000 | Staged. |
| `sn rescore deuterium_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T141343Z` | $0.000000 | Staged. |
| `sn rescore helium_4_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T141352Z` | $0.000000 | Staged. |
| `sn rescore hot_neutral_temperature --stage-only` | 0 | `sn-rescore-20260902T141401Z` | $0.000000 | Staged. |
| `sn rescore hydrogen_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T141409Z` | $0.000000 | Staged. |
| `sn rescore lithium_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T141417Z` | $0.000000 | Staged. |
| `sn rescore neon_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T141425Z` | $0.000000 | Staged. |
| `sn rescore normalized_poloidal_flux_coordinate --stage-only` | 0 | `sn-rescore-20260902T141433Z` | $0.000000 | Staged. |
| `sn rescore poloidal_magnetic_flux --stage-only` | 0 | `sn-rescore-20260902T141441Z` | $0.000000 | Staged. |
| `sn rescore tungsten_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T141449Z` | $0.000000 | Staged. |
| `sn rescore xenon_density_at_plasma_boundary --stage-only` | 0 | `sn-rescore-20260902T141458Z` | $0.000000 | Staged. |
| `sn run --name <13 exact names> --skip-global-maintenance --cost-limit 13` | 0 | `283db39c-a930-473a-9de3-5dfbb361efc5`; scope `c0c97850-8d9d-4037-ac1e-685c9cb4df7a` | **$1.396644 exact** | 13 name reviews, one docs generation, two docs reviews, one docs refinement; no eligible work remained. |
| `sn edit line_integrated_... --hint ... --axis name --scope self --stage-only` | 0 | `sn-edit-20260902T142708Z` | $0.000000 | Governed hint staged. |
| Three accepted-name `sn edit --hint ... --stage-only` attempts | 2 each | none | $0.000000 | CLI requires complete rename for accepted identities. |
| `sn edit phase_of_... --rename ion_cyclotron_heating_antenna_phase --scope self --stage-only` | 0 | `sn-edit-20260902T142836Z` | $0.000000 | Valid successor drafted; old identity superseded. |
| `sn edit toroidal_coordinate_of_detector_pixel --rename toroidal_coordinate_at_detector_pixel --scope self --stage-only` | 0 | `sn-edit-20260902T142845Z` | $0.000000 | Valid successor drafted; old identity superseded. |
| `sn edit vertical_coordinate_of_line_of_sight --rename vertical_coordinate_at_line_of_sight --scope self --stage-only` | 1 | none | $0.000000 | Source migration CAS refused conflicting bindings. |
| `sn edit volume_integrated_total_electron_density --rename total_electron_density_inside_flux_surface --scope self --stage-only` | 2 | none | $0.000000 | Replacement rejected by ISN grammar round-trip. |
| `sn run --only review --edits --cost-limit 10` | 1 after SIGINT | `d435f0df-d0fa-4b10-b0b3-d779ca6cd817` | $0.000000 recorded, **inexact** | Claimed unrelated open edits; interrupted; seven claims released; receipt degraded. |
| GraphClient final lifecycle, ledger-resolution, cost, and legacy census | 0 | none | $0.000000 | 28/28 originals plus successor controls; approved 0, contested 0. |

The authorized docs commands were **not run** after the time fence: the
regression route `sn run --batch west_production_dd_paths --only review_docs`
and a scoped docs drain for `spectral_wavelength_of_optical_element`.

## Exact 28-name lifecycle and route census

Triplets are `name_stage / docs_stage / validation_status`.

| Standard name | Prior reason | Before | After | Route and measured result |
|---|---|---|---|---|
| `beryllium_density_at_plasma_boundary` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Quorum 1.000; accepted descendant blocks cascade. |
| `beta` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Semantic gate 0.300. |
| `breakdown_initial_time` | resolution unrecorded | accepted / accepted / valid | accepted / accepted / valid | Ledger inspection; applied docs edit and contested text require export writeback. |
| `carbon_density_at_plasma_boundary` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Quorum 1.000; accepted descendant blocks cascade. |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | resolution unrecorded | accepted / accepted / valid | accepted / accepted / valid | Ledger inspection; live review methods exist. |
| `deuterium_density_at_plasma_boundary` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Quorum 1.000; accepted descendant blocks cascade. |
| `deuterium_deuterium_neutron_flux` | invalid validation | drafted / pending / quarantined | drafted / pending / quarantined | Legitimately withheld. |
| `helium_4_density_at_plasma_boundary` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Quorum 0.96875; accepted descendant blocks cascade. |
| `hot_neutral_temperature` | name not accepted | reviewed / accepted / valid | reviewed / accepted / valid | Semantic gate 0.300. |
| `hydrogen_density_at_plasma_boundary` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Quorum 1.000; accepted descendant blocks cascade. |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | invalid validation | drafted / pending / quarantined | drafted / pending / quarantined | Grammar gap. |
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | invalid validation | drafted / pending / quarantined | drafted / pending / quarantined | Hint staged; edit open. |
| `lithium_density_at_plasma_boundary` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Quorum 1.000; accepted descendant blocks cascade. |
| `neon_density_at_plasma_boundary` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Quorum 1.000; accepted descendant blocks cascade. |
| `normalized_poloidal_flux_coordinate` | name not accepted | reviewed / pending / valid | **accepted / accepted / valid** | Name 1.000; refined docs accepted 0.8875. |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | resolution unrecorded; docs regression | accepted / pending / valid | accepted / pending / valid | Open source-drift docs hint; drain not run. |
| `phase_of_ion_cyclotron_heating_antenna` | invalid validation | accepted / accepted / quarantined | superseded / accepted / quarantined | Successor `ion_cyclotron_heating_antenna_phase` is drafted / pending / valid. |
| `poloidal_magnetic_flux` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Quorum 1.000; accepted descendant blocks cascade. |
| `radial_coordinate_of_conductor_cross_section` | resolution unrecorded | accepted / accepted / valid | accepted / accepted / valid | Ledger inspection; live review methods exist. |
| `radial_outline_of_plasma_boundary` | invalid validation | accepted / accepted / quarantined | accepted / accepted / quarantined | Legitimately withheld. |
| `radial_outline_of_wall` | invalid validation | accepted / accepted / quarantined | accepted / accepted / quarantined | Legitimately withheld. |
| `spectral_wavelength_of_optical_element` | docs not accepted | accepted / reviewed / valid | accepted / reviewed / valid | Docs pass not run. |
| `toroidal_coordinate_of_detector_pixel` | invalid validation | accepted / accepted / quarantined | superseded / accepted / quarantined | Successor `toroidal_coordinate_at_detector_pixel` is drafted / pending / valid. |
| `tritium_tritium_neutron_flux` | invalid validation | drafted / pending / quarantined | drafted / pending / quarantined | Legitimately withheld. |
| `tungsten_density_at_plasma_boundary` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Quorum 1.000; accepted descendant blocks cascade. |
| `vertical_coordinate_of_line_of_sight` | invalid validation | accepted / accepted / quarantined | accepted / accepted / quarantined | Rename refused by source-binding CAS. |
| `volume_integrated_total_electron_density` | invalid validation | accepted / accepted / quarantined | accepted / accepted / quarantined | Proposed spelling rejected by grammar. |
| `xenon_density_at_plasma_boundary` | name not accepted | reviewed / pending / valid | reviewed / pending / valid | Quorum 1.000; accepted descendant blocks cascade. |

## Frozen export-ledger conditions

All four frozen records say: `resolution_unrecorded: docs-axis reviews exist
but no winning group records a method`.

| Name | Missing resolution and live evidence |
|---|---|
| `breakdown_initial_time` | Frozen winning method absent. Six docs reviews have `quorum_consensus` and `max_cycles_reached`; applied docs edit plus contested resolution says the shorter wording preserves breakdown-onset semantics. Governed export-ledger/edit writeback is required. |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | Frozen winning method absent. Ten live docs reviews have `quorum_consensus`, `authoritative_escalation`, and `max_cycles_reached`; no open edit/contest. Regenerate the winning-group ledger. |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | Frozen winning method absent; docs pending. Nine prior docs reviews have all three methods, but an open source-drift hint records the DD path change to `plasma_profiles/ggd/j_parallel/values`. Drain docs, then regenerate. |
| `radial_coordinate_of_conductor_cross_section` | Frozen winning method absent. Seven live docs reviews have all three methods; no open edit/contest. Regenerate the winning-group ledger. |

## Quarantine classification

### Grammar gap: 1

| Name | Recorded issue | Classification |
|---|---|---|
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | `parse_error: grammar round-trip failed for inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | No evidence-grounded replacement is supplied; keep quarantined. |

### Clear corrections routed through governed edit: 5

| Original | Recorded issue | Correction and result |
|---|---|---|
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | `integrated_` prefix forbidden; use `_inside_flux_surface` suffix. | Governed hint staged; review open. |
| `phase_of_ion_cyclotron_heating_antenna` | `phase_of_<X>` noncanonical; validator names `ion_cyclotron_heating_antenna_phase`. | Exact governed rename staged; valid successor drafted. |
| `toroidal_coordinate_of_detector_pixel` | `_of_detector_pixel` must be `_at_detector_pixel`. | Exact governed rename staged; valid successor drafted. |
| `vertical_coordinate_of_line_of_sight` | `_of_line_of_sight` must be `_at_line_of_sight`. | Rename attempted; source compare-and-set refused two live bindings. |
| `volume_integrated_total_electron_density` | `integrated_` prefix forbidden; use `_inside_flux_surface` suffix. | `total_electron_density_inside_flux_surface` failed grammar and was not applied; identity needs governance. |

### Legitimately withheld: 4

| Name | Recorded issue | Why withheld |
|---|---|---|
| `deuterium_deuterium_neutron_flux` | Multiple subjects `['deuterium_deuterium', 'neutron']`. | Isotope pair role is unresolved. |
| `tritium_tritium_neutron_flux` | Multiple subjects `['tritium_tritium', 'neutron']`. | Isotope pair role is unresolved. |
| `radial_outline_of_plasma_boundary` | `outline` does not identify the described entity; locus prefers `_at_plasma_boundary`. | `radial_outline_at_plasma_boundary` fixes only the preposition and leaves the semantic defect. |
| `radial_outline_of_wall` | `outline` does not identify the described entity; locus prefers `_at_wall`. | `radial_outline_at_wall` fixes only the preposition and leaves the semantic defect. |

## GraphClient legacy-state census

The query was aimed at `StandardName.name_stage` and carried candidate/property
positive controls. Governed successors and interrupted-run maintenance changed
the population.

| Snapshot | Candidates | With id | With name_stage | Accepted | Approved | Contested |
|---|---:|---:|---:|---:|---:|---:|
| Before | 4,688 | 4,688 | 4,688 | 2,336 | **0** | **0** |
| Final | 4,690 | 4,690 | 4,690 | 2,335 | **0** | **0** |

The required legacy census is proven: **approved 0, contested 0**. Positive
controls prove the zeros are not missing-property artifacts.

## Safety and remaining work

No direct acceptance, graph-text Cypher mutation, or manual text write was
performed. All lifecycle changes used `sn rescore`, `sn run`, or governed
`sn edit`. Remaining work:

1. Enforce name/id scope for edit review, then review only the opacity,
   antenna-phase, and detector-pixel-coordinate edits.
2. Resolve the line-of-sight source's two bindings before retrying its rename.
3. Govern a grammar-valid electron-content identity; do not apply the rejected
   spelling.
4. Run the WEST batch `review_docs` drain and a scoped docs drain for
   `spectral_wavelength_of_optical_element`.
5. Regenerate the export ledger and verify the winning docs method for all four
   frozen rows.

Unmet done-when conditions are the two docs drains, all five completed edit
reviews, recovery beyond 4/28, and an exact receipt for the interrupted run.
