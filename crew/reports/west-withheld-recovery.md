NEEDS-HELP: the required `sn rescore` route refuses five accepted WEST targets before any graph mutation, so the exact recovery contract cannot be completed from this report-only node.

tried: Read the complete live `sn-west-catalog-release` plan at version 31 and `crew/reports/west-batch-drain.md`; verified the current CLI help; captured all 28 target lifecycles; then ran `imas-codex sn rescore breakdown_initial_time --dry-run`. It exited 2 because `breakdown_initial_time` is `name_stage='accepted'`, while `sn rescore` admits only `reviewed` or `exhausted` names. The same predicate excludes the other three `resolution_unrecorded` names and the accepted/docs-pending regression.

options: (1) widen the node's code/test scope and extend `sn rescore` with an explicitly governed accepted-name recovery mode for missing docs-resolution authority; (2) amend the execution contract to authorize the existing accepted-name/docs routes, then redispatch the operational node; or (3) split the 13 eligible reviewed names into a rescore node and keep the five accepted names visibly blocked until their route is settled.

leaning: Option 2. The implementation already separates name-axis rescore from accepted-name/docs recovery, and substituting another accepted-name command without plan authority would violate the fixed route more seriously than reporting the mismatch.

cost-if-wrong: Option 1 adds and validates a new public lifecycle transition and may duplicate an existing accepted-name mechanism; option 3 creates a partially recovered cohort and requires a second before/after reconciliation. No live recovery or LLM call was made here, so the current rollback cost is zero.

# WEST withheld-name recovery refusal

## Outcome

No graph write or LLM call was made. The required route is internally
inconsistent with the live lifecycle state:

- Thirteen `name_not_accepted` names are `reviewed` and individually eligible
  for `sn rescore`.
- The four `resolution_unrecorded` names are `accepted / accepted / valid`.
  `sn rescore` refuses every accepted name by design.
- The source-drift regression is `accepted / pending / valid`; it is likewise
  refused by `sn rescore`.
- The one `documentation_not_accepted` name and all governed quarantine edits
  were left untouched after the hard preflight refusal. Running a partial paid
  wave would not satisfy the exact 28-name recovery measure.

The current fully accepted lifecycle-triplet count is therefore **3 of 28**,
unchanged across this node. The count newly made fully accepted is **0**.
Exact spend is **$0.000000 of the authorized $30.000000**. No `SNRun` was
created, so there is no `SNRun` id to report.

## Command receipt

| Command or read | Exit status | SNRun id | Spend | Result |
|---|---:|---|---:|---|
| Reckon `read_plan`, raw view, `sn-west-catalog-release` | 0 | none | $0.000000 | Live plan version 31 read completely from the worktree. |
| `imas-codex sn rescore --help` | 0 | none | $0.000000 | Confirms only `reviewed`/`exhausted` non-accepted names are eligible. |
| `imas-codex sn review --help` | 0 | none | $0.000000 | Existing docs review surface inspected; not substituted for the mandated rescore route. |
| `imas-codex sn run --help` | 0 | none | $0.000000 | Existing name/docs scoped-pool surfaces inspected; not substituted. |
| `imas-codex sn edit --help` | 0 | none | $0.000000 | Governed hint route and mandatory reason confirmed. |
| GraphClient before snapshot | 0 | none | $0.000000 | Matched all 28 exact target ids; positive controls covered all 4,688 `StandardName.id` and `name_stage` properties. |
| `imas-codex sn rescore breakdown_initial_time --dry-run` | **2** | none | $0.000000 | Refused: accepted names are live; rescore only recovers `reviewed` or `exhausted`. |
| GraphClient after snapshot | 0 | none | $0.000000 | Matched all 28; byte-identical to before, SHA-256 `98c90e97cbcf6b59f94f627f47f7e8a2439a0c23310276cf950e47c7f4f9d75b`. |

The exact refusal was:

> `breakdown_initial_time` is `name_stage='accepted'` — rescore only recovers
> `exhausted` or `reviewed` names (accepted names are already live;
> superseded names should be recovered via their successor).

## Exact 28-name lifecycle and route census

The lifecycle columns are `name_stage / docs_stage / validation_status`. Since
the preflight failed before mutation, every before and after triplet is equal.

| Standard name | Prior export reason | Before | After | Required route and observed result |
|---|---|---|---|---|
| `beryllium_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `beta` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `breakdown_initial_time` | `resolution_unrecorded` | `accepted / accepted / valid` | `accepted / accepted / valid` | Required rescore dry-run refused with exit 2. |
| `carbon_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | `resolution_unrecorded` | `accepted / accepted / valid` | `accepted / accepted / valid` | Required rescore ineligible: accepted stage. |
| `deuterium_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `deuterium_deuterium_neutron_flux` | `invalid_validation_status` | `drafted / pending / quarantined` | `drafted / pending / quarantined` | Classified legitimately withheld; no clear-cut edit. |
| `helium_4_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `hot_neutral_temperature` | `name_not_accepted` | `reviewed / accepted / valid` | `reviewed / accepted / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `hydrogen_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | `invalid_validation_status` | `drafted / pending / quarantined` | `drafted / pending / quarantined` | Classified grammar gap; no edit. |
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | `invalid_validation_status` | `drafted / pending / quarantined` | `drafted / pending / quarantined` | Clear-cut governed hint candidate; not staged after preflight blocked. |
| `lithium_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `neon_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `normalized_poloidal_flux_coordinate` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | `resolution_unrecorded` | `accepted / pending / valid` | `accepted / pending / valid` | Required rescore ineligible: accepted stage; this is the docs-pending regression. |
| `phase_of_ion_cyclotron_heating_antenna` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` | Clear-cut governed hint candidate; not staged after preflight blocked. |
| `poloidal_magnetic_flux` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `radial_coordinate_of_conductor_cross_section` | `resolution_unrecorded` | `accepted / accepted / valid` | `accepted / accepted / valid` | Required rescore ineligible: accepted stage. |
| `radial_outline_of_plasma_boundary` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` | Classified legitimately withheld; `_at_` alone leaves the semantic defect. |
| `radial_outline_of_wall` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` | Classified legitimately withheld; `_at_` alone leaves the semantic defect. |
| `spectral_wavelength_of_optical_element` | `documentation_not_accepted` | `accepted / reviewed / valid` | `accepted / reviewed / valid` | Docs review pass not run after cohort preflight blocked. |
| `toroidal_coordinate_of_detector_pixel` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` | Clear-cut governed hint candidate; not staged after preflight blocked. |
| `tritium_tritium_neutron_flux` | `invalid_validation_status` | `drafted / pending / quarantined` | `drafted / pending / quarantined` | Classified legitimately withheld; no clear-cut edit. |
| `tungsten_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |
| `vertical_coordinate_of_line_of_sight` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` | Clear-cut governed hint candidate; not staged after preflight blocked. |
| `volume_integrated_total_electron_density` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` | Clear-cut governed hint candidate; not staged after preflight blocked. |
| `xenon_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` | Rescore eligible; not run after cohort preflight blocked. |

## Quarantine classification

The recorded `validation_issues` were classified without changing graph text.
No direct rename is asserted where the issue does not determine the intended
physics. The five clear-cut corrections remain proposed `sn edit --hint`
directions, not applied graph state, because the node stopped at its hard
route mismatch.

### Grammar gap: 1

| Standard name | Recorded issue | Classification reason |
|---|---|---|
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | Grammar round-trip parse failure. | The issue identifies parser/grammar non-round-trip but supplies no semantically complete replacement; changing the identity would guess beyond the evidence. |

### Semantic defects with a clear-cut governed hint: 5

| Standard name | Recorded issue | Governed correction direction and reason |
|---|---|---|
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | Cumulative-prefix audit rejects `integrated_`; a DD `_inside` quantity must use `_inside_flux_surface` after the quantity. | Hint toward `spectral_wave_opacity_inside_flux_surface_at_ece_channel_emission_position`; reason: the DD quantity is inside a flux surface, so the canonical aggregation is a suffix and `line_integrated_` asserts the wrong integration form. |
| `phase_of_ion_cyclotron_heating_antenna` | Canonical-form audit rejects `phase_of_<X>` and calls for noun-suffix form. | Hint toward `ion_cyclotron_heating_antenna_phase`; reason: the quantity is the antenna phase, whose canonical identity places `phase` after its owner. |
| `toroidal_coordinate_of_detector_pixel` | Canonical-locus audit calls for `_at_detector_pixel`, not `_of_detector_pixel`. | Hint toward `toroidal_coordinate_at_detector_pixel`; reason: the coordinate is evaluated at the pixel, not possessed by it. |
| `vertical_coordinate_of_line_of_sight` | Canonical-locus audit calls for `_at_line_of_sight`, not `_of_line_of_sight`. | Hint toward `vertical_coordinate_at_line_of_sight`; reason: the coordinate is evaluated at that locus. |
| `volume_integrated_total_electron_density` | Cumulative-prefix audit rejects `integrated_`; a DD `_inside` quantity must use `_inside_flux_surface` after the quantity. | Hint toward `total_electron_density_inside_flux_surface`; reason: the DD source denotes content inside a flux surface and the grammar expresses that aggregation as a suffix. |

### Legitimately withheld from automatic editing: 4

| Standard name | Recorded issue | Why no edit is clear-cut |
|---|---|---|
| `deuterium_deuterium_neutron_flux` | Multiple subjects `deuterium_deuterium` and `neutron`. | The issue proves the identity invalid but does not settle whether the isotope pair is a process, source reaction, population, or other governed role. |
| `radial_outline_of_plasma_boundary` | `outline` does not identify the entity whose path or boundary is described; locus audit prefers `_at_plasma_boundary`. | `radial_outline_at_plasma_boundary` would repair only the preposition and retain the semantic ambiguity, so that tempting spelling is explicitly rejected as insufficient. |
| `radial_outline_of_wall` | `outline` does not identify the entity whose path or boundary is described; locus audit prefers `_at_wall`. | `radial_outline_at_wall` would repair only the preposition and retain the semantic ambiguity, so that tempting spelling is explicitly rejected as insufficient. |
| `tritium_tritium_neutron_flux` | Multiple subjects `tritium_tritium` and `neutron`. | The issue proves the identity invalid but does not settle the semantic role of the isotope pair. |

## GraphClient legacy-state census

The same query carried positive controls and was aimed explicitly at
`StandardName.name_stage`:

| Snapshot | Candidates | With `id` | With `name_stage` | `accepted` | `approved` | `contested` |
|---|---:|---:|---:|---:|---:|---:|
| Before | 4,688 | 4,688 | 4,688 | 2,336 | **0** | **0** |
| After | 4,688 | 4,688 | 4,688 | 2,336 | **0** | **0** |

The nonzero candidate and property-coverage controls prove the absence scan
fired against the intended key. Approved is **0** and contested is **0**.

Snapshot artifacts:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T134805682558-n-west-withheld-recovery/before.json`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T134805682558-n-west-withheld-recovery/after.json`
- Both files: SHA-256 `98c90e97cbcf6b59f94f627f47f7e8a2439a0c23310276cf950e47c7f4f9d75b`

## Safety statement

This node performed no direct acceptance, no graph-text Cypher mutation, no
`sn edit`, no review call, and no paid recovery run. The graph was read only
through `GraphClient`; the sole lifecycle command used `--dry-run` and refused
before mutation. The exact unresolved condition is authorization for a route
that can lawfully re-establish docs-review resolution on accepted names, or a
scoped code change that makes the mandated `sn rescore` route support that
case.
