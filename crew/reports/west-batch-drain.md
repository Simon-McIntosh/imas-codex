# WEST production DD-path batch drain

## Outcome

The single authorized drain completed with process exit status `0`, but it did
not advance any of the 28 pre-cut non-structural withheld names to a newly fully
accepted lifecycle. Twenty-seven lifecycle triplets were unchanged. One name,
`parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive`,
moved backward from `accepted / accepted / valid` to
`accepted / pending / valid` when startup source-drift refresh reset its
documentation. The count of fully accepted lifecycle triplets in this cohort
therefore fell from 4 to 3; the count that **became** fully accepted was **0**.

This is not evidence that the next cut will publish more of the 355 batch
sources. The run reported that 323 focused paths already had a live accepted
name, seeded the remaining 32 sources, and completed two compositions, five
name reviews, and three name refinements. Nevertheless, the graph-wide accepted
name count remained 2,336 and none of the exact 28 export-blocking identities
advanced. The four entries classified as `resolution_unrecorded` by the pre-run
export ledger illustrate why lifecycle triplets alone are not release
authority: all four began `accepted / accepted / valid`, but their docs-axis
reviews did not record a winning resolution method.

## Run receipt

| Field | Value |
|---|---|
| Command | `imas-codex sn run --batch west_production_dd_paths --cost-limit 25` |
| Process exit status | `0` |
| SNRun id | `06585023-175b-41b0-98a5-4123fc9c05b6` |
| Focus scope run id | `ea6ef04d-26a8-49b6-a199-50f17cc189d4` |
| Graph status | `completed` |
| Stop reason | `no_eligible_work` |
| Exact spend | `$1.684869` of `$25.000000` (`6.739476%`; `$23.315131` unspent) |
| Cost exactness | `cost_is_exact=true`; `review_cost=$0.805316`, `compose_cost=$0.879553` |
| Work counters | composed 2; enriched 0; reviewed 5; regenerated 3; events 55 |
| Started / ended | `2026-09-02T13:09:56.600255Z` / `2026-09-02T13:39:11.965000000+00:00` |
| Elapsed | `1672.726 s` |
| Transcript | `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T130422511580-n-west-batch-drain/sn-run.log` (SHA-256 `5b61fcb9d5c945c01063466830cb943f9cb9f2663399a3e743474749021ebb31`) |

The command recovered from two `generate_name` worker errors whose recorded
exception was `RuntimeError: source claim changed during transactional name
persistence`. Its final pool health had zero pending and zero in-flight work in
every pool, but retained `generate_name.error_count=2`. The successful process
exit and `SNRun.status=completed` therefore coexist with this operational
caveat.

## Exact 28-name lifecycle census

The target was aimed at the 28 distinct `StandardName.id` values in the latest
WEST export ledger at
`/home/ITER/mcintos/.cache/imas-codex/staging/.export_report.json`, restricted to
the four non-structural reasons shown below. Both snapshots matched all 28 ids.
The before and after columns are `name_stage / docs_stage /
validation_status`.

| Standard name | Pre-run export-ledger reason | Before | After |
|---|---|---|---|
| `beryllium_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `beta` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `breakdown_initial_time` | `resolution_unrecorded` | `accepted / accepted / valid` | `accepted / accepted / valid` |
| `carbon_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | `resolution_unrecorded` | `accepted / accepted / valid` | `accepted / accepted / valid` |
| `deuterium_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `deuterium_deuterium_neutron_flux` | `invalid_validation_status` | `drafted / pending / quarantined` | `drafted / pending / quarantined` |
| `helium_4_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `hot_neutral_temperature` | `name_not_accepted` | `reviewed / accepted / valid` | `reviewed / accepted / valid` |
| `hydrogen_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` | `invalid_validation_status` | `drafted / pending / quarantined` | `drafted / pending / quarantined` |
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | `invalid_validation_status` | `drafted / pending / quarantined` | `drafted / pending / quarantined` |
| `lithium_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `neon_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `normalized_poloidal_flux_coordinate` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | `resolution_unrecorded` | `accepted / accepted / valid` | `accepted / pending / valid` |
| `phase_of_ion_cyclotron_heating_antenna` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` |
| `poloidal_magnetic_flux` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `radial_coordinate_of_conductor_cross_section` | `resolution_unrecorded` | `accepted / accepted / valid` | `accepted / accepted / valid` |
| `radial_outline_of_plasma_boundary` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` |
| `radial_outline_of_wall` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` |
| `spectral_wavelength_of_optical_element` | `documentation_not_accepted` | `accepted / reviewed / valid` | `accepted / reviewed / valid` |
| `toroidal_coordinate_of_detector_pixel` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` |
| `tritium_tritium_neutron_flux` | `invalid_validation_status` | `drafted / pending / quarantined` | `drafted / pending / quarantined` |
| `tungsten_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |
| `vertical_coordinate_of_line_of_sight` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` |
| `volume_integrated_total_electron_density` | `invalid_validation_status` | `accepted / accepted / quarantined` | `accepted / accepted / quarantined` |
| `xenon_density_at_plasma_boundary` | `name_not_accepted` | `reviewed / pending / valid` | `reviewed / pending / valid` |

Snapshot artifacts:

- Before: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T130422511580-n-west-batch-drain/before.json` (SHA-256 `c212405ab7c6f3b46429acf44c5f4c7f6ee18a61f10168dabc1d28f2a2fc9f26`).
- After: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260902T130422511580-n-west-batch-drain/after.json` (SHA-256 `3e8ff372af133e5a028ed0adfddd827c41394e45761cb2cf4f5afcac506d7055`).

## Remaining quarantine reasons

All 10 names that began quarantined remained quarantined. The graph's
`quarantine_reason` property was null for each; the actionable reasons are in
`validation_issues`:

- `deuterium_deuterium_neutron_flux`: multiple subjects
  `deuterium_deuterium` and `neutron`.
- `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width`:
  grammar round-trip parse failure.
- `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position`:
  cumulative-prefix audit rejects `integrated_`; a DD `_inside` quantity must
  use the `_inside_flux_surface` suffix after the quantity.
- `phase_of_ion_cyclotron_heating_antenna`: canonical-form audit rejects the
  `phase_of_<X>` prefix; it calls for noun-suffix form
  `ion_cyclotron_heating_antenna_phase`.
- `radial_outline_of_plasma_boundary`: semantic validation says `outline` does
  not identify the entity whose path or boundary is described, and the locus
  audit calls for `radial_outline_at_plasma_boundary` rather than `_of_`.
- `radial_outline_of_wall`: semantic validation says `outline` does not identify
  the entity whose path or boundary is described, and the locus audit calls for
  `radial_outline_at_wall` rather than `_of_`.
- `toroidal_coordinate_of_detector_pixel`: canonical-locus audit calls for
  `toroidal_coordinate_at_detector_pixel` rather than `_of_`.
- `tritium_tritium_neutron_flux`: multiple subjects `tritium_tritium` and
  `neutron`.
- `vertical_coordinate_of_line_of_sight`: canonical-locus audit calls for
  `vertical_coordinate_at_line_of_sight` rather than `_of_`.
- `volume_integrated_total_electron_density`: cumulative-prefix audit rejects
  `integrated_`; a DD `_inside` quantity must use the
  `_inside_flux_surface` suffix after the quantity.

These are diagnostic spellings from the graph, not accepted rename proposals;
any correction still has to pass the governed `sn edit` and review route.

## GraphClient legacy-state census

The same GraphClient query supplied its own positive controls and was explicitly
aimed at `StandardName.name_stage`:

| Snapshot | StandardName candidates | With `id` | With `name_stage` | `accepted` | `approved` | `contested` |
|---|---:|---:|---:|---:|---:|---:|
| Before | 4,683 | 4,683 | 4,683 | 2,336 | **0** | **0** |
| After | 4,688 | 4,688 | 4,688 | 2,336 | **0** | **0** |

The nonzero candidate and key-coverage controls prove the scan fired against the
intended property. The required absence result is therefore approved **0** and
contested **0** both before and after the drain.

## Follow-on implication

The prescribed batch command is gap-only: it skips paths already attached to a
live accepted name and does not by itself re-stage these pre-existing reviewed,
quarantined, or resolution-unrecorded identities. A separately authorized
recovery must use the governed lifecycle routes appropriate to each class
(fresh quorum rescore/refine for sound reviewed names, `sn edit` for semantic or
grammar corrections, and docs review for missing docs authority). No direct
acceptance or graph-text edit is supported by this evidence.
