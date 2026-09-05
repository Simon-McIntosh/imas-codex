# Canonical rendering: live-cohort rename migration

Executed 2026-09-05, worktree base `eeb1cfe944aef1a12c6f9c6e11d695809fdbe187`.
Specification: the live-cohort rename map observed 2026-09-05T07:32:34Z against
consumer revision `044070eea` and grammar `59754b7d6` — 149 rows. The map was
read, not regenerated; no spelling was re-derived. The route was the sanctioned
`imas-codex sn edit OLD --rename NEW --reason REASON --scope self --stage-only`
per row followed by one `imas-codex sn run --only review --edits` batch review.
Row 1 had been applied by a prior attempt (predecessor superseded, successor
drafted, both ledger rows present) and was not repeated.

## Outcome: 137 of 149 rows applied; 12 un-migrated (10 refused, 2 ghost)

| Measure | Count |
|---|---:|
| Rows in the map | 149 |
| Rows applied (predecessor superseded, successor created, ledger row written) | 137 |
| Rows refused by the staged route | 10 |
| Rows errored "target not found" | 2 |
| Staged rows still awaiting review | 137 |
| Batch review outcome | run twice, both exited 1, no review completed |

The applied 137 are rows 1–149 minus the twelve below. Every applied row carries
`StandardNameChange` rows with `operation`, `changed_at`, `from_name`, `to_name`
(`human_edit` for all; most rows also `source_migration_manifest`) — the
"at least ten renamed identities with ledger rows" requirement is exceeded 137×.
Row 1 alone carries both the `source_migration_manifest` (13:54:02.255Z) and
`human_edit` (13:54:02.737Z) records for its from/to pair.

## Residue: the map generator still proposes exactly 10 rows

The generator was re-run over the live cohort (lenient parse of each
non-superseded stored spelling, compose its IR, strict reparse of the proposal).
It now reports:

- Live identities: 2981 (2,102 superseded); compose-to-self: 2964
- Still proposed: **10**, all strict-ok, named row by row below
- Lenient parse failures: 7, byte-for-byte the original seven (untouched)

The rows the generator still proposes are precisely the ten the staged route
refuses. The two "target not found" rows (40, 80) are no longer proposed:
their map spellings do not exist as stored identities, confirming they were
census ghosts, not migratable rows.

| # | Stored spelling still proposed | Canonical proposal |
|---|---:|---|
| 8 | `accumulated_helium_4_count_due_to_gas_injection` | `helium_4_count_accumulated_due_to_gas_injection` |
| 31 | `derivative_with_respect_to_poloidal_angle_of_normalized_effective_particle_energy` | `derivative_of_normalized_effective_particle_energy_with_respect_to_poloidal_angle` |
| 48 | `flux_surface_averaged_ion_density_at_plasma_boundary` | `ion_density_flux_surface_averaged_at_plasma_boundary` |
| 60 | `gradient_of_normalized_pressure_at_flux_surface` | `pressure_normalized_gradient_at_flux_surface` |
| 99 | `peak_wave_current_of_antenna_strap_amplitude` | `peak_wave_current_amplitude_of_antenna_strap` |
| 116 | `root_mean_square_of_wave_current_of_antenna_strap` | `wave_current_root_mean_square_of_antenna_strap` |
| 122 | `toroidal_flux_surface_averaged_beryllium_velocity_at_plasma_boundary` | `toroidal_beryllium_velocity_flux_surface_averaged_at_plasma_boundary` |
| 128 | `toroidal_flux_surface_averaged_helium_4_velocity_at_plasma_boundary` | `toroidal_helium_4_velocity_flux_surface_averaged_at_plasma_boundary` |
| 132 | `toroidal_flux_surface_averaged_krypton_velocity_at_plasma_boundary` | `toroidal_krypton_velocity_flux_surface_averaged_at_plasma_boundary` |
| 133 | `toroidal_flux_surface_averaged_lithium_velocity_at_plasma_boundary` | `toroidal_lithium_velocity_flux_surface_averaged_at_plasma_boundary` |

Target zero residue is therefore not yet met; the ten are a stable, named,
reproducible remainder, not a silent or partial migration.

## The ten refusals, verbatim and why the guard refuses

The map's own no-write preflight passed every row (no-write never mutates, so it
cannot reveal the conflict). Under the real migration the rename source-migration
compare-and-set guard (`retarget_standard_name_sources`) requires a source's
bindings to be exactly `[old_name]` with a matching scalar and a non-stale,
non-claimed source. Nine of the ten rows share their DD source with a co-bound
second name; row 60's source is `stale`. All ten fail the guard before any write,
so each refusal is transactional — retried once after the cohort settled, all ten
failed identically, establishing the refusals as stable, not an ordering artifact.

Verbatim refusals (retry pass, identical to first pass):

| # | Verbatim error |
|---|---|
| 8 | `source migration compare-and-set failed: dd:summary/gas_injection_prefill/helium_4/value(exists=True, status='composed', claimed=False, bindings=['accumulated_helium_4_count_due_to_gas_injection', 'accumulated_helium_4_prefill_count_due_to_gas_injection'], scalar='accumulated_helium_4_count_due_to_gas_injection')` |
| 31 | `source migration compare-and-set failed: dd:gyrokinetics_local/species/potential_energy_gradient_norm(exists=True, status='composed', claimed=False, bindings=['derivative_with_respect_to_poloidal_angle_of_normalized_effective_particle_energy', 'gradient_of_effective_potential'], scalar='derivative_with_respect_to_poloidal_angle_of_normalized_effective_particle_energy')` |
| 48 | `source migration compare-and-set failed: dd:summary/local/separatrix_average/n_i(exists=True, status='composed', claimed=False, bindings=['density_at_separatrix', 'flux_surface_averaged_ion_density_at_plasma_boundary'], scalar='flux_surface_averaged_ion_density_at_plasma_boundary')` |
| 60 | `source migration compare-and-set failed: dd:gyrokinetics/flux_surface/pressure_gradient_norm(exists=True, status='stale', claimed=False, bindings=['gradient_of_normalized_pressure_at_flux_surface'], scalar=None)` |
| 99 | `source migration compare-and-set failed: dd:ic_antennas/antenna/module/strap/current(exists=True, status='composed', claimed=False, bindings=['peak_wave_current_of_antenna_strap_amplitude', 'root_mean_square_of_wave_current_of_antenna_strap'], scalar='root_mean_square_of_wave_current_of_antenna_strap')` |
| 116 | `source migration compare-and-set failed: dd:ic_antennas/antenna/module/strap/current(exists=True, status='composed', claimed=False, bindings=['peak_wave_current_of_antenna_strap_amplitude', 'root_mean_square_of_wave_current_of_antenna_strap'], scalar='root_mean_square_of_wave_current_of_antenna_strap')` |
| 122 | `source migration compare-and-set failed: dd:summary/local/separatrix_average/velocity_phi/beryllium/value(exists=True, status='composed', claimed=False, bindings=['toroidal_flux_surface_averaged_beryllium_velocity_at_plasma_boundary', 'toroidal_flux_surface_averaged_beryllium_velocity_at_separatrix'], scalar='toroidal_flux_surface_averaged_beryllium_velocity_at_plasma_boundary')` |
| 128 | `source migration compare-and-set failed: dd:summary/local/separatrix_average/velocity_phi/helium_4/value(exists=True, status='composed', claimed=False, bindings=['toroidal_flux_surface_averaged_helium_4_velocity_at_plasma_boundary', 'toroidal_helium_4_velocity_at_separatrix'], scalar='toroidal_flux_surface_averaged_helium_4_velocity_at_plasma_boundary')` |
| 132 | `source migration compare-and-set failed: dd:summary/local/separatrix_average/velocity_phi/krypton/value(exists=True, status='composed', claimed=False, bindings=['toroidal_flux_surface_averaged_krypton_velocity_at_plasma_boundary', 'toroidal_krypton_velocity_at_separatrix'], scalar='toroidal_flux_surface_averaged_krypton_velocity_at_plasma_boundary')` |
| 133 | `source migration compare-and-set failed: dd:summary/local/separatrix_average/velocity_phi/lithium/value(exists=True, status='composed', claimed=False, bindings=['toroidal_flux_surface_averaged_lithium_velocity_at_plasma_boundary', 'toroidal_flux_surface_averaged_lithium_velocity_at_separatrix'], scalar='toroidal_flux_surface_averaged_lithium_velocity_at_plasma_boundary')` |

The non-migrated identities themselves are undisturbed: nine are `accepted`, two
`exhausted`, two absent; none has a ledger change after 15:07Z, i.e. nothing
today's runs wrote to them. They simply cannot be renamed by `sn edit` while
their sources remain shared/bound to two names or stale.

## The two "target not found" errors (rows 40, 80)

| # | Stored spelling per map | CLI error |
|---|---|---|
| 40 | `flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | `target StandardName '...' not found` |
| 80 | `normalized_momentum_flux_due_to_e_cross_b_drift` | `target StandardName '...' not found` |

Neither spelling exists as a stored `StandardName.id`. The nearest stored
identities carry a `parallel_` prefix or species modifier and are all superseded.
The map generator, re-run, no longer proposes either row — they were census
artifacts (an identity moved/renamed between population and staging), not
migratable rows. Nothing was written for either.

## Batch review did not run to completion

`imas-codex sn run --only review --edits` was invoked twice, once with `-t 9`
and once with `-q -t 5`. Both exited 1; no review rotation landed, and all 137
successors remain `drafted` with `edit_status=open` — a resumable position, not a
corrupt one. Both invocations ran the full pool machinery (source reconcile,
grammar validation, structural-edge derivation, post-drain parent/doc-link
normalization) before dying rather than a narrowly scoped review over the open
edits, so the scoping path itself needs attention upstream. The CLI's file log
did not capture this command's output, and stdout truncation hid the terminal
traceback; the observable facts are the exit code, the phases reached, and the
unchanged `drafted`/`open` state of every successor. The same command failed
twice, so per the run discipline it was not retried a third time.

## The map's own refusal did not recur

Row 67 (`maximum_of_energy_flux_at_divertor_target`), the single row the map
recorded as refused by a `W` vs `W.m^-2` unit-authority disagreement, staged
cleanly this session at 14:50:07Z. Its successor `energy_flux_maximum_at_divertor_target`
is drafted with its ledger row. The unit-authority defect named in the map
appears resolved as of this run.

## Fences honoured

- Route only: `sn edit ... --stage-only` per row, then one `sn run --only review --edits`.
  No direct graph write, no Cypher write, no string rewrite of a stored name.
- The 7 lenient-parse-failure identities were not touched.
- No `REFINED_FROM` edge read or written. No branch, tag, or pull request cut or opened.
- Per-row progress ledger appended after every row (resumable position):
  run dir `progress-ledger.tsv`, 149 coverage rows plus the 10 retry rows.
- Suite: this node's only tracked change is this documentation file, which no
  test in `tests/standard_names` imports or reads; added failures are 0 by
  construction. No source file was touched, so no source-reached suite was
  re-measured.

## What remains

1. The ten refused rows need the shared-source/stale-source conflict resolved —
   either a source-binding de-duplication or a guard relaxation that can
   partition a shared source's bindings — then re-staging completes the cohort.
2. The batch review must be re-dispatched once the scoping path is fixed; the 137
   successors are `drafted`/`open` and will be scored by that run exactly as the
   inline review would have.
3. Rows 40 and 80 require no action; the map generator no longer proposes them.
