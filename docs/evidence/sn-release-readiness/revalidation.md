# Accepted-name deterministic revalidation

Measured against the live production `codex` graph on 2026-08-23. This run
changed only validation stamps and results for an explicit 28-identity scope.
It made no LLM call and incurred no LLM cost.

## Outcome

The sanctioned scoped admission path cleared all 28 identities that the
grounded audit regression cohort identified. All 28 were quarantined at the
start, all 28 were re-stamped `validation_status='valid'`, and none was
re-quarantined. The complete accepted-and-docs-accepted quarantine cohort fell
from **42 to 15**. The corresponding valid cohort rose from **2,022 to 2,049**.

The total `StandardName` count was **4,666 before and 4,666 after** (delta
**0**). This is the exact operation-window sentinel: concurrent refinement had
already raised the count from the earlier 4,395-node evidence snapshot, but no
successor was minted during this revalidation.

| Measure | Before | After | Change |
|---|---:|---:|---:|
| Explicit revalidation scope still quarantined | 28 | 0 | -28 |
| Accepted/approved + docs accepted + quarantined | 42 | 15 | -27 |
| Accepted/approved + docs accepted + valid | 2,022 | 2,049 | +27 |
| Production `_fetch_candidates` eligibility, before score gate | 537 | 540 | **+3** |
| All `StandardName` nodes | 4,666 | 4,666 | 0 |

The `-27`, rather than `-28`, in the two lifecycle-cohort rows is real:
`magnetic_field_at_pedestal_top_low_field_side` was already
`docs_stage='drafted'` at the start of this run, so it was outside the
accepted-docs cohort even though it was one of the 28 validation targets.

The exact production eligibility payoff is **+3**, not an inferred +28. The
three newly eligible identities are:

- `electron_density_at_pellet_path`
- `energy_confinement_enhancement_factor`
- `perturbed_plasma_velocity`

Of the other 25 validation-cleared identities, 24 remain excluded by
`documentation_review_unresolved` because their accepted documentation has no
`docs_review_resolution_method`; the remaining
`magnetic_field_at_pedestal_top_low_field_side` remains excluded because its
documentation is drafted, not accepted. This live state contradicts the earlier
assumption that every cleared row would immediately satisfy every production
export predicate. Validation clearance is complete; documentation-resolution
repair is separate work.

## Sanctioned path used

The operation used `default_audit_revalidate` in
`imas_codex/standard_names/campaign.py:746`. That function clears only the
named identities' validation/claim stamps and calls the ID-scoped drain. The
drain is `drain_validation_for_ids` in
`imas_codex/standard_names/workers.py:4857`; its claim function at line 4803
matches only IDs in the supplied list, requires the cleared validation stamp,
uses a claim token, and delegates classification to the shared
`validate_name_candidate` admission gate at line 4657. The marker write remains
inside the existing token-verified `mark_names_validated` path.

This is the production same-gate path, not a status override: the 28 rows first
had `validated_at` cleared, were reclaimed by exact identity, passed the current
grammar, ISN, description, and codex audit layers, and were then re-stamped from
their computed results. The returned receipt was:

- `cleared`: 28
- `valid_ids`: 28
- `requarantined_ids`: 0

No Cypher statement directly set `validation_status='valid'`, no audit was
bypassed, no name or documentation text changed, and no model entry point was
called.

## Per-identity result

Each row below carried the named stored audit finding before revalidation. A
pass means that finding is absent from the freshly computed issue list and the
shared admission gate returned `valid`.

| Identity | Former audit | Current audit result | Status after |
|---|---|---|---|
| `perturbed_electrostatic_potential_imaginary_part` | `name_description_consistency_check` | PASS | valid |
| `perturbed_electrostatic_potential_real_part` | `name_description_consistency_check` | PASS | valid |
| `perturbed_plasma_mass_density` | `name_description_consistency_check` | PASS | valid |
| `perturbed_plasma_mass_density_imaginary_part` | `name_description_consistency_check` | PASS | valid |
| `perturbed_plasma_pressure` | `name_description_consistency_check` | PASS | valid |
| `perturbed_plasma_pressure_imaginary_part` | `name_description_consistency_check` | PASS | valid |
| `perturbed_plasma_pressure_real_part` | `name_description_consistency_check` | PASS | valid |
| `perturbed_plasma_velocity` | `name_description_consistency_check` | PASS | valid |
| `poloidal_perturbed_plasma_magnetic_field_imaginary_part` | `name_description_consistency_check` | PASS | valid |
| `poloidal_perturbed_plasma_magnetic_field_real_part` | `name_description_consistency_check` | PASS | valid |
| `poloidal_perturbed_plasma_velocity` | `name_description_consistency_check` | PASS | valid |
| `poloidal_perturbed_vacuum_magnetic_field` | `name_description_consistency_check` | PASS | valid |
| `poloidal_perturbed_vacuum_magnetic_field_imaginary_part` | `name_description_consistency_check` | PASS | valid |
| `radial_perturbed_plasma_velocity` | `name_description_consistency_check` | PASS | valid |
| `radial_perturbed_plasma_velocity_real_part` | `name_description_consistency_check` | PASS | valid |
| `radial_perturbed_vacuum_magnetic_field_imaginary_part` | `name_description_consistency_check` | PASS | valid |
| `toroidal_perturbed_plasma_magnetic_field_imaginary_part` | `name_description_consistency_check` | PASS | valid |
| `toroidal_perturbed_plasma_velocity` | `name_description_consistency_check` | PASS | valid |
| `toroidal_perturbed_vacuum_magnetic_field` | `name_description_consistency_check` | PASS | valid |
| `toroidal_perturbed_vacuum_magnetic_field_real_part` | `name_description_consistency_check` | PASS | valid |
| `energy_confinement_enhancement_factor` | `name_unit_consistency_check` | PASS | valid |
| `tendency_of_rotation_frequency_of_neoclassical_tearing_mode` | `name_unit_consistency_check` | PASS | valid |
| `tendency_of_total_thermal_plasma_internal_energy` | `name_unit_consistency_check` | PASS | valid |
| `ion_field_line_average_temperature_over_scrape_off_layer` | `implicit_field_check` | PASS | valid |
| `magnetic_field_at_pedestal_top_low_field_side` | `implicit_field_check` | PASS | valid |
| `poloidal_electron_beta_at_pedestal_top_high_field_side` | `implicit_field_check` | PASS | valid |
| `poloidal_electron_beta_at_pedestal_top_low_field_side` | `implicit_field_check` | PASS | valid |
| `electron_density_at_pellet_path` | `multi_subject_check` | PASS | valid |

Grouped result: 20/20 name-description rows passed, 3/3 name-unit rows
passed, 4/4 implicit-field rows passed, and 1/1 multi-subject row passed.

## Quarantines that remain

None of the 28 scoped identities remains quarantined. The complete live
accepted/approved + docs-accepted quarantine population now contains the 15
rows below; these retained findings were not force-cleared.

| Identity | Current quarantine authority |
|---|---|
| `cumulative_inside_flux_surface_torque` | `cumulative_prefix_check` |
| `perturbed_particle_pressure` | `name_unit_consistency_check` |
| `perturbed_pressure` | `name_unit_consistency_check` |
| `phase_of_ion_cyclotron_heating_antenna` | `amplitude_of_prefix_check` |
| `plasma_internal_energy` | `unit_dimension_check`; `name_unit_consistency_check` |
| `radial_outline_of_flux_surface` | ISN semantic ERROR; `canonical_locus_check` |
| `radial_outline_of_plasma_boundary` | ISN semantic ERROR; `canonical_locus_check` |
| `radial_outline_of_wall` | ISN semantic ERROR; `canonical_locus_check` |
| `rotation_frequency_due_to_e_cross_b_drift` | `latex_def_check` |
| `thermal_plasma_internal_energy` | `unit_dimension_check`; `name_unit_consistency_check` |
| `toroidal_angle_of_active_limiter_point` | `name_unit_consistency_check` |
| `toroidal_coordinate_of_detector_pixel` | `canonical_locus_check` |
| `vertical_coordinate_of_line_of_sight` | `canonical_locus_check` |
| `volume_integrated_runaway_electron_density` | `cumulative_prefix_check` |
| `volume_integrated_total_electron_density` | `cumulative_prefix_check` |

## Three-row caveat re-checked

The earlier audit commentary warned against assuming the census disposition or
its separate baseline reading for three identities. Their current live status
was checked both before and after this operation:

| Identity | Current status | Current audit |
|---|---|---|
| `cumulative_inside_flux_surface_torque` | quarantined | `cumulative_prefix_check` |
| `volume_integrated_runaway_electron_density` | quarantined | `cumulative_prefix_check` |
| `volume_integrated_total_electron_density` | quarantined | `cumulative_prefix_check` |

All three therefore remain excluded. They were outside the 28-ID grounded-audit
scope and were not mutated. This current result differs from the earlier note
that they were valid in a separate baseline; the live graph, not either prior
reading, is reported here.

## Verification record

- Focused regression and sanctioned-path tests:
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/standard_names/test_audit_false_positives.py tests/standard_names/test_campaign.py -q`
  — PASS, exit 0, 100% of the selected tests.
- Operation receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T161945785708-n-revalidate/logs/revalidation-operation.json`
- Start-state rows and stored former findings:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T161945785708-n-revalidate/logs/start-snapshot.json`
- Post-validation production-gate partition:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T161945785708-n-revalidate/logs/post-gate-partition.json`
- Remaining accepted-and-docs-accepted quarantines:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T161945785708-n-revalidate/logs/remaining-quarantine.json`
- Test log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T161945785708-n-revalidate/logs/scoped-validation-tests.log`

