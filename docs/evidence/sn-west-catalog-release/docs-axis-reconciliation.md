# Docs-axis reconciliation

A lifecycle scalar can be reset by unrelated work while the review history that
justified it survives. Thirty-one names sat in exactly that state on 2026-09-09:
every one carried surviving docs-axis `HAS_REVIEW` edges and a null
`reviewer_score_docs` with `docs_stage='pending'`, and they were restored by
hand-written Cypher, one value per row. This records the named reconciliation
that replaces the hand patch, and what that rule reproduces.

## The selection rule, in one sentence

A name's docs axis is taken from its **winning surviving docs-axis review** — the
highest-ranked surviving docs review group (a canonical group ahead of a
non-canonical one, then the newest group, then the newest winning record inside
it), which is the same selection the review-eligibility predicate already uses.

The statement does not restate that ordering. It calls the existing
`_docs_review_winner_query_body`, so the reconciliation and the eligibility
predicate cannot drift apart on which surviving review is authoritative — a
second traversal would have been a second source of truth for the same question.

## What the rule does

| Half | Behaviour |
|---|---|
| Restore | A name in scope (`name_stage='accepted'`, not quarantined) with a winning surviving docs review has `docs_stage` set to `accepted` and `reviewer_score_docs` set to the winner's score. |
| Refuse | A name with **no** docs-axis review at all has no winner, is not in the statement's row set, and is left untouched. This is the half that stops the function manufacturing false acceptances. |
| Idempotent | Apply mode filters on the mirror disagreeing, so a re-run writes nothing. |

**This node wrote no graph data.** `graph_mutations: 0`; every measurement below
is a dry run.

## Reproduction against the 31 hand-patched rows

Per-row verdicts, from the rule itself (one dry run per name), alongside the
value the row currently stores:

| Verdict | Count | Names |
|---|---|---|
| agrees with the stored value | 5 | `electron_density_at_plasma_boundary`, `magnetic_shear_at_flux_surface`, `poloidal_magnetic_flux_at_measurement_position`, `radial_coordinate_of_magnetic_axis`, `toroidal_magnetic_flux` |
| disagrees | 26 | `effective_charge`, `faraday_angle`, `frequency_of_ion_cyclotron_heating_antenna`, `gap_at_outboard_midplane`, `initial_polarization_ellipticity_of_polarimeter_beam`, `launched_power_of_lower_hybrid_antenna`, `line_integrated_electron_number_density`, `maximum_magnetic_field_magnitude`, `normalized_plasma_internal_inductance`, `normalized_toroidal_flux_coordinate_at_measurement_position`, `plasma_current`, `poloidal_angle_of_flux_surface`, `poloidal_angle_of_measurement_position`, `poloidal_magnetic_flux_at_flux_surface`, `poloidal_magnetic_flux_of_flux_loop`, `radial_coordinate_of_geometric_axis`, `radial_coordinate_of_strike_point`, `radial_outline_of_antenna_strap`, `safety_factor`, `toroidal_beta`, `toroidal_magnetic_flux_due_to_diamagnetic_drift`, `total_power_due_to_ion_cyclotron_heating`, `vertical_coordinate_of_camera`, `vertical_coordinate_of_geometric_axis`, `vertical_coordinate_of_strike_point`, `volume_averaged_electron_density` |
| refused (no winning docs review) | 0 | none — all 31 carry a surviving winning docs review |

So **the rule reproduces 5 of the 31** and does **not** reproduce the other 26.
That is a negative result about the hand patch, not about the rule, and it
matches what the plan already found: no single rule covers the cohort, because
the hand-written pass mixed selection criteria per row. Two of the 26 are
diagnostic — for `faraday_angle` (stored 0.975, best surviving 0.975) and
`radial_coordinate_of_strike_point` (stored 0.9375, best 0.9375) the stored value
equals the **highest-scoring** surviving docs review while the named rule selects
the newest/canonical group's winner, which scores lower. The hand patch appears
to have preferred the best-scoring review; the named rule prefers the winning
group. Where the two criteria coincide, the values agree (the 5 above).

Neither value is fabricated: every stored score appears among its own row's
docs-axis review scores. The rule is the right one to name because it is the one
the pipeline's own eligibility predicate already applies; re-deriving through it
makes the mirror agree with the pipeline by construction. Re-deriving the 31
through it would change 26 stored scores, which is a data decision this node did
**not** take — see *Open decisions*.

## The refusal half, shown

Two names that carry **no** docs-axis review at all were used as the control:

| Name | Stored | Rule |
|---|---|---|
| `alpha_critical_parameter` | `docs_stage='pending'`, score null | refused |
| `alpha_reconstructed_parameter` | `docs_stage='pending'`, score null | refused |

`with_winner 0`, `refused_no_review 2`, and both left unchanged. This is the
guarded thing made to happen: without this half the function would set
`docs_stage='accepted'` for any name it scans, which is the false-acceptance
projection the cohort's own followup warns about.

## An instrument defect caught mid-measurement

The first aggregate dry run reported `with_winner 30, refused_no_review 1` over a
31-name list, which read as "one row the rule cannot cover". It was my probe, not
the graph. The cohort literal in that probe carried a cosmetic
`.replace("cXamera", "amera")` edit over `vertical_coordinate_of_camera`, which
consumed the leading `c` and produced the **nonexistent id**
`vertical_coordinate_of_amera`; the batch matched 30 nodes and the arithmetic
`len(ids) - with_winner` invented the one refusal. The per-row pass, over a list
with no such edit, finds all 31 rows present and none refused.

The lesson is the ordinary one and worth restating because it very nearly became
a reported number: an implausible measurement is a claim about the instrument
first. A `refused_no_review` of 1 inverting to 0 was the tell, and the
discrepancy is visible only because the check was run per row as well as in
aggregate.

## Evidence

| Item | Value |
|---|---|
| Function | `reconcile_docs_axis_from_reviews` in `imas_codex/standard_names/graph_ops.py` |
| Shared rule reused | `_docs_review_winner_query_body` (extended to carry the winner's score) |
| Commit | `8d3a4e66d` |
| Test | `tests/standard_names/test_docs_axis_reconciliation.py` — 4 tests |
| Test command | `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv UV_NO_SYNC=1 PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/standard_names/test_docs_axis_reconciliation.py` |
| Test result | `4 passed, 1 warning in 8.24s`, exit 0 |
| Cohort dry run | `{"agree": 5, "repair": 26, "with_winner": 31, "refused_no_review": 0}`, 0.74 s |
| Control dry run | `{"agree": 0, "repair": 0, "with_winner": 0, "refused_no_review": 2}` |
| Graph mutations | 0 |

The test proves both halves: it restores a row whose docs scalar was regressed
while its surviving docs review stands, and it refuses a row with no docs-axis
review, leaving that row's `docs_stage` and score untouched. It also asserts that
a second apply writes nothing.

## Open decisions

The rule disagrees with 26 of the 31 stored values. Running the apply mode would
overwrite them with the winning group's score. That is a live-data decision for
the release owner, not a side effect of landing the function, so the function
landed with apply mode available and this node ran only the dry run.