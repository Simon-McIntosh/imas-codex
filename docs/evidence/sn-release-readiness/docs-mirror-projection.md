# Documentation review-resolution projection

## Outcome

The sanctioned writer `project_docs_review_resolution_methods` now projects a
docs-axis winning review group's resolution method onto an already accepted
`StandardName` without restaging its documentation, changing its score, or
calling an LLM. The writer lives beside `persist_reviewed_docs`, the existing
owner of `docs_review_resolution_method`, and uses a null-only compare-and-set.

The live backfill mirrored **1,400** rows. The null accepted-docs population
fell from **1,516 to 116**, and a second invocation wrote **0** rows. The
`StandardName` population stayed **4,666 before and after**.

| Measure | Before | After |
|---|---:|---:|
| `StandardName` nodes | 4,666 | 4,666 |
| Accepted names with accepted docs and a null method | 1,516 | 116 |
| Null-method rows with both docs score and timestamp | 1,408 | 9 |
| Rows mirrored by winning method | 0 | 1,400 |
| Rows mirrored on immediate replay | — | 0 |

The complete identity-level receipt is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T201417422211-n-mirrormech/docs-mirror-projection-receipt.json`.
It contains every mirrored Standard Name id, source review-group id, source
review id, and mirrored method. Its method distribution is 1,140
`quorum_consensus`, 133 `authoritative_escalation`, and 127 `single_review`.

## Schema-owned admission and the non-winning guard

The winning set is parsed from the `ReviewResolutionMethod` enum description in
`imas_codex/schemas/standard_name.yaml`; it is not duplicated as a method list
in the writer. New enum values therefore remain excluded until the schema
explicitly declares them eligible. Focused tests assert that both
`max_cycles_reached` and `retry_item` are outside the derived set and that a
synthetic future terminal value is not silently admitted.

The guard is deliberately identity-wide over docs-axis terminal history. The
first operational pass showed why: each of the two rows with a reachable
`max_cycles_reached` group also had older winning history. Filtering to winning
groups before checking the full history would therefore recover an older win
and hide the unresolved terminal review. The module-owned correction path
restored both mirrors to null, the final writer vetoed them, and replay remained
a zero-write operation.

The two identities that remain null for this reason are:

- `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_electron_density_at_pedestal_maximum`
- `tritium_density`

Both still have `docs_review_resolution_method = null`; neither became
export-eligible through a review history containing a non-winning terminal
group.

## The 116 rows left null

The residual is a complete, non-overlapping partition:

| Reason | Rows | Disposition |
|---|---:|---|
| No docs-axis review exists | 107 | Separate targeted docs rescore; catalog provenance is not review authority |
| Docs reviews exist but no terminal resolution method was recorded | 7 | Historical write defect; no method can be inferred by this projection |
| A non-winning terminal method is reachable | 2 | Intentionally left null by the schema guard |
| **Total** | **116** | |

The seven score-and-timestamp rows with no terminal method are
`accumulated_ethylene_count`, `line_averaged_deuterium_tritium_density`,
`power_at_divertor_target_due_to_conduction`, `power_due_to_first_orbit_loss`,
`radial_coordinate_of_filter_window`,
`vertical_coordinate_of_poloidal_magnetic_field_probe`, and
`wetted_area_of_divertor_target`. The durable row-level residual census is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T201417422211-n-mirrormech/null-method-residuals.json`.

## Independent score-mirror defect

`krypton_density_at_magnetic_axis` is the one accepted, docs-accepted,
`catalog_edit` identity with attached docs review history and a null
`reviewer_score_docs`. This projection did cover its missing resolution method:
it now carries `quorum_consensus` from its winning docs group and appears in the
production export. It deliberately did **not** invent a score or timestamp;
`reviewer_score_docs` and `reviewed_docs_at` remain null. A score projection
needs its own sanctioned, axis-specific path and evidence because it changes a
different authority slot. No such write was forced here.

## Production export measure

Both measurements used the real `run_export` path with `min_score=0.85`,
`skip_gate=True`, `force=True`, and `include_sources=False`. They did not
reconstruct the release predicate by hand.

| Production-path measure | Recorded baseline | Fresh before | After |
|---|---:|---:|---:|
| Emitted identities | 537 | 559 | **1,947** |
| Accepted/approved export population | 2,335 | 2,335 | 2,335 |
| Exclusion accounting | passed | passed | passed |
| `documentation_review_unresolved` | 1,508 | 1,508 | **116** |

The graph had moved by 22 emitted identities before this node wrote anything,
so the mirror's attributable production-path gain is **1,388** from the fresh
559-row baseline, while the change against the recorded 537-row baseline is
1,410. Twelve of the 1,400 newly mirrored rows remain excluded by independent
release gates or catalog validation, so a method mirror does not imply emission.

Durable export reports:

- Before: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T201417422211-n-mirrormech/before-export/.export_report.json`
- After: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T201417422211-n-mirrormech/after-export/.export_report.json`

No LLM function was called, no review was rerun, and no review score,
timestamp, documentation string, lifecycle stage, or identity was changed.


---

**Correction, 2026-08-25.** The present-tense claims in this record about
`docs_review_resolution_method` no longer hold. Commit `23c4b55a` deleted the field
together with its clearing function and every writer, and
`project_docs_review_resolution_method` returns zero hits across `imas_codex/`. Read
the assertions above as describing the state at the time of measurement, which they
recorded correctly, not as current behaviour. The measurements themselves stand.
