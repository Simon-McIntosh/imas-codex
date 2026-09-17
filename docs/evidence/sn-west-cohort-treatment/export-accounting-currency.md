# WEST export accounting: currency of the eighteen-name claim

Re-measurement of the plan's §9b assertion that **eighteen candidate names left the
WEST export with both declared exclusion counters reading zero and the
`exclusion_accounting` gate passing**, against the current WEST manifest cohort.

## Result

The instrument is `run_export` invoked directly against the current cohort: the
committed manifest's terminal name identities handed in as `review_batch`, and the
manifest's source rows as `manifest_sources`. It writes a staging tree on local disk
and reads the graph; it performs no graph mutation. No hand-rolled predicate stands
in for it.

```text
N  candidate count handed to the export     230
E  emitted                                  230
   per-reason exclusions                       0
   accounted_exclusions                        0
   accounting_residue (N - E - exclusions)      0

E + sum(per-reason exclusions) = 230 + 0 = 230 = N   exactly
```

**No identity left the export uncounted. The shortfall is zero, so there is no
silently-dropped-identity list to enumerate.** Every gate passed:
`catalog_status`, `exclusion_accounting`, `identity_token_collision`,
`manifest_source_accounting`.

The plan's asserted figure is **stale**. The cohort it described no longer exists:
the committed manifest carries 342 source rows (not 355), resolving to 340 terminal
identities and 230 unique candidate names (not 214); and the accounting defect it
described is closed — where eighteen names once left through neither declared
counter, zero now leave through any uninstrumented path.

The two declared counters the plan quotes still both read zero, but for a different
reason than it recorded: zero names are withheld at all, not zero counted among
eighteen withheld. A reader who takes the plan's "eighteen" as current will expect
a defect that is no longer present.

## Why the figure is stale, decomposed

| Quantity | Plan §9b as asserted | Current measurement | Reading |
| --- | ---: | ---: | --- |
| Manifest source rows | 355 | 342 | drifted |
| Unique candidate names handed to the export | 214 | 230 | drifted |
| Names leaving with both declared counters zero | 18 | 0 | stale; defect closed |
| `accounting_residue` | 0 (by coincidence of the two miscounted paths) | 0 (by construction) | current |

The residue reading is the one that carries the meaning. Before the exporter fix,
residue closed at zero only because the two null-`physics_domain` identities were
folded into a generic `invalid_catalog_entry`; every other exit was uncounted. Now
every exit is instrumented, so a residue of zero means what it says.

## The eighteen historic identities, traced individually

Re-measured against the current cohort rather than restated from the earlier
evidence document:

| Outcome | Count | Identities |
| --- | ---: | --- |
| Present in the current cohort and emitted | 16 | the twelve accepted-and-scored names plus the two null-domain names and two further entries listed in `export-accounting-closure.md` |
| Absent from the current terminal projection | 2 | `hot_neutral_temperature`, `inner_normalized_toroidal_flux_coordinate_hard_xray_emissivity_peak_half_width` |
| Silently dropped by the export | 0 | — |

The two absent identities are absent by the selection boundary, not by an export
exit. Both read `status: superseded` and `name_stage: superseded` in the graph, so
the manifest's terminal projection no longer reaches them. The current cohort's
two residual manifest sources — `calorimetry/group/component/energy_total/data`
(`extracted`) and `camera_x_rays/camera/camera_dimensions` (`failed`) — are the
same two the 2026-09-17 source-side audit recorded, and neither is an export
exclusion.

## The accounting path fires: a deliberate probe

A cohort in which every candidate is published cannot show the ledger working — a
guard that has nothing to catch is indistinguishable from a guard that does not
fire. So the same instrument was re-run with `min_score` raised to `0.99`, which
forces real exclusions:

```text
N  candidates   230
E  emitted 90
   below_name_score    52
   bound_adjacent      88
   accounted_exclusions 140
   accounting_residue     0

90 + 140 = 230 = N   exactly
```

`exclusion_accounting` passed at both settings. Every identity in every exclusion
reason is enumerated in the report's `exclusion_ledger`, so each is recoverable from
the recorded output. At the release threshold (`0.65`) the ledger is empty because
nothing is excluded; at `0.99` it carries 140 identities with their reasons. The
same instrument produced both readings.

## Instrument and method

- **Instrument:** `run_export` (`imas_codex/standard_names/export.py`), called with
  `review_batch=<the manifest's 230 unique terminal identities>`,
  `manifest_sources=<the manifest's 342 resolved rows>`, `skip_gate=True`,
  `include_sources=False`, and `gate_only=False` writing to a scratch staging
  directory. The full release path is not the instrument: `run_review_release`
  returns at its dry-run branch before reaching the export, so it cannot measure
  the accounting at all.
- **`gate_only=True` is not a valid setting for this measure.** It returns before
  the emission loop, so `exported` is structurally zero and every candidate shows
  as residue (measured: residue 230). That is an artefact of the mode, not a
  shortfall. Recorded so the next worker does not misread it.
- **Cohort resolution:** `load_focus_file` → `fetch_manifest_source_release_rows`.
  342 of 342 manifest rows resolved; 340 carry a terminal identity; 230 are unique
  candidate names; 2 rows are unmatched sources. This reproduces the 2026-09-17
  source-side audit exactly, so the resolution is a working control, not a blind
  projection.
- **Known-present control:** the emitted set is 230, not zero, and the `0.99` probe
  produces nonzero per-reason counts. Neither zero reading in the primary result is
  an artefact of an instrument seeing nothing.
- **Graph mutation:** none. `export.py` contains no `MERGE`, `execute_write` or
  write session; the only output is the local staging tree and the report JSON.
  Graph access used the login-node-local tunnel, as the graph cannot be reached
  from a SLURM partition.
- **Bounded work:** every query is scoped to the 342 manifest rows or to a named
  2-identity probe; no whole-graph traversal and no statement approaching the
  ten-second ceiling.

## Logs and artifacts

All under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260917T110637356274-n-the-export-accounts-for-every-name-it-drops/`:

| Path | Content | Exit |
| --- | --- | ---: |
| `export-accounting-current.log` | first attempt, `gate_only=True` (residue-230 artefact) | 0 |
| `export-accounting-current-full.log` | primary measurement at `min_score=0.65` | 0 |
| `export-accounting-probe.log` | `0.99` probe, ledger fires with 140 exclusions | 0 |
| `check-two.log` | lifecycle of the two absent historic identities | 0 |
| `probe.log` | membership of the historic eighteen in the current cohort | 0 |
| `measure-current/export_report.json` | primary report, machine-readable | — |
| `measure-current/cohort.json` | resolved cohort, 230 names + 2 unmatched | — |
| `measure-probe/export_report.json` | probe report, machine-readable | — |
| `driver.py`, `check-two.py` | the exact drivers that produced the above | — |

## Limits

- This re-measures the accounting currency for the current cohort only. It does not
  re-cut, publish or publish-enable anything, and it creates no candidate, branch,
  tag or pull request.
- `skip_gate=True` was used so the broad Gate A suite would not launch from the
  login node; Gate C (score filtering) and the accounting path still run under that
  flag. Merged-head verification of the export suite belongs to a separately
  dispatched test node.
- The 233 pruned internal documentation links and the `0.9.3-2-ge93fe62` grammar
  checkout identity in the report are advisory and count-independent; they are
  visible in the retained log and did not affect the accounting.
- `hot_neutral_temperature` carries a null `superseded_by` alongside its
  `superseded` status. That is the graph-wide scalar-rot condition already recorded
  elsewhere, not a finding of this measurement.

## Verdict

**Stale.** The plan's eighteen-name claim describes a cohort of 214 names and an
accounting defect that no longer exist. The current cohort is 230 names, all of
which the export accounts for, with `E + sum(per-reason exclusions) = N` exactly.
The two residual manifest sources remain the source-side tail the 2026-09-17 audit
recorded, and they are not export exclusions.