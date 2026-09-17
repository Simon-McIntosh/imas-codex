# Drafted successor exposure at the cut

Measure: how many renamed successors a cut made today would lose because they sit
at `drafted` rather than `accepted`. The followup
`f-wcr-a-cut-now-loses-the-drafted-successors` asserts that the 2026-09-05
whole-cohort canonical migration renamed 137 identities whose successors "sit at
`drafted`", that the withdrawn candidate carried eighteen such names, and that the
successor candidate's size must be measured against 431.

**Verdict: drifted.** The rename count reproduces at exactly 137; the state does
not. Only 2 of the 137 successors sit at `drafted` on the name axis and none on
the documentation axis. The live count of successors not accepted on both axes is
86, and the number of those inside the committed WEST roster — the names a cut
made now would actually lose — is **0**.

## The cohort

The rename lineage is carried by `StandardNameChange` rows, not by
`HAS_SUCCESSOR` (that relationship carries 48 edges graph-wide and is not the
migration's record). The cohort is the change rows dated 2026-09-05 whose
operation is `human_edit` and whose `from_name` differs from its `to_name`:

```cypher
MATCH (c:StandardNameChange)
WHERE c.changed_at >= datetime('2026-09-05') AND c.changed_at < datetime('2026-09-06')
  AND c.operation = 'human_edit' AND c.from_name <> c.to_name
OPTIONAL MATCH (p:StandardName {id: c.from_name})
OPTIONAL MATCH (s:StandardName {id: c.to_name})
RETURN c.from_name AS predecessor, c.to_name AS successor,
       coalesce(s.name_stage,'<null>') AS name_stage,
       coalesce(s.docs_stage,'<null>') AS docs_stage,
       coalesce(s.status,'<null>') AS status,
       p IS NOT NULL AS pred_present, s IS NOT NULL AS suc_present
ORDER BY c.to_name
```

| Measure | Count |
|---|---|
| cohort rows (renames) | **137** |
| predecessor identity present | 137 |
| successor identity present | 137 |
| successor accepted on both axes | **51** |
| successor not accepted on both axes | **86** |
| successor identity absent | 0 |

The vertical pair is the split, and the 137 reproduces the followup's figure
exactly. It is corroborated the same day by `realign_grammar_segments: 137`, the
grammar pass over the same identities.

## Where the 86 actually sit

| successor `name_stage` / `docs_stage` | count |
|---|---|
| superseded / accepted | 50 |
| reviewed / accepted | 10 |
| exhausted / accepted | 10 |
| superseded / pending | 5 |
| exhausted / pending | 5 |
| reviewed / pending | 2 |
| **drafted** / pending | **1** |
| exhausted / exhausted | 1 |
| superseded / exhausted | 1 |
| **drafted** / accepted | **1** |

The dominant class is a successor that was itself superseded by a later rename —
56 of the 86 carry `name_stage=superseded`. They are middle links of a two-step
lineage, not abandoned drafts. The
documentation axis is accepted for 68 of the 86. Two rows are at `drafted` on the
name axis — the two the followup's description fits, against 137 asserted.

Read graph-wide, the claim "137 successors sit at drafted" is untenable
independently of the cohort: the whole graph carries **22** identities at
`name_stage=drafted` (accepted 2528, superseded 2163, exhausted 293, reviewed 113,
pending 11) and **10** at `docs_stage=drafted` (accepted 3343, pending 1734,
reviewed 27, exhausted 12, superseded 2, null 2). The instruction the followup
derives from its figure — run `sn run --only review --edits` so the 137 move off
`drafted` — would find at most two rows to move.

## What a cut made now would lose

The committed WEST artifacts are the review roster
`v0.10.0rc1+west-task-2e.sn_names.yaml` (214 names), the manifest
`west_production_dd_paths.yaml`, and the withdrawn candidate
`v0.4.0rc4+west-task-2e.sn_names.yaml` (431 names — the size the followup names).

| Overlap test | count |
|---|---|
| not-accepted successors (137 cohort) whose either spelling is in the **committed** roster | **0** |
| not-accepted successors (568-row whole-day cohort) whose either spelling is in the **committed** roster | **0** |
| cohort names appearing in the committed roster at all | 8 |
| not-accepted successors (137 cohort) whose either spelling is in the **withdrawn** roster | 8 |
| cohort names appearing in the withdrawn roster at all | **18** |

The asserted eighteen reproduces against the **withdrawn** 431-name candidate and
only there. The committed roster's eight cohort names are all cases whose
successor is accepted on both axes, so the batch carries a resolvable name and a
cut loses nothing. The exposure of the committed cut to this migration is 0,
against eighteen asserted for the withdrawn one.

The eight withdrawn-roster rows, quoted in full:

| predecessor | successor | name / docs stage |
|---|---|---|
| `flux_surface_averaged_electron_density_at_plasma_boundary` | `electron_density_flux_surface_averaged_at_plasma_boundary` | superseded / accepted |
| `parallel_flux_surface_averaged_current_density_due_to_wave_driven_current_drive` | `parallel_current_density_flux_surface_averaged_due_to_wave_driven_current_drive` | exhausted / accepted |
| `accumulated_deposited_energy_of_plasma_facing_component` | `deposited_energy_accumulated_of_plasma_facing_component` | exhausted / accepted |
| `accumulated_total_particle_count_due_to_gas_injection` | `total_particle_count_accumulated_due_to_gas_injection` | exhausted / accepted |
| `derivative_with_respect_to_normalized_poloidal_flux_coordinate_of_area_of_flux_surface` | `derivative_of_area_of_flux_surface_with_respect_to_normalized_poloidal_flux_coordinate` | exhausted / accepted |
| `per_toroidal_mode_launched_power_of_lower_hybrid_antenna` | `launched_power_per_toroidal_mode_of_lower_hybrid_antenna` | reviewed / accepted |
| `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` | `spectral_signal_to_noise_ratio_logarithm_of_spectrometer_channel` | **drafted** / accepted |
| `line_integrated_spectral_wave_opacity_at_ece_channel_emission_position` | `spectral_wave_opacity_line_integrated_at_ece_channel_emission_position` | exhausted / pending |

## Positive controls

A zero is only evidence if the same instrument sees something known present.

Same predicate, within the 137 cohort — the successor-acceptance test returns
**51** accepted on both axes against 86 not — both directions non-zero on one
cohort.

Same predicate, whole change log (15,548 rows):

```cypher
MATCH (c:StandardNameChange) OPTIONAL MATCH (s:StandardName {id: c.to_name})
RETURN sum(CASE WHEN coalesce(s.name_stage,'')='accepted'
                 AND coalesce(s.docs_stage,'')='accepted' THEN 1 ELSE 0 END) AS accepted_both,
       sum(CASE WHEN s IS NOT NULL AND NOT (coalesce(s.name_stage,'')='accepted'
                 AND coalesce(s.docs_stage,'')='accepted') THEN 1 ELSE 0 END) AS not_accepted,
       sum(CASE WHEN s IS NULL THEN 1 ELSE 0 END) AS absent
```

| | count |
|---|---|
| accepted on both axes | **7211** |
| present, not accepted on both | **4109** |
| successor identity absent | **4228** |

The roster overlap reader is likewise known to fire: the same predicate finds 18
names in the withdrawn roster and 8 in the committed one, so the committed roster
is being read and not silently missed.

## Instrument qualifications, stated rather than omitted

- The whole-day cohort (568 change rows on 2026-09-05, all operations) is reported
  beside the 137 rename cohort. Its "not accepted" figure of 317 in the per-row
  pass counts rows whose successor has no identity at all under the same branch
  (`!= 'accepted'` over a null placeholder), which inflates it against the clean
  Cypher split for the same day: 568 rows, 499 successors present, **248 present
  and not accepted**, **69 absent**. The 69 absent are dominated by test fixtures
  (`__cleartest__sn_single_*`, `__contestedtest___over`) and by rows where
  `from_name` equals `to_name` under operations that are not renames. The roster
  overlap of 0 holds under every one of these readings.
- Manifest-source extraction from `west_production_dd_paths.yaml` yielded 22
  paths, so the overlap against *DD source paths* (as opposed to roster names) is
  not measured here. The followup's eighteen and this record's overlap are both
  name-based, which is the spelling a roster carries.
- No figure: the write fence for this node is three documentation paths and
  `docs/evidence/figures/` is outside it, so no graphic was written. The
  relationships carried here are tabular (a count per stage pair, a count per
  roster) and are stated as tables.

## Reproductions

| artifact | path | exit |
|---|---|---|
| rename cohort (137) and both roster overlaps | `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260917T185218284145-n-the-drafted-successors-are-counted-before-any-cut/recovered-census-137.txt` | 0 |
| whole-day cohort (568), per-day operations, whole-log positive control | `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260917T185218284145-n-the-drafted-successors-are-counted-before-any-cut/recovered-census-568.txt` | 0 (recorded in the file's own head) |

Both were run read-only over the live graph from the login node, which is the
sanctioned exception for a login-local endpoint (`NEO4J_URI` resolves through a
login-node tunnel); every query is an indexed read over a named day range or over
the change log, and none exceeded the ten-second ceiling. The two captures are
this node's own run output, preserved beside the manifest by the coordinator after
the worker process ended. **No graph mutation was performed.**

## What the measure changes

The gate the followup asks for — measure the successor candidate against 431 and
account for the difference before the request is opened — is worth running, and it
will find a different problem from the one asserted. The exposure is not a batch
of drafted successors waiting on `--edits`; it is 56 successors in the 137 cohort
that are themselves superseded, i.e. a two-step rename lineage whose middle links
the roster must not carry. The `--only review --edits` precondition and the
eighteen-name loss estimate are both stale and should not gate the cut.