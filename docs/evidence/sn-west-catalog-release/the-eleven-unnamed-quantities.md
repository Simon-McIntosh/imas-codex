# The eleven unnamed WEST quantities: naming campaign evidence

Companion to the settled classification ledger
[`the-unnamed-manifest-sources.md`](the-unnamed-manifest-sources.md): that
document classified all 25 manifest rows without a terminal standard name into
dispositions A (11, valid source wrongly withheld), B (13, genuinely
ineligible), and C (1, already represented).  This document records what the
candidate, review, and documentation stages did to each of the 11
disposition-A paths.

## Result (measured from the graph)

Five of the eleven reached an accepted name with accepted documentation.  Four
more carry a composed name at `name_stage=drafted` (two of those already hold
accepted documentation).  Two produced no name and are reported below with
their exact stage and cause.

The release projection re-measurement: `load_sources_file()` expands the WEST
manifest to 342 eligible paths and `fetch_manifest_source_release_rows()`
returns 342 rows; the count with no terminal `standard_name_id` fell from **25**
(pre-exclusions, ledger) → **12** (post-exclusions, ledger) → **3** now, the
three being the two unnamed disposition-A paths below and the one
disposition-C path (`spectrometer_visible/channel/isotope_ratios/signal_to_noise`,
binding collision, out of scope).

| Path | Source before → after | Terminally produced name | name_stage | docs_stage | Name score (agg) | Docs score |
|---|---|---|---|---|---|---|
| `barometry/gauge/pressure` | extracted(0) → composed | `neutral_pressure` | drafted | accepted | — (mean 0.771) | 0.85 |
| `calorimetry/group/component/energy_total/data` | extracted(0) → composed | `total_energy` | drafted | pending | — (mean 0.612) | — |
| `calorimetry/group/component/power` | extracted(0) → composed | `coolant_absorbed_power_of_calorimetry_component` | accepted | accepted | 1.0 | 1.0 |
| `camera_ir/channel/camera/frame/apparent_temperature` | extracted(0) → composed | `surface_temperature` | drafted | accepted | — (mean 0.795) | 0.869 |
| `camera_x_rays/camera/camera_dimensions` | failed(5) → failed(5) | none | — | — | — | — |
| `camera_x_rays/detector_humidity` | skipped(1) → composed | `relative_humidity_of_detector` | accepted | accepted | 0.9 | 1.0 |
| `equilibrium/time_slice/profiles_1d/darea_dpsi` | failed(5) → extracted(4) | none | — | — | — | — |
| `equilibrium/time_slice/profiles_1d/pressure` | extracted(0) → attached | `total_plasma_pressure` | accepted | accepted | 0.9125 | 0.95 |
| `gas_injection/valve/flow_rate` | extracted(0) → composed | `gas_flow_of_valve` | accepted | accepted | 1.0 | 1.0 |
| `hard_x_rays/emissivity_profile_1d/half_width_external` | extracted(0) → composed | `outer_hard_xray_half_width_at_emissivity_peak` | accepted | accepted | 1.0 | 1.0 |
| `ic_antennas/antenna/module/strap/distance_to_conductor` | extracted(0) → composed | `distance_of_antenna_strap` | drafted | pending | — (mean 0.637) | — |

Five exportable (accepted name + accepted docs): `outer_hard_xray_half_width_at_emissivity_peak`,
`coolant_absorbed_power_of_calorimetry_component`, `gas_flow_of_valve`,
`total_plasma_pressure`, `relative_humidity_of_detector`.

## The sibling pair came out right

`outer_hard_xray_half_width_at_emissivity_peak` is symmetric with the
previously accepted `inner_hard_xray_half_width_at_emissivity_peak`: same base
(`half_width`), same position locus (`emissivity_peak`), same `at` relation,
opposite direction qualifier (`outer` vs `inner`).  The external counterpart
reached 1.0 and accepted docs.

Its first compose attempt, `outer_hard_xray_half_width`, scored **0.5375** and
was exhausted before the locus-bearing form reached 1.0 — the refine-style
rotation working as designed rather than a failure.  Equally, the candidate
`coolant_absorbed_power_of_plant_component_port` was superseded at **0.675**
in favour of the calorimetry-specific spelling
`coolant_absorbed_power_of_calorimetry_component` (accepted, 1.0).

## The three signal-STRUCTURE quantities were admitted, no refusal

The classification ledger flagged three of the eleven as signal `STRUCTURE`
quantities whose generic-container exclusion might wrongly withhold them:
`barometry/gauge/pressure` (Pa), `calorimetry/group/component/power` (W), and
`camera_x_rays/detector_humidity` (dimensionless, `1`).  No eligibility
refusal occurred at this revision: the live extractor admitted all three, and
each went on to a composed name.  One is fully landed
(`coolant_absorbed_power_of_calorimetry_component`, 1.0/1.0 and
`relative_humidity_of_detector`, 0.9/1.0); `barometry/gauge/pressure` has a
drafted name with accepted docs and a review mean of 0.771, short of the 0.85
acceptance floor.

## Duplication checks before composition

`equilibrium/.../pressure`: no existing representative under it at the time;
`total_plasma_pressure` is the kinetic-sum quantity derived from
`derived:particle_pressure`, and `thermal_plasma_pressure` excludes the fast
population, so a new identity for the equilibrium flux-surface pressure was
justified.  The source attached to the accepted `total_plasma_pressure`
identity (created 2026-08-23, score 0.9125, docs 0.95).

`calorimetry/group/component/energy_total/data`: nearest accepted name,
`accumulated_coolant_absorbed_energy_of_calorimetry_component`, is bound to a
different path (`calorimetry/group/component/energy_cumulated` — cumulative
since pulse start) whereas `energy_total` is the whole-discharge energy
including post-pulse thermal equilibrium; distinct identity justified.  The
composition produced `total_energy` (drafted, pending docs) — see the
candidate review below.

## The four drafted names — per-reviewer name votes

All four are at `name_stage=drafted` with review means below the 0.85 floor;
none yet has an accepted aggregate.  Per-reviewer name-axis votes:

- `neutral_pressure` (mean 0.771, 15 votes): grok-4.5 0.6/0.5625/0.725/0.6625
  (primary, cycle 0); gpt-5.6-luna 0.925/0.9125/0.8375/0.75 (cycle 1);
  claude-sonnet-5 0.825/0.75/0.6625 (cycle 2).  Docs already accepted (0.85,
  reviewed by minimax/qwen).
- `surface_temperature` (mean 0.795, 14 votes): grok-4.5 0.9/0.7625/0.85/0.8125
  (primary); gpt-5.6-luna 0.7/0.775/0.8/0.65; claude-sonnet-5 0.8375/0.75.
  Docs already accepted (0.869).
- `total_energy` (mean 0.612, 5 votes): claude-sonnet-5 0.5375, qwen3.7-max
  0.725, minimax-m3 0.6125, grok-4.5 0.55, gpt-5.5 0.6375.  Docs pending.
- `distance_of_antenna_strap` (mean 0.637, 5 votes): qwen3.7-max 0.575,
  grok-4.5 0.65, minimax-m3 0.65, gpt-5.6-luna 0.725, gpt-5.5 0.5875.  Docs
  pending.  Reviewers consistently flag the missing conductor/wall endpoint
  (`gap_of_antenna_strap` is the accepted near-sibling at sim 0.96).

## The two that produced no name

Both consumed multiple attempts.  This section records their exact stage and
cause now; the diagnosis follows the candidate work below.

- `camera_x_rays/camera/camera_dimensions` — `status=failed`, `attempt_count=5`,
  `last_error='compose claim-attempt cap reached'`, `skip_reason='vocab_gap'`,
  `skip_reason_detail='position:camera_dimensions'`.  The compose LLM cannot
  express a registered camera-dimensions position locus under the current
  grammar snapshot (0.9.3).
- `equilibrium/time_slice/profiles_1d/darea_dpsi` — `status=extracted`,
  `attempt_count=4`, no `last_error`, no `skip_reason`.  It returned to the
  claimable pool after the diagnosed retry and has consumed four further
  attempts without a produced name and without a recorded refusal, which is
  itself the anomaly to explain.

## Treatment by group

Per the ledger these are not one undifferentiated batch:

1. **Signal-STRUCTURE quantities** (barometry/gauge/pressure, calorimetry power,
   detector_humidity) — eligible; the classifier already admits them and each
   composed.  Treatment: normal composition.
2. **Retried failed/skipped sources** (camera_dimensions, darea_dpsi after a
   diagnosed retry; detector_humidity after a skip retry) — retried because the
   0.9.3 grammar supplies the vocabulary the 0.8.x-era failures lacked
   (`derivative_of_area_of_flux_surface_with_respect_to_toroidal_flux_coordinate`,
   `extent_of_soft_xray_detector`, …).  detector_humidity landed; the two
   structure/geometry cases are reported individually, not blindly re-attempted
   a sixth time.
3. **Previously-extracted quantities** (the remaining paths) — composed normally
   and either accepted or carried to a reviewed draft.

## Spend

Campaign total before this node: **103.82 USD** against the authorised
**250.00 USD** campaign ceiling.  This node's cap: **30.00 USD**.  The
pipeline's LLM spend ledger for the enclosing run (LLMCost cluster
`180c8780-0e01-427c-b4b2-3a7a78087846`, 42 events, 2026-09-14 22:10–22:31 UTC)
totals **0.88 USD** — composition resolving to the free local seat, review and
documentation the billed part.  Campaign total after this node: **104.70 USD**.
No spend cap was approached; the node consumed a small fraction of its
authorised budget.

## Remaining work (candidates drive)

- Drive the four drafted names to acceptance: all four need the name axis to
  clear 0.85; `total_energy` and `distance_of_antenna_strap` also need docs.
- Then record the diagnosis for `camera_dimensions` (vocab gap, cap) and
  `darea_dpsi` (unexplained extracted-with-4-attempts) and the decision for
  each.
