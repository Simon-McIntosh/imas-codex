# Tail A restated against the live roster

**Plan section:** `sn-west-cohort-treatment` §9a — Tail A: source paths that reached no name
**Rule enforced:** what Tail A still owes must be stated against the roster that exists now,
not against the frozen table the section counted on 2026-09-07.
**Roster identity:** the committed manifest `west_production_dd_paths`
(`imas_codex/standard_names/manifests/west_production_dd_paths.yaml`), re-resolved live.
**Readout:** live graph, login node, `2026-09-17`; every query bounded to the manifest's
342 source paths, each returning in well under 10 s.

## 1. The headline

**Zero manifest rows end without either a name or a stated cause.**

The current manifest carries **342** source paths (the section counted 355). Resolved live
through `fetch_manifest_source_release_rows`: **340 carry a name, 2 do not, and both of
those 2 state a cause.** So the count of rows ending without a name *or* a stated cause
is **0**.

A zero is only meaningful if the query is aimed, so the same read is shown in two forms.
The breakdown query `MATCH (source:StandardNameSource {id: 'dd:' + path}) … RETURN path AS
source_path, source.status AS source_status, source.skip_reason AS skip_reason,
source.last_error AS last_error ORDER BY source_path` — which requires the row pattern to
match — **returned 342 rows**, so the residue of 0 is a measured zero rather than an
unaimed pattern that matched nothing.

The two unnamed rows, each with the cause it states:

| source_path | status | stated cause |
|---|---|---|
| `calorimetry/group/component/energy_total/data` | `extracted` | `vocab_gap_nonactionable: :` — a vocabulary gap, non-actionable (the detail field is empty) |
| `camera_x_rays/camera/camera_dimensions` | `failed` | `compose claim-attempt cap reached` — a terminal compose failure |

## 2. Why 355 became 342

Thirteen paths that the section's Tail A counted are no longer manifest members. The
section's table names eleven of them individually, and none is in the current roster:

`summary/disruption/time/value`, `camera_x_rays/detector_humidity/time`,
`camera_x_rays/frame/time`, the four `equilibrium/time_slice/constraints/<probe>/weight`
rows, `equilibrium/time_slice/contour_tree/node/z`,
`equilibrium/time_slice/convergence/iterations_n`.

They remain live sources in the graph, and this document reports their live state below,
but they are outside the cohort the manifest now drives. Sibling evidence
(`export-accounting-currency`) established the same 355 → 342 drift independently; the
figure is reproduced here rather than restated.

## 3. Each of the section's nine row groups, re-counted

Verdict vocabulary: **current** — the live measurement equals the asserted figure and the
mechanism the table names is what the live state shows; **drifted** — the figure no longer
matches and the section did not anticipate the direction; **stale** — the asserted figure
can no longer be reproduced, because the rows it counted have been resolved or have left
the cohort.

| # | Rows asserted | Recorded state | Live measurement | Verdict |
|---|---:|---|---|---|
| 1 | 15 | `extracted`, no reason — never composed | **0** unnamed-and-uncaused; the two rows that remain unnamed both state a cause. The extract-backlog signature no longer exists in the cohort. | stale |
| 2 | 1 | `camera_x_rays/detector_humidity` declined by the skip rule | **0 remaining**: status `composed`, named `relative_humidity_of_detector`. | stale |
| 3 | 1 | `…/signal_to_noise` declined on an absent vocabulary base | **0 remaining**: status `attached`, named `spectral_signal_to_noise_ratio_of_spectrometer_channel`. | stale |
| 4 | 1 | `summary/disruption/time/value`, the disruption instant | **1**, and it still reads `non_nameable_coordinate:time: bare non-nameable token: time` — the same value the table records. | current |
| 5 | 3 | three `…/time` rows reporting the compose skip | **0 remaining** in the cohort reporting `compose_model_skipped`; the two named rows now carry the coordinate rule, and the third was already correct when it was counted. | stale |
| 6 | 4 | time axes, correctly declined and correctly explained | the mechanism holds — the rows still carry `non_nameable_coordinate:time` / `temporal_coordinate`, but every one of them is outside the current manifest roster, so the cohort-level count is **0**. | drifted |
| 7 | 4 | four fit-constraint weights, one concept with two verdicts | **one verdict, four rows**: all four now record `dd_node_category_ineligible: Backing DD node category fit_artifact cannot realize a StandardName`. The group's rows are no longer manifest members. | stale |
| 8 | 2 | `contour_tree/node/z` and `convergence/iterations_n`, one group | **1 of 2 still unexplained**: `iterations_n` now reads `dd_node_category_ineligible` / `fit_artifact`; `contour_tree/node/z` still reads only `compose_model_skipped`. | drifted |
| 9 | 1 | `camera_x_rays/camera/camera_dimensions`, a terminal failure with nothing recorded | **1**, still unnamed, and it now states its cause (`compose claim-attempt cap reached`). | drifted |

Verdict counts: **5 stale, 3 drifted, 1 current — 9, the number of groups.**

## 4. The row whose cause still cannot be stated

`equilibrium/time_slice/contour_tree/node/z` remains the one Tail A row that ends with
neither a name nor a cause it can honestly be given. It reads only the placeholder
`compose_model_skipped`. The section groups it with `convergence/iterations_n` and
prescribes "classification repair, as above" — the category-based verdict that fitted
`iterations_n`. That verdict does **not** fit this row: its backing DD node carries
`node_category = quantity` with unit `m`, so the exclusion is by locus, not by category,
and asserting `dd_node_category_ineligible` would be false. Stating a cause needs a
vocabulary/locus decision the section does not settle. This is the residual Tail A owes,
and it is one row.

## 5. Work belonging to another plan

Two of the section's recoveries are not pipeline work and are named here so they are not
mistaken for this batch's remainder.
- `spectrometer_visible/channel/isotope_ratios/signal_to_noise`: admitting
  `signal_to_noise_ratio` as a vocabulary base is **owned by the grammar plan** (the
  grammar repository). It is satisfied for this batch by name, not by this plan.
- `summary/disruption/time/value`: whether the vocabulary admits an **event-instant base**
  is a **grammar question, not a pipeline one**. The row is unchanged and correct as the
  coordinate rule; a restatement as an event instant depends on that decision.

## 6. Limits

- The roster is re-derived from the committed manifest, so these are live figures, not the
  frozen 355-row artifact's. No graph write was made by this node; nothing outside the
  roster was read, and every query was scoped to the 342 manifest paths.
- The live graph is written by concurrent peers, so the two unnamed rows could be named or
  newly failed between this readout and a later node's read. The measurement carries its
  own date for that reason.
- `calorimetry/group/component/energy_total/data` states a cause but records no detail
  text after the colon; it is counted as stated because the reason token is present.