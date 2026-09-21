# The WEST manifest, re-measured against the live graph

Re-measurement of the WEST manifest's export-gate generability, taken live
against the live graph at base `ea566ff15935` from a detached worktree inside a
SLURM step. It answers one question with a number: **does the manifest now pass
the gate chain?** It does not.

```
manifest_size 355   carried 340   uncarried 15   generable False
all_gates_passed False   gate_failures 1   failed_gates ['manifest_generability']
mechanism_counts {'recorded_refusal': 15}
```

<figure>
  <img src="/imas-codex/figures/unbound-source-backlog/west-gate-chain.svg"
       alt="The WEST manifest gate chain: 355 manifest source paths read live,
       projected to 340 carried and 15 uncarried, with all 15 withheld on
       recorded_refusal and zero on waived, yielding generable False"
       style="max-width:100%">
  <figcaption>The decision path, drawn from the verdict JSON rather than
  hand-written, so it cannot drift from the numbers. The withheld set is
  exhaustive on one mechanism, <code>recorded_refusal</code>; the
  <code>waived</code> count is zero, which is why the verdict stays
  <code>False</code>.</figcaption>
</figure>

## Cohort — what was read, and how it was bounded

The cohort is the **355 `manifest_sources` of
`imas_codex/standard_names/manifests/reviews/v0.4.0rc7+west-task-2e.sn_names.yaml`**
and nothing else. Every statement below is scoped to that cohort: the manifest's
own 355 `source_path` strings are the only input, and no traversal reaches a
node outside the set of identities those paths resolve to.

The read is bounded three ways, all stated before the query ran:

- **By name. ** The cohort is named (355 paths) before any statement is issued,
  and no statement enumerates a label. `StandardNameSource` is entered by
  `id: 'dd:' + path` — one indexed lookup per cohort member, 355 in total.
- **By the work ceiling.** One indexed projection, then one successor-chain walk
  seeded only by the identities those 355 rows resolved to (340 seeds). The walk
  is anchored at `start.id IN $seed_ids`; it is not a label scan and not a
  cartesian product. Neither statement approached ten seconds.
- **At the caller.** `fetch_manifest_source_release_rows` returns one row per
  cohort path (355), and the gate consumes that list; the result is fixed-size
  at the cohort, not at the graph.

No statement exceeds the cohort. Splitting a scan into many fast queries is not
used as a way around the ceiling, because there is no scan.

### The named queries

Both live in `fetch_manifest_source_release_rows`
(`imas_codex/standard_names/graph_ops.py:12893`), which is the release's own
cohort projection — reused rather than re-written, so the measurement reads
exactly what a release would read.

| # | Statement | Bound | Rows |
|---|---|---|---|
| 1 | `UNWIND $paths AS path OPTIONAL MATCH (source:StandardNameSource {id: 'dd:' + path})` … `RETURN source.status, skip_reason, skip_reason_detail, last_error, produced_sn_id, collect(direct.id)` | `$paths` = the 355 manifest paths | 355 |
| 2 | `MATCH (start:StandardName) WHERE start.id IN $seed_ids` … `MATCH path=(start)-[:HAS_SUCCESSOR*0..]->(target) WHERE NOT (target)-[:HAS_SUCCESSOR]->()` `RETURN target.name_stage` | `$seed_ids` = the 340 resolved identity ids | 340 keys / 355 rows resolved |

Live projection totals: 355 rows returned; 340 carry a terminal identity; 340
carry a terminal stage; 15 carry a non-nameable reason. Live `source_status`
counts: `composed 181, attached 159, extracted 1, failed 1, skipped 11,
not_physical_quantity 2`.

The export leg then runs over that cohort into a scratch tree
(`run_export(staging_dir=<tmp>, force=True, review_batch=<340 dedup ids, 230
accepted>, manifest_sources=<the 355 live rows>)`), reproducing the release's
own dry-run path. **No graph write anywhere**: the export leg writes files into a
`TemporaryDirectory` that is removed, and the cohort query is read-only.

## Against the recorded baseline

The prior audit (`export-gate-audit.md`, at `83e93c42c`) recorded the same
instrument over the same manifest:

| | recorded baseline | this re-measurement |
|---|---|---|
| manifest_size | 355 | 355 |
| carried | 330 | **340** |
| uncarried | 25 | **15** |
| generable | False | **False** |
| mechanism_counts | `{recorded_refusal: 21, composition_not_scheduled: 4}` | `{recorded_refusal: 15}` |
| all_gates_passed | (not recorded) | False |

**Ten sources moved from uncarried to carried** — four previously reported
`composition_not_scheduled` (no terminal identity) now resolve to a terminal
identity, and six recorded refusals cleared as the pipeline produced names for
them. The verdict did not move: `generable` is still **False**, the single failed
gate is still `manifest_generability`, and it remains the only gate that fails.

## The uncarried table — one row per identity

Fifteen source identities remain uncarried, every one on the same mechanism:
`recorded_refusal`, which `_manifest_source_mechanism` (`export.py:2465`) returns
for a `documented_non_nameable` disposition. The **withholding clause** is the
recorded reason on each row; the disposition for all fifteen is
`documented_non_nameable`.

| # | source identity | status | withholding clause (recorded reason, abridged) | permanent? |
|---|---|---|---|---|
| 1 | `equilibrium/time_slice/constraints/b_field_pol_probe/weight` | not_physical_quantity | `dd_node_category_ineligible: fit_artifact` | **permanent** |
| 2 | `equilibrium/time_slice/constraints/flux_loop/weight` | not_physical_quantity | `dd_node_category_ineligible: fit_artifact` | **permanent** |
| 3 | `equilibrium/time_slice/constraints/faraday_angle/weight` | skipped | `dd_node_category_ineligible: fit_artifact` | **permanent** |
| 4 | `equilibrium/time_slice/constraints/n_e_line/weight` | skipped | `dd_node_category_ineligible: fit_artifact` | **permanent** |
| 5 | `equilibrium/time_slice/convergence/iterations_n` | skipped | `dd_node_category_ineligible: fit_artifact` | **permanent** |
| 6 | `camera_x_rays/detector_humidity/time` | skipped | `non_nameable_coordinate:time` | **permanent** |
| 7 | `camera_x_rays/detector_temperature/time` | skipped | `non_nameable_coordinate:time` | **permanent** |
| 8 | `camera_x_rays/frame/time` | skipped | `non_nameable_coordinate:time` | **permanent** |
| 9 | `core_profiles/profiles_1d/time` | skipped | `non_nameable_coordinate:time` | **permanent** |
| 10 | `hard_x_rays/emissivity_profile_1d/time` | skipped | `non_nameable_coordinate:time` | **permanent** |
| 11 | `summary/disruption/time/value` | skipped | `non_nameable_coordinate:time` | **permanent** |
| 12 | `equilibrium/time_slice/time` | skipped | `temporal_coordinate: nested time axis` | **permanent** |
| 13 | `calorimetry/group/component/energy_total/data` | extracted | `vocab_gap_nonactionable` | not this pipeline's to close |
| 14 | `camera_x_rays/camera/camera_dimensions` | failed | `compose claim-attempt cap reached` | outstanding (budget spent) |
| 15 | `equilibrium/time_slice/contour_tree/node/z` | skipped | `compose_model_skipped` | outstanding |

Twelve of the fifteen are withholdings the pipeline has **permanently refused by
design**: five are `fit_artifact` backings — a fit artifact is not a physical
quantity, and the plan's opening section lists that class as correctly excluded — and seven
are time axes (six bare `time` tokens and one nested time coordinate), a
coordinate rather than a quantity. No amount of pipeline work will turn any of
these twelve into a name; they are refused, correctly, forever.

The remaining three are not permanent refusals. One is a recorded non-actionable
vocabulary gap owned by the other repository; one is a spent compose attempt budget
(`compose claim-attempt cap reached`); one is a compose skip
(`compose_model_skipped`), i.e. work nothing has scheduled yet.

## The `waived` question — the two fit_artifact rows do NOT carry it

The brief asks to confirm specifically that the two fit_artifact rows that made
the manifest permanently ungenerable before `waived` existed now carry `waived`
and no longer block. **They do not.** The measurement is unambiguous:

- All four `fit_artifact` `constraints/*/weight` rows (and
  `convergence/iterations_n`) report `mechanism = recorded_refusal`; **zero rows
  in the whole uncarried set report `waived`**, and `mechanism_counts` is
  `{'recorded_refusal': 15}` with no `waived` key at all.
- `generable` is `all(mechanism == "waived")`, so it is **False**.

The code half of the repair *did* land, and it is what the brief remembers: the
`waived` mechanism was added at `8c33f2003` ("Keep explicitly waived sources
visible in uncarried accounting without blocking generability"), and
`_manifest_source_mechanism` returns `"waived"` for a row whose disposition says
so (`export.py:2476-2477`). But **nothing produces such a row.** The export leg's
only disposition assignments are `documented_non_nameable`, `emitted` and
`excluded` (`export.py:3281-3287`); there is no fourth branch. `waived` is a
declared vocabulary member with a mapping but no producer anywhere in
`imas_codex/` — `grep -rn waived --include=*.py imas-codex/` outside `export.py`
returns nothing, and its only construction in the tree is
`SourceDispositionRecord(disposition="waived")` **in the test file**
(`tests/standard_names/test_manifest_generability.py:299`).

There is also no channel through which a waiver could enter. The manifest row
schema carries no `disposition` key (keys are `source_path`, `source_status`,
`standard_name_id`, `terminal_stage`, `non_nameable_reason`), the live cohort
projection reads no waiver property (`source.status`, `skip_reason`,
`skip_reason_detail`, `last_error`, `produced_sn_id` only), and the manifest loop
never consults one. So a waiver would have to be authored on the disposition row
inside the export leg, and no path does it.

The consequence is the one the prior audit named as defect 5, still live: **a
manifest containing a correctly-excluded source is permanently ungenerable, with
no route to green except removing the source from the manifest or waiving it —
and no one can waive it yet.** The repair made the gate *satisfiable in
principle*; it is still *unsatisfiable in fact*, because the data half requires a
waiver no component writes.

## Follow-ons (found, not fixed — this node reads only)

| # | identity / site | mechanism |
|---|---|---|
| 1 | `export.py:3281-3287` | No producer of a `waived` disposition. The vocabulary member and its mapping exist, but the export leg cannot emit one and no other module constructs one, so a correctly-excluded source can never be waived. **A waiver-writing path (or an accepted manifest-level waiver channel) is required before any manifest carrying a by-design exclusion can be generable.** |
| 2 | `camera_x_rays/camera/camera_dimensions` | Reports `recorded_refusal`; its recorded reason is `compose claim-attempt cap reached` and its terminal stage is absent, so `attempt_budget_exhausted` (`export.py:2478`, fires only on `terminal_stage == "exhausted"`) is never reached. Same misattribution as audit defect 4: a spent search reports as a standing refusal. |
| 3 | `calorimetry/group/component/energy_total/data` | `vocab_gap_nonactionable` — a vocabulary gap recorded as non-actionable on this side; ownership of the closing work sits with the ISN grammar, and the row should be corroborated against it before any cut relies on it. |

Item 1 is the one that decides whether the WEST cut can ever clear the gate. It
is a code/data repair, not a measurement, so it belongs to a write node with an
adjudication mandate; this node found it and did not touch it.

## Reproduction

```
srun --partition=all_debug --time=00:59:00 --cpus-per-task=4 --mem=32G bash -lc '
  export TMPDIR=/tmp
  UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH=$PWD uv run python /home/ITER/mcintos/usb-west-gate/measure_west_generability.py'
```

Script: `/home/ITER/mcintos/usb-west-gate/measure_west_generability.py`
Result: `/home/ITER/mcintos/usb-west-gate/verdict.json` (28,486 bytes)
SLURM job `1275229` on `all_debug`, COMPLETED, exit 0. The instrument is
deterministic over the live graph, so a re-run at an unchanged graph reproduces
the counts above.

> **Citation corrected 2026-09-21.** The terminal-stage check that gates
> `attempt_budget_exhausted` is at `export.py:2478`, returning at `:2479`; `:2482` is the
> `documented_non_nameable` branch. An independent review caught the four-line slip and
> confirmed, by reading the whole mechanism function, that the behaviour described here is
> correct. The coordinator had repeated the wrong citation before checking it.
