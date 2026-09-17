# Roster reason repair: every unnamed row ends with a name or a stated cause

**Plan section:** sn-west-cohort-treatment §9a (Tail A) — the roster's reason field
**Rule enforced:** an unnamed row must end with a name or a stated cause; an
empty or placeholder reason is not a neutral absence.
**Roster identity:** frozen review batch `v0.10.0rc1+west-task-2e`
(`manifest_sources`, 355 rows) for the WEST production manifest
`west_production_dd_paths.yaml`.

## 1. The two figures, and which one the change moves

The census command is `/tmp/census_roster_reason.py` (read-only; source in the
run directory as `census_roster_reason.py`). It prints the frozen artifact's
standing counts **and** the live projection re-derived from the graph
(`fetch_manifest_source_release_rows`), because the two answer different questions.

| read | before | after |
|---|---|---|
| frozen artifact rows, no name and empty `non_nameable_reason` | 18 | 18 |
| live projected rows, no name and empty `non_nameable_reason` | 0 | 0 |
| live projected rows, no name and a placeholder reason (`compose_model_skipped`) | 6 | 1 |

- The **artifact** figure is the plan's 18 and did not move: the artifact is the
  frozen batch, and this node's write scope is the graph plus three documents.
  Its roster builder had already been repaired to transcribe recorded causes, so
  a re-cut reads the same causes this node now records.
- The **live empty** figure is 0 both sides. It was already 0 before this node:
  the roster builder now falls back to `last_error` / `skip_reason` /
  `skip_reason_detail` and writes `cause not recorded` rather than `""`. The
  zero is aimed, not blind: the same command resolves all 355 rows (no missing-node
  artefact) and 340 of them to a name; the 15 unnamed are listed below.
- The **placeholder** figure is the axis this node moves: 6 → 1.

## 2. What was written (the only graph write is the reason field)

Five rows recorded only the bare placeholder `compose_model_skipped`, which
names no mechanism. Each write was compare-and-set guarded: the source had to
carry `status = skipped`, the placeholder reason, no claim, and a backing DD node
whose `node_category` matches the verdict written. All five guards held and each
write returned `affected = 1`.

| source path | written `skip_reason` / `skip_reason_detail` | backing DD `node_category` |
|---|---|---|
| `camera_x_rays/detector_humidity/time` | `non_nameable_coordinate:time` / `bare non-nameable token: time` | `coordinate`, unit `s`, doc "Time" |
| `camera_x_rays/frame/time` | `non_nameable_coordinate:time` / `bare non-nameable token: time` | `coordinate`, unit `s`, doc "Time" |
| `equilibrium/time_slice/constraints/faraday_angle/weight` | `dd_node_category_ineligible` / `Backing DD node category fit_artifact cannot realize a StandardName` | `fit_artifact`, unit `1` |
| `equilibrium/time_slice/constraints/n_e_line/weight` | `dd_node_category_ineligible` / same detail | `fit_artifact`, unit `1` |
| `equilibrium/time_slice/convergence/iterations_n` | `dd_node_category_ineligible` / same detail | `fit_artifact`, unit ``, `INT_0D` |

Writer: `mark_source_skipped` (`imas_codex/standard_names/graph_ops.py`) — the
canonical reason writer. For all five the prior status was already `skipped`, so
the write changed no status: the substantive write is the reason field only.

## 3. The four fit-constraint rows now carry one verdict for one concept

All four are `equilibrium/time_slice/constraints/<probe>/weight`, all measure
`fit_artifact` with unit `1` and identical documentation. Before this node they
recorded **two** verdicts: two read `not_physical_quantity`, two read
`compose_model_skipped`. The census's fit-weight block now prints **one** value
four times:

```
equilibrium/time_slice/constraints/b_field_pol_probe/weight | not_physical_quantity | dd_node_category_ineligible: Backing DD node category fit_artifact cannot realize a StandardName
equilibrium/time_slice/constraints/faraday_angle/weight      | skipped               | dd_node_category_ineligible: Backing DD node category fit_artifact cannot realize a StandardName
equilibrium/time_slice/constraints/flux_loop/weight          | not_physical_quantity | dd_node_category_ineligible: Backing DD node category fit_artifact cannot realize a StandardName
equilibrium/time_slice/constraints/n_e_line/weight           | skipped               | dd_node_category_ineligible: Backing DD node category fit_artifact cannot realize a StandardName
```

One concept, one recorded verdict. Two rows keep the deterministic
`not_physical_quantity` status, two keep `skipped`: that status difference is
a source-lifecycle value, not the exclusion verdict, and the fence for this node
permits only the reason field. It is recorded in the manifest as a `follow_ons`
item rather than silently normalised.

The same rule repaired the time axes: `equilibrium/time_slice/time` keeps the
nested-array reason, and the two bare `camera_x_rays/*/time` axes now read the
same coordinate token as the four rows that were already correct.

## 4. The one row that cannot be given a cause

`equilibrium/time_slice/contour_tree/node/z` still records only
`compose_model_skipped`. The DD backing node carries `node_category = quantity`,
unit `m`, documentation "Height" — so the deterministic
`dd_node_category_ineligible` verdict that fits the other solver rows is
**false** (the category *is* eligible). The plan groups this row with
`convergence/iterations_n` as "solver structure … classification repair, as
above", but the category-based verdict it points at does not apply: the exclusion
is by locus, not category. Stating it requires a vocabulary/locus decision the
plan does not settle, so the row is listed here individually rather than given a
cause that the recorded evidence contradicts.

## 5. Reproduction

```
# before (2026-09-17)
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" \
  uv run --no-sync python /tmp/census_roster_reason.py   # exit 0
# after the five writes
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" \
  uv run --no-sync python /tmp/census_roster_reason.py   # exit 0
```

Captured runs: `/tmp/census_before.log` and `/tmp/census_after.log` (exit 0 each).
The census script, the guarded writer and the backing-category probe are copied
verbatim into the run directory as `census_roster_reason.py`, `apply_reasons.py`
and `backing_category.py`.



## 6. Residual

No Python changed, so no test suite is affected; this node changed no code.
