# Raw DD rebuild versus active resolution scope

## Outcome

The current graph has **49 Data Dictionary paths** whose active DD 4.1.1 resolution changes the raw declaration. All 49 resolutions are for `unit`, and a raw-DD rebuild of any of those paths is therefore capable of overwriting the reviewed effective unit with the raw DD unit.

Tonight's exact **54-path** re-derivation cohort intersected that exposure surface on **5 paths**. Every one of those five now has live unit `1`, matching the raw DD declaration and disagreeing with the active resolution's effective value. The graph therefore **disagrees with its own active resolution records today on exactly 5 rows**.

The requested populations remain separate:

- Cohort-touched paths whose live unit equals raw while the active resolution differs: **5**.
- Paths outside tonight's cohort already showing that condition: **0**.

No source file, graph value, holdout row, or resolution record was changed by this investigation.

## Fixed version and cohort authority

Every probe was pinned to **DD 4.1.1**:

- `pyproject.toml` declares `[tool.imas-codex.data-dictionary].version = "4.1.1"`.
- `get_dd_version()` returned `4.1.1` under the assigned project environment.
- The graph query `MATCH (v:DDVersion {is_current: true}) RETURN collect(v.id) AS current_versions` returned only `4.1.1`.
- `extract_paths_for_version("4.1.1")` produced 44,150 raw paths.
- For all 49 active current-version resolutions, the exact extracted raw unit matched `DDResolution.published_value`: **0 missing raw paths and 0 mismatches**. This makes `published_value` a verified raw-DD comparator in the queries below, rather than an assumed alias.

The 54-path cohort was reconstructed with the exact predicate used by the re-derivation receipt:

```python
parts = path.split("/")
constraint_artifact = "constraints" in parts and (
    parts[-1] == "weight"
    or any(
        part == "reconstructed" or part.endswith("_reconstructed")
        for part in parts
    )
)
convergence_iterations = (
    parts[-1] == "iterations_n" and "convergence" in parts
)
selected = (
    path.split("/", 1)[0] in {"equilibrium", "transport_solver_numerics"}
    and (constraint_artifact or convergence_iterations)
)
```

The sorted, newline-delimited cohort has SHA-256 `2e2a131d3ba026260e0fccd2dffa289feef0e4050ebdae702fb46dfb2f2c2742`. The queries below receive that exact 54-element list as `$cohort_paths`; they do not infer membership from the post-write category. This distinction matters because one selected path, `transport_solver_numerics/solver_1d/equation/convergence/iterations_n`, currently carries `lifecycle_status = "alpha"`; an unrelated `active` filter would incorrectly recount the historical cohort as 53.

## Count one — active resolutions differing from raw DD

**Result: 49 distinct paths and 49 resolution rows.** The only field represented is `unit`.

Verbatim query:

```cypher
MATCH (v:DDVersion {id: '4.1.1', is_current: true})
MATCH (r:DDResolution {status: 'active', dd_version: '4.1.1'})
WHERE r.published_value <> r.effective_value
RETURN count(DISTINCT r.path) AS paths_with_effective_raw_difference,
       count(r) AS resolution_rows,
       collect(DISTINCT r.field) AS fields
```

Result:

```json
{
  "fields": ["unit"],
  "paths_with_effective_raw_difference": 49,
  "resolution_rows": 49
}
```

This is the full current-version overwrite exposure: a raw rebuild of these paths would select the raw side of 49 active raw/effective disagreements unless the rebuild applies resolution authority.

## Count two — differing resolutions touched by tonight's cohort

**Result: 5 paths.**

Verbatim query, with `$cohort_paths` bound to the exact 54-element cohort described above:

```cypher
MATCH (v:DDVersion {id: '4.1.1', is_current: true})
MATCH (r:DDResolution {status: 'active', dd_version: '4.1.1'})
WHERE r.published_value <> r.effective_value
  AND r.path IN $cohort_paths
RETURN count(DISTINCT r.path) AS touched_paths_with_effective_raw_difference
```

Result:

```json
{"touched_paths_with_effective_raw_difference": 5}
```

## Count three — per-path live/raw/effective comparison

**Result: 5 of 5 touched paths now match raw DD, and 0 of 5 match the resolution effective value.** Each path also has exactly one `HAS_UNIT` target, and that target is `1`, consistent with the live scalar.

| Path | Raw DD 4.1.1 | Active resolution effective | Live unit | Live matches |
|---|---:|---:|---:|---|
| `equilibrium/time_slice/constraints/j_parallel/reconstructed` | `1` | `A.m^-2` | `1` | raw DD |
| `equilibrium/time_slice/constraints/j_phi/reconstructed` | `1` | `A.m^-2` | `1` | raw DD |
| `equilibrium/time_slice/constraints/n_e/reconstructed` | `1` | `m^-3` | `1` | raw DD |
| `equilibrium/time_slice/constraints/pressure/reconstructed` | `1` | `Pa` | `1` | raw DD |
| `equilibrium/time_slice/constraints/pressure_rotational/reconstructed` | `1` | `Pa` | `1` | raw DD |

Verbatim query:

```cypher
MATCH (v:DDVersion {id: '4.1.1', is_current: true})
MATCH (r:DDResolution {status: 'active', dd_version: '4.1.1', field: 'unit'})
WHERE r.published_value <> r.effective_value
  AND r.path IN $cohort_paths
MATCH (n:IMASNode {id: r.path})
OPTIONAL MATCH (n)-[:HAS_UNIT]->(u:Unit)
WITH r, n, collect(u.id) AS unit_targets
RETURN r.path AS path,
       r.published_value AS raw_dd_value,
       r.effective_value AS resolution_effective_value,
       n.unit AS live_unit,
       unit_targets,
       CASE
         WHEN r.published_value = ('"' + coalesce(n.unit, '') + '"') THEN 'RAW_DD'
         WHEN r.effective_value = ('"' + coalesce(n.unit, '') + '"') THEN 'RESOLUTION_EFFECTIVE'
         ELSE 'NEITHER'
       END AS live_matches
ORDER BY path
```

`published_value` and `effective_value` are canonical JSON strings in the graph; the table above displays their decoded scalar values.

## Count four — touched and outside-cohort raw-live populations

The two populations are deliberately not summed:

- `touched_live_equals_raw`: **5**.
- `outside_cohort_live_equals_raw`: **0**.

Verbatim query:

```cypher
MATCH (v:DDVersion {id: '4.1.1', is_current: true})
MATCH (r:DDResolution {status: 'active', dd_version: '4.1.1', field: 'unit'})
MATCH (n:IMASNode {id: r.path})
WHERE r.published_value <> r.effective_value
  AND r.published_value = ('"' + coalesce(n.unit, '') + '"')
RETURN count(DISTINCT CASE
         WHEN r.path IN $cohort_paths THEN r.path END
       ) AS touched_live_equals_raw,
       count(DISTINCT CASE
         WHEN NOT (r.path IN $cohort_paths) THEN r.path END
       ) AS outside_cohort_live_equals_raw
```

Result:

```json
{
  "touched_live_equals_raw": 5,
  "outside_cohort_live_equals_raw": 0
}
```

There is therefore no evidence tonight that this exact live-raw/resolution-effective disagreement predates the 54-path re-derivation outside its cohort. That is a statement about the measured current graph, not a claim that rebuild bypass is impossible elsewhere: the 44 remaining active raw/effective differences currently have live values matching their effective resolutions.

## Does the graph disagree with its active resolutions today?

**Yes: exactly 5 resolution rows, representing exactly 5 paths.** They are the five cohort paths enumerated above. A repair restoring effective resolution authority would therefore be sized to **5 currently disagreeing rows**, while a guard preventing raw rebuild overwrite must cover the broader **49-path active-resolution exposure**.

Verbatim query:

```cypher
MATCH (v:DDVersion {id: '4.1.1', is_current: true})
MATCH (r:DDResolution {status: 'active', dd_version: '4.1.1', field: 'unit'})
MATCH (n:IMASNode {id: r.path})
WHERE r.published_value <> r.effective_value
  AND r.effective_value <> ('"' + coalesce(n.unit, '') + '"')
RETURN count(r) AS rows_where_live_disagrees_with_active_resolution,
       count(DISTINCT r.path) AS paths_where_live_disagrees_with_active_resolution,
       collect(r.path) AS paths
```

Result:

```json
{
  "rows_where_live_disagrees_with_active_resolution": 5,
  "paths_where_live_disagrees_with_active_resolution": 5,
  "paths": [
    "equilibrium/time_slice/constraints/pressure_rotational/reconstructed",
    "equilibrium/time_slice/constraints/pressure/reconstructed",
    "equilibrium/time_slice/constraints/n_e/reconstructed",
    "equilibrium/time_slice/constraints/j_phi/reconstructed",
    "equilibrium/time_slice/constraints/j_parallel/reconstructed"
  ]
}
```

## Suite evidence and failure attribution

The historical stated baseline remains the last cleanly attributable pre-wave baseline:

- Revision `b95b136009b33e901892b3783730e9e3fb89da70`.
- Command: `env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync pytest -p no:cacheprovider tests/standard_names/`.
- Exit 1 with **5 pre-existing failures**:
  1. `tests/standard_names/test_docs_review_eligibility.py::test_winning_methods_are_derived_from_schema`
  2. `tests/standard_names/test_docs_review_eligibility.py::test_export_gate_and_population_use_shared_traversal`
  3. `tests/standard_names/test_docs_review_eligibility.py::test_pending_count_and_claim_use_the_same_atomic_predicate`
  4. `tests/standard_names/test_docs_review_eligibility.py::test_stranded_promotion_uses_shared_traversal`
  5. `tests/standard_names/test_edit_prompt_injection.py::test_no_edit_render_matches_golden`

The suite was run once at this worker's merged head:

- Revision `e84ea0b0ece5ec07609bc8e4caf2211c4a01e317`.
- The same command with `PYTHONDONTWRITEBYTECODE=1` added.
- Exit 1 with **6 failed, 6997 passed, 8 skipped, 318 deselected** in 258.07 seconds.
- The same five baseline failures remain.
- Exactly **one newly added failure** remains: `tests/standard_names/test_docs_holdout_set.py::test_docs_holdout_physics_authority_matches_dd_path_bindings`.
- Candidate attribution: commit `23d1465f121a1256255618a8e6f458689ae27910`, specifically its associated live DD 4.1.1 cohort re-derivation. The commit did not edit the holdout test or the DD build helper; the failure is a live graph side effect of invoking the helper over the cohort.

No failure is attributed to this report-only node because it changed no repository or graph state.

## Durable evidence

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T233520799580-n-sli-does-a-rebuild-bypass-a-resolution/resolution-probe.log` — exact DD/version checks, raw-DD cross-check, all verbatim queries, and all results; SHA-256 `0c2b4d999d86e6bc74dee256b1c008ed7cffac0566549317c609db9941af6c40`.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T233520799580-n-sli-does-a-rebuild-bypass-a-resolution/after-suite.log` — merged-head Standard Names suite; SHA-256 `b9e9e38f689dd035b107d3b048cffc277ed079880cf9bc030f7b29d481da9be1`.
- Historical baseline log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T215001801849-n-sgr-consumer-tests-read-the-grammar-they-resolve/after-suite.log`; SHA-256 `086e48ae1ba07113960df7320f4fe72adcd279371a04a6a8ea5a9e29f3be7d69`.

## Sequencing boundary

Do not refresh the stale holdout yet and do not rewrite these resolution records as part of this measurement. The evidence distinguishes two possible follow-on changes that need the coordinator's authority and separate write scopes:

1. Prevent raw rebuild paths from bypassing active `DDResolution` authority across the 49-path exposure surface.
2. After that closure is proven, reconcile the 5 live graph rows that currently disagree with their active effective resolution.

Applying the holdout's raw value now would encode `1` precisely where a resolution-authoritative repair would restore `A.m^-2`, `m^-3`, or `Pa`; this investigation therefore leaves the disagreement visible.
