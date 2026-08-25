# Full default-selection baseline

The first complete default-selection run available after installing the test
extra is **red**. Collection and execution reached 100%, but 11 selected tests
failed, so this record is a baseline and not a green terminal gate.

- Tested revision: `1d6cca57e241cc1326072277e1f9e7aaa816ad89`
- Command: `uv run --no-sync pytest`
- Worktree environment: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv`, with the worktree first on `PYTHONPATH`
- Exit status: `1`
- Captured log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T004953060596-n-fullsuitebaseline/pytest-default-selection.log`
- Run duration reported by pytest: 492.64 seconds (8 minutes 12 seconds)

Verbatim final pytest summary line:

```text
11 failed, 11956 passed, 141 skipped, 750 deselected, 1 xfailed, 35 warnings in 492.64s (0:08:12)
```

## Selection size against the narrower baseline

The default selection produced **12,109 selected outcomes**: 11,956 passed,
11 failed, 141 skipped, and 1 expected failure. That is **4,891 more selected
outcomes than the 7,218 tests reported by the narrower
`tests/standard_names/ tests/graph/` selection**, or 1.678 times as many
(67.8% more). Pytest separately reported 750 deselected tests. Comparing pass
counts alone, this run has 11,956 passes, 4,738 more than 7,218.

## Failing pytest node IDs

```text
tests/units/test_dd_unit_exceptions.py::TestUnitsAgree::test_dd_side_bug_charge_number
tests/units/test_dd_unit_exceptions.py::TestUnitsAgree::test_dd_side_bug_unit_vector
tests/units/test_dd_unit_exceptions.py::TestUnitsAgree::test_dd_side_bug_charge_state_bundle_bounds
tests/units/test_dd_unit_exceptions.py::TestUnitsAgree::test_dd_side_bug_ggd_value_copies
tests/units/test_dd_unit_exceptions.py::TestUnitsAgree::test_dd_side_bug_wave_vector_tagged_as_electric_field
tests/units/test_dd_unit_exceptions.py::TestUnitsAgree::test_dd_side_bug_gas_flow_rate
tests/units/test_dd_unit_exceptions.py::TestGraphUnitCorrection::test_legacy_reconstructed_constraint_sentinels_are_rewritten
tests/units/test_dd_unit_exceptions.py::TestGraphUnitCorrection::test_poloidal_angle_sentinel_is_rewritten
tests/units/test_dd_unit_exceptions.py::TestGraphUnitCorrection::test_phase_space_source_dimensionality_is_rewritten
tests/units/test_dd_unit_exceptions.py::TestGraphUnitCorrection::test_active_manifest_retires_matching_graph_and_comparator_authority
tests/units/test_dd_unit_exceptions.py::TestGraphUnitCorrection::test_nonresolved_legacy_rule_keeps_exact_behavior
```

All 11 failures share the same observable failure path: graph-backed DD
resolution loading raises `neo4j.exceptions.AuthError` with
`Neo.ClientError.Security.Unauthorized`, then surfaces as
`DDResolutionManifestInvalid: cannot read DD resolution graph authority`.
The detached worktree had no `.env` link during this first run. This records
the result as produced; the suite was not rerun with credentials, and the
common traceback does not by itself determine whether the required repair is
test isolation, credential setup, or graph-authority fallback behavior.
