# Repaired-head consumer-suite measurement

## Outcome

The combined repaired head holds the stated consumer baseline. At consumer revision `c058283ae7d4948bf5ad8f33cbf92a8e09fb8fa0`, the complete `tests/standard_names/` gate produced exactly the six failures accounted for before execution and no seventh failure. The complete `tests/core/` gate remained clean at 924 passed and zero failures.

| Gate | Exit | Result | Stated failure base | Added failures |
|---|---:|---|---:|---:|
| `tests/standard_names/` | 1 | 6 failed, 6997 passed, 8 skipped, 318 deselected | 6 accounted ids | 0 |
| `tests/core/` | 0 | 924 passed, 2 skipped | 0 | 0 |

The nonzero Standard Names exit is entirely explained by the six known failures below. It is not a seventh or unaccounted regression.

## Measured revision and repair lineage

- Assigned worktree revision: `c058283ae7d4948bf5ad8f33cbf92a8e09fb8fa0`.
- Main checkout revision at orientation, suite completion, and final audit: `c058283ae7d4948bf5ad8f33cbf92a8e09fb8fa0`.
- Eligibility-fixture repair: `a23929bad55fff5bd0c7bc6b7e53e4b00bf8ee5e`, confirmed an ancestor of the measured head.
- Grammar-assertion repair: `324a1f1aa5f1520e59fad4c499ba68f3c0965ef3`, confirmed an ancestor of the measured head.
- The worktree was clean before and after measurement. No source file was changed, no dependency was installed, and the shared environment was never synchronized.

The earlier merged-head full-suite measurement at `d89f837bdb9ac6bb906728e3354471f3138ba148` contained 32 failures. The two isolated repair measurements reported 19 and 7 removals respectively, but that arithmetic was not treated as combined-head evidence. This run closes that gap: all 26 targeted failures are absent when both repairs are present together, and the only remaining failures are the six accounted ids.

## Exact commands and captured logs

Primary gate, executed once:

```text
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync pytest -p no:cacheprovider tests/standard_names/
```

- Exit status: `1`.
- Summary: `6 failed, 6997 passed, 8 skipped, 318 deselected, 34 warnings in 260.68s`.
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T233633838135-n-sgr-the-repaired-head-holds/standard-names-suite.log`.
- SHA-256: `50729deb7c2be1fce5295b9d9628819359c87c138b7cc7e27ff62ee7f0344ea5`.

Secondary gate, executed once:

```text
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync pytest -p no:cacheprovider tests/core/
```

- Exit status: `0`.
- Summary: `924 passed, 2 skipped, 1 warning in 13.57s`.
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T233633838135-n-sgr-the-repaired-head-holds/core-suite.log`.
- SHA-256: `a78a8fb677e096137cc96fd80d418d477507d5e26511f35a7ac720cbb8f463a2`.

Both logs contain the complete captured pytest output followed by an explicit `EXIT_STATUS` line. Results were read from those files after execution; neither suite was piped through a filter or rerun.

## Pre-existing and intentionally retained failures

Five failures predate tonight's grammar work:

1. `tests/standard_names/test_docs_review_eligibility.py::test_winning_methods_are_derived_from_schema`
2. `tests/standard_names/test_docs_review_eligibility.py::test_export_gate_and_population_use_shared_traversal`
3. `tests/standard_names/test_docs_review_eligibility.py::test_pending_count_and_claim_use_the_same_atomic_predicate`
4. `tests/standard_names/test_docs_review_eligibility.py::test_stranded_promotion_uses_shared_traversal`
5. `tests/standard_names/test_edit_prompt_injection.py::test_no_edit_render_matches_golden`

One additional failure is intentionally retained while the graph-authority question is measured:

6. `tests/standard_names/test_docs_holdout_set.py::test_docs_holdout_physics_authority_matches_dd_path_bindings`

The complete tracebacks for all six are preserved in the Standard Names log. These failures are listed here as the accounted baseline, not attributed to either repair commit.

## Added failures and attribution

There are no added failures in either suite:

```json
{}
```

Accordingly, there is no candidate commit to name in `failure_attribution`. In particular, no interaction failure appeared between `a23929ba` and `324a1f1a`.

For removal attribution only:

- `a23929ba` accounts for the 19 eligibility-fixture failures removed by supplying the node category required by each fixture scenario.
- `324a1f1a` accounts for the 7 grammar-assertion failures removed by making the assertions read the merged canonical grammar while retaining their tested properties.
- Combined delta from the prior 32-failure head to the repaired head: 26 failures removed, 0 unaccounted failures added, 6 accounted failures remaining.

## Environment and coverage boundary

No optional dependency was installed. `torch` and `sentence-transformers` remain absent by design. The previously measured six-file coverage gap caused by that difference between the declared test extra and the shared working environment was not re-enumerated here because this node's exact checks were `tests/standard_names/` and `tests/core/`; that gap remains an independent recorded finding rather than being folded into either suite result.

## Attribution conclusion

The repaired consumer head satisfies the quantitative done-when: Standard Names has exactly six accounted failures and zero added failures, while core has 924 passes and zero failures. The two repairs therefore hold together at the measured merged head. The remaining six failures retain their pre-existing or intentionally unresolved ownership and were not repaired by this measurement node.
