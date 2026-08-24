# Quiescence baseline verification

The requested baseline is **not green**. The single full run completed with
seven failures, so it cannot establish a green attribution boundary for later
changes.

- Tested HEAD: `b794924bd572cb66ff09045b941c42d3ac73efe4`
- Command: `uv run --no-sync pytest tests/standard_names/ tests/graph/`
- Exit status: `1`
- Captured log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260824T061908191750-n-baselinegreen/baseline-standard-names-graph-pytest.log`
- Recorded at: `2026-08-24T08:26:18+02:00`

Verbatim final pytest summary line:

```text
7 failed, 7207 passed, 8 skipped, 654 deselected, 34 warnings in 213.64s (0:03:33)
```

Failing pytest node IDs:

```text
tests/standard_names/test_docs_resolution_mirror.py::test_projection_is_docs_scoped_canonical_and_idempotent
tests/standard_names/test_docs_resolution_mirror.py::test_non_winning_repair_is_exact_and_axis_scoped
tests/standard_names/test_export_filters.py::TestQuorumShortfallIsNotExportable::test_full_export_requires_docs_quorum_authority
tests/standard_names/test_export_filters.py::TestQuorumShortfallIsNotExportable::test_batch_export_requires_docs_quorum_authority
tests/standard_names/test_review_docs_stages.py::test_docs_shortfall_blocks_refine_claim
tests/graph/test_dd_lifecycle.py::TestReconcileSourcesGate::test_reconcile_never_revives_removed_node_sources
tests/graph/test_dd_resolution_schema.py::test_resolution_schema_has_typed_lifecycle_and_immutable_receipt
```

## `graph_ops.py` worktree scan

All 22 worktrees registered by `git worktree list --porcelain` were checked
with `git status --porcelain -- imas_codex/standard_names/graph_ops.py`.
The count of worktrees holding uncommitted edits to `graph_ops.py` was **0**.
