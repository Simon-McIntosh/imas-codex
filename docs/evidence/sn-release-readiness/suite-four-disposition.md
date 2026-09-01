# Standard Names suite-four disposition

Date: 2026-09-01

Baseline reproduction at worktree head `0dc0bf4d` ran the four plan-named test IDs together and produced **4 failed, 0 passed** in 9.24 s. After correcting the stale assertions, the identical four-ID selection produced **4 passed, 0 failed** in 6.12 s. No production path was changed.

## Per-ID dispositions

### `test_catalog_layout_hierarchy.py::TestRoundTripByteStability::test_byte_stable_round_trip`

Disposition: **stale-test-asserting-a-changed-contract**.

Deciding evidence: the exporter deliberately treats `roles` as graph-only rendered metadata. `_write_domain_yaml` removes `roles` before strict canonical ISN ordering and appends it afterward; the old test parsed emitted YAML and sent `roles` directly to `reorder_entry_dict`, whose contract is to reject every field absent from `CANONICAL_KEY_ORDER`. That mismatch reproduced as `UnknownCatalogKeyError: Unknown catalog key(s) ['roles']`. The test now re-emits through the same two-step ordering as the exporter.

Export-path verdict: **sound**. The test exercises the real `_write_domain_yaml` output, parses it, canonicalizes it, re-applies the export ordering including `roles`, and proves byte-for-byte YAML equality. It passes after the assertion correction, so graph-only role metadata does not make the published catalog drift on a round trip.

### `test_export_exclusion_ledger.py::test_non_quantity_token_collision_blocks_export_with_registry_citation`

Disposition: **stale-test-asserting-a-changed-contract**.

Deciding evidence: the test hardcoded `minimum_safety_factor` as a locus token, but the installed ISN grammar reports no segment for that spelling. The gate therefore correctly returned `passed=True`; there was no governed collision to reject. The corrected test derives a current token from `get_grammar_context()["grammar"]["vocabularies"]["locus_registry"]`; the deterministic current fixture token is `active_limiter_point`.

Export-path verdict: **sound and fail-closed**. For the live governed locus token, the real fixture export reports `identity_token_collision.passed=False`, makes `all_gates_passed=False`, writes no `catalog.yml`, emits exactly one issue naming identity and token `active_limiter_point`, classifies it as segment `locus`, and cites `locus_registry.yml:<line>`. This proves both the refusal and its auditable registry authority rather than merely changing an expected spelling.

### `test_component_system.py::test_repair_normalization_peel_parent_units_scoping`

Disposition: **stale-test-asserting-a-changed-contract**.

Deciding evidence: `repair_normalization_peel_parent_units` is no longer a direct Cypher function. It is intentionally exposed as a `functools.partial` of `apply_signed_manifest`, so `inspect.getsource` correctly raises `TypeError` and a direct `MagicMock` graph call asserts a retired mutation route. The test now pins the governed interface: adapter `normalization-peel-unit-repair`, mutation kind `clear-normalization-peel-parent-unit`, the two explicit guards, the repair reason, and `apply=True`. The disposable-graph behavioral suite remains the owner of mutation predicate coverage.

### `test_sn_help_no_reconcile.py::TestSnHelpNoLegacyVerbs::test_no_resolve_links_as_command`

Disposition: **stale-test-asserting-a-changed-contract**.

Deciding evidence: the test matched any help line beginning with `resolve`, so it rejected the legitimate current `resolve` command while intending only to forbid the retired `resolve-links` command. It now parses exact top-level command names using the same Click help-table shape as the adjacent reconcile check and asserts only that `resolve-links` is absent. The adjacent invocation test still proves `sn resolve-links` is rejected as an unknown command.

## Full-suite capture

Command:

`UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/standard_names`

Result: **6,817 passed, 1 failed, 8 skipped, 306 deselected, 34 warnings** in 270.76 s. The four named IDs contributed **0 failures**.

The one newly failing ID is:

`tests/standard_names/test_pool_registry_imports.py::test_pool_phase_import_orders_succeed_in_fresh_process[import imas_codex.cli.sn; import imas_codex.standard_names.turn as t; print(t.TURN_PHASES)]`

It failed because its fresh-process subprocess exceeded the global 30.0 s pytest timeout. That path is outside this node's exclusive write scope; it is named here for follow-on rather than hidden or rerun for a luckier result.

Full capture: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T134852783119-n-suitefourfix/standard-names-full.log`.
