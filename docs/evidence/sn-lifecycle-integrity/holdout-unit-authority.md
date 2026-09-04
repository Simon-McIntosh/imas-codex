# Reconstructed electron-density unit authority

## Verdict

**The tracked holdout is stale and should be refreshed from `m^-3` to `1`.**

For the authority rule assigned to this node—the unit declared by the exact path in the configured, current Data Dictionary—the live graph is right. DD **4.1.1** declares `1` for `equilibrium/time_slice/constraints/n_e/reconstructed`; the live `IMASNode.unit`, its `units` alias, and its sole `HAS_UNIT` target all read `1`. Commit `23d1465f121a1256255618a8e6f458689ae27910` is the supported candidate attribution for the newly exposed holdout failure because its accompanying live re-derivation rewrote this path from DD 4.1.1 through `_batch_create_path_nodes`. This is a graph side effect associated with that commit, not a source diff to the holdout or authority query.

No source file, holdout row, resolution record, or graph value was changed by this node.

## 1. What the Data Dictionary says

The repository configuration and live graph independently identify **DD 4.1.1** as current:

- `[tool.imas-codex.data-dictionary].version = "4.1.1"` in `pyproject.toml`.
- The live `DDVersion {is_current: true}` row is `4.1.1`.
- The exact DD 4.1.1 XML field `equilibrium/time_slice/constraints/n_e/reconstructed` declares `units="1"`.
- `extract_paths_for_version("4.1.1", {"equilibrium"})` returns `units="1"` for that exact path.

The version history explains why a plausible older answer is different:

| DD version range | Raw DD spelling | Effective extraction result |
|---|---:|---:|
| 3.22.0–3.39.0 | `as_parent` | `m^-3` at 3.39.0, inherited from the `n_e` parent |
| 3.40.0–4.0.0 | `-` | `-`, the DD's dimensionless sentinel and semantically equivalent to `1` |
| 4.1.0–4.1.1 | `1` | `1` |

The graph's own `IMASNodeChange` row records the boundary explicitly: `equilibrium/time_slice/constraints/n_e/reconstructed:units:3.40.0` changed from `m^-3` to `-`. Both the tracked holdout and live graph have null COCOS transformation fields for this path, so COCOS does not distinguish the two sides.

There is one important collision to preserve in the evidence: the path still has an active `DDResolution` for DD 4.1.1 whose published value is `1` and effective value is `m^-3`, with `retiring_release="none-yet"`. That record captures the physically motivated self-contradiction treatment, but the node's assigned decision rule explicitly asks which unit the exact current DD declares. Under that rule, the published DD value `1` wins. The resolution record is therefore follow-on authority debt; it does not make the tracked row current under the raw-DD rule fixed for this investigation.

## 2. Whether tonight's re-derivation wrote the unit

**Yes.** The re-derivation selected 54 current DD 4.1.1 paths, including the reconstructed constraint leaves, and called `extract_paths_for_version` followed by `_batch_create_path_nodes`.

The write is not category-only. `_batch_create_path_nodes` unconditionally includes the extracted unit in each batch row and executes `SET path.unit = p.unit`. It then deletes existing `HAS_UNIT` edges for every path in the batch and re-creates the edge from the path-aware normalized value. For this exact input:

- DD 4.1.1 extraction supplied `p.unit = "1"`.
- The resulting node has `unit="1"` and `units="1"`.
- The resulting authority edge is exactly one `HAS_UNIT -> Unit {id: "1"}`.
- The live node category is `fit_artifact`, as intended by the classification repair.

The written unit therefore **matches DD 4.1.1 exactly**. Commit `23d1465f` itself changed only the node classifier, DD qualifier, and their tests; it did not edit the holdout row, the holdout authority query, `build_dd.py`, or unit registries. The causal mechanism is the live cohort re-derivation performed as part of that commit's delivery: the build helper rewrote more than `node_category`, so the unit moved with the refreshed DD backing data.

The tracked row was originally added by `52764ef77632a494a33de4a3fd4736ad6f209ea0` as a snapshot of the then-live graph-backed value `m^-3`. Its age is not why it loses; it loses because the exact current DD path now declares `1` and the re-derived graph mirrors that declaration.

## 3. Which side is right

| Side | Value | Decision |
|---|---:|---|
| Tracked holdout | `m^-3` | stale under the exact-current-DD authority rule |
| Live graph scalar + edge | `1` | correct; matches DD 4.1.1 |

**Required disposition: refresh the holdout, not the graph.** The mechanism is current-DD re-derivation: `extract_paths_for_version("4.1.1")` reads the explicit dimensionless unit and `_batch_create_path_nodes` replaces both the scalar and edge. A reconstructed constraint often would carry the parent quantity's unit, and the active resolution record preserves that physical interpretation, but this field stopped inheriting `m^-3` in DD 3.40.0 and DD 4.1.1 explicitly says `1`. The requested authority is what the DD declares, not the value physics intuition would choose.

## Whole-suite evidence and attribution

The completed baseline and merged-head whole-suite logs were produced by the measuring node and are reused because checking out the old revision against today's mutated live graph would not reproduce the historical graph state.

### Baseline suite

- Revision: `b95b136009b33e901892b3783730e9e3fb89da70`
- Result: exit 1; **5 failures**.
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T215001801849-n-sgr-consumer-tests-read-the-grammar-they-resolve/after-suite.log`
- Digest: `sha256:086e48ae1ba07113960df7320f4fe72adcd279371a04a6a8ea5a9e29f3be7d69`

Pre-existing failures:

1. `tests/standard_names/test_docs_review_eligibility.py::test_winning_methods_are_derived_from_schema`
2. `tests/standard_names/test_docs_review_eligibility.py::test_export_gate_and_population_use_shared_traversal`
3. `tests/standard_names/test_docs_review_eligibility.py::test_pending_count_and_claim_use_the_same_atomic_predicate`
4. `tests/standard_names/test_docs_review_eligibility.py::test_stranded_promotion_uses_shared_traversal`
5. `tests/standard_names/test_edit_prompt_injection.py::test_no_edit_render_matches_golden`

### Merged-head suite

- Revision: `d89f837bdb9ac6bb906728e3354471f3138ba148`
- Result: exit 1; **32 failures, 6970 passed, 8 skipped, 318 deselected**.
- Added against baseline: **27 failures**.
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T223502625859-n-sgr-the-merged-head-holds/standard-names-suite.log`
- Digest: `sha256:2e4ea76425be7578b027005a5f1e8cae28f7b6526dc6723579439aae0bfb4336`

Added failures and candidate attribution:

- Candidate `dd7c3aec` — 6 failures:
  - `tests/standard_names/test_audit_check_precision.py::test_registered_field_compounds_are_not_bare_fields`
  - `tests/standard_names/test_audit_check_precision.py::test_registered_field_locus_is_not_token_repetition`
  - `tests/standard_names/test_audit_false_positives.py::test_unit_audit_uses_physical_base_and_operator_dimensions[tendency_of_rotation_frequency_of_neoclassical_tearing_mode-s^-2]`
  - `tests/standard_names/test_derivation.py::test_maximum_of_temperature_at_plasma_boundary`
  - `tests/standard_names/test_logarithmic_unit_prefix_audit.py::test_decibel_quantity_with_logarithm_prefix_is_quarantined`
  - `tests/standard_names/test_operator_compose_round_trip.py::test_projection_and_transformation_follow_public_canonical_order`
- Candidate `eefcdf86` — 1 failure:
  - `tests/standard_names/test_gap_rule_verdicts.py::TestOrdinalSamples::test_the_rule_fires_on_no_registered_token`
- Candidate `23d1465f` — 19 fixture failures caused by the newly authoritative `node_category` eligibility predicate:
  - `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_quantity_path_passes_through`
  - `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_deduplicates_multi_cluster_rows`
  - `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_all_clusters_collected`
  - `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_primary_cluster_attached`
  - `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_no_cluster_gives_none_primary`
  - `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_preserves_enrichment_fields`
  - `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_duplicate_cluster_id_not_double_counted`
  - `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_mixed_quantity_and_skip`
  - `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_rows_with_empty_path_skipped`
  - `tests/standard_names/test_enrichment.py::TestIntegration::test_full_flow`
  - `tests/standard_names/test_enrichment.py::TestIntegration::test_mixed_units_split`
  - `tests/standard_names/test_enrichment.py::TestIntegration::test_unclustered_flow`
  - `tests/standard_names/test_enrichment.py::TestMagneticsDomainReclassification::test_enrich_paths_reclassifies_magnetics`
  - `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_eligible_rows_kept`
  - `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_ineligible_rows_removed`
  - `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_mixed_unit_rejected`
  - `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_string_type_rejected`
  - `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_process_path_rejected`
  - `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_skip_records_written`
- Candidate `23d1465f` live graph side effect — 1 failure, now attributed:
  - `tests/standard_names/test_docs_holdout_set.py::test_docs_holdout_physics_authority_matches_dd_path_bindings`

The current observed head `edc24f187dd5221f9eace42e07f11cf36e4c160a` was also checked with the whole suite, once. It has the **same 32 failure IDs**, with 6971 passed, 8 skipped, and 318 deselected; the extra pass is from later unrelated source history, while the failure population is unchanged. Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T230600682031-n-sli-which-unit-authority-is-right/current-standard-names-suite.log`, digest `sha256:508a77669ad437d2f8d6dc93dd6cc4a8262d52c761e6884d2f8a515e89eb195c`.

## Evidence logs

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T230600682031-n-sli-which-unit-authority-is-right/dd-version.log` — configured DD 4.1.1 and installed-version census; exit 0.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T230600682031-n-sli-which-unit-authority-is-right/dd-raw-unit-history.log` — exact XML unit over every installed DD version; exit 0.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T230600682031-n-sli-which-unit-authority-is-right/dd-extracted-unit-history.log` — extraction behavior around the 3.39→3.40 boundary and current DD; exit 0.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T230600682031-n-sli-which-unit-authority-is-right/live-unit-authority-compact.log` — live current DD, node scalar/edge, active resolution, and graph-recorded unit change; exit 0.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T230600682031-n-sli-which-unit-authority-is-right/check-index.md` — exact commands, exit statuses, and log paths for every command run.

One first live-graph query failed at parse time because its Cypher parameter was over-escaped; it made no graph request beyond syntax validation and no write. The corrected read succeeded once. Both logs and commands are retained in the check index.

## Follow-on boundary

The requested holdout refresh and reconciliation of the still-active `1 → m^-3` `DDResolution` are outside this read-only node. Do not mutate either blindly: the next node should first state whether repository policy continues to distinguish raw published DD authority (`1`) from reviewed effective authority (`m^-3`), then update the holdout and resolution consistently with that ruling. Under this node's fixed raw-DD rule, the immediate tracked-row change is `m^-3` → `1`.
