# Merged-head consumer measurement

## Outcome

The corrected measurement is complete at the node's assigned imas-codex revision `d89f837bdb9ac6bb906728e3354471f3138ba148`, with the editable grammar resolving from upstream revision `e00af1c2bdc1ff936b4b5c1fc535c3be40f592b9`.

- Primary `tests/standard_names/` gate: **32 failed, 6970 passed, 8 skipped, 318 deselected; exit 1**. The five known baseline failures remain, and **27 failures were added**.
- Secondary `tests/core/` gate: **924 passed, 2 skipped; exit 0**. Added failures against its zero-failure base: **0**.
- Required strict reparse: `power_flux_density_maximum_at_inner_divertor_target` still fails with `UnknownBaseTokenError`; the closed vocabulary still owes `power_flux_density`.
- No optional dependency was installed. The shared environment intentionally has neither `torch` nor `sentence-transformers`.
- No source file was changed and no failure was repaired.

The 27 added Standard Names failures do not all belong to the two grammar commits. The evidence-supported attribution is:

| Cause | Added failures | Evidence |
|---|---:|---|
| `dd7c3aec` — tail-gated operator placement | 6 | Previously accepted spellings are now non-canonical or unparseable; one compose expectation pins the former operator order. |
| `eefcdf86` — six closed-vocabulary tokens | 1 | `s0`, `s1`, `s2`, and `s3` are now registered, invalidating a test that requires them to be absent. |
| `23d1465f1` — consumer DD node-category eligibility guard | 19 | Older enrichment and qualification fixtures omit `node_category`; the new guard rejects them all as `node_category_ineligible`. |
| Live graph authority; no honest commit attribution | 1 | The tracked holdout says `m^-3` while the live graph reports `1` for one exact DD path. |

## Orientation and candidate identity

- Live plan read: `imas-codex:sn-grammar-refinement`, version 13.
- Consumer assigned worktree `HEAD`: `d89f837bdb9ac6bb906728e3354471f3138ba148`.
- Consumer main checkout `main` at orientation and suite start: `d89f837bdb9ac6bb906728e3354471f3138ba148`, matching the assigned worktree.
- Consumer main checkout after the measurements completed: `edc24f187dd5221f9eace42e07f11cf36e4c160a`. The intervening commits concern graph-pull targeting and export failure propagation; the worker did not switch, merge, or rerun outside its assigned revision.
- Grammar main and origin/main: `e00af1c2bdc1ff936b4b5c1fc535c3be40f592b9`.
- Vocabulary merge: `908f413`; candidate `eefcdf86` is an ancestor of grammar `HEAD`.
- Tail-rule merge: `e00af1c`; candidate `dd7c3aec` is an ancestor of grammar `HEAD`.
- Lead-supplied merged grammar gate: 1195 passed, exit 0.
- Resolved editable import: `/home/ITER/mcintos/Code/imas-standard-names/imas_standard_names/__init__.py`.

## Primary gate: `tests/standard_names/`

Exact command:

```text
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest -p no:cacheprovider tests/standard_names/
```

- Revision: `d89f837bdb9ac6bb906728e3354471f3138ba148`.
- Exit status: 1.
- Result: 32 failed, 6970 passed, 8 skipped, 318 deselected, 34 warnings in 232.23 seconds.
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T223502625859-n-sgr-the-merged-head-holds/standard-names-suite.log`.
- Log SHA-256: `2e4ea76425be7578b027005a5f1e8cae28f7b6526dc6723579439aae0bfb4336`.
- Baseline population: the same `tests/standard_names/` directory at `b95b136009b33e901892b3783730e9e3fb89da70`, previously measured as 5 failed and 6991 passed.
- Added failures: 27.

### Unchanged pre-existing failures: 5

1. `tests/standard_names/test_docs_review_eligibility.py::test_winning_methods_are_derived_from_schema`
2. `tests/standard_names/test_docs_review_eligibility.py::test_export_gate_and_population_use_shared_traversal`
3. `tests/standard_names/test_docs_review_eligibility.py::test_pending_count_and_claim_use_the_same_atomic_predicate`
4. `tests/standard_names/test_docs_review_eligibility.py::test_stranded_promotion_uses_shared_traversal`
5. `tests/standard_names/test_edit_prompt_injection.py::test_no_edit_render_matches_golden`

### Added failures attributed to `dd7c3aec`: 6

1. `tests/standard_names/test_audit_check_precision.py::test_registered_field_compounds_are_not_bare_fields`
   - Traceback result: `implicit_field_check(...)` returns one issue instead of `[]` because the old spelling falls back to lexical analysis and sees bare `field` after `low`.
   - Parser evidence: `magnetic_field_at_pedestal_top_low_field_side_magnitude` now raises `NonCanonicalNameError`; the canonical form is `magnetic_field_magnitude_at_pedestal_top_low_field_side`.
2. `tests/standard_names/test_audit_check_precision.py::test_registered_field_locus_is_not_token_repetition`
   - Traceback result: `repeated_token_check(...)` returns a duplicate-`field` issue instead of `[]` for the same now-non-canonical spelling.
3. `tests/standard_names/test_audit_false_positives.py::test_unit_audit_uses_physical_base_and_operator_dimensions[tendency_of_rotation_frequency_of_neoclassical_tearing_mode-s^-2]`
   - Traceback result: the structured parse no longer suppresses the lexical frequency-unit rule, so `s^-2` is rejected as a frequency unit.
   - Parser evidence: the former spelling now raises `UnknownBaseTokenError` for residue `tendency_of_rotation_frequency`.
4. `tests/standard_names/test_derivation.py::test_maximum_of_temperature_at_plasma_boundary`
   - Traceback result: `derive_edges(...)` returns no `HAS_PARENT` edge, so `len(co)` is 0 rather than 1.
   - Parser evidence: `maximum_of_temperature_at_plasma_boundary` now raises `ParseError`; it retains the former prefix-before-operand order across a locus tail.
5. `tests/standard_names/test_logarithmic_unit_prefix_audit.py::test_decibel_quantity_with_logarithm_prefix_is_quarantined`
   - Traceback result: the audit finds zero matching logarithmic-prefix issues rather than one because its old spelling no longer parses.
   - Parser evidence: `logarithm_of_spectral_signal_to_noise_ratio_of_spectrometer_channel` now raises `ParseError`.
6. `tests/standard_names/test_operator_compose_round_trip.py::test_projection_and_transformation_follow_public_canonical_order`
   - Traceback result: composer output is `perpendicular_momentum_flux_normalized_due_to_e_cross_b_drift`; the assertion still expects `perpendicular_normalized_momentum_flux_due_to_e_cross_b_drift`.

The four direct parser probes and the live holdout comparison are captured in `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T223502625859-n-sgr-the-merged-head-holds/attribution-diagnostics-corrected.log` (SHA-256 `147c036f20deaef39e0d669f8bfb744ecb4118bab8fe963b355f75c06d557ff3`).

### Added failure attributed to `eefcdf86`: 1

1. `tests/standard_names/test_gap_rule_verdicts.py::TestOrdinalSamples::test_the_rule_fires_on_no_registered_token`
   - Traceback result: `assert not offenders` receives `['s0', 's1', 's2', 's3']`.
   - Mechanism: the vocabulary candidate deliberately registered those four Stokes component indices. The consumer assertion still defines them as examples with no registered token.

### Added failures attributed to consumer commit `23d1465f1`: 19

Commit `23d1465f1` added a fail-closed guard in `qualify_dd`: a candidate whose `metadata.node_category` is absent or outside `SN_SOURCE_CATEGORIES` returns `node_category_ineligible`. The older fixtures in `test_enrichment.py` default `node_category` to `None`, and the `_row` helper in `test_qualify_sources.py` omits it entirely. Their rows are therefore filtered before the behavior each test intends to exercise.

The 13 enrichment failures are:

1. `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_quantity_path_passes_through`
2. `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_deduplicates_multi_cluster_rows`
3. `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_all_clusters_collected`
4. `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_primary_cluster_attached`
5. `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_no_cluster_gives_none_primary`
6. `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_preserves_enrichment_fields`
7. `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_duplicate_cluster_id_not_double_counted`
8. `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_mixed_quantity_and_skip`
9. `tests/standard_names/test_enrichment.py::TestEnrichPaths::test_rows_with_empty_path_skipped`
10. `tests/standard_names/test_enrichment.py::TestIntegration::test_full_flow`
11. `tests/standard_names/test_enrichment.py::TestIntegration::test_mixed_units_split`
12. `tests/standard_names/test_enrichment.py::TestIntegration::test_unclustered_flow`
13. `tests/standard_names/test_enrichment.py::TestMagneticsDomainReclassification::test_enrich_paths_reclassifies_magnetics`

Their tracebacks uniformly show empty output where one or two enriched rows were expected, or `IndexError` when an assertion indexes the now-empty result.

The 6 qualification failures are:

14. `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_eligible_rows_kept`
15. `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_ineligible_rows_removed`
16. `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_mixed_unit_rejected`
17. `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_string_type_rejected`
18. `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_process_path_rejected`
19. `tests/standard_names/test_qualify_sources.py::TestQualifySources::test_skip_records_written`

Five tracebacks show zero kept rows where one or two were expected. The sixth shows the recorded reason changed from expected `duplicate_ids` to actual `node_category_ineligible`. The captured logger confirms that every fixture population was filtered to zero.

### Added failure with no supported commit attribution: 1

1. `tests/standard_names/test_docs_holdout_set.py::test_docs_holdout_physics_authority_matches_dd_path_bindings`
   - Traceback result: tracked `declared_unit='m^-3'` differs from live graph authority `declared_unit='1'`.
   - Exact mismatch: DD path `equilibrium/time_slice/constraints/n_e/reconstructed`, catalog name `electron_density`; tracked unit `m^-3`, live graph unit `1`; both COCOS fields are null.
   - Attribution: live graph authority changed after the five-failure baseline. No source diff between `b95b1360` and `d89f837b` changes this holdout row or its authority query, so assigning this failure to either grammar commit or to a consumer commit would be unsupported.

## Secondary gate: `tests/core/`

Exact command:

```text
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run --no-sync pytest -p no:cacheprovider tests/core/
```

- Revision: `d89f837bdb9ac6bb906728e3354471f3138ba148`.
- Exit status: 0.
- Result: 924 passed, 2 skipped, 1 warning in 12.73 seconds.
- Failure count: 0.
- Failure IDs: none.
- Added failures against the lead-supplied zero-failure base: 0.
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T223502625859-n-sgr-the-merged-head-holds/core-suite.log`.
- Log SHA-256: `e85d3d1411c01f959a9520faa37111be5decb34f3a26582a2f1add061117c766`.

## Tests unavailable in the ordinary shared environment

No package installation was attempted. A direct probe confirms `torch=false` and `sentence_transformers=false`. The declared `test` extra includes both packages, while the plain shared environment intentionally includes neither.

The exact census is in `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T223502625859-n-sgr-the-merged-head-holds/optional-dependency-gap.log` (SHA-256 `adc76c44d2310dabb2e72ac333362faed3d696f3546250039d45d08e58d5c0fb`).

### Hard collection fence for missing `torch`: 3 files

`tests/conftest.py` names these three files in `_TEST_EXTRA_DEPENDENT_SUITES`. Selecting any of their collected items while `torch` is absent exits pytest with usage error 4 before any test body runs:

1. `tests/embeddings/test_prompt_name.py` — 8 declared test functions.
2. `tests/features/test_search.py` — 12 declared test functions.
3. `tests/test_imas_search_scoring.py` — 34 declared test functions.

Total: **3 files and 54 statically declared test functions**. Parametrization can make the eventual collected-item count larger; the hard guard prevents an honest runtime item count without providing the dependency, so 54 is deliberately reported as the source-level function count rather than mislabelled as collected cases.

### Module-level collection skips for missing `sentence-transformers`: 2 files

These modules call `pytest.importorskip('sentence_transformers')` at import time and therefore do not collect their tests when the package is absent:

1. `tests/embeddings/test_embedding_encoder.py` — 3 declared test functions.
2. `tests/embeddings/test_model_comparison.py` — 1 declared test function.

Total: **2 files and 4 declared test functions skipped at module collection**. All four functions are also marked `slow`, directly or through a module mark, so the default `-m 'not slow and not graph'` population would deselect them even if the package were installed.

### Per-test skips for missing `sentence-transformers`: 1 additional file

`tests/embeddings/test_encoder.py` still collects, but three tests are individually guarded by `skipif(not _has_sentence_transformers)`:

1. `tests/embeddings/test_encoder.py::TestEncoder::test_initialization_default_config` — unmarked, so it is skipped rather than executed in the ordinary default population.
2. `tests/embeddings/test_encoder.py::test_encoder_embed_texts_basic` — marked `slow`, therefore normally deselected first.
3. `tests/embeddings/test_encoder.py::test_encoder_build_document_embeddings_cache_integration` — marked `slow`, therefore normally deselected first.

Across both dependencies, **5 files cannot contribute any collected/executed tests without the optional packages, and 1 additional file has 3 dependency-skipped tests**. This is a six-file coverage gap. The declared test environment includes both dependencies, but the environment used by the ordinary suite does not; these tests therefore provide no passing behavioral signal in the routine gate today.

The earlier deliberately over-broad full-tree attempt recorded the hard fence in `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T223502625859-n-sgr-the-merged-head-holds/merged-head-suite.log`: exit 4 after 41.17 seconds, before test execution. That attempt is retained only as evidence of the coverage gap and is not compared with either corrected suite population.

## Required strict reparse

Exact command:

```text
env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python -c 'from imas_standard_names.grammar import parse_standard_name; print(parse_standard_name("power_flux_density_maximum_at_inner_divertor_target"))'
```

- Grammar state: both candidate commits merged at grammar `HEAD` `e00af1c2bdc1ff936b4b5c1fc535c3be40f592b9`; merged grammar suite reported by the lead as 1195 passed, exit 0.
- Exit status: 1.
- Outcome: expected continued failure.
- Parser detail: residue `power_flux_density_maximum` matches no `physical_base` or `geometry_carrier`; nearest candidate is `power_density`. The public parser then raises `UnknownBaseTokenError` against the 212-token closed vocabulary.
- Interpretation: `power_flux_density` remains the owed physical-base token. This failure predates both candidates and is not attributed to `dd7c3aec` or `eefcdf86`.
- Log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T223502625859-n-sgr-the-merged-head-holds/strict-reparse.log`.
- Log SHA-256: `625692b8e6c5886408c9f4c639eb3eb1ca2883adf4f76cc17bbdc61ca09d808f`.

## Follow-on repair scopes

This node made no repairs. The measured follow-ons are separable:

1. Update the six tail-rule-sensitive consumer assertions/fixtures to the canonical grammar without weakening their audit and round-trip properties; candidate owner `dd7c3aec`.
2. Update the single Stokes absence assertion to reflect the newly registered `s0`–`s3` component tokens; candidate owner `eefcdf86`.
3. Repair 19 older DD-source fixtures so they state an eligible `node_category` when testing downstream behavior, while retaining negative tests for the new fail-closed category guard; candidate owner `23d1465f1`.
4. Reconcile the tracked holdout unit for `equilibrium/time_slice/constraints/n_e/reconstructed` against its live graph authority before assigning any source commit.
5. Add and validate the still-owed `power_flux_density` physical base in a separately attributable upstream vocabulary change.
6. Decide explicitly whether the six-file optional-dependency coverage population belongs in a routinely provisioned gate, a separately provisioned scheduled gate, or a smaller dependency-free test design. Installing the roughly 2 GB extra into the shared environment was not authorized and was not done.
