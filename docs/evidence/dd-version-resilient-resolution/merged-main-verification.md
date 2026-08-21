# Integrated main verification

Date: 2026-08-21

Verdict: **FAILED — integrated `main` is not fully green.** Model generation
and the full Standard Names suite pass. The credentialed graph suite has ten
failures, and both Ruff checks fail. This verification node changed no product
or test code and attempted no repair.

## Integrated checkout

- Tested `HEAD`: `7400ea2ae36b29ec5c5ce889a0651ec78da04960`
- `main` and the canonical checkout matched that SHA at preflight.
- Canonical `main` advanced after the run began; the final audit observed
  `f8521d841a10450921d03149d28c4f7106c7ddbc`. This report certifies only the
  dispatched detached base `7400ea2a`, not that later concurrent tip.
- The row-convergence change `3d68f699e8e3ed57a004d37a1739394e2cda0d27`
  and repair-authority schema change
  `30ee0249b49fdbc0b9c137039d244928ca6a8637` are both ancestors of the tested
  `HEAD`.
- `git status --porcelain` emitted no output before the runs and after all four
  runs. Generated model files remained ignored and were never staged.
- The canonical gitignored `.env` supplied the Neo4j credential to the graph
  subprocess without being copied into this worktree or recorded in a log.

All Python commands reused `/home/ITER/mcintos/Code/imas-codex/.venv` with
`UV_PROJECT_ENVIRONMENT` and `PYTHONPATH="$PWD"`; `--no-sync` prevented this
detached worker from mutating the shared environment.

## Run results

### Model generation — PASS

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync build-models --force
```

- Exit code: `0`
- Five `Generated:` artifact lines were emitted.
- Headline schema-reference output, verbatim:
  `[gen-schema-ref] Generated .../agents/schema-reference.md (92 labels, 17 indexes)`
- Headline schema-context output, verbatim:
  `[gen-schema-context] Generated .../imas_codex/graph/schema_context_data.py (92 labels, 17 vector indexes, 5 fulltext indexes, 6 task groups)`
- The run warned that `RepairAuthorityArtifact.schema` shadows a parent-model
  attribute; it did not fail generation.

### Standard Names suite — PASS

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/standard_names/ -q
```

| Outcome | Count |
|---|---:|
| Passed | 6,542 |
| Skipped | 8 |
| Failed | 0 |
| Errors | 0 |
| Total outcomes | 6,550 |

Exit code: `0`.

The repository already configures quiet pytest output, and the invocation adds
`-q`, so pytest suppresses its prose summary line. The complete captured result
stream contains 6,542 `.` markers, 8 `s` markers, no `F` or `E` markers, and
`EXIT_CODE=0`.

#### Comparison with the ddconv baseline

The recorded baseline at `3d68f699` is **6,536 passed / 8 skipped / 0
failed**. The integrated result is **+6 passed, +0 skipped, +0 failed, +0
errors**. `git diff 3d68f699..7400ea2a -- tests/standard_names` adds exactly
one test module, and collection identifies these six new passing test IDs:

1. `tests/standard_names/test_repair_authority_schema.py::test_committed_authority_validates_without_resigning[catalog-edit-dual-binding-adjudication.json-5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e]`
2. `tests/standard_names/test_repair_authority_schema.py::test_committed_authority_validates_without_resigning[refused-target-orphan-adjudication.json-2c2d38f3241ec3057d24a5d05c27840f5e4ffe99520063059ab31c1e9d4bca36]`
3. `tests/standard_names/test_repair_authority_schema.py::test_committed_authority_validates_without_resigning[owner-geometry-rc66-partition.json-dbb37f7be12ba99d7e85bf13b9d63e6c19cb6c20bd35fe687e590f798e2dc85b]`
4. `tests/standard_names/test_repair_authority_schema.py::test_committed_authority_validates_without_resigning[stale-source-lifecycle.json-f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad]`
5. `tests/standard_names/test_repair_authority_schema.py::test_canonical_authority_fields_cover_the_recorded_extensions`
6. `tests/standard_names/test_repair_authority_schema.py::test_mutation_vocabulary_includes_lifecycle_and_edge_removal`

No baseline test was removed, changed from pass to skip, or changed from pass
to failure.

### Credentialed graph suite — FAIL

The Neo4j credential was loaded from the canonical gitignored `.env` before
executing:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider -m graph tests/graph/ -q
```

| Outcome | Count |
|---|---:|
| Passed | 277 |
| Skipped | 65 |
| Expected failures | 1 |
| Failed | 10 |
| Errors | 0 |
| Total outcomes | 353 |

Exit code: `1`. The selection executed; this is not pytest exit code `5` and
is not a “not run” result.

Failing test IDs and headline findings:

1. `tests/graph/test_data_quality.py::TestDescriptionQuality::test_no_empty_string_descriptions`
   — 10 `SignalEpoch` nodes have empty descriptions.
2. `tests/graph/test_grammar_graph_compliance.py::TestSegmentOrderCompliance::test_has_segment_edge_positions_match_isn_index`
   — live `HAS_SEGMENT.position` values disagree with current ISN indexes.
3. `tests/graph/test_schema_compliance.py::TestLabelsAndRelationships::test_no_undeclared_properties`
   — `StandardNameChange.manifest_sha256` exists in the graph but is undeclared
   in LinkML.
4. `tests/graph/test_sn_edge_integrity.py::TestStandardNameEdgeIntegrity::test_dd_attachment_unit_agrees_with_name`
   — `electron_source_rate` has unit `s^-1` while two attached DD paths have
   unit `m^-3.s^-1`.
5. `tests/graph/test_sn_graph.py::TestStandardNameGraph::test_cocos_dependent_names_linked`
   — 1 of 984 COCOS-dependent names lacks both the integer mirror and
   `HAS_COCOS` edge.
6. `tests/graph/test_sn_integrity_ratchets.py::test_sources_with_multiple_live_targets_do_not_regrow`
   — 27 dual-bound sources exceed the ceiling of 23.
7. `tests/graph/test_sn_integrity_ratchets.py::test_stale_sources_with_live_bindings_do_not_regrow`
   — 9 stale bound sources exceed the ceiling of 3.
8. `tests/graph/test_sn_semantic_source_invariant.py::TestSemanticSourceLedgerInvariant::test_no_semantic_source_mirror_disagreements`
   — 28 mirror violations: 27 multiple-live-target rows plus one scalar/edge
   disagreement.
9. `tests/graph/test_sn_unit_integrity.py::TestSNUnitIntegrity::test_sn_unit_matches_linked_dd_path_unit`
   — the same two `electron_source_rate` attachments disagree with canonical DD
   units.
10. `tests/graph/test_structural.py::TestClusterIntegrity::test_clusters_have_members`
    — one `IMASSemanticCluster` has no `IN_CLUSTER` member.

The full assertion text, including every offending source identity, is retained
in the graph log.

### Ruff — FAIL

The fourth measured run executed both required checks over `imas_codex` and
`tests` in one log:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ruff check --no-cache imas_codex tests
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ruff format --check --no-cache imas_codex tests
```

- `ruff check`: exit code `1`; headline output verbatim: `Found 1 error.`
  The finding is `B017` at
  `tests/standard_names/test_axis_catch_and_promote.py:59`, where
  `pytest.raises(Exception)` asserts a blind exception.
- `ruff format --check`: exit code `1`; headline output verbatim:
  `8 files would be reformatted, 1101 files already formatted`.
- Combined fourth-run exit code: `1`.

The eight format findings are the four generated files
`imas_codex/config/models.py`, `imas_codex/graph/dd_models.py`,
`imas_codex/graph/models.py`, and
`imas_codex/graph/schema_context_data.py`, plus these tracked tests:

- `tests/standard_names/test_cli_release.py`
- `tests/standard_names/test_sources_manifest.py`
- `tests/standard_names/test_tombstone_supersede.py`
- `tests/standard_names/test_vocab_gap_triage.py`

Ruff ran in check-only mode and changed none of them.

## Durable logs

| Run | Log | SHA-256 |
|---|---|---|
| Model generation | `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T105756125941-mergedverify/build-models.log` | `530c84472401d92de662e9c6ae78909553958d031bb38a3d725db954f2192b6e` |
| Standard Names | `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T105756125941-mergedverify/standard-names.log` | `e1fe7293a60b49837fc9dd0584cba7960af367709f16fccf1b3237cf22489590` |
| Graph | `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T105756125941-mergedverify/graph-tests.log` | `0a7feb7ca58a76c2886f4b2e61e0ef9ade7ecaf90695dd5f9bbd41602972721b` |
| Ruff | `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T105756125941-mergedverify/ruff.log` | `09fa85d3033d8157df2dbbdbffd7a5053ab8824a0287482dcbc46d57f478509c` |

## Disposition

The integrated Python Standard Names behavior is green and its six-test growth
is fully accounted for. Integration cannot be declared wholly green because
the live graph and Ruff gates fail. Fixing graph data, schemas, or test style is
outside this verification-only node and no such change was attempted.
