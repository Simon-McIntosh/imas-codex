# Remaining suite-failure triage

## Result

The fresh targeted run leaves **1 of the original 29 failures still failing** once the declared `test` extra is installed. The verdict census is:

| Verdict | Count |
|---|---:|
| `defect-to-fix-here` | 0 |
| `owned-elsewhere` | 0 |
| `environmental-artifact` | 28 |
| `wrong-premise` | 1 |

The 27 failures whose original trace ended in `ModuleNotFoundError: No module named 'torch'` all passed after provisioning the repository's declared test extra. The remote-cleanup timeout also passed, in 3.66 seconds. The one remaining failure is the graph-marker test: its subprocess ran the graph-marked test successfully instead of exercising the asserted credentialless branch.

## Fresh-run evidence

- Environment provisioning: `uv sync --extra test`, run in `/home/ITER/mcintos/Code/imas-codex`; this selected the configured `pytorch-cpu` index and installed `torch==2.10.0+cpu` in the shared main-checkout environment.
- Environment identity: `/home/ITER/mcintos/Code/imas-codex/.venv/bin/python3`; `torch.__version__ == "2.10.0+cpu"`.
- Command:

  ```text
  env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/cli/test_services_remote.py tests/embeddings/test_prompt_name.py tests/features/test_search.py tests/ids/test_mapping_e2e.py tests/integration/test_workflows.py tests/test_graph_marker_gate.py tests/test_imas_search_scoring.py
  ```

- Exit status: **1**.
- Outcome: **162 passed, 1 failed, 6 skipped** in **64.87 s**.
- Original-census failures still failing with the test extra: **1 / 29**.
- Full log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T202010483788-n-suite-triage-29/logs/pytest-census-files.log`.

The node fence says “exactly nine files,” but the live plan's measured census names ten files in total, three of which are the already-repaired Standard Names suites. The 29 unresolved failures are therefore distributed across exactly the seven files in the command above (1 + 2 + 10 + 2 + 7 + 1 + 6 = 29). There is no nine-file selection in the live census; adding two files would invent evidence inputs outside the authoritative set.

## Per-test verdicts

Each row covers one of the 29 IDs from the opening census and gives one line of evidence from the original traceback and fresh run.

| Test ID | Verdict | Evidence |
|---|---|---|
| `tests/cli/test_services_remote.py::TestLlmCliBase64::test_stale_cleanup_no_base64` | `environmental-artifact` | Original run hit the global 30 s timeout inside `subprocess.communicate`; unchanged test passed in 3.66 s in the fresh run. |
| `tests/embeddings/test_prompt_name.py::test_embed_texts_local_passes_prompt_name` | `environmental-artifact` | Original encoder construction raised `ModuleNotFoundError: torch`; it passed with the declared test extra and CPU torch installed. |
| `tests/embeddings/test_prompt_name.py::test_embed_texts_local_no_prompt_name_omitted` | `environmental-artifact` | Original encoder construction raised `ModuleNotFoundError: torch`; it passed with the declared test extra and CPU torch installed. |
| `tests/features/test_search.py::TestSearchFeatures::test_basic_search_functionality` | `environmental-artifact` | Original search returned `ToolError` solely after `ModuleNotFoundError: torch`; the same test passed in the provisioned fresh run. |
| `tests/features/test_search.py::TestSearchFeatures::test_filtered_search_by_physics_domain` | `environmental-artifact` | Original search returned `ToolError` solely after `ModuleNotFoundError: torch`; the same test passed in the provisioned fresh run. |
| `tests/features/test_search.py::TestSearchFeatures::test_search_result_quality` | `environmental-artifact` | Original search returned `ToolError` solely after `ModuleNotFoundError: torch`; the same test passed in the provisioned fresh run. |
| `tests/features/test_search.py::TestSearchFeatures::test_search_with_different_query_types` | `environmental-artifact` | Original search returned `ToolError` solely after `ModuleNotFoundError: torch`; the same test passed in the provisioned fresh run. |
| `tests/features/test_search.py::TestSearchFeatures::test_search_result_limits` | `environmental-artifact` | Original search returned `ToolError` solely after `ModuleNotFoundError: torch`; the same test passed in the provisioned fresh run. |
| `tests/features/test_search.py::TestSearchFeatures::test_search_performance_basic` | `environmental-artifact` | Original search returned `ToolError` solely after `ModuleNotFoundError: torch`; the same test passed in the provisioned fresh run. |
| `tests/features/test_search.py::TestSearchPathsResultStructure::test_search_result_consistency` | `environmental-artifact` | Original search returned `ToolError` solely after `ModuleNotFoundError: torch`; the same test passed in the provisioned fresh run. |
| `tests/features/test_search.py::TestSearchPathsResultStructure::test_search_metadata_completeness` | `environmental-artifact` | Original search returned `ToolError` solely after `ModuleNotFoundError: torch`; the same test passed in the provisioned fresh run. |
| `tests/features/test_search.py::TestSearchErrorHandling::test_search_very_long_query` | `environmental-artifact` | Original search returned `ToolError` solely after `ModuleNotFoundError: torch`; the same test passed in the provisioned fresh run. |
| `tests/features/test_search.py::TestSearchErrorHandling::test_search_special_characters` | `environmental-artifact` | Original search returned `ToolError` solely after `ModuleNotFoundError: torch`; the same test passed in the provisioned fresh run. |
| `tests/ids/test_mapping_e2e.py::TestPipelineOrchestrator::test_gather_context` | `environmental-artifact` | Original `gather_shared_context` failed while constructing `Encoder` because torch was absent; it passed after provisioning the test extra. |
| `tests/ids/test_mapping_e2e.py::TestGatherContextFields::test_context_has_enriched_source_fields` | `environmental-artifact` | Original `gather_shared_context` failed while constructing `Encoder` because torch was absent; it passed after provisioning the test extra. |
| `tests/integration/test_workflows.py::TestUserWorkflows::test_discovery_workflow` | `environmental-artifact` | Original search became `ToolError` after `ModuleNotFoundError: torch`; the workflow passed in the provisioned fresh run. |
| `tests/integration/test_workflows.py::TestUserWorkflows::test_research_workflow` | `environmental-artifact` | Original first search became `ToolError` after `ModuleNotFoundError: torch`; the workflow passed in the provisioned fresh run. |
| `tests/integration/test_workflows.py::TestUserWorkflows::test_comprehensive_exploration_workflow` | `environmental-artifact` | Original terminal assertion failed because search became `ToolError` after missing torch; the accepted cluster-tool `ToolError` was incidental, and the workflow passed once torch was present. |
| `tests/integration/test_workflows.py::TestWorkflowPerformance::test_workflow_total_time` | `environmental-artifact` | Original search became `ToolError` after `ModuleNotFoundError: torch`; the workflow passed in the provisioned fresh run. |
| `tests/integration/test_workflows.py::TestWorkflowPerformance::test_concurrent_tool_usage` | `environmental-artifact` | Original concurrent search became `ToolError` after `ModuleNotFoundError: torch`; both-result assertions passed in the provisioned fresh run. |
| `tests/integration/test_workflows.py::TestWorkflowErrorRecovery::test_workflow_continues_after_error` | `environmental-artifact` | Original nominal search became `ToolError` after `ModuleNotFoundError: torch`; it returned `SearchPathsResult` in the provisioned fresh run. |
| `tests/integration/test_workflows.py::TestWorkflowDataConsistency::test_search_consistency` | `environmental-artifact` | Original search became `ToolError` after `ModuleNotFoundError: torch`; it returned `SearchPathsResult` in the provisioned fresh run. |
| `tests/test_graph_marker_gate.py::test_missing_credential_explicit_graph_selection_exits_not_run` | `wrong-premise` | Original run reached a different ambient branch (“configured graph unavailable”); fresh run found the graph and returned 0 with one graph test passed, proving that deleting only `IMAS_CODEX_TEST_NEO4J_URI` does not establish the claimed credentialless premise. |
| `tests/test_imas_search_scoring.py::TestGraphSearchToolHybrid::test_hybrid_boost_dual_match` | `environmental-artifact` | Original hybrid search returned `ToolError` after `ModuleNotFoundError: torch`; the scoring assertion passed with the declared test extra installed. |
| `tests/test_imas_search_scoring.py::TestGraphSearchToolHybrid::test_text_only_match_included` | `environmental-artifact` | Original hybrid search returned `ToolError` after `ModuleNotFoundError: torch`; the scoring assertion passed with the declared test extra installed. |
| `tests/test_imas_search_scoring.py::TestGraphSearchToolHybrid::test_full_searchhit_metadata` | `environmental-artifact` | Original hybrid search returned `ToolError` after `ModuleNotFoundError: torch`; the metadata assertion passed with the declared test extra installed. |
| `tests/test_imas_search_scoring.py::TestGraphSearchToolHybrid::test_vector_query_includes_node_category_filter` | `environmental-artifact` | Original vector-search path failed after `ModuleNotFoundError: torch`; the query-filter assertion passed with the declared test extra installed. |
| `tests/test_imas_search_scoring.py::TestGraphSearchToolHybrid::test_no_results_returns_empty` | `environmental-artifact` | Original hybrid search returned `ToolError` after `ModuleNotFoundError: torch`; the empty-result assertion passed with the declared test extra installed. |
| `tests/test_imas_search_scoring.py::TestGraphSearchToolHybrid::test_ids_coverage_in_summary` | `environmental-artifact` | Original hybrid search returned `ToolError` after `ModuleNotFoundError: torch`; the coverage-summary assertion passed with the declared test extra installed. |

## Repair implication

The remaining failure should not be repaired by changing an environment-specific message. The test must make its missing-credential condition deterministic: isolate every credential/profile input used by `resolve_neo4j()` or mock the credential/probe boundary, then assert the explicit-selection behavior. The current test name says “missing credential,” but its setup allows both an unavailable configured graph and a reachable configured graph; the two measured runs exercised those two ambient outcomes.

The 28 environmental verdicts do not weaken the locked provisioning decision. They show that the product/test assertions are green when the declared extra is actually installed; CI and documented test entry points still need to guarantee that provisioned environment and fail once, loudly, when it is absent.
