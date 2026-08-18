# Verified typed DD resolution consumer coverage

Only boundaries with a test assertion on the typed seam are claimed here.

| Consumer | Public entry point | Asserting test | Typed assertion |
|---|---|---|---|
| Bulk DD extraction | `sources.dd.extract_dd_candidates` | `test_public_extractors_return_typed_active_and_pass_through_rows` | Active rows expose effective unit, raw unit, resolution ids, and marker; pass-through rows retain raw/effective equality and an empty applied-id set. |
| Targeted DD extraction | `sources.dd.extract_specific_paths` | `test_public_extractors_return_typed_active_and_pass_through_rows` | The targeted public boundary makes the same raw/effective and marker assertions as bulk extraction. |
| Extraction candidate query | `graph_ops.get_extraction_candidates_dd` | `test_extraction_candidates_preserve_graph_row_when_authority_is_empty` | Every original graph value survives and the additive raw context plus typed marker are asserted. |
| Review enrichment | `review.pipeline.enrich_review_worker` | `test_review_entry_point_propagates_authority_errors`; `test_review_entry_point_propagates_compose_context_authority_error` | The public worker is asserted to raise `DDResolutionError` for both typed-resolution routes rather than falling back. |
| Documentation refinement | `workers.process_refine_docs_batch` | `test_refine_docs_entry_point_propagates_authority_errors` | The public worker is asserted to propagate malformed and absent authority before model work. |
| Semantic source snapshot | `source_authority.authority_snapshot` | `test_semantic_authority_snapshot_uses_effective_unit` | Effective unit, exact raw unit, applied ids, authority digest, and marker are asserted together. |
| Source refresh and drift | `source_refresh.refresh_drifted_sources` | `test_public_refresh_restamps_raw_convergence_without_steering` | A raw-value and applied-to-converged identity transition with unchanged effective value is detected, re-stamped, and asserted not to steer documentation. |
| Raw release-fact loading | `dd_gaps.load_raw_unit_release_facts` | `test_effective_context_is_rejected_as_raw_release_evidence` | List, path-keyed object, and direct root-object effective projections are asserted to fail closed on the typed marker. |

## Deferred claims

The earlier table's worker re-injection, compose rendering, generated-document
context, drain-manifest, attachment-audit, attachment-integrity, and release-note
rows are not claimed here. Their cited tests exercise nearby behavior but do not
assert the typed DD marker, raw/effective values, resolution identities, or
authority-error propagation through the advertised public boundary. They require
dedicated public-boundary tests before they can be restored to this coverage
table.
