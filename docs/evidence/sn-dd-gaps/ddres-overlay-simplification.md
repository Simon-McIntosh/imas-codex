# DD resolution graph-value cutover evidence

## Identity precondition

The read-only live-graph probe in
`/tmp/ddres-overlay-simplification-identity.log` inspected every
`IMASNode-[:BRIDGED_BY]->DDResolution` binding before the code change. It found
37 bridged paths, 37 graph unit scalars equal to the resolution effective value,
37 `HAS_UNIT` relationships targeting that same effective value, and zero
mismatches. No graph write was performed.

## Runtime contract

`resolve_dd_context` now treats the incoming DD row as an already-effective
graph snapshot. An exact bridge may attach provenance only when that graph value
equals the resolution's effective value; a published or third value at this
boundary raises `DDResolutionStale`. The context retains all prior serialized
provenance fields and additionally exposes:

- `graph`: the direct graph context used for effective fields;
- `published`: the context reconstructed from each bridge's published value;
- `resolution_provenance`: the typed bridge records, including upstream
  references and the plain who, when, and why trail;
- `published_dd_context`: the additive pipeline serialization of the published
  context.

The field-level resolver remains available for release-fact comparison, where a
published release value is intentionally compared with an effective bridge.
Attachment validation now calls the provenance-only graph-field reader. That
reader never substitutes a value: it returns the graph value when it matches the
effective bridge and raises `DDResolutionStale` when an active exact bridge sees
the published or a third value.

Exact-version bridge identities are reported as applied and not converged.
Prior-version bridge identities are reported as converged and not applied when
their effective value matches the graph. Source refresh persists the bridge's
published context in the raw provenance fields, while its effective snapshot
continues to carry the direct graph value.

## Simplified consumer sites

Thirteen context sites inherit direct graph-value semantics through the one
typed row boundary. The fourteenth, attachment validation, validates its direct
graph unit through the non-substituting field boundary. Every custom projection
now retains `published_dd_context` beside the effective graph context.

| Site | Location | Final contract | Covering regression |
|---|---|---|---|
| Bulk DD extraction | `sources/dd.py:322` | Direct row values plus published bridge provenance | `test_consumer_boundaries_call_typed_authority` |
| Explicit-path seed | `pool_adapter.py:205` | Direct row values plus published bridge provenance | `test_graph_context_reads_effective_value_and_reports_published_provenance` and pool-adapter public tests |
| Source refresh | `source_refresh.py:65` | Effective graph snapshot plus persisted published context and split identity sets | `test_source_refresh_persists_published_bridge_provenance` and `test_public_refresh_restamps_raw_convergence_without_steering` |
| Source authority snapshot | `source_authority.py:383` | Direct row values plus published bridge provenance | source-authority public snapshot tests and complete suite |
| Extraction candidates | `graph_ops.py:692` | Additive direct rows plus published bridge provenance | extraction-candidate public tests and complete suite |
| Manifest drain plan | `graph_ops.py:1101` | Direct row values plus published bridge provenance | manifest-drain public tests |
| Source snapshot pinning | `graph_ops.py:9504` | Direct row values plus published bridge provenance | focus-reseed and source-pinning public tests |
| Review context | `review/pipeline.py:699` | Direct row values plus published context and applied/converged identity sets | prompt-context public tests |
| Batch enrichment | `workers.py:2197` | Direct row values plus published bridge provenance | worker enrichment public tests |
| Attachment validation | `workers.py:2701` | Direct graph unit validated without substitution; stale active rows refuse | `test_attachment_refuses_published_value_on_active_graph_bridge` |
| Name-review member context | `workers.py:7281` | Direct row values plus published bridge provenance | name-review context public tests |
| Name-review parent context | `workers.py:7333` | Direct row values plus published bridge provenance | parent-context public tests |
| Documentation context | `workers.py:8677` | Direct row values plus published context and applied/converged identity sets | documentation-context public tests |
| Refine-documentation context | `workers.py:10210` | Direct row values plus published bridge provenance | refine-documentation public tests |

Release evidence now lists each bridge's id, path, field, published and effective
typed values, upstream reference, and retiring release beside the authority
digest and record count.

## Verification

- Initial red proof: the public-boundary regression failed because the context
  had no distinct graph and published surfaces.
- Corrective public regressions: four passed, covering stale attachment,
  published source-refresh persistence, exact applied identity, and prior-version
  converged identity semantics.
- Corrective focused verification: 102 passed, 7 deselected, and zero failed.
- Credentialed graph consumer probe: 6 passed, 5 deselected, and zero failed.
- Complete credential-less Standard Names suite: 6,508 passed, 8 skipped, 198
  deselected, and zero failed in 180.78 seconds.
