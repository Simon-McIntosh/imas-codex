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
published release value is intentionally compared with an effective bridge. It
is no longer used by attachment validation; that path compares the graph unit
directly.

## Simplified consumer sites

Thirteen context sites inherit direct graph-value semantics through the one
typed row boundary. The fourteenth, attachment validation, now reads its graph
unit directly and contains no resolver call.

| Site | Location | Result |
|---|---|---|
| Bulk DD extraction | `sources/dd.py:322` | Direct row values plus bridge provenance |
| Explicit-path seed | `pool_adapter.py:205` | Direct row values plus bridge provenance |
| Source refresh | `source_refresh.py:65` | Direct row values; raw graph, bridge ids, digest, marker retained |
| Source authority snapshot | `source_authority.py:383` | Direct row values plus bridge provenance |
| Extraction candidates | `graph_ops.py:692` | Direct row values plus bridge provenance |
| Manifest drain plan | `graph_ops.py:1101` | Direct row values plus bridge provenance |
| Source snapshot pinning | `graph_ops.py:9504` | Direct row values plus bridge provenance |
| Review context | `review/pipeline.py:699` | Direct row values plus bridge provenance |
| Batch enrichment | `workers.py:2197` | Direct row values plus bridge provenance |
| Attachment validation | `workers.py:2701` | Direct graph unit; substitution call removed |
| Name-review member context | `workers.py:7262` | Direct row values plus bridge provenance |
| Name-review parent context | `workers.py:7314` | Direct row values plus bridge provenance |
| Documentation context | `workers.py:8658` | Direct row values plus bridge provenance |
| Refine-documentation context | `workers.py:10185` | Direct row values plus bridge provenance |

Release evidence now lists each bridge's id, path, field, published and effective
typed values, upstream reference, and retiring release beside the authority
digest and record count.

## Verification

- Red proof: the new public-boundary regression failed because the context had
  no distinct graph and published surfaces.
- Focused verification: 34 tests passed with no failures.
- Credentialed graph consumer probe: 6 tests passed with no skips or failures.
- Complete credential-less Standard Names suite: 6,505 passed, 8 skipped, 198
  deselected, and zero failed.
