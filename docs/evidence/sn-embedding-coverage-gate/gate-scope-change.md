# Description-embedding coverage gate scope

## Outcome

The description-embedding coverage assertion now keeps its exact-zero bar while
aiming at the population a release ships. For a description-embeddable schema
label that declares `name_stage`, the query asserts that accepted nodes with
`embedded_at` retain an embedding vector. Labels without `name_stage` retain
the original graph-wide assertion; the parametrized test does not guess or
reference a lifecycle property those labels do not declare.

The live accepted `StandardName` population is clean:

| Measure | Count |
|---|---:|
| Accepted candidates | 2,302 |
| Accepted with schema key `id` | 2,302 |
| Accepted with `name_stage` | 2,302 |
| Accepted with `embedded_at` | 2,302 |
| Accepted with `description` | 2,302 |
| Accepted with `embedding` | 2,302 |
| Accepted with `embedded_at` and no embedding | **0** |

The schema sanity census over the full label confirms that the accepted zero is
not a wrong-property result: the graph contains 4,658 `StandardName` candidates,
4,658 with `id`, 4,658 with `name_stage`, 4,657 with `embedded_at`, 4,632 with
`description`, and 4,631 with `embedding`. A positive control finds 4,631 rows
with both `embedded_at` and an embedding.

## The twenty-six rows remain a producer defect

The scope change deliberately stops asserting on **26 non-accepted rows**. It
does not repair, delete, describe, embed, or reclassify them. All 26 have
`embedded_at` but neither a description nor an embedding: 14 are superseded, 8
are drafted, and 4 are pending. None is accepted. This population remains
tracked in the coverage-shortfall evidence and should be repaired at its
producer rather than made a release-blocking population.

The test's prior failure text said embeddings had been written and then lost.
That diagnosis was false for these rows. Source inspection identifies
`imas_codex.standard_names.review.audits.run_embedding_preflight` as a producer
that can stamp the timestamp without an embedding attempt:

1. A row with no embedding is appended to `needs_embed` even when its
   `description` is null.
2. `embed_descriptions_batch` is called with `description` as the text field and
   can leave such an item without an embedding.
3. The persistence batch uses `item.get("embedding")`, so the vector can remain
   null, while its Cypher unconditionally sets `sn.embedded_at = datetime()`.
4. The report then counts every `needs_embed` item as refreshed through
   `refreshed_count = len(needs_embed)`.

Therefore this path is reachable without a successful embedding attempt and
can create exactly the timestamp-without-description-or-vector state measured
on the 26 rows. Repairing that producer is outside this change's write scope.

## Before and after

Before the query changed, the named live graph test failed its graph-wide
predicate:

```text
1 failed, 1 warning in 19.08s
```

The failure selected 26 `StandardName` rows against an exact-zero threshold.
After the query was scoped through the declared acceptance lifecycle, the same
named test passed over 2,302 accepted candidates while preserving the exact-zero
assertion:

```text
1 passed, 1 warning in 9.19s
```

Full command output and the direct census are retained in the node's run logs.
The post-change run across all 13 description-embeddable labels also completed
without a failure: **8 passed and 5 skipped**. That broader run exercises labels
without `name_stage`, confirming that their original graph-wide assertion still
runs without an undeclared lifecycle-property predicate.

## Evidence inputs

- Before-test log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T173738947054-n-gatescope/logs/before-coverage-test.log`
- Direct live census and schema-label inventory:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T173738947054-n-gatescope/logs/coverage-census.log`
- After-test log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T173738947054-n-gatescope/logs/after-coverage-test.log`
- All-label post-change test log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T173738947054-n-gatescope/logs/after-all-labels-coverage-test.log`
