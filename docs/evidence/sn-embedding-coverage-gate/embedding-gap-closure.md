# Accepted-corpus embedding gap closure

## Outcome

The single accepted identity with a real description and a null embedding,
`ratio_of_toroidal_ion_velocity_to_magnetic_field_magnitude`, was re-embedded
through the existing dedicated embed-worker path:
`claim_embed_batch` → `process_embed_batch` → `persist_embed_batch`.
`process_embed_batch` persisted exactly **1** identity. It generated no
description, changed no threshold, and embedded none of the 26 non-accepted
description-gap rows.

Accepted-corpus description-embedding coverage changed from **2,301 of 2,302**
to **2,302 of 2,302**:

| Observation | Accepted identities | Accepted with schema key `id` | Accepted with description | Accepted with embedding |
|---|---:|---:|---:|---:|
| Before | 2,302 | 2,302 | 2,302 | **2,301** |
| After | 2,302 | 2,302 | 2,302 | **2,302** |

The graph-wide schema sanity census proves these are measurements over the
declared properties rather than plausible zeros from a wrong key. Before the
write, the graph held **4,658 `StandardName` candidates, 4,658 with `id`, 4,632
with `description`, 4,630 with `embedding`, and 4,656 with `embedded_at`**.
After the write, candidate, key, and description counts were unchanged while
embedding and timestamp coverage each increased by exactly one: **4,631 with
`embedding` and 4,657 with `embedded_at`**. The instrument is aimed at
`StandardName.id`, `StandardName.description`, and `StandardName.embedding`.

The target retained `name_stage=accepted`. Its description was unchanged
before and after, exactly:

> Signed ratio of the charge-state-averaged toroidal ion bulk velocity to the local magnetic-field magnitude, retaining the toroidal flow contribution.

Its vector state changed from null to present, `embed_text_hash` changed from
null to `a2a3e5468e339791`, and `embedded_at` was set to
`2026-08-25T16:35:31.263000000+00:00`.

## Sanctioned path and cost

The claim was fenced by the target's existing run identifier. The sanctioned
claim returned seven eligible identities in that run; the six non-target claims
were released immediately and were not processed. The one target item was
passed to `process_embed_batch`, which called the existing local embedding
service and then `persist_embed_batch`. The worker emitted one `embed_name`
event for the target and reported one persisted row.

Provider cost was exactly **USD 0.000000**. The embed worker does not call an
LLM or charge the budget. As an independent graph check, the target's associated
`LLMCost` population remained unchanged at six rows and USD 0.789556 in total;
the before/after deltas were **0 rows** and **USD 0.000000**.

No embedding vector was invented or written through hand-authored Cypher. No
description was generated or modified. No gate threshold changed.

## Named graph check

The named check was executed against the live graph:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH="$PWD" uv run --no-sync pytest -m graph \
  'tests/graph/test_data_quality.py::TestDescriptionEmbeddingCoverage::test_description_embedding_coverage[StandardName]'
```

It still reports **26** `StandardName` rows with `embedded_at` present and a
null vector, and therefore fails its exact-zero assertion. This is the expected
separate signal established by the prior census: all 26 are non-accepted rows
that lack descriptions, whereas the repaired accepted identity previously had
a real description and no `embedded_at`. This node did not reconcile those 26
rows toward a pass.

## Evidence inputs

- Re-embedding record, including before/after censuses, unchanged description,
  claim release counts, persisted count, and cost delta:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T163230354600-n-reembedgap/reembed.log`
- Named graph-check result:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T163230354600-n-reembedgap/named-check-graph.log`
- Initial deselected invocation showing the graph marker requirement:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T163230354600-n-reembedgap/named-check.log`
