# Embedding preflight stamping repair

## Outcome

`run_embedding_preflight` now persists only candidates for which the embedding
call produced a non-null vector. The same filtered batch controls both
`embedded_at` stamping and `EmbeddingReport.refreshed_count`, so the report
counts vectors actually written rather than candidates considered.

The producer previously appended every missing-embedding row to `needs_embed`,
including rows with a null description. It then persisted
`item.get("embedding")` even when that remained null, stamped `embedded_at`
unconditionally, and reported `len(needs_embed)` as refreshed. That mechanism
accounts for the measured 26 rows carrying `embedded_at` with neither a
description nor an embedding.

This repair does not generate descriptions, embed graph data, or alter any live
row. In particular, it does not clear `embedded_at` on those 26 existing rows;
their retrospective disposition remains a separate data-repair decision.

## Regression proof

The regression uses two candidates and a controlled embedding function:

- `missing_description` has no description and receives no vector.
- `described_name` has a description and receives one vector.

The fake graph client models the query's timestamp side effect, proving the
test observes the persisted batch rather than merely matching query text.
Against the unguarded producer, the null-description candidate was present in
that batch and the test failed at the timestamp assertion. Verbatim summary:

```text
1 failed, 1 warning in 26.60s
```

With the persistence guard in place, exactly one vector is written,
`refreshed_count == 1`, and the null-description candidate remains unstamped.
Verbatim summary:

```text
1 passed, 1 warning in 15.00s
```

The first attempted counterfactual run did not reach the test because the
repository's autouse setup exceeded its default 30-second timeout while
importing `litellm` from the shared environment. Its distinct harness summary
was `1 warning, 1 error in 50.10s`. Adding the repository-supported per-test
timeout override allowed the subsequent counterfactual and repaired runs to
exercise the producer behavior.

## Affected-suite validation

Commands used the main checkout's shared environment with `--no-sync` and
disabled the pytest cache in this detached worktree.

```text
pytest -p no:cacheprovider tests/standard_names/test_embedding_preflight_stamping.py
1 passed, 1 warning in 15.00s

pytest -p no:cacheprovider tests/standard_names/test_review_pipeline.py
22 passed, 1 warning in 34.11s
```

Scoped Ruff check and format validation also passed; Ruff changed neither
source nor test after the implementation edit.

Full logs:

- `logs/counterfactual-before-fix.log` — setup-timeout attempt.
- `logs/counterfactual-before-fix-reached.log` — honest pre-repair failure.
- `logs/regression-after-fix.log` — repaired regression.
- `logs/review-pipeline-suite.log` — existing affected suite.

The log paths are relative to the node run envelope at
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T180951950774-n-embedstamp/`.
