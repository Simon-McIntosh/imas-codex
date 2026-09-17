# The approval unwind restores every property the approval wrote

Measured 2026-09-17 at base `3baece092dca04a8ef0b8ce6704f8de2ca4b82bb`.

## The named test

`tests/standard_names/test_approval_unwind_restores_approval_properties.py:76`,
`test_undo_approval_restores_every_property_the_approval_wrote`.

## Symbols called, with file and line

| Symbol | Location |
|---|---|
| `mark_catalog_name_approved` | `imas_codex/standard_names/promote.py:1627` |
| `undo_approval` | `imas_codex/standard_names/promote.py:1549` |
| `GraphClient` | `imas_codex/graph/client.py:119` |
| `GraphClient.session` | `imas_codex/graph/client.py:210` |

Both production functions run their Cypher unchanged. The test passes a `gc`
whose `query(cypher, **params)` runs on the transaction it owns.

## Substrate

The live graph, not an isolated substrate: the running Neo4j behind the
login-local bolt tunnel, one explicit transaction, rolled back and never
committed. Every statement executes against a real Cypher engine, so a `SET`
clause dropped from the production text makes the test fail. Because nothing
committed, the demonstration costs no graph state.

## What is measured

1. Inside the transaction, create `unwind_proof_<12 hex>`, stage `accepted`,
   status `draft`; snapshot `properties(sn)` as the before map.
2. `mark_catalog_name_approved` returns true; all five catalog fields set; the
   identity reads `approved` / `active`.
3. `undo_approval` reports the identity demoted; all five fields read null.
4. The after map equals the before map, and the difference-set check proves the
   only property that differs is `updated_at`, the stamp both writes re-stamp.

## Result

Exit 0, 1 passed in 14.18s. The five catalog fields — `catalog_approved_at`,
`catalog_pr_number`, `catalog_pr_url`, `catalog_merge_commit_sha`,
`catalog_reviewer_actor` — are all null after the unwind.

## Negative control

The same function compares the after map with `catalog_pr_number` left set and
asserts the maps differ, so a comparison that can never fail fails the run.

## Residual the test does not assert away

The approval creates a `StandardNameChange` row (`internal: true`,
`origin='catalog_promotion'`). The unwind is a property-level revert and leaves
that audit row in graph history.

## Cost to the graph, read after the run

| Query | Result |
|---|---|
| `StandardName` ids starting `unwind_proof_` | 0 |
| `StandardNameChange` origin `catalog_promotion` naming PR 4242 | 0 |

Both checks come from an instrument proved to see present rows: 5,130 `StandardName` rows and 1,143 `origin='catalog_promotion'` change rows exist, and the first control shows the population the absence was measured against.

## Command

```
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv \
  PYTHONPATH=<worktree> uv run --no-sync --directory <worktree> \
  pytest <worktree>/tests/standard_names/test_approval_unwind_restores_approval_properties.py \
  -m graph -p no:cacheprovider
```

## Log

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260917T155320155643-n-the-approval-unwind-leaves-no-lasting-change/unwind-test.log`