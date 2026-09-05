# Canonical rendering, cohort rows 1–50

Attempted 2026-09-05, worktree base `85581a231b70a4457e686974de74d0a0d9c1236f`.
Specification: the live-cohort rename map observed 2026-09-05T07:32:34Z against
consumer revision `044070eea` and grammar `59754b7d6` — 149 rows, of which this
record covers rows 1 through 50. The map was read, not regenerated; no spelling
was re-derived.

## Outcome: 1 of 50 rows applied, then stopped on the time fence

The sanctioned route `imas-codex sn edit OLD --rename NEW --reason REASON` is
*stage plus inline review in one command*. Staging — the rename itself — is
fast. The inline review that follows it is not: it runs the reviewer rotation
against the successor and is the dominant cost of the command.

| Measure | Value |
|---|---:|
| Rows in this node's slice | 50 |
| Rows applied (rename written) | 1 |
| Rows refused | 0 |
| Rows skipped (time fence) | 49 |
| Wall clock consumed by row 1 alone | 8 m 20 s, and it had not finished |

Row 1 was run with a 500-second command timeout. The rename landed inside the
first few seconds; the inline review was still running when the timeout killed
the command. Extrapolating the observed floor of 8 m 20 s per row, the slice
needs at least 7 hours against a 55-minute fence, so rows 2–50 were never
started rather than started and abandoned.

## What row 1 actually left in the graph

The rename is a real, provenance-carrying migration and not a string rewrite:

| Fact | Observed value |
|---|---|
| Predecessor `accumulated_carbon_count_due_to_gas_injection` | `name_stage=superseded`, `origin=catalog_edit` |
| Successor `carbon_count_accumulated_due_to_gas_injection` | `name_stage=drafted`, `origin=pipeline` |
| Ledger row | `operation=human_edit`, `changed_at=2026-09-05T13:54:02.737Z`, `from_name=accumulated_carbon_count_due_to_gas_injection`, `to_name=carbon_count_accumulated_due_to_gas_injection` |
| Source migration | `operation=source_migration_manifest`, same from/to pair, `changed_at=2026-09-05T13:54:02.255Z` |

The successor sits at `drafted` rather than `accepted` because its review was
truncated. That is a recoverable position, not a corrupt one: a later batch
review over the open edits scores it exactly as the inline review would have.
Nothing else in the slice was touched, so rows 2–50 are byte-identical to the
map's "old spelling" column and the map remains a valid work list for them.

## The measure this run could not produce

The done-when asks for fifty applied rows and a before/after
`tests/standard_names` comparison with zero added failures. The row count is
one, not fifty. The suite was measured at the base revision in this session and
is reported here rather than inherited:

| Suite measurement | Value |
|---|---|
| Command | `uv run --no-sync pytest tests/standard_names -q -p no:cacheprovider` |
| Revision | `85581a231` |
| Exit status | 1 |
| Failures | 6, all pre-existing |
| Wall clock | about 33 minutes, not the ~90 seconds the repository notes quote |

The six are four in `test_docs_review_eligibility.py`, one golden-render
assertion in `test_edit_prompt_injection.py`, and one fresh-process import-order
case in `test_pool_registry_imports.py`. None is attributable to this node: the
only tracked change here is this document, which no test in that suite imports
or reads, so added failures are zero by construction. A second run for the
after-measurement did not fit what remained of the budget once the baseline had
consumed thirty-three minutes of it.

## What the next attempt needs

The blocker is a budget-versus-route mismatch, not a defect in the route. The
CLI documents the shape that fits a bulk migration — stage every row, then
review the staged set once — and the per-row inline review is what does not
fit. Choosing between "same route, a fence sized to seven hours" and
"stage-then-batch-review" is a scope decision this node was not authorized to
make, so it is referred upward rather than taken.

A resumable position is on disk: the per-row ledger records row 1 and its real
outcome, so a continuation starts at row 2 without re-reading the graph.
