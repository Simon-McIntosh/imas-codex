# Graph test tier green

Recorded 2026-09-18 against implementation commit `ac35b9455f15ea81771604c0590d08d3be0745bc`.

## Outcome

The same `tests/graph` command was run on `all_debug` before and after the repair. The baseline reproduced the recorded defect exactly; the committed result has no failures.

| Revision | Selected | Passed | Failed | Deselected | Duration |
|---|---:|---:|---:|---:|---:|
| `e9a4e7c0035aa64358a7f32318710af0b715ddf2` | 521 | 517 | 4 | 355 | 43.01 s |
| `ac35b9455f15ea81771604c0590d08d3be0745bc` | 522 | 522 | 0 | 355 | 38.13 s |

The pass count rises by five because all four failing tests now pass and the digest-mismatch regression adds one selected test. The focused gate over the property audit, grammar synchronization, timestamp coverage, and generated-model currency was independently green at **22 passed**.

## Causes and repairs

| Failing test | Cause established before editing | Repair |
|---|---|---|
| `test_repository_cypher_literals_have_declared_properties` | `ISNGrammarVersion.content_digest` was used at two Cypher sites but absent from LinkML; `StandardNameSource.parked_disposition` was used at one write and two reads but absent from LinkML; five retired `StandardName.quarantine_reason` occurrences remained registered as runtime exceptions after their source strings disappeared. | Declared both durable fields, typed `parked_disposition` with the exact six-member code vocabulary, and removed only the five obsolete registry occurrences while leaving the audit armed. |
| `test_auto_sync_skips_when_in_sync` | Commit `ff540084d3` changed freshness from version equality to content-digest equality, but the in-sync fixture still returned only a version. It therefore exercised the deliberately stale missing-digest branch and called synchronization. | The fixture supplies `grammar_content_digest()` for the in-sync case. Missing and mismatched digests separately assert that synchronization still fires. |
| `test_standard_name_package_matches_updated_at_debt_baseline` | The scanner found five identities while the named baseline carried two: the real `graph_ops.py` defect plus unrecorded provenance/coordination writes in `publish.py` and `workers.py`. | Repaired `graph_ops.py`; named the two legitimate surviving identities and their mechanisms. The measured post-repair baseline is exactly four: catalog import receipt, drain lease heartbeat, export receipt, and unspent-refinement accounting. |
| `test_graph_ops_standard_name_writes_have_hard_zero` | `ON CREATE SET sn.status = 'draft'` modified a substantive `StandardName` property without stamping `sn.updated_at` in that same clause. | Appended `sn.updated_at = datetime()` to the create clause. The hard-zero audit now reports no `graph_ops.py` finding. |

## The five retired adjudications

The runtime registry contained these exact source locations:

1. `campaign.py:302` — `StandardName.quarantine_reason`
2. `campaign.py:665` — `StandardName.quarantine_reason`
3. `campaign.py:712` — `StandardName.quarantine_reason`
4. `graph_ops.py:23584` — `StandardName.quarantine_reason`
5. `signed_manifest.py:1786` — `StandardName.quarantine_reason`

An exact-boundary `rg` for `quarantine_reason` returned exit 1 independently for `campaign.py`, `graph_ops.py`, and `signed_manifest.py`. The same invocation shape over the same files using `validation_status` as a positive control returned exit 0 in all three, with 10, 77, and 8 matches respectively. The absence is therefore a statement about the retired property, not an instrument that failed to see the files. No other registry row changed.

## Parked disposition is declared without claiming a complete lifecycle

The schema enum and `CAP_PARKED_DISPOSITIONS` compare equal at six members:

- `name_produced`
- `upstream_quantity_removed`
- `compose_not_applicable`
- `vocabulary_gap`
- `attempt_budget_exhausted`
- `cause_not_recorded`

The peer-provided live census reports five populated values and an empty `attempt_budget_exhausted` bucket; the empty member is retained because the closed set comes from the classifier, not from current data. The field description also records the current limitation: it is written once, a non-null value is skipped by later classification, and no implemented path clears or recomputes it. The clear-path defect remains owned by the lifecycle-integrity work rather than being hidden by this declaration.

## Timestamp debt after repair

The post-repair audit returns exactly four named findings:

| File | Properties | Why no content timestamp is written |
|---|---|---|
| `catalog_import.py` | `catalog_commit_sha`, `imported_at` | Catalog provenance receipt under a constrained import authority. |
| `orphan_sweep.py` | `drain_scope_claimed_at` | Lease heartbeat, not name-content modification. |
| `publish.py` | `exported_at` | Delivery receipt applied from the successfully published manifest. |
| `workers.py` | `refine_attempts` | Returns a charged rotation when an unchanged pinned rename consumed no improvement attempt. |

## Reproduction and verification

- Baseline log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260918T162636270064-n-the-graph-test-tier-is-green-on-main/graph-tier-baseline.log`
- Focused log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260918T162636270064-n-the-graph-test-tier-is-green-on-main/graph-tier-focused.log`
- After log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260918T162636270064-n-the-graph-test-tier-is-green-on-main/graph-tier-after.log`
- Command shape: `srun --partition=all_debug ... uv run --no-sync pytest -p no:cacheprovider tests/graph`
- Generated models were rebuilt inside the worktree before verification and remained gitignored.

No test was widened, disabled, skipped, or marked expected-to-fail.
