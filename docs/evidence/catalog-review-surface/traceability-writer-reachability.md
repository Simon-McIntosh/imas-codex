# The catalog traceability writer: receipt completeness and reachability

## Result

`mark_catalog_name_approved` (`imas_codex/standard_names/promote.py:1627`) writes
**every field a reviewer receipt needs, in one update on one identity**, and it
refuses the write when its declared stage preconditions are unmet. A production
path reaches it: the `sn approve` verb calls `run_approval`, which calls the
writer twice. The writer has never fired against the live graph, and the one
identity carrying a reviewer actor was populated by a different writer.

## The receipt the writer sets, and its gate.

The writer declares its stage preconditions in the same statement that writes the
receipt, so the write and the gate cannot disagree:

```text
promote.py:1649-1652   WHERE sn.name_stage IN ['accepted', 'approved']
                             AND sn.docs_stage = 'accepted'
                             AND coalesce(sn.status, 'draft') = 'draft'
                             AND coalesce(sn.validation_status, 'valid') <> 'quarantined'
promote.py:1653-1661   SET   sn.name_stage = 'approved',
                             sn.docs_stage = 'accepted'
                             sn.status = 'active',
                             sn.catalog_pr_number = $pr_number,
                             sn.catalog_pr_url = $pr_url,
                             sn.catalog_merge_commit_sha = $merge_commit,
                             sn.catalog_reviewer_actor = $reviewer_actor,
                             sn.catalog_approved_at = coalesce(sn.catalog_approved_at, datetime()),
                             sn.updated_at = datetime()
```

Five receipt fields, one `SET`, one `MATCH`: there is no ordering in which a
reader can see a request number without its URL, or an actor without its request.
`promote.py:1638-1639` additionally raises `ValueError` on an incomplete merged
request tuple before the statement is issued.

The actor is not inferred from `edit_origin`. `run_approval` fills it from
`_RESOLVED_PR_ACTORS` (`promote.py:1142-1143`), which is keyed by the same
(number, URL, merge commit) tuple that the writer stamps, so the actor and the
request number arrive from one authoritative reading rather than two.

## Every call site, and whether production reaches it.

| call site | caller | editorial outcome |
| --- | --- | --- |
| `imas_codex/standard_names/promote.py:1306` | `run_approval`, accepted content edit | `content_edit` |
| `imas_codex/standard_names/promote.py:1388` | `run_approval`, untouched batch identities | `unchanged_ratification` |

**Plainly: yes, a production path calls it.** `imas_codex/cli/sn.py:5262` calls
`run_approval` from `sn_approve`, the `sn approve` verb declared at
`imas_codex/cli/sn.py:5101`. Those two interior lines are the writer's only call
sites anywhere in the tree; the call sites under `tests/standard_names/` patch or
drive it, and none of them is production.

The reachability has one silent branch worth stating, because it bounds what a
"receipt fell out of an existing path" claim can mean. `approval_values` at
`promote.py:1122-1126` holds the request number, URL and merge commit only, and
the writer is skipped when all three are absent — an `sn approve` invocation with
no `--pr` metadata. In that case the accepted name is reported accepted with no
catalog receipt at all. The actor is deliberately not part of that guard: it may
be `None`, in which case the writer still runs and stamps a **null** reviewer
actor alongside a real request number.

## The actor field has a second, non-receipt writer.

`catalog_reviewer_actor` is written by one other path, and it is the path that
produces the shape the plan recorded:

- `promote.py:908`, inside `_contest` — reached from `run_approval`'s contested
  branch (`promote.py:1341`) and from `review_triage.py:365-374`
  (`route_triage`), the latter passing `catalog_pr_number=None`.

So a non-null actor with a null request number is not this writer's product: the
receipt writer cannot produce it, because it stamps the actor and the request
number in the same `SET`.

## Live graph measurement, 2026-09-17.

One bounded read over the whole `StandardName` label, on the login node because
the graph tunnel is login-local; it completed inside the ten-second ceiling.

| graph field | populated rows of 5,130 |
| --- | ---: |
| `catalog_reviewer_actor` | **1** |
| `catalog_pr_number` | **0** |
| `catalog_approved_at` | **0** |
| `catalog_merge_commit_sha` | **0** |
| `exported_at` (publication receipt, not this writer) | 221 |

The single actor-bearing identity is `net_power_due_to_ion_cyclotron_heating`,
actor `Simon McIntosh`, request number **null**, approval time **null**. It is not
this writer's product, and the read-back discriminates the mechanism: the row
retains `contested_reason` and `contested_at` (`2026-09-08T12:51:51Z`) from a
contest, and its `catalog_approved_at` is null although the receipt writer always
sets it (`promote.py:1660`). The row's contested state was resolved by a path that
returns an identity to an accepted stage without clearing the actor, so the actor
outlives the contest that wrote it.

**A null request number on all 5,130 rows is the load-bearing figure: the receipt
writer is reachable in production and has never run against the live graph.**

## The gate.

`tests/standard_names/test_catalog_traceability_fields.py` now holds two tests
that together assert both halves of the writer's contract, so neither can regress
without failing:

- `test_the_receipt_tuple_is_written_together_on_one_identity` — an accepted,
  accepted-docs identity is approved, and all five receipt fields are asserted
  present together.
- `test_the_write_is_refused_when_a_stage_precondition_is_unmet` — a `drafted` or
  `contested` name stage, and a `drafted` or `contested` docs stage, each return
  `False` with **no** receipt field written.

The graph double the new tests use (`_StageGuardedGraph`) refuses exactly where
the `WHERE` clause refuses, which is the property the pre-existing double could
not show: that double applied any write it was handed, so a stage gate that
stopped gating would have kept passing.

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" \
  uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_catalog_traceability_fields.py \
  tests/standard_names/test_sn_approve.py \
  tests/standard_names/test_sn_approve_preflight.py \
  tests/standard_names/test_promote_change_row.py \
  tests/standard_names/test_promote_origin_guard.py \
  tests/standard_names/test_promote_reviewer_actor.py \
  tests/standard_names/test_frozen_stage_pool_guard.py \
  tests/standard_names/test_cli_approve.py \
  tests/standard_names/test_provenance_lifecycle.py
99 passed, 2 deselected, 1 warning in 25.60s   exit 0
```

## What this makes measurable that was not.

The who-changed-it receipt is now gated by a test, its writer's stage
preconditions are gated separately, and its reachability is stated rather than
assumed: the writer is reachable from `sn approve`, it sets the complete tuple or
nothing, and no live run has yet produced a readable receipt.