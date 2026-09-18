# A binding conflict is adjudicable, not a hard failure

`retarget_standard_name_sources` compares and sets a source's `PRODUCED_NAME`
binding. Its preflight already refused every ineligible row, but it refused
them all the same way: an anonymous `RuntimeError`. A source that already binds
a *live* identity other than the migration's predecessor is not a fault at all
— it is an ordinary state on a supersede path, and repeating the migration
cannot change it. Reported as a bare fault, it is indistinguishable from a
transient one, so a caller that retries on failure spends a retry budget on a
question only an operator can answer.

## What changed

`retarget_standard_name_sources` now classifies the refusal instead of
flattening it:

| Preflight state | Outcome | Why |
|---|---|---|
| Source binds a live identity other than the predecessor | `SourceBindingConflictError`, carrying one `SourceBindingConflict` per holder: `source_id`, `holding_identity`, `holding_stage`, `source_status` | The operator must decide which spelling carries the quantity; the source does not change between attempts |
| Concurrent claim, stale row, lost binding, scalar mismatch, partially applied manifest | `RuntimeError` (unchanged) | Transient or malformed — a retry can legitimately succeed |
| Source bound only to a retired sibling | stays `pending` (unchanged) | A retired binding is a prior generation of a surviving name, not a competing claim |

`SourceBindingConflictError` derives from `RuntimeError`, so callers that do not
yet distinguish the branches keep working unchanged; the type is the signal, not
the inheritance. The conflict is decided in the preflight, so the refusing run
issues exactly one statement — the read — and no mutation.

## Evidence

The two branches were driven on one source row, `dd:example/predicted_path`,
carrying `old_name` at stage `drafted` alongside a live `live_holder` at stage
`reviewed`, against the migration code as committed at `a15768122` and against
the worktree:

| Branch | Outcome |
|---|---|
| HEAD | `RuntimeError`, no named payload: `source migration compare-and-set failed: dd:example/predicted_path(exists=True, status='attached', claimed=False, bindings=['live_holder', 'old_name'], scalar='old_name')` |
| Worktree | `SourceBindingConflictError` naming `live_holder` at stage `reviewed`; one statement issued before the refusal |

### The retry budget

The measure the node exists for is that the conflict does not consume an
attempt. A passing suite never shows a budget being spent, so the budget is
carried by a fake source that records every write it receives and whose counter
is proved to move when an attempt is charged:

| Measure | Value |
|---|---|
| `attempt_count` before the conflict | 3 |
| `attempt_count` after the conflict | 3 |
| Write statements issued by the refusing run | 0 |
| Control — same instrument, one charged attempt | 5 → 6 |

A zero would have proved nothing on its own; the control shows the counter is a
live integer, so an unmoved 3 is a measurement rather than a constant.

### Gate

```
uv run --no-sync pytest tests/standard_names/test_source_migration_conflict.py \
                        tests/standard_names/test_live_binding_migration.py
15 passed in 8.70s, exit 0
```

Log: `logs/gate.log`. The adjacent module is the pre-existing coverage for the
same preflight and is unchanged: its two live-binding cases still raise, now
through the named type, which is what keeps the existing contract.

The wider `tests/standard_names` suite was also run: **7453 passed, 8 skipped,
1 failed**, the failure being
`test_error_siblings.py::test_reconcile_orphans_error_siblings`, which asserts a
literal Cypher string owned by `signed_manifest.py` / `graph_ops.py` — files
this change does not touch.

## Not in this node

The attempt is actually charged at **claim** time, not at the migration:
`claim_explicit_standard_name_sources` writes `attempt_count = coalesce(...) + 1`
in `graph_ops.py`, outside this node's write scope. Withholding the charge
belongs with that writer; this node supplies the named outcome a caller uses to
distinguish the two, and the live population still owes a governed reset.