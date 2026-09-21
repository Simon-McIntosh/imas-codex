# The graph-writing changes landed this sprint, reviewed statement by statement

Scope: exactly `git diff 3f8c06fcef82b8a769ee5ec5d81ae83b4fe3c5c5..HEAD --
imas_codex/standard_names/graph_ops.py` — **591 added lines, 51 removed, 19
hunks**, read only through `git diff` and targeted `sed -n` ranges. The file was
never opened whole.

**Result: one CONFIRMED defect, two PLAUSIBLE, and three hypotheses that the
live graph refuted.** No statement in this diff can delete or retarget an
existing binding without a guard, and every `MATCH` property was checked to
exist on its target label. The refuted hypotheses are recorded with their
measurements, because each was the exact defect shape this review was told to
hunt and the instrument, not the argument, is what settled them.

Live-graph verification ran on `all_debug`; probes and logs at
`~/.config/reckon/crew/runs/graphops-review-042900/probe{,2}.{py,log}`.

## Property existence, checked before any claim about a statement

Every `MATCH` and map projection added by this diff resolves. Measured:

| Property / pattern the diff relies on | Measured | Verdict |
|---|---|---|
| `SNRun.id` | 733 of 733 rows | present |
| `SNRun.status` | 733 of 733 rows | present |
| `(:StandardNameSource)-[:PRODUCED_NAME]->(:StandardName)` | 5,493 edges | correct direction |
| the reverse direction | **0 edges** | the diff does not use it |
| `IMASNode.node_category` | 61,366 rows | present |
| `IMASNode.lifecycle_status` | 22,179 rows | present, partial — see below |
| `IMASNode {id: sns.source_id}` resolves for the parked cohort | 234 of 234 | present |

The `IMASNode.lifecycle_status` column is populated on only 22,179 of 61,366
nodes, but `classify_parked_source` reads it with `or ""` and compares against
`"removed"`, so an absent value falls through to the next test rather than
short-circuiting. That is the correct direction for the absent-key shape.

## CONFIRMED — `produced` counts dead bindings, so a retired name reads as work already done

`_CAP_PARKED_QUERY` projects

```cypher
size([(sns)-[:PRODUCED_NAME]->(:StandardName) | 1]) AS produced
```

and `classify_parked_source` documents that field as **"the number of *live*
produced names"** and branches on it first:

```python
if int(evidence.get("produced") or 0) > 0:
    return "name_produced"     # "no composition is owed"
```

The pattern applies no stage filter, so a source whose only producer is
`superseded` or `exhausted` — a binding the pipeline itself retired — is
classified as already satisfied and drops out of triage permanently.

**This is the same sprint disagreeing with itself.** The claim guard added a few
hundred lines earlier in the same diff defines
`_SETTLED_SOURCE_BINDING_STAGES = {superseded, exhausted}` and filters exactly
these stages out of its own holder test, on the stated reasoning that a settled
binding does not hold the source. The classifier omits that filter.

**Concrete failure, measured live rather than constructed.** `dd:focs/spun`
carries `attempt_count >= 5`, `status = 'stale'`, and exactly one
`PRODUCED_NAME` edge to a name at stage `exhausted`. Its stored
`parked_disposition` is **`name_produced`**. The graph state is wrong in the way
that matters: the row asserts a name exists for this source when the only name
it ever had was retired, so a triage census reports it as needing nothing.

**Blast radius today is 1 of 234**, and the branch it hides behind is the widest
one in the classifier:

| disposition | rows |
|---|---|
| `name_produced` | 164 |
| `cause_not_recorded` | 30 |
| `compose_not_applicable` | 21 |
| `upstream_quantity_removed` | 18 |
| `vocabulary_gap` | 1 |
| `attempt_budget_exhausted` | 0 |

164 of 234 parked rows (70%) are decided by this one unfiltered test. The count
is 1 today because supersedes have not yet accumulated inside the parked cohort;
it rises monotonically with every supersede and every exhaust, and nothing
re-examines a row once it is classified.

**Repair** is one clause, matching the guard in the same diff:
`size([(sns)-[:PRODUCED_NAME]->(n:StandardName) WHERE NOT
coalesce(n.name_stage,'') IN ['superseded','exhausted'] | 1])`. Verified against
the live graph: `contested` producers in this cohort number **0**, so the
two-stage set is sufficient and matching `_SETTLED_SOURCE_BINDING_STAGES`
exactly is correct.

## PLAUSIBLE — the fail-closed claim above the fallback is documented as stronger than it is

`_TERMINAL_RUN_STATUSES` carries:

> Every other status — `started`, a status written by a future release, or **a
> run row that cannot be read at all** — leaves the claim held, so the reading
> here fails closed: only a run known to have stopped releases its claim.

The last clause does not hold. `_claim_disposition` ends:

```python
status = run_statuses.get(str(candidate.get("run_id") or ""))
if status is not None:
    return _CLAIM_RELEASED if status in _TERMINAL_RUN_STATUSES else _CLAIM_HELD
return _CLAIM_RELEASED if candidate.get("claim_stale", False) else _CLAIM_HELD
```

A run that cannot be read reaches the age fallback, which **does** release a
stale claim. That is the deliberate, correct design — it is stated accurately
three times elsewhere in the same change — but the sentence above says the
opposite, and it is the sentence a reader checking whether this guard fails
closed will find first. An `SNRun` node that exists with no `status` property
lands in the same branch, because `row.get("status")` stores `None` and the
`is not None` test cannot distinguish an unstored key from an absent run.

No live instance: all 733 `SNRun` rows carry a status. Recorded as a
documentation defect rather than a behaviour one.

## PLAUSIBLE — the disposition write reports its input, not its receipt

```python
if not dry_run and written:
    client.query(_CAP_DISPOSITION_WRITE, rows=written)
...
return {"parked": len(written), "written": 0 if dry_run else len(written), ...}
```

`_CAP_DISPOSITION_WRITE` ends `RETURN count(sns) AS written`, and that value is
discarded. The reported `written` is the length of the list handed in. A row
whose id no longer matches — the zero-row trap, since `MATCH
(sns:StandardNameSource {id: row.id})` on a missing id is not an error — is
reported as written when nothing was.

Mitigated, and deliberately: `census_parked_dispositions` re-reads the stored
property rather than the computed values, and its own docstring says so. So the
receipt exists; it is simply not the one this function returns. Severity is
low precisely because the independent readback is there.

## Three hypotheses the live graph refuted

Recorded because each was the defect shape this review was directed to find, and
in each case only the measurement settled it.

1. **`_EXACT_NAME_SCOPE_RUN_STATUS_QUERY` matching a property `SNRun` does not
   carry.** `MATCH (owner:SNRun {id: run_id})` would return zero rows for every
   run, `run_statuses` would always be empty, every claim would fall back to age,
   and the entire run-status mechanism would be a silent no-op that its own tests
   could not detect. **Refuted: 733 of 733 `SNRun` rows carry both `id` and
   `status`.**

2. **`_EXPLICIT_SOURCE_CLAIM_QUERY` traversing `PRODUCED_NAME` in the wrong
   direction.** `(sns)-[:PRODUCED_NAME]->(holder:StandardName)` with the edge
   stored the other way would make `holding` empty for every source, so
   `size(holding) = 0` would be universally true and every source would be
   claimed — a guard that reads as present and withholds nothing. **Refuted:
   5,493 edges in the written direction, 0 in the reverse.** The guard is also
   not inert: of the 37 `extracted` sources carrying any producer, **3 carry a
   non-settled one and are now withheld** where they were previously claimed and
   charged an attempt.

3. **`classify_parked_source` testing for a phrase the pipeline never writes.**
   `"vocabulary gap" in reason.lower()` looked wrong, because this project's
   vocabulary everywhere else is the token `vocab_gap` — a `StandardNameSource`
   status, a node id prefix, `vocab_gap_nonactionable`. **Refuted for this
   field:** across the parked cohort, `last_error` values containing `vocab_gap`
   number **0** and values containing `vocabulary gap` number **1**, and the one
   instance reads `'non-actionable vocabulary gap: β is the complementary second
   inclination angle …'`. The `vocab_gap`-token spelling lives on a different
   field. The substring matches what this field actually carries.

## Statements checked and found sound

- **`_EXACT_NAME_SCOPE_STAMP_QUERY`'s new `OR name.id IN $released_ids` clause.**
  It admits a *fresh* claim whose owning run is terminal, which is the point. The
  danger would be stamping a new `run_id` while leaving the dead run's claim
  fields behind: the row would then resolve the new, `started` run on every later
  read and be held forever. The `SET` clears both — `name.claimed_at = null,
  name.claim_token = null` — so the row returns to the unclaimed shape the first
  clause already admits. No retarget without a guard.
- **`_claim_disposition`'s absent-key handling.** `claimed_at is None` with a
  token present returns `_CLAIM_HELD`, so a token carrying no timestamp is never
  released — matching the orphan sweep, which never clears one either. The
  absent-key branch fails closed, which is the opposite of the shape this review
  was hunting.
- **`_TERMINAL_RUN_STATUSES` membership.** `{completed, interrupted, failed,
  degraded, stale}` is exactly `SNRunStatus` minus `started`, and the schema
  states "`started` is in-progress; others are terminal". `degraded` is a
  completed run with an undrained cost write, and `stale` is written by the
  orphan sweep for a dead process — both genuinely release. No live worker's
  claim is released by this set.
- **`_claimed_run_ids` bounding.** Collects only from candidates already in the
  preflight snapshot, so the follow-on read is bounded by the requested scope
  rather than by the claim population, as its docstring says.
- **`deletion_change_cypher("parent", param_prefix="reset_skeleton_")` in
  `clear_standard_names`.** The helper accepts `param_prefix`, validates it with
  `isidentifier()`, and namespaces all four parameters, so the two receipts in
  one statement do not collide. This refactor is a **strict improvement**: the
  literal `CREATE (:StandardNameChange {...})` it replaces recorded no edge
  inventory, and the helper captures `deleted_edge_inventory` before the
  `DETACH DELETE`.
- **`write_standard_names`' `ON CREATE SET sn.status = 'draft', sn.updated_at =
  datetime()`.** Additive; the following unconditional `SET sn.updated_at`
  overwrites it with the same value. Harmless, and it makes the created row
  well-formed if the later `SET` is ever narrowed.
- **`reconcile_docs_axis_from_reviews`.** The refusal is load-bearing and real: a
  name with no surviving docs review cannot reach the update because the winner
  subquery matches nothing for it, so absence never infers acceptance. The
  `dry_run` branch omits the disagreement filter deliberately so `agree` and
  `repair` are both countable, and the write branch applies it, which is what
  makes the pass idempotent.
- **`_docs_review_winner_query_body` gaining `score`.** Purely additive to two
  map projections; no predicate reads it, so no selection changes.

## One thing outside this node's scope, reported not triaged

`_CAP_PARKED_QUERY` and `_CAP_DISPOSITION_READBACK` both open `MATCH
(sns:StandardNameSource)` with a `coalesce(sns.attempt_count, 0) >= $cap`
filter — a whole-label scan with no index and no caller-side bound, against
~10,000 rows. It completed well inside the ten-second ceiling on `all_debug`, so
it is not a present problem, but it is an unbounded read in a file whose other
reads state their row set.
