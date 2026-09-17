# The cap cohort carries a triage disposition

Sources parked at the compose claim-attempt cap used to carry one error string,
so a source that had already produced a name and a
source that can never be named read identically. Every parked row now carries a
disposition drawn from a closed set, and a census reports the count per class
with no residue.

Measured against the live graph on 2026-09-17 with the pass and the census in
`imas_codex/standard_names/graph_ops.py`.

## The cohort

| quantity | count |
|---|---|
| rows parked at `attempt_count >= 5` | 234 |
| of those, at exactly 5 | 217 |
| carrying the sweep string `compose claim-attempt cap reached` | 192 |
| carrying no reason at all | 41 |
| carrying any other reason | 1 |

The single row with a substantive reason is
`dd:iron_core/segment/geometry/oblique/beta`, whose recorded cause is a
non-actionable vocabulary gap.

## The closed set

`CAP_PARKED_DISPOSITIONS` declares six dispositions. Every row at or past the cap
lands in exactly one.

| disposition | rows | evidence it rests on |
|---|---|---|
| `name_produced` | 164 | at least one `PRODUCED_NAME` edge, so the cap bounds further claims and not the outcome |
| `cause_not_recorded` | 30 | no reason, or only the cap sentence, and no structural fact either, so the row cannot be triaged until the write path records one |
| `compose_not_applicable` | 21 | the backing `IMASNode` has `node_category` geometry or coordinate |
| `upstream_quantity_removed` | 18 | the backing `IMASNode` has `lifecycle_status` removed |
| `vocabulary_gap` | 1 | the recorded reason names a vocabulary gap, which no retry produces. |
| `attempt_budget_exhausted` | 0 | a spent budget with a recorded non-vocabulary cause: the class is reachable and empty today, and it is the class a revival would act on |
| **unclassified** | **0** | |

A recorded cause outranks the node category, and the category outranks silence.
The category branch exists to sort the rows whose cause was lost, so a specific
reason always decides wherever one is recorded. That precedence is what moved
`dd:iron_core/segment/geometry/oblique/beta` from `compose_not_applicable` on the
first pass to `vocabulary_gap` on the re-run: it records a specific
cause that the category could not have supplied.

## The two operations

The pass reads the cohort and the evidence the classifier consumes in one
indexed statement:

```cypher
MATCH (sns:StandardNameSource)
WHERE coalesce(sns.attempt_count, 0) >= $cap
OPTIONAL MATCH (node:IMASNode {id: sns.source_id})
RETURN sns.id AS id, sns.last_error AS last_error,
       node.lifecycle_status AS lifecycle_status,
       size([(sns)-[:PRODUCED_NAME]->(:StandardName) | 1]) AS produced,
       node.node_category AS node_category
ORDER BY sns.id
```

The read-back is by the ordered set:

```cypher
MATCH (sns:StandardNameSource)
WHERE coalesce(sns.attempt_count, 0) >= $cap
RETURN sns.id AS id, sns.parked_disposition AS disposition
ORDER BY sns.id
```

## The writer records it as it parks

The operation that parks a source now writes the disposition for it, so the
cohort does not depend on a later pass. `record_parked_dispositions` is the one
classify-and-write step; the reconcile pass and the sweep tick both call it, so
the precedence above exists once. When the sweep's park statement parks at least
one row, the tick follows it with a read scoped by the reason the writer itself
writes and by the absence of the disposition:

```cypher
MATCH (sns:StandardNameSource)
WHERE sns.status = 'failed'
  AND sns.last_error = $cap_reason
  AND sns.parked_disposition IS NULL
OPTIONAL MATCH (node:IMASNode {id: sns.source_id})
RETURN sns.id AS id, sns.last_error AS last_error,
       node.lifecycle_status AS lifecycle_status,
       size([(sns)-[:PRODUCED_NAME]->(:StandardName) | 1]) AS produced,
       node.node_category AS node_category
ORDER BY sns.id
```

The tick stamps every row that read returns. A tick that parks nothing issues
neither the read nor a write, so the pairing costs one query only on a tick that
parks something.

![The parking writer's disposition path, before and after the change](/imas-codex/figures/attempt-cap-disposition/writer-path.svg)

### The live check

Measured against the live graph on 2026-09-17. One row
(`dd:camera_x_rays/camera/camera_dimensions`) was picked out of the cohort and
its disposition removed, which is what a freshly parked row looks like. The
writer's own read then returned exactly that row with the evidence it holds —
the cap sentence as its reason, no lifecycle removal, `produced` 0 and
`node_category` quantity — the stamp path wrote one row, and the property read
back off the graph as `cause_not_recorded`. The census before and after the
strip are identical: 234 parked, 0 unclassified, 0 unexpected, distribution
unchanged at `name_produced` 164, `cause_not_recorded` 30,
`compose_not_applicable` 21, `upstream_quantity_removed` 18, `vocabulary_gap` 1.

The strip is the instrument: with the property removed, the census reports the
row unclassified and the writer's own read sees it. The function exercised live
is the same one the tick calls with the same predicate; the tick's wiring of it
is what the stubbed tests cover, and no live parking event was synthesized to
drive the tick end to end.

## The refusal

`disposition_parked_sources` raises before writing anything when a row has no
place in the closed set, and `census_parked_dispositions` reads the stored
property back rather than the values this process just computed, so the census
is a statement about the write and not about the classifier that produced it.

Eight tests live in `tests/standard_names/test_attempt_cap_disposition.py`. The
one that fails when the classification is removed is
`test_pass_refuses_a_row_the_classifier_cannot_place`: it replaces
`classify_parked_source` with a classifier that returns the empty string and
asserts the pass raises with no write at all. Removing the classification fails
the test instead of parking an unclassified row, and
`test_each_class_is_a_member_and_is_reached` fails if the closed set and the
cases it asserts ever disagree.

Three tests drive the sweep's own park path:
`test_parking_writer_stamps_a_disposition_as_it_parks` parks two rows with
different evidence and asserts each arrives with a member of the closed set
classified from its own row, `test_parking_writer_writes_nothing_when_it_parks_nothing`
asserts the read and the write are both skipped on a tick that parks nothing,
and `test_parking_writer_refuses_a_row_it_cannot_place` asserts the tick raises
with no write when a row has no place in the set. The first fails against the
writer as it was, which issued no disposition write at all, so it measures the
new write path and not the classifier alone.

## What this does not establish

The disposition states the row's triage class and nothing more: no pass here
raises the cap, revives a source or deletes a row, and no row was revived. The
30 `cause_not_recorded` rows are the standing gap that remains: 25 carry only
the cap sentence and 5 carry nothing at all, so no evidence the row itself holds
can place them further, and the only repair is a write path that records a cause
where one exists. `attempt_budget_exhausted` is the one class with no members in
this cohort, so no parked source is a revival candidate on the evidence stored.

The refusal in the writer's path is unreachable today, because the classifier is
total over the evidence shape the park statement produces. If it ever fired,
the row would already be parked and the failing tick would leave it undisposed
until the reconcile pass runs; the sweep loop logs the exception and continues.

## Artifacts

| item | value |
|---|---|
| code | `imas_codex/standard_names/graph_ops.py` and `imas_codex/standard_names/orphan_sweep.py` |
| tests | `tests/standard_names/test_attempt_cap_disposition.py`, 8 passed; `tests/standard_names/test_orphan_sweep.py`, 13 passed |
| writer path figure | `docs/figures/attempt-cap-disposition/writer-path.svg` |
| pass and census output | run directory `scratch/disposition2.log` |
| live writer check | run directory `scratch/writer_proof.log` |
| reason breakdown | run directory `scratch/breakdown.log` |
