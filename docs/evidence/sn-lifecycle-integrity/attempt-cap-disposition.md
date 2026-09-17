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

## The refusal

`disposition_parked_sources` raises before writing anything when a row has no
place in the closed set, and `census_parked_dispositions` reads the stored
property back rather than the values this process just computed, so the census
is a statement about the write and not about the classifier that produced it.

Five tests live in `tests/standard_names/test_attempt_cap_disposition.py`. The
one that fails when the classification is removed is
`test_pass_refuses_a_row_the_classifier_cannot_place`: it replaces
`classify_parked_source` with a classifier that returns the empty string and
asserts the pass raises with no write at all. Removing the classification fails
the test instead of parking an unclassified row, and
`test_each_class_is_a_member_and_is_reached` fails if the closed set and the
cases it asserts ever disagree.

## What this does not establish

The pass classifies the cohort as found and nothing more: it does not raise the
cap, revive a source or delete a row. `orphan_sweep.py` still writes the bare
sentence when it parks a source, and that file is outside this node's scope, so
a row parked after this run carries no disposition until the pass runs again.
The 30 `cause_not_recorded` rows measure that gap today: 25 carry only the cap
sentence and 5 carry nothing at all, and neither can be triaged further from
evidence the row itself holds. `attempt_budget_exhausted` is the one class with
no members in this cohort, so no parked source is a revival candidate on the
evidence stored, and no row was revived.

## Artifacts

| item | value |
|---|---|
| code | `imas_codex/standard_names/graph_ops.py` at commit ffa0c9359 |
| tests | `tests/standard_names/test_attempt_cap_disposition.py`, 5 passed |
| pass and census output | run directory `scratch/disposition2.log` |
| reason breakdown | run directory `scratch/breakdown.log` |
