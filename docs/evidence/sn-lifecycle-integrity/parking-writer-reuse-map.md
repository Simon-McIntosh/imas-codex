# The parking writer's disposition path — reuse map

Every line reference below was read at HEAD `9705954c264bd4c51c78d68d9be23be54178c2ac`
(orientation base).

## Outcome

**ALREADY-FIXED at HEAD** for the missing disposition in
`f-sli-the-parking-writer-records-a-disposition-itself`. The single live writer
that parks a source at the compose claim-attempt cap already records the
disposition in the same tick, the classifier exists once, and three tests drive
the park path. Nothing here needs re-implementing: a repair node dispatched to
this followup would re-implement a working guard. The residual work is a wiring
gap named under *What is still open* — the reconcile net is not called by the
pipeline — and that is a different node from "the writer records a disposition".

The manager of this path is reusable as-is; the parking statement, the
classifier and the sweep tick are the pieces a follow-on node would extend, not
replace.

## The followup's claim, and whether it still holds

The followup's wording, quoted from the plan:

> `orphan_sweep.py` parks a source with the bare cap sentence alone, so any row
> parked after that pass carries no disposition until someone runs it again.

At HEAD, the sweep's tick pairs the parking statement with the disposition
write, so the quoted behaviour does not reproduce. Landing commit:
`73242777f` — *"feat(standard_names): record a disposition as the parking writer
parks"* (2026-09-17); `git log -1 -S "record_dispositions_for_undisposed_parks"
-- imas_codex/standard_names/orphan_sweep.py` returns that commit.

The pairing at HEAD is `_orphan_sweep_tick`:

- `orphan_sweep.py:162` — `if counts.get(_PARKING_LABEL):`
- `orphan_sweep.py:163` — `recorded = record_dispositions_for_undisposed_parks(gc=gc)`

A tick that parks nothing issues neither the read nor the write, so the pairing
costs one query only on a tick that parks something.

## Every write site in `imas_codex/standard_names/orphan_sweep.py`

Read at HEAD; the file is 434 lines. `SET` clauses were located exhaustively
(grep for `SET ` in the file), and each one is placed in the table below, so
"every write site that parks a source" is answered against the whole file and
not against a filtered view. Only one of them parks.

| # | site (label / function) | Cypher `SET` at | what it writes | parks a source? |
|---|---|---|---|---|
| 1 | `_SWEEP_QUERIES[0]` `name_refining` | `orphan_sweep.py:45` | `name_stage` refining → reviewed, clears claim | no |
| 2 | `_SWEEP_QUERIES[1]` `docs_refining` | `orphan_sweep.py:59` | `docs_stage` refining → reviewed, clears claim | no |
| 3 | `_SWEEP_QUERIES[2]` `stale_token_sn` | `orphan_sweep.py:75` | clears stale name claim token | no |
| 4 | `_SWEEP_QUERIES[3]` `stale_token_source` | `orphan_sweep.py:88` | clears stale source claim token | no |
| 5 | `_SWEEP_QUERIES[4]` `_PARKING_LABEL` | **`orphan_sweep.py:114-118`** | `status='failed'`, `failed_at`, `last_error='compose claim-attempt cap reached'`, clears claim | **YES** |
| 5a | `_orphan_sweep_tick`, the paired write (`:162-168`) | `graph_ops.py:11566` (statement `_CAP_DISPOSITION_WRITE`, `:11563`) | `parked_disposition` on the rows site 5 just parked | no — it classifies rows the park created |
| 6 | `refresh_manifest_drain_scope` | `orphan_sweep.py:247`, `:252` | drain-scope lease timestamp | no |
| 7 | `recover_manifest_drain_scope` | `orphan_sweep.py:329`, `:364` | reverts stuck refining stages, clears stale claims, removes drain-scope properties | no |
| 8 | `run_manifest_drain_heartbeat_loop` | (delegates to site 6) | — | no |

The parking statement, in full as it stands at HEAD (`orphan_sweep.py:109-121`):

```cypher
MATCH (s:StandardNameSource)
WHERE s.status = 'extracted'
  AND coalesce(s.attempt_count, 0) >= $max_compose_attempts
SET s.status     = 'failed',
    s.failed_at  = datetime(),
    s.last_error = 'compose claim-attempt cap reached',
    s.claim_token = null,
    s.claimed_at  = null
RETURN count(*) AS n
```

The statement's own comment (`orphan_sweep.py:105-108`) records the pairing:
*"The tick follows this statement with `_CAP_UNDISPOSED_QUERY`, which stamps a
triage disposition on the rows it just parked — the reason above says only that
the budget ran out, and a cohort carrying only that string cannot be triaged
from it."*

The label is named once as `_PARKING_LABEL` (`orphan_sweep.py:30`) precisely so
the tick can pair the two writes by label rather than by re-spelling the
statement.

The shape of the paired path, before and after the write path landed, is drawn
in the existing figure
`docs/figures/attempt-cap-disposition/writer-path.svg` (publication path
`/imas-codex/figures/attempt-cap-disposition/writer-path.svg`); no new figure is
drawn here because this node's write fence carries no `docs/figures/` path.

## The classify-and-write path, as it exists at HEAD

| symbol | file:line | role | verdict |
|---|---|---|---|
| `_COMPOSE_CAP_REASON` | `graph_ops.py:11337` | the one reason string the park writes and the sweep's read is scoped by | reuse-as-is |
| `CAP_PARKED_DISPOSITIONS` | `graph_ops.py:11343` | the closed set (see below) | reuse-as-is |
| `classify_parked_source` | `graph_ops.py:11369` | the classifier; total by construction | reuse-as-is |
| `_CAP_PARKED_QUERY` | `graph_ops.py:11445` | cohort read for the reconcile pass | reuse-as-is |
| `record_parked_dispositions` | `graph_ops.py:11457` | the single classify-and-write step; refuses a row outside the set before any write | reuse-as-is |
| `disposition_parked_sources` | `graph_ops.py:11494` | the reconcile net over the whole cohort | extend — not wired (below) |
| `_CAP_UNDISPOSED_QUERY` | `graph_ops.py:11523` | the writer's own read: scoped by the cap reason **and** absence of the disposition, so it can only see rows a parking event produced | reuse-as-is |
| `record_dispositions_for_undisposed_parks` | `graph_ops.py:11537` | the sweep's call into the classify-and-write step | reuse-as-is |
| `_CAP_DISPOSITION_WRITE` | `graph_ops.py:11563` | `SET sns.parked_disposition = row.disposition` | reuse-as-is |
| `census_parked_dispositions` | `graph_ops.py:11579` | re-reads the stored property rather than this process's classification, so the census is a statement about the graph | reuse-as-is |
| `_MAX_COMPOSE_CLAIM_ATTEMPTS = 5` | `graph_ops.py:15914` | the cap, shared by the park statement and the cohort reads | reuse-as-is |

The rejected alternative is worth keeping in view, because it is the design the
map exists to prevent: expressing the precedence a second time as a Cypher
`CASE` inside the park statement would put the branch order in two languages,
and the one defect this beat already found — a category test outranking a
recorded cause — would be reachable again with no single place to fix it.
The precedence lives once, in `classify_parked_source`.

## The closed disposition set

`CAP_PARKED_DISPOSITIONS` (`graph_ops.py:11343`) declares **six** members, of
which **five** have members in the live cohort — the five the followup names:

| disposition | live rows (2026-09-17 census) | evidence it rests on |
|---|---|---|
| `name_produced` | 164 | at least one `PRODUCED_NAME` edge |
| `cause_not_recorded` | 30 | no reason, or only the cap sentence, and no structural fact |
| `compose_not_applicable` | 21 | backing `IMASNode` category geometry or coordinate |
| `upstream_quantity_removed` | 18 | backing `IMASNode` lifecycle_status removed |
| `vocabulary_gap` | 1 | the recorded reason names a vocabulary gap |
| `attempt_budget_exhausted` | 0 (declared, reachable, empty) | spent budget with a recorded non-vocabulary cause — the class a revival would act on |

The classifier symbol that produces them is **`classify_parked_source` at
`imas_codex/standard_names/graph_ops.py:11369`**. Its branch order, which is the
mechanism the map should carry forward, is: produced name → upstream removed →
recorded reason (vocabulary gap / spent budget) → node category → silence.
A recorded cause outranks the node category and the category outranks silence;
that ordering is why `dd:iron_core/segment/geometry/oblique/beta` is classified
`vocabulary_gap` rather than `compose_not_applicable`.

`attempt_budget_exhausted` is a member with zero rows, deliberately: it is the
class a revival acts on, and it is reachable rather than missing. A reader who
counts "five classes" and a reader who reads the declaration's six are both
right about different things, and the difference is that one class exists for a
population that is empty today.

## Verification at HEAD

Receipt: `UV_PROJECT_ENVIRONMENT=<main-checkout>/.venv PYTHONPATH=$PWD
UV_NO_SYNC=1 uv run --no-sync pytest -p no:cacheprovider
tests/standard_names/test_attempt_cap_disposition.py
tests/standard_names/test_orphan_sweep.py`

`21 passed, 12 deselected, 1 warning in 9.82s`, exit 0 — 8 tests in
`test_attempt_cap_disposition.py`, 1 fails when the classification is removed;
13 in `test_orphan_sweep.py`. Log:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260918T090405201486-n-sli-scout-parking-writer/focused-tests.log`.
This receipt measures this node's own read of a landed path; it is not the
merged-result verification, which belongs to a separately dispatched test node.

The three tests that drive the sweep's own park path are, from the prior
record and re-read in the test file at HEAD:

- `test_parking_writer_stamps_a_disposition_as_it_parks` — fails against the
  writer as it was, which issued no disposition write at all, so it measures the
  refusal and the write rather than the classifier alone;
- `test_parking_writer_writes_nothing_when_it_parks_nothing` — asserts the read
  and the write are both skipped on a tick that parks nothing;
- `test_parking_writer_refuses_a_row_it_cannot_place`.

## What is still open (found while reading, outside this node's write scope)

1. **The reconcile net is never called by the pipeline.** `disposition_parked_sources`
   (`graph_ops.py:11494`) and `census_parked_dispositions` (`graph_ops.py:11579`)
   have no callers in `imas_codex/` — grep for `record_parked_dispositions|disposition_parked_sources|census_parked_dispositions` returns the definitions
   and their uses inside `graph_ops.py` plus three calls in
   `tests/standard_names/test_attempt_cap_disposition.py`, and nothing else. So
   the writer's own read (scoped to the cap reason) is the only automatic heal,
   and a row parked by any route writing a *different* reason is outside it.
   The followup's measure — "a repeat census after a later parking event returns
   no unclassified rows without a pass being run" — therefore holds for a
   sweep-parked row and only for a sweep-parked row.
2. **`parked_disposition` is never cleared.** Grep over `imas_codex/` finds
   exactly one write (`graph_ops.py:11566`) and two reads (`:11527`, `:11574`).
   A source revived through the sanctioned route — `retry_failed_sources`
   (`graph_ops.py:11616`) — keeps its old disposition, and
   `_CAP_UNDISPOSED_QUERY` requires `parked_disposition IS NULL`, so a
   re-parked row carrying a stale value would not be restamped. No live row is
   wrong today (the class is a property of a parking event and the revive path
   is operator-driven), but the round trip park → revive → park has no
   disposition clearing step.
3. **`mark_sources_failed` (`graph_ops.py:11398`) writes `status='failed'` at
   its own `max_attempts` and writes no disposition, but it is not a live
   parking route.** Grep for the symbol over `imas_codex/` returns the
   definition and no caller; its only callers are in
   `tests/standard_names/test_standard_name_source.py:414` and
   `tests/standard_names/test_failed_source_reason.py:182`. It is named here
   because a reader scanning for parking writers will find it: the shape looks
   like a second route, and the reason it is not one is measured rather than
   asserted.

## The exclusive write-path set

**At the code level, one chain owns the property, and it is exclusive today.**
The disposition value has one producer and one writer:

- produced by `classify_parked_source` (`graph_ops.py:11369`) — the only place
  the branch order exists;
- written by `_CAP_DISPOSITION_WRITE` (`graph_ops.py:11563`), issued only from
  `record_parked_dispositions` (`graph_ops.py:11457`), which refuses a value
  outside the closed set before writing;
- reached on the live path only by the sweep tick (`orphan_sweep.py:162-168`)
  through `record_dispositions_for_undisposed_parks` (`graph_ops.py:11537`);
- read back only via `census_parked_dispositions` (`graph_ops.py:11579`).

So the exclusive write path is: park (`orphan_sweep.py:109-121`) → tick pairing
(`:162-168`) → `record_dispositions_for_undisposed_parks` →
`record_parked_dispositions` → `classify_parked_source` → one `SET`. A future
parking route is expected to join that chain, exactly as the plan's §6 comment
argued a repair should: `update on the first run, no-op on every run after`. It
must not write `parked_disposition` itself, and must not re-*express* the
precedence as Cypher — that is the one change to this path that has already
produced a wrong classification once. (The rejected alternative was a Cypher
`CASE` inside the park statement, which would have put the branch order in two
languages and made that defect reachable again.)

**Proposed file-level exclusive write-path set for a node taking the residual
work.** This is a proposal, not a decision: the coordinator owns fencing.

| file | why | measured constraint |
|---|---|---|
| `imas_codex/standard_names/graph_ops.py` | the classifier, the classify-and-write step and the two cohort reads live here; interpolating a disposition clear on the revive path (`retry_failed_sources`, `:11616`) lands here too | 27,234 lines / 1,139,822 bytes — a declared write path of this size is a dispatch-window hazard: probe the dispatch with `--dry_run` before committing to it, and if the window refuses, put the new helper in a smaller module and fence accordingly |
| `imas_codex/standard_names/orphan_sweep.py` | only if the writer's own read scope changes; any edit here must preserve the label-based pairing (`_PARKING_LABEL`, `:30`) rather than re-spelled re-derivation of the same rule | 434 lines |
| `tests/standard_names/test_attempt_cap_disposition.py` | the gate for classification, refusal and the writer's park path (8 tests at HEAD) | — |
| `tests/standard_names/test_orphan_sweep.py` | the tick-level tests, including `test_parking_writer_writes_nothing_when_it_parks_nothing` (13 tests at HEAD) | — |

If the residual work is only the *wiring* of the reconcile net, the minimal set
is the module that owns the `sn run` maintenance sequence plus the two test
files — and that set cannot be named from here, because the maintenance
sequence's call site was not read in this node and this node does not write
outside its fence.