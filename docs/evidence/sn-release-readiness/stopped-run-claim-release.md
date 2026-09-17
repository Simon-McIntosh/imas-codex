# A stopped run releases its claim without waiting for age

The exact-name scope preflight decided whether a claim was contention by
comparing its timestamp to the orphan-sweep cutoff (1800 s). The claim that
matters, however, is not how long ago a worker took a row — it is whether that
worker still exists. A claim whose owning run has stopped is abandoned the
instant the run stops, at any age; a claim whose run is still executing is held
at any age. Age is now the fallback for a claim whose owning run cannot be
resolved, rather than the primary rule.

## The decision

`_claim_disposition` in `imas_codex/standard_names/graph_ops.py` returns one of
three outcomes per preflight candidate:

| Claim state | Owning `SNRun.status` | Disposition | Effect |
|---|---|---|---|
| no `claimed_at`, no `claim_token` | — | absent | nothing to release |
| `claim_token` only, no `claimed_at` | any | held | refuses; the orphan sweep never clears a token with no timestamp either |
| `claimed_at` set | `started` | held | renders `live`, whatever the claim's age |
| `claimed_at` set | a status this reader does not recognise | held | fails closed: only a run known to have stopped releases |
| `claimed_at` set | `completed`, `interrupted`, `failed`, `degraded`, `stale` | released | reclaimable at any age |
| `claimed_at` set | run row absent, or no `run_id` | age rule | released when older than the sweep cutoff, held otherwise |

The stamp is unchanged in shape: the preflight read and the stamp share one
transaction, so the released set the preflight computed cannot go stale between
the two. The stamp admits a row carrying no claim, a claim older than the sweep
cutoff, or a claim the preflight released by name (`$released_ids`), and it
nulls `claimed_at`/`claim_token` while writing `run_id`.

`_EXACT_NAME_SCOPE_RUN_STATUS_QUERY` resolves the owning runs, and it is issued
only when a claim is present, so the read is bounded by the requested scope
rather than by the claim population. An unresolvable `run_id` — the model does
not need to distinguish "absent row" from "no run_id" — leaves the claim to the
age rule.

## Evidence

**Test:** `tests/standard_names/test_exact_name_claim_run_release.py` (16 tests).
Both halves are asserted: a fresh claim (inside the sweep window, so age alone
refused it) owned by a terminal run is reclaimed, and an old claim (outside the
window, so age alone released it) owned by a live run refuses. Each half fails
against the age-only rule independently.

**Passing run:**

```bash
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" \
  uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_exact_name_claim_run_release.py \
  tests/standard_names/test_exact_name_claim_age_preflight.py
```

Exit status `0` — `24 passed, 1 warning in 7.39s` (16 new, 8 from the
predecessor age-relative suite, unchanged and still green). Log:
`/tmp/node2-green.log`.

**Failing before the change.** The pre-change `graph_ops.py` (from `HEAD`) was
placed in a scratch package overlay
(`/tmp/prechange-node2/imas_codex/standard_names/graph_ops.py`, the rest of the
package symlinked to the worktree) and the same test file re-run against it
(`PYTHONPATH=/tmp/prechange-node2`). Exit status `1` — `13 failed, 2 passed`.
Log: `/tmp/node2-red.log`.

The failures split, and the split is stated because it bounds the claim:

- **3 behavioural failures** — the assertions this node exists to make:
  - `TestStoppedRunReleasesItsClaim::test_fresh_claim_held_by_a_stopped_run_is_reclaimed`
    → `ExactNameScopeConflict: held: current worker claim (live)` — a claim
    owned by a `completed` run refused on age alone.
  - `TestStoppedRunReleasesItsClaim::test_released_claim_is_admitted_by_the_stamp`
    → the same refusal, so the stamp was never reached.
  - `TestLiveRunHoldsItsClaim::test_old_claim_held_by_a_live_run_refuses_atomic`
    → `Failed: DID NOT RAISE` — an old claim owned by a `started` run passed the
    preflight, i.e. the old rule released a live run's claim.
- **10 signature failures** — `_exact_name_scope_refusals` gained its
  `run_statuses` parameter in this change, so those call sites raise
  `TypeError` against the old module. They are not behavioural evidence and are
  counted separately rather than folded into the 13.

## Live census

Taken on the login node through the login-local bolt tunnel (`NEO4J_URI` resolves
to `bolt://localhost:17687`, which a compute node cannot establish for itself);
every statement below is an aggregate over an indexed predicate, the run is
logged, and the whole pass is far below the ten-second ceiling per query.

```cypher
// A: the claim cohort this change governs
MATCH (sn:StandardName)
WHERE sn.claimed_at IS NOT NULL
OPTIONAL MATCH (run:SNRun {id: sn.run_id})
RETURN coalesce(run.status, '<no-run-row>') AS run_status,
       count(*) AS claims, count(sn.run_id) AS with_run_id
ORDER BY claims DESC
```

| Reading | Value |
|---|---|
| StandardName rows currently claimed | **0** |
| Claims currently held by a run that has stopped | **0** |
| Claims held by a live run | **0** |

So the live count this node was asked to report is zero, and it is zero *because
the claim population itself is empty right now* — no run is mid-flight. A zero
here therefore carries no information about whether the predicate works, and
reporting it as an answer for the specified control the fence asked for would be
the error this evidence is meant to avoid.

**Control, as specified, could not be produced.** The fence asked for a control
showing the same predicate returning non-zero over live-run claims. There are no
live-run claims to show it with, and none can be manufactured without writing
claims into the live graph, which is outside this node. Two substitutes, both
taken, both non-zero:

```cypher
// C: the resolution arm, exercised over rows that DO carry a run_id
MATCH (sn:StandardName)
WHERE sn.run_id IS NOT NULL
OPTIONAL MATCH (run:SNRun {id: sn.run_id})
RETURN coalesce(run.status, '<no-run-row>') AS run_status, count(*) AS names
ORDER BY names DESC
```

| `run_status` resolved for a name row's `run_id` | Names |
|---|---|
| `<no-run-row>` (unresolved — the age-fallback cohort) | 557 |
| `failed` | 9 |
| `stale` | 6 |
| `completed` | 4 |

19 name rows resolve to a terminal run; the join the change depends on returns
non-zero and discriminates, and 557 rows resolve to nothing at all, which is the
population the age fallback exists for. None of the 19 carries a claim right now
(`claimed_now = 0`), so the release arm would fire on no row today — the release
is triggered by a claim, and there are no claims; the change alters behaviour at
the next mid-run stop, not retroactively.

```cypher
// D: the run population the resolution reads
MATCH (run:SNRun) RETURN coalesce(run.status, '<null>') AS status,
       count(*) AS runs ORDER BY runs DESC
```

| `started` | `completed` | `stale` | `degraded` | `interrupted` | `failed` |
|---|---|---|---|---|---|
| 1 | 507 | 104 | 84 | 26 | 11 |

621 runs, 620 terminal, 1 `started`.

## What this does not fix, measured

The `started` run is `8c781962-6c62-42e6-a2b6-d6f25538390e`, started
`2026-09-15T19:19:27Z`, observed `2026-09-17T19:35:25Z` — **48 hours
in-progress with `last_heartbeat: null`**. It is a process that died without
finalising, and its row still says the run is executing.

Under this change a claim held by that run refuses at **any** age, where the age
rule would have released it after 1800 s. That is the trade the fence states
explicitly — "a claim whose owning run is still alive refuses regardless of
age — the second is the half that stops this from being a deletion of the check"
— and it is the correct direction, because the alternative releases work a live
run may still be doing. It does mean the un-finalised-run wedge is now handled
by the run-level finalisation path (`mark_orphaned_standard_name_runs_stale`,
which ages on `coalesce(last_heartbeat, created_at, started_at, stopped_at)`)
rather than by the claim rule, and that path had not run for this row in 48
hours. Nothing here reads `last_heartbeat`; a claim whose owning run is 48 hours
past its start with a null heartbeat is indistinguishable, to this reader, from
one being written now.

## Scope

`SNRun` claims do not natively carry the claiming run id: `claimed_at` and
`claim_token` are set by the phase claim paths, and the row's `run_id` is
whatever the scope stamp last wrote — the run that bound the row, which is the
run executing its phases. The census confirms the field is populated (576 of
5130 names carry one) and that 557 of those do not resolve to a run row. The
design's premise, "a claimed row carries `run_id`", therefore holds for rows
bound by a scope and not for a row claimed outside one, which is exactly the
population the age fallback is retained for.