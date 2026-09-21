# Local-lane graph-write review

provisional: false

Diff under review: `3f8c06fcef82b8a769ee5ec5d81ae83b4fe3c5c5..af0fb6e07da1bb909f03f647b021466f83ec9b96`,
restricted to six files. Read with `git diff` per file; `graph_ops.py` was never opened
(it exceeds the context window) and every reference to it below comes from a `grep -n`
with bounded context, never a whole-file read.

| File | Insertions | Deletions |
| --- | --- | --- |
| `imas_codex/graph/client.py` | 65 | 1 |
| `imas_codex/standard_names/orphan_sweep.py` | 28 | 5 |
| `imas_codex/standard_names/promote.py` | 96 | 0 |
| `imas_codex/standard_names/provenance_lifecycle.py` | 108 | 6 |
| `imas_codex/standard_names/review/audits.py` | 45 | 2 |
| `imas_codex/standard_names/signed_manifest.py` | 151 | 3 |

Total 493 insertions, 17 deletions.

## Findings

### 1 — CONFIRMED — a transport failure in the new PR notice aborts the catalog correction

`imas_codex/standard_names/promote.py:1524` (`notify_refusal_on_pull_request`), reached
from `run_approval` at `promote.py:1414`.

The function's own docstring states that "a rejected write [is] reported rather than
raised: the fold-back itself has already committed its decisions, and losing them to a
transport failure would be the worse outcome". It delivers that only for a *rejected*
write — a non-2xx status, which `_github_request` returns rather than raises. It does not
deliver it for a *failed* one. The only `except` clause is the `except ValueError` at line 1541, around
`parse_pull_request_url`. Everything below that line can raise:

- `imas_codex/graph/ghcr.py:453-459` — `resolve_api_token` (the raise is line 456) raises `GitHubRestError` when
  neither `GITHUB_TOKEN` nor `GH_TOKEN` is set and the local credential store is not
  authenticated.
- `imas_codex/graph/ghcr.py:469-475` — `_github_request` (the sentence is line 473)'s docstring says outright that
  "only a transport-level failure escapes". `urllib.error.URLError`, a DNS failure and a
  socket timeout all escape; only `HTTPError` is converted to a status.

Failure scenario: `run_approval` is invoked non-dry-run on a merged catalog PR in an
environment with no GitHub token (a compute step, or a shell that did not export one).
The fold-back has already written its accepted and refused decisions to the graph. One
outcome carries `promotion_refused`, so `refusal_notice_body` returns a body and
`_pull_request_comment` is reached. `resolve_api_token` raises `GitHubRestError`, which
propagates out of `notify_refusal_on_pull_request`, out of `run_approval`, and past
`promote.py:1415-1424` — so `_commit_catalog_correction` never runs. Resulting state: the
graph holds the refusals, the catalog still holds the unapproved additions that the
correction commit exists to remove, the reviewer is told nothing, and the caller sees a
GitHub-token error rather than a fold-back result. The one thing the notice was added to
prevent — the catalog and the graph disagreeing silently — is now reachable through the
notice itself.

Two independent repairs, either sufficient: wrap the call site in the exception classes
`_github_request` admits it lets through, or move the `if not dry_run:` notice to after
the `catalog_delta` block so a notice failure cannot preempt a graph-consistency write.

### 2 — CONFIRMED — the reconstruction receipt silently drops a role the archive census does not name

`imas_codex/standard_names/signed_manifest.py:7212` (`_archive_role_outcomes`), the comprehension at line 7234.

```python
reinstated_roles = {role: live[role] for role in archived if role in live}
```

The comprehension iterates `archived` — the archive record's census — and keeps only the
roles that census names. A role that the reconstruction edges genuinely put back, and
that the live read at `signed_manifest.py:7414` genuinely observed, is absent from the
receipt's `reinstated` map whenever the archive record for that identity omits it.

Failure scenario: an archive record supplies `archive_roles` for identity `A` naming only
`HAS_DOCS_REVIEW_ADMISSION`, while the manifest's reconstruction edges for `A` also carry
`HAS_REVIEW_ADMISSION`. `_load_archive_role_counts` admits this — it refuses only the
opposite direction, `reconstructable < count`, refused at line 6946 — and the parity guard at line
7417 passes, because `expected` there is built from the closure and both roles came back.
The receipt then reports `identity_roles[A]["reinstated"] == {"HAS_DOCS_REVIEW_ADMISSION": 1}`.
A reader auditing the restore from the receipt concludes one role came back when two did.
This is the shape the comment at `signed_manifest.py:242-248` says the widened role set
exists to prevent, applied in the other direction: the receipt is silent about what it
restored rather than about what it lost.

Note the guard itself is *not* weakened: the `any(...)` rewrite at line 7417 iterates
`expected.items()`, whose keys are still exactly `_ARCHIVE_EDGE_COUNTERPARTS`, so it
compares the same role set the pre-change `!=` compared. The defect is confined to the
receipt.

### 3 — CONFIRMED — the parity table and the parity guard disagree about `expected`

`imas_codex/standard_names/signed_manifest.py:7247` (`_archive_role_parity`), the `expected` expression at line 7272, against the
guard at `signed_manifest.py:7410-7419`.

The guard's `expected` for a role is the count the *reconstruction closure* carries. The
receipt's `expected` is `archived.get(role, closure.get(role, 0))` — the archive census
where one exists, the closure otherwise. `_load_archive_role_counts` refuses only
`reconstructable < count`, so `reconstructable > count` is an admitted state.

Failure scenario: an archive record declares `{"A": {"HAS_REVIEW_ADMISSION": 1}}` while
the manifest's edges carry two `HAS_REVIEW_ADMISSION` rows for `A`. The load admits it
(2 >= 1). The guard requires the live count to equal 2, reads 2, and commits. The receipt
then prints `{"HAS_REVIEW_ADMISSION": {"expected": 1, "observed": 2}}` — a parity row that
reads as a defect on an apply that succeeded by design. An operator diffing the parity
table for non-equal rows gets a false positive on every such identity, and a table that
produces routine false positives stops being read.

### 4 — PLAUSIBLE — the parking disposition is claimed atomic and is not

`imas_codex/standard_names/orphan_sweep.py:162-168`.

The rewritten docstring at lines 133-139 claims "whatever this tick parks arrives already
classified, so the parked cohort never holds a row the writer left unclassified". The
parking statement and the disposition write are separate transactions: the sweep loop
commits each query in its own transaction (lines 152-160), and only then is
`record_dispositions_for_undisposed_parks` called. Nothing wraps the pair.

Failure scenario: a tick parks 40 sources at the compose claim-attempt cap, that
transaction commits, and the disposition write then fails — a Neo4j timeout, a transient
disconnect, an exception inside `record_dispositions_for_undisposed_parks`. The exception
propagates out of `_orphan_sweep_tick`. The 40 rows are parked and unclassified, which is
exactly the state the docstring says cannot occur.

The recovery hole compounds it: the next tick's disposition call is gated on
`if counts.get(_PARKING_LABEL)`, that is, on *this* tick having parked something new. A
tick that parks zero rows never calls the catch-up write, even though the function is
named `record_dispositions_for_undisposed_parks` and is evidently capable of finding the
backlog. So the 40 undisposed rows stay undisposed until some later tick happens to park
at least one more.

Marked PLAUSIBLE rather than CONFIRMED: the atomicity claim and the gating are both
confirmed by reading the diff, but the consequence depends on
`record_dispositions_for_undisposed_parks` in `graph_ops.py`, which this review is fenced
from opening whole. The cheap repair needs no such reading — drop the `if counts.get(...)`
gate so every tick reconciles the backlog, which makes the write self-healing instead of
event-triggered.

### 5 — CONFIRMED — the adjudicable binding conflict is masked by any other malformed row

`imas_codex/standard_names/provenance_lifecycle.py:652-656`.

```python
if conflicts or (pending and completed):
    ...
    raise RuntimeError(f"source migration compare-and-set failed: {details}")
if binding_conflicts:
    raise SourceBindingConflictError(binding_conflicts)
```

`SourceBindingConflictError` exists so a caller "can withhold a retry budget instead of
treating a decision as a transient fault" (docstring, `provenance_lifecycle.py:66-78`). The generic branch is
tested first, so the named type is raised only when *every* other row in the cohort is
clean.

Failure scenario: a migration cohort of ten sources. One source is `status='stale'` and so
lands in `conflicts`; two others are bound to a live foreign identity and land in
`binding_conflicts`. The raise is the anonymous `RuntimeError` naming only the stale row.
A caller that classifies retry budget by exception type sees a transient-looking fault,
retries, and gets the identical result every time — the outcome the named type was
introduced to remove — while the two rows that actually need an operator decision are not
named in the message at all.

## Explicit checks the fence names

**Writes that delete or retarget an existing binding without a guard.** Inside the
reviewed diff: none. The added Cypher in `provenance_lifecycle.py` is receipt
construction (`deletion_change_cypher`), which creates a `StandardNameChange` node and
deletes nothing; the `param_prefix` addition is validated with `isidentifier()` at line
157, so it cannot inject. The `signed_manifest.py` additions are read-and-compare only
(`_archive_edge_counts` is a `count(...)`; `_archive_role_*` are pure functions over
already-loaded data). `orphan_sweep.py` adds no Cypher of its own. `client.py` writes
nothing.

One unguarded retarget exists in a file the diff touches but is **pre-existing, outside
this diff**, and is reported under follow-ons rather than triaged here: the migration
apply at `provenance_lifecycle.py:696` runs
`OPTIONAL MATCH (dd)-[dd_old:HAS_STANDARD_NAME]->(:StandardName) DELETE dd_old`, which
deletes the DD node's binding to *any* standard name rather than to `old` — the anonymous
`(:StandardName)` end carries no `{id: $old_name}` predicate, unlike the
`(source)-[prior:PRODUCED_NAME]->(old)` line seven rows above it (line 689), which is bound.
`git log -L` places these lines before the review base.

**A guard that swallows its own exception.** Inside the reviewed diff: none of that
shape. The two candidates both fail *open in the other direction*, which is finding 1 and
finding 4 above — an exception that escapes where the code claims it is contained, rather
than one contained where it should escape. `client.py:103` is explicit about the
distinction and gets it right: `_slurm_service_uri` deliberately lets an unreadable
location raise, because returning `None` there would silently restore the loopback
endpoint the function exists to replace.

**A guard that never fires.** `answer_quarantine_question`
(`review/audits.py:41`, `NoReturn`) is unconditional, so it cannot be a dormant guard, and
it is exercised: `tests/standard_names/test_quarantine_instrument_agreement.py:44-45` asserts
the refusal. Verified by `grep -rn` over `imas_codex/` and `tests/`, which returns the
definition and that one test — no production caller, which is correct for a refusal whose
purpose is to be unavailable rather than to be called.

## What was examined and found sound

- `client.py` `_resolve_graph_uri` / `_slurm_service_uri` — the precedence is right
  (`NEO4J_URI` outranks discovery), the SLURM branch is gated on `SLURM_JOB_ID` so the
  workstation path is untouched, and the fallback returns the profile URI rather than a
  fabricated address. Nothing here writes to the graph.
- `signed_manifest.py` `_load_archive_role_counts` input validation — the `bool`
  exclusion before the `int` check (line 6939) is correct and easy to get wrong;
  `isinstance(True, int)` is `True`, so without it a JSON `true` would have passed as a
  count of 1.
- `provenance_lifecycle.py` `_live_foreign_bindings` — excluding `_RETIRED_NAME_STAGES`
  is right: a superseded generation of a surviving name is not a competing claim.
- `provenance_lifecycle.py` `deletion_change_cypher(param_prefix=...)` — the identifier
  validation covers the new parameter, and the default `""` preserves every existing call.
- `review/audits.py` — the module adds no writes; the two question constants travel on
  `AuditReport` and the refusal names its authority.

## Negative control

The atomicity claim in finding 4 and the exception claim in finding 1 are both assertions
about paths a passing suite does not exercise. Neither was reproduced live: the fence for
this node is a read of the diff, and forcing a GitHub transport failure or a mid-tick
Neo4j disconnect needs the execution surface a test node owns. Both are recorded with the
exact line and the exact exception class so a test node can drive them; finding 1 in
particular is reproducible offline by clearing `GITHUB_TOKEN`/`GH_TOKEN` and calling
`notify_refusal_on_pull_request` with a report carrying one `promotion_refused` outcome
and a well-formed PR URL.
