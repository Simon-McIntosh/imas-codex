# An unreachable embedder does not become a finding — at three sites, not one

A peer session reported one fail-open on the review surface: a downed embedder
lets `sn review` report clean. Confirmed at HEAD, and while confirming it I found
**two further sites of the same shape**, one of which nests inside the other. By
the repo's own rule, three sites of one shape is a repo-wide defect rather than
three fixes.

The common shape: **each site returns a well-formed answer on failure.** Nothing
raises, so no caller, test or gate learns the service was unavailable, and every
check that validates the *shape* of a result passes. The log is not the problem —
two of the three sites log with `exc_info=True`. The problem is the *result*.

## Site 1 — the embedding preflight (the reported one)

`imas_codex/standard_names/review/audits.py`, the re-embed preflight. On failure
it logs a warning and returns its report. `EmbeddingReport` carries counts and
nothing else:

```python
total  missing_count  stale_count  refreshed_count  missing_ids  stale_ids
```

There is **no field for unavailability**, so the two states are
indistinguishable to every consumer:

| embedder | report |
|---|---|
| healthy | `missing_count=N, refreshed_count=N` |
| unreachable | `missing_count=N, refreshed_count=0` |

`imas_codex/cli/sn.py:6320` prints both numbers and continues. A reader sees
*"N missing, 0 refreshed"*, which reads as benign. The peer's own correction to
their report is the precise one: it is not silent in the log, it is silent in the
result.

## Site 2 — `semantic_similarity_check`, which is a CRITICAL check

`imas_codex/standard_names/audits.py:3737`. On an embed failure, and on an embed
that returns `None`:

```python
except Exception:
    logger.debug("semantic_similarity_check: embed failed for %s", name, exc_info=True)
    return None, []
```

An empty issues list is *no issues* — a pass. This check is a member of
`CRITICAL_CHECKS` (`audits.py:295`), the set whose failure demotes a name to
`quarantined`. **So a check severe enough to quarantine a name reports pass when
its input cannot be obtained**, and at `DEBUG` level. This is a different file
and a different function from site 1; it was found only because the reported line
number resolved into the wrong `audits.py` (there are two, 200,916 bytes and
24,062 bytes) and the misread landed on an independent instance.

## Site 3 — the caller swallows the check as well

`imas_codex/standard_names/workers.py:8551` wraps the site-2 call in a second
handler:

```python
try:
    sem_sim, sem_issues = await _asyncio.to_thread(semantic_similarity_check, ...)
except Exception:
    logger.debug("review_name: semantic_similarity_check failed for %s", sn_id, exc_info=True)
```

So even if site 2 were repaired to raise, this caller would convert the refusal
back into silence. **A repair at site 2 alone is invisible**, which is the reason
these are one defect: closing the inner handler without the outer one produces a
guard that still cannot speak.

## The remedy is not a log line

Adding a message fixes nothing, because two sites already log. The remedy is the
one that closes the other invisibility class in the same incident: **assert
something only the healthy path could produce.**

- Give the report an explicit unavailability state and make the consumers read
  it, so an unreachable embedder is a *finding* rather than a count of zero.
- Add a **control input whose expected verdict is not clean** — a pair of
  near-duplicate descriptions the embedder must flag — so an audit run against a
  dead embedder fails on a *missing* finding instead of passing on an empty one.
- Close site 3 in the same change as site 2, or the repair cannot be observed.

A test that patches the embedder to raise and asserts the report carries an
unavailability finding would have covered the whole of the outage window that
prompted this.

### The obvious repair is worse than the fail-open

**The unavailability signal must not be emitted inside the
`audit:<critical-check>:` namespace.** Quarantine is decided by substring, with
no severity field:

```python
def has_critical_audit_failure(issues: list[str]) -> bool:
    for issue in issues:
        for check in CRITICAL_CHECKS:
            if f"audit:{check}:" in issue:
                return True
```

`semantic_similarity_check` is a member of that set. So an issue string reading
`audit:semantic_similarity_check: embedder unavailable` would **quarantine every
name the outage touched** — on 2026-09-23 that is the whole review population —
not because any name is defective but because a GPU job was cancelled.

The current fail-open returns `(None, [])`, a silent pass. Emitting a finding in
that namespace converts it into a **silent mass demotion**, and the second is far
harder to undo than the first. The repository has the precedent: a property
written for one purpose later acted as a delete permission across 2,096 rows.
**A string meaning *this name is defective* must not be reachable by *I could not
tell*.**

The distinction the code lacks is **check failed** versus **check could not
run**. Two shapes hold it, and the choice belongs to whoever takes the repair:
keep the unavailability signal outside the critical namespace so it reports
without demoting, or refuse before any name is judged — which fits
`EmbeddingReport` having no state to branch on, and is closer to stopping rather
than grading. Locked as a decision on the owning plan.

### Measure what the fail-open was holding up, before writing the change

Making one guard in this repository raise instead of swallow exposed **38
failures and 5 setup errors across 13 files**, none of them regressions — every
one green only because the guard had been switching itself off. Expect that shape
here: flip it closed in a scratch copy and count first, so the number is in the
brief rather than discovered by the worker, and split the node if it cannot
absorb what it exposes. Never widen the swallow to reach green.

## Exposure, measured rather than assumed

The embedding server had been up 8 days 11 hours when it was cancelled at
2026-09-23T04:12Z, and was restored about 35 minutes later. No `sn review` ran
inside that window — no project log was written after 04:00Z and no crew run was
live — so the actual damage is nil. **That bounds the incident, not the defect:**
the fail-opens are reachable on every future outage, and the exposure was narrow
by luck of timing rather than by any guard.

## Provenance and ownership

Reported by a peer session, which confirmed it reproduces at HEAD `d5b0cf1f8`
and has explicitly handed the repair over rather than half-holding it; it will
stay off `imas_codex/standard_names/review/audits.py`. Sites 2 and 3 are this
session's, found while verifying site 1. The original finding lived as a plan
comment with no followup id, so nothing in pending work would have surfaced it —
which is why this record is paired with one.
