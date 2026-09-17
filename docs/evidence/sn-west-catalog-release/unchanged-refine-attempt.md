# An unchanged resubmission returns the rotation it did not spend

## Outcome

The refine pool charges a rotation when it **claims** a name, before any model
call, so the charge buys an improvement attempt.

**Resolved.**

The return is now issued from the pinned-rename branch of
`process_refine_name_batch` in `imas_codex/standard_names/workers.py`, fenced on
the same claim token and `'refining'` stage the resubmission checks, so a
rotation can only be given back while the claim that charged it is still live.

This closes one of the two defects the followup
`f-swcr-an-unchanged-resubmission-still-spends-a-rotation` records. The second
— the reviewer noise band — is not measurable from this node's change and is
recorded under *What remains*.

## The defect, reproduced

The reproduction disables the return (rebinds the module attribute to a no-op
that issues no write — the pre-change behaviour) and runs the named test:

```
$ python scripts/reproduce.py off      # return disabled
tests/standard_names/test_unchanged_refine_attempt.py F             [100%]
tests/standard_names/test_unchanged_refine_attempt.py:150: in test_counter_is_left_where_the_claim_found_it
    assert graph.rotation_returns == 1
E   assert 0 == 1
======================== 1 failed, 1 warning in 14.69s =========================
exit status 1

$ python scripts/reproduce.py on       # return enabled
======================== 1 passed, 1 warning in 13.51s =========================
exit status 0
```

Logs: `logs/reproduce-off.log`, `logs/reproduce-on.log` in the run directory.

The assertion is on the **write**, not on the absence of an error: `RotationCounter.query`
raises `AssertionError("unmodelled write reached the counter: ...")` for any
`SET`/`MERGE`/`CREATE`/`DELETE` it does not model, so a run that never issues the
return cannot pass by doing nothing.

## The named test

`tests/standard_names/test_unchanged_refine_attempt.py` — three cases, driving
`process_refine_name_batch` with a claim-carrying pinned-rename item:

| Case | What it pins |
| --- | --- |
| `TestUnchangedResubmissionReturnsTheRotation::test_counter_is_left_where_the_claim_found_it` | The resubmission fires, the model is never called, the counter returns to where the claim found it, and the emitted receipt carries the returned rotation at zero cost |
| `...::test_a_claim_this_pool_no_longer_holds_is_not_rewritten` | A stage the pool no longer owns (`name_stage='reviewed'`) is refused by the fence — zero returns, counter untouched |
| `TestRewrittenRefineKeepsItsCharge::test_a_different_spelling_keeps_the_charged_rotation` | An ordinary rewrite that produces a different spelling keeps its charge |

Command and result:

```
uv run --no-sync pytest -p no:cacheprovider \
  tests/standard_names/test_unchanged_refine_attempt.py \
  tests/standard_names/test_refine_name_chain.py \
  tests/standard_names/test_pinned_rename_refine.py \
  tests/standard_names/test_refine_attempt_budget.py
======================== 80 passed, 1 warning in 20.97s ========================
exit status 0
```

## The live census

Read from the graph on the login node through the project client, bounded to one
label and counter-filtered so each statement is a small scan of an indexed
property rather than a traversal. Script `scripts/census.py`, log
`logs/census.log`.

The cap cohort uses the **same expression the eligibility gate uses**
(`REFINE_NAME_ATTEMPTS_SPENT` = `coalesce(refine_attempts, coalesce(chain_length, 0))`),
so the figure counts the population the gate actually excludes rather than a
near-miss of it.

| Figure | Count |
| --- | --- |
| Identities at or beyond the rotation cap (3) | **293** |
| …of those, carrying at least one unchanged resubmission (`review_resubmit_count > 0`) | **11** |
| …of those, pinned renames (`edit_mode='rename'`) | **68** |

**Controls** — each counting predicate shown to see something known present
before any figure above is used:

| Control | Count |
| --- | --- |
| `REFINE_NAME_ATTEMPTS_SPENT = 1` | 1284 |
| `REFINE_NAME_ATTEMPTS_SPENT = 2` | 396 |
| `review_resubmit_count > 0` (the resubmission instrument, unaided) | 31 |
| `edit_mode = 'rename'` (the pinned-rename instrument, unaided) | 493 |
| `REFINE_NAME_ATTEMPTS_SPENT < 0` (deliberately unsatisfiable) | 0 |

The 11 is **non-zero**, so this defect is a live population and not a
single-name anecdote, and the 0 in the last row shows the instrument can report
zero when the world is empty, so a zero would have been a fact rather than a
silence.

The named case from the plan's own record reproduces exactly as written:

```
inner_hard_xray_peak_width
  refine_attempts=3/3, review_resubmit_count=1, edit_mode='rename'
  name_stage='superseded', refine_stop_reason='attempts_exhausted'
  reviewer_score_name=0.625
```

All eleven are pinned renames carrying `refine_attempts=3`. Eight rest at
`exhausted` with `refine_stop_reason='attempts_exhausted'`; one is the named
identity at `superseded` with the same reason; and two carry `accepted` — one of
them still holding `refine_stop_reason='attempts_exhausted'`, so a live stage and
a terminal stop reason disagree in the same record and the two fields cannot both
be read as a verdict.

## What the return deliberately does not do

The ordinary **identical-spelling** path is untouched. That path already ends at
the terminal/exhausted handling whose purpose is to stop a paid loop; returning
the rotation there would make the stop query see a zero counter and re-open the
loop it exists to close. The pinned branch has the opposite property: it produces
no candidate at all, so there is nothing for the loop to iterate on, which is
what makes the return safe here and unsafe there.

## What remains

- **The reviewer noise band.** Measured on the same identity: per-reviewer scores
  `0.7125/0.875/0.750` then `0.6875/0.750/0.625` for identical text — a 0.125
  aggregate swing with nothing changed. An `exhausted` verdict reached across
  that band is partly a coin flip. This node's change does not touch scoring, and
  the requirement — a repeated-measure rule or an explicit noise allowance before
  a terminal stage is treated as settled — is a separate decision the plan
  records. Not measured here.
- **The identity itself.** `inner_hard_xray_peak_width` stays `superseded` at
  `0.625` with `docs_stage` pending. Its remaining obstacle is the missing
  `half_width` physical base in the grammar registry, which is an
  `imas-standard-names` change and is already recorded against that plan. **This
  return does not recover it, and is not a route to recovering it** — an
  identity already past the cap is not re-claimed by a counter change; it needs
  the governed rescore path.
- **The residue this leaves in the counters.** The 293 at-cap identities keep
  whatever counter history they hold; the change is forward-only; it does not
  rewrite a spent counter, by design, because refunding a spent counter in bulk
  would re-open the paid loop for every name that genuinely exhausted its budget.

## Sources

- Change: `imas_codex/standard_names/workers.py` at `5e8e7bfd6`
- Test: `tests/standard_names/test_unchanged_refine_attempt.py` at `5e8e7bfd6`
- Census: `scripts/census.py`, log `logs/census.log` in the node run directory
- Reproduction: `scripts/reproduce.py`, logs `logs/reproduce-off.log` and
  `logs/reproduce-on.log`