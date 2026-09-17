# The batch failure cause reaches the recorded outcome

## Authority and question

The semantic authority is the live `imas-codex:sn-release-readiness` plan and
its followup `f-srr-the-batch-failure-cause-reaches-the-outcome` — *carry the
already-computed batch failure cause into the recorded outcome, so an endpoint
drop stops reading as a clean empty run*. The predecessor node
([an endpoint drop is distinguishable from a model refusal](/imas-codex/evidence/sn-release-readiness/endpoint-drop-distinguishable))
measured that the two failures reach the per-batch handler as different
exceptions and that the distinction is discarded there.

Applied and measured on **2026-09-17** against the worktree at
`4ba57a134b15849389a85ae5e9e115423b81c728`.

Answer: **the cause is now recorded, the applied result is the preserved patch
unchanged, and no adaptation to the current revision was needed.**

## The preserved patch and where it came from

The candidate patch and its test were preserved by the predecessor run at

```
/home/ITER/mcintos/.config/reckon/crew/runs/r-20260917T185454259985-n-an-endpoint-drop-is-distinguishable-from-a-model-refusal/artifacts-endpoint-drop/
    candidate.patch                 1868 bytes
    test_batch_failure_causes.py    2690 bytes
```

Both files were copied into this worktree from that directory and nowhere else;
neither was retyped. The patch edits
`imas_codex/standard_names/benchmark.py` in three hunks: a
`batch_error_causes: dict[str, int]` field on `ModelResult`, a module-level
`_classify_batch_error` helper, and a write of the classified cause beside the
existing `result.batch_errors += 1`.

## Application, and whether the applied result differs

The patch carries repo-root-relative paths with **no `a/`/`b/` prefixes**, so
the default strip level is wrong and reports

```
$ git apply --check -v .../candidate.patch
error: standard_names/benchmark.py: No such file or directory
```

which is a path-prefix complaint, not a content complaint: the default `-p1`
strips `imas_codex/` from both sides. At the correct strip level the patch
applies cleanly.

```
$ git apply -p0 --check .../candidate.patch    -> exit 0, no output
$ git apply -p0 .../candidate.patch            -> exit 0, 3/3 hunks applied
```

**Statement the fence asks for: the applied result does not differ from the
preserved patch, in none of the three hunks.** The comparison is by content
rather than by the absence of a reject file:

| Check | Result |
|---|---|
| hunks in the patch | 3 |
| hunks rejected or fuzzy-matched | 0 — `git apply` reports no offset and writes no `.rej` |
| added lines in the patch | 23 |
| added lines in the committed diff against the base revision | 23 |
| the two added-line sets compared line for line | identical |

So the patch, written against the earlier revision the predecessor measured,
still matches this revision exactly; the `-p1` failure above is the only
difference encountered and it is about how the patch is invoked, not about what
it contains.

## The classifier, and what it can and cannot separate

```text
_classify_batch_error(exc)
  response_count = getattr(exc, "response_count", None)
  |
  +-- response_count is not None
  |     +-- == 0  -> endpoint_drop   (transport failed, no answer)
  |     +-- >= 1  -> model_refusal   (an answer arrived, unusable)
  |
  +-- response_count is None
        +-- ConnectionError / TimeoutError / OSError -> endpoint_drop
        +-- otherwise                                -> provider_error
```

`response_count` is stored on the terminal exception by the structured-call
layer (`imas_codex/discovery/base/llm.py:588`, set at `:190`, `:200`, `:495`,
`:509`, `:559`, `:571`, `:596`, `:610`), which is the distinguishing signal the
predecessor node identified and which this patch reads rather than re-derives.
The `None` branch covers errors raised before that layer could attach a count.

## The test, its exit status, and the refusal it produces

`tests/standard_names/test_batch_failure_causes.py` (copied from the same
`artifacts-endpoint-drop/` directory) drives `_run_model` against a simulated
endpoint drop (`ConnectionError`) and a simulated model refusal (a structured
call error built with `response_count=2`), and asserts the recorded outcome
names which occurred.

| Run | Command | Exit | Result |
|---|---|---|---|
| without the patch | reverse-applied, same suite | **1** | `3 failed, 1 warning in 19.41s` |
| with the patch | worktree as committed | **0** | `3 passed, 1 warning in 10.71s` |
| focused gate | `test_batch_failure_causes.py test_benchmark.py test_benchmark_roles.py` | **0** | `118 passed, 1 warning in 8.40s` |

Logs: `/tmp/batchcause_test_unpatched.log`, `/tmp/batchcause_test.log` and
`/tmp/batchcause_gate.log`. The patch was reverse-applied for the first row and
re-applied afterwards, and `diff /tmp/benchmark_patched.py` against the restored
file reported the two identical, so the refusal run did not disturb the
committed state.

The failing run is the evidence the test can fail: without the patch both
scenarios collapse onto the same recorded `ModelResult` and all three tests
report it, including the one whose whole assertion is that the two recorded
outcomes differ (`batch_errors` equal at `1` on both sides, the cause mapping
present on neither). A suite that only ever passes would not have shown that.

## What the change does not do

It records the cause; it does not act on it. No retry policy, report gate or
run disposition reads `batch_error_causes` yet, so an endpoint drop still
produces a run whose composition contributed nothing — the difference is recorded
and available to a reader, not prevented.

## Limits

- **This gate measures this node's change only.** Verifying the merged result
  belongs to a separately dispatched test node, so no suite-wide claim is made
  here. The focused gate covers the module the change lives in plus its two
  neighbours; the `118 passed` figure is that subset at this revision.
- **The patch reads a stored attribute rather than re-deriving the signal.**
  If `response_count` is ever dropped from the structured-call error, the
  classifier silently falls through to the type-based branch and a model
  refusal would be reclassified as a provider error. Nothing in this node
  guards that; the test pins the attribute being present.
- The simulated failures are constructed in-process, not produced by a live
  endpoint, so the test confirms the classification of the two shapes rather
  than the shape a real drop or refusal actually raises.