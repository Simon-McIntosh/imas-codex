# An endpoint drop is distinguishable from a model refusal

## Authority and question

The semantic authority is the live `imas-codex:sn-release-readiness` plan and
its followup `f-srr-a-retried-batch-hides-an-endpoint-drop` ("Benchmark
composition retries past provider errors, so an endpoint drop cannot be
distinguished from a clean run"). The question the plan fixes: when a benchmark
composition batch produces nothing usable, does the artifact the run leaves
behind say *which* of the two happened — the endpoint dropped, or the model
answered and could not be turned into a batch?

Measured on **2026-09-17** against the worktree at
`b7da331577b56ae3ba59dc3c79ce26ac76b1d4f5`.

Verdict: **the two are not distinguishable in the recorded outcome, and the
discriminating signal already exists one layer below the point where it is
discarded.**

## How many attempts the path makes before reporting

Composition calls the LLM layer once per batch with `max_retries=2`
(`imas_codex/standard_names/benchmark.py:1647`). The attempt loop that number
drives is `for attempt in range(max_retries)`, `imas_codex/discovery/base/llm.py:2393` —
so `max_retries=2` means **two provider attempts: the original and one retry**.

A retry happens only when `_is_retryable(error_msg)` is true
(`llm.py:2473`). The retryable token set is at `llm.py:645-672` and the
non-retryable overrides at `llm.py:677-682`. That distinction, not the failure
itself, decides the attempt count — and it is *orthogonal to which failure
occurred*. Measured directly against `_is_retryable` (probe log, table below):

| terminal message | matches a retryable token | attempts |
|---|---|---|
| `APIConnectionError: Connection error.` | no | 1 |
| `Connection error.` | no | 1 |
| `litellm.InternalServerError: The server had an error` | yes | 2 |
| `ReadTimeout: request timed out` | yes | 2 |
| `EOFError: EOF while parsing a JSON string` | yes | 2 |
| `ValidationError: 1 validation error for batch` | yes | 2 |
| `value 'foo' is not a registered token` | no (non-retryable override) | 1 |

So a connection refusal is **not retried at all** — it is reported after a
single attempt — while a timeout or a provider 500 is retried once. Both lose
the batch and both report the same way.

## What an operator sees today, for each case

Driving the real loop and then the real per-batch handler (probe log
`/tmp/endpoint-drop/probe.log`, exit 0) gives:

| case | attempts | terminal exception | terminal `response_count` | `ModelResult.batch_errors` | any other recorded field |
|---|---|---|---|---|---|
| endpoint drop (connection refused) | 1 | `ConnectionError` | absent | `1` | none |
| endpoint drop (timeout) | 2 | `ValueError("LLM failed after 2 attempts: …")` | absent | `1` | none |
| model refusal (response received, unparseable) | 2 | `LLMStructuredCallError` | `2` | `1` | none |

The three `ModelResult`s are identical in every grading field:

```json
{"status": "pending", "error": "", "compose_error": "",
 "batch_errors": 1, "candidate_count": 0}
```

**What the operator sees in the report**: the error column reads `1`
(`benchmark.py:1877` renders `str(r.batch_errors)`), and nothing else on the
row. The report carries **no field recording that a failure occurred at all**
beyond that count, and no field recording that a retry happened — the
retry is invisible in the artifact.

**What the operator sees in the log**: a `WARNING` at
`benchmark.py:1683` — `LLM call failed for model <model> batch <group_key>: <exc>` —
whose message text *does* differ between the three cases. That is the only
place the distinction survives, it is unstructured prose, and it is not read by
anything.

## Where the discriminator already exists

`LLMStructuredCallError` (`llm.py:528-548`) already carries `response_count`,
`attempt_count` and `telemetry_states`, and `response_count` is incremented the
moment a provider response is received, *before* its content is parsed
(`llm.py:2431`). So the LLM layer already knows whether the failure happened
before or after the provider answered: zero responses is an endpoint drop, one
or more is an answered call whose answer was unusable. The terminal raise at
`llm.py:2499-2520`) even chooses the exception *type* on that basis — a bare
`ValueError` when nothing was received, a telemetry-bearing
`LLMStructuredCallError` when something was.

`benchmark.py:1688` then catches `Exception` and keeps only
`result.batch_errors += 1`. **The distinction is discarded at exactly one
line**, which is why the fix is small.

## The test, and its two directions

The test drives benchmark composition against a simulated endpoint drop and
against a simulated model refusal and asserts the recorded outcome names which
occurred. Its source is reproduced below (`/tmp/endpoint-drop/test_batch_failure_causes.py`
on disk; it is a scratch path, not a repo path — see *Write fence* below).

```python
def _endpoint_drop_error() -> BaseException:
    """A transport failure observed before any provider response arrives."""
    return ConnectionError("Connection error.")


def _model_refusal_error() -> BaseException:
    """A provider response arrived and could not be turned into a valid batch."""
    return llm_mod._structured_call_error(
        "1 validation error for StandardNameComposeBatch",
        cost=0.0, input_tokens=11, output_tokens=7,
        cache_read_tokens=0, cache_creation_tokens=0, response_count=2,
    )


@pytest.mark.parametrize(
    ("exc", "expected"),
    [(_endpoint_drop_error(), "endpoint_drop"),
     (_model_refusal_error(), "model_refusal")],
)
def test_batch_failure_cause_is_recorded(exc, expected):
    result = asyncio.run(_recorded(exc))
    assert getattr(result, "batch_error_causes", {}).get(expected) == 1


def test_the_two_failures_are_not_recorded_identically():
    drop = asyncio.run(_recorded(_endpoint_drop_error()))
    refusal = asyncio.run(_recorded(_model_refusal_error()))
    assert drop.batch_errors == refusal.batch_errors == 1
    assert getattr(drop, "batch_error_causes", None) != getattr(
        refusal, "batch_error_causes", None
    ), "the recorded outcome is the same for an endpoint drop and a model refusal"
```

`_recorded()` patches `imas_codex.discovery.base.llm.acall_llm_structured` to
raise the terminal exception the measured loop actually produces for each case
(both exceptions above were reproduced by driving the real loop — see the
attempt table), and calls the real `benchmark._run_model`.

Both directions were run, with both logs on disk:

| direction | command | exit | log |
|---|---|---|---|
| **before** (tree unmodified) | `python -m pytest -p no:cacheprovider -q test_batch_failure_causes.py` with `PYTHONPATH=<worktree>` | **1** — 3 failed | `/tmp/endpoint-drop/test_before.log` |
| **after** (candidate patch applied to a scratch copy) | same command with `PYTHONPATH=/tmp/endpoint-drop/after:<worktree>` | **0** — 3 passed | `/tmp/endpoint-drop/test_after.log` |

The before-failure is the assertion itself, not an import error:

```
>       assert getattr(drop, "batch_error_causes", None) != getattr(
E       AssertionError: the recorded outcome is the same for an endpoint drop and a model refusal
E       assert None != None
3 failed in 12.77s
```

Regression check on the candidate patch, same file both sides:

| state | command | result |
|---|---|---|
| base | `pytest -q tests/standard_names/test_benchmark.py` | `82 passed` (exit 0) |
| patched | same, overlay on `PYTHONPATH` | `82 passed` (exit 0) |

There are **no added failures**, and the marker proves the patched tree was the
one under test: the handler line moves from `benchmark.py:1682` to
`benchmark.py:1705` in the patched run's warnings.

## The change the test requires

```diff
     batch_errors: int = 0
+    # Why each failed batch failed, counted by cause so a report can be
+    # gated on the mechanism rather than on the bare failure count.
+    batch_error_causes: dict[str, int] = field(default_factory=dict)

+def _classify_batch_error(exc: BaseException) -> str:
+    """Name why a compose batch produced no usable result.
+
+    A structured call reports how many provider responses it received before
+    failing. Zero responses means the transport failed before the provider
+    answered - an endpoint drop - whereas one or more responses means the
+    provider answered and its answer could not be turned into a batch.
+    """
+    response_count = getattr(exc, "response_count", None)
+    if response_count is not None:
+        return "endpoint_drop" if response_count == 0 else "model_refusal"
+    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
+        return "endpoint_drop"
+    return "provider_error"

             except Exception as exc:
                 logger.warning(...)
                 result.batch_errors += 1
+                cause = _classify_batch_error(exc)
+                result.batch_error_causes[cause] = (
+                    result.batch_error_causes.get(cause, 0) + 1
+                )
```

This is the first remedy the plan names ("record per-batch retry and error
counts on `ModelResult` so a report can be gated on them after the fact"),
minus the retry half: the retry is already accounted inside
`LLMStructuredCallError.attempt_count`, and recording the *cause* is what makes
the artifact gateable at all. A report can then refuse itself — the
"zero mid-run generation failures" condition becomes provable from the artifact
instead of resting on an operator watching the log.

## Write fence, and what is therefore not landed

This node's exclusive write paths are three documentation files. `benchmark.py`
and a `tests/standard_names/` test file are **both outside that fence**, so the
change above is a candidate patch reproduced for the coordinator and **not
landed here**: no source file in this worktree was modified, and the
before/after evidence was produced by running the test against a scratch copy
of the package under `/tmp/endpoint-drop/after/`.

Consequences, stated plainly:

1. The done-when's "failing before the change and passing after" is satisfied
   **as a demonstration** — both logs exist, exit 1 then exit 0 — but the
   change and its test are not in the repo, so the node cannot be closed on
   this evidence alone. A follow-on implementation node with
   `imas_codex/standard_names/benchmark.py` and
   `tests/standard_names/test_benchmark.py` in its fence lands the diff above
   and the test verbatim.
2. The measurement half of the done-when is complete and independent of that
   fence: attempts per case, exception types, the single line that discards the
   distinction (`benchmark.py:1688`), and what an operator sees today.

## Reproduction

```bash
W=<worktree>
# attempt counts and terminal errors, driving the real loop with a stubbed transport
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$W \
  /home/ITER/mcintos/Code/imas-codex/.venv/bin/python /tmp/endpoint-drop/probe.py \
  > /tmp/endpoint-drop/probe.log 2>&1; echo EXIT=$?     # EXIT=0

# the test, before and after the candidate patch
UV_PROJECT_ENVIRONMENT=... PYTHONPATH=$W \
  .../python -m pytest -p no:cacheprovider -q /tmp/endpoint-drop/test_batch_failure_causes.py
```

Scratch tree (not committed): `/tmp/endpoint-drop/probe.py`,
`test_batch_failure_causes.py`, `apply_candidate.py`, `base_benchmark.log`,
`after_benchmark.log`, `test_before.log`, `test_after.log`, `probe.log`,
`candidate.patch`.

## Limit of this measure

The simulated failures are the exception objects the loop produces, replayed
into the handler; the loop itself was measured separately with the transport
stubbed. No live endpoint was dropped and no paid provider call was made —
this node's question is about what is *recorded*, and both the recording path
and the retry classification are fully local. What remains unmeasured, and is
outside this node: whether the report-facing table should also render the
cause, and whether composing should refuse rather than record (the second
remedy the plan names, an opt-in fail-on-first-provider-error mode).