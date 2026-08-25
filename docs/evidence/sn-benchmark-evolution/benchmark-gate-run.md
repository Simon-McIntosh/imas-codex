NEEDS-HELP: the benchmark cannot be run against locked gates because no admissible frozen population or numeric advisory-gate thresholds exist at the dispatched source revision.

tried: Read the complete live plan at version 12 and source commit `95930c027757858cd0a149c10e43aa7a5af6907b`, its superseded benchmark plan, the landed reuse map, the frozen-runner and panel implementations, their tests, repository configuration, and the committed eval inventory. The runner/panel verification selection passed 12 tests with one warning. The only committed eval inventory has 51 positive slots but just 20 populated positives and 31 `TODO` placeholders; the reuse map explicitly rejects it as the frozen batch. The plan names five advisory measures but supplies no numeric thresholds, and the runner emits judgments, hashes, and costs without defining gate calculations or a repeatability protocol.

options: (1) Commit the independently labelled frozen advisory population, numeric thresholds for all five named measures, and the within-run repeatability protocol, then resume this node; (2) if those authorities already exist outside this source revision, integrate them and redispatch or resume this worker with the exact paths and commit; (3) explicitly change the benchmark contract to use the incomplete legacy inventory and its three composer gates, accepting that they do not measure the advisory reviewer described by the live plan.

leaning: Option 1. It follows the locked `repo-json` baseline decision, preserves the meaning of the three required hashes, and keeps the census authorization fail-closed.

cost-if-wrong: Running against an invented population or thresholds can spend up to the USD 25 ceiling and produce a plausible-looking receipt that could wrongly authorize a graph-wide census. Any such result would have to be discarded and the candidate plus all three hosted judges rerun after the real population and gates are frozen.

# Frozen advisory benchmark gate record

## Outcome

No billable candidate or judging call was made. The hard spend ceiling is intact at **USD 0.000000 spent of USD 25.000000 authorized**. Because the population and gate authority are missing, the benchmark evidence fence is unmet and the graph-wide advisory census is **not authorized**.

This is a fail-closed blocker, not a negative measurement of DSv4 quality. The local candidate and configured production panel were not invoked, so no model-quality conclusion is supported.

## Authority and input audit

- Plan: `docs/sn-benchmark-evolution.html`, plan version 12, read in full at repository commit `95930c027757858cd0a149c10e43aa7a5af6907b`.
- Candidate route that the plan intends to assess: `hosted_vllm/deepseek-v4-flash` from the local Standard Names compose seat.
- Configured production judging profile: `openrouter/x-ai/grok-4.5`, `openrouter/openai/gpt-5.6-luna`, and `openrouter/anthropic/claude-sonnet-5`; disagreement threshold 0.20.
- Runner: `imas_codex/standard_names/advisory_benchmark.py`; it freezes a caller-supplied population, samples deterministically, calls the candidate and panel, and publishes raw rows, three hashes, provenance, and artifact-local costs.
- Gate gap: the runner defines no gate schema or computation. The plan names critical-defect recall, clean-name false positives, calibration, repeatability, and hosted-quorum agreement, but neither the plan nor a committed baseline gives those measures numeric thresholds.
- Repeatability gap: the runner calls the candidate once. The plan does not define whether repeatability means multiple candidate draws within one campaign, a deterministic replay, score tolerance, label agreement, or an exact-output comparison.
- Population gap: `tests/standard_names/eval_sets/benchmark.json` contains 51 positive slots, of which 20 are populated and 31 remain `TODO`, plus 20 negatives. Its file SHA-256 is `4e2309f0801d1d0c35fb05a90908c3645283c73089362e84b3ed9bdff75c30f7`. The landed reuse map explicitly says not to treat this file as the frozen batch because its positive inventory is incomplete and its legacy loop self-judges.
- Legacy-threshold mismatch: the superseded composer benchmark carries pass-at-1 at least 0.80, positive mean reviewer score at least 0.75, and negative rejection rate at least 0.90. Those gates assess composition output and do not define the five advisory-review measures in the current plan, so applying them here would silently change the experiment.

## Gate matrix

| Locked measure named by the live plan | Measured value | Locked threshold available? | Verdict |
|---|---:|---:|---|
| Critical-defect recall | Not measured | No | **BLOCKED — fail-closed** |
| Clean-name false-positive rate | Not measured | No | **BLOCKED — fail-closed** |
| Calibration | Not measured | No metric or threshold | **BLOCKED — fail-closed** |
| Repeatability | Not measured | No protocol or threshold | **BLOCKED — fail-closed** |
| Hosted-quorum agreement | Not measured | No metric or threshold | **BLOCKED — fail-closed** |
| Cost ceiling | USD 0.000000 | USD 25.000000 maximum | **PASS** |

No advisory quality gate passed, qualified, or failed on model evidence because no authorized measurement could begin. Treating “not measured” as a quality failure would also be inaccurate; the operative verdict is **blocked and fail-closed**.

## Frozen hashes

The three required run hashes were deliberately not manufactured:

| Required hash | Value | Reason |
|---|---|---|
| Population hash | **not produced** | No committed admissible frozen population exists. |
| Ordered-input hash | **not produced** | Sampling an unauthorized population would not define a valid benchmark input. |
| Ordered-result hash | **not produced** | No candidate or judging calls were made. |

The SHA-256 above identifies the rejected legacy eval file only. It is not any of the three benchmark-run hashes and must not be promoted as one.

## Spend

| Spend component | Actual USD | Authorized ceiling | Remaining USD |
|---|---:|---:|---:|
| Candidate | 0.000000 | — | — |
| Hosted judging panel | 0.000000 | — | — |
| **Total** | **0.000000** | **25.000000** | **25.000000** |

No OpenRouter request and no local candidate request was made. No graph-backed budget reservation or cost record was created.

## Verification evidence

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/standard_names/test_advisory_benchmark.py tests/standard_names/test_advisory_panel.py -q
```

Result: **exit 0; 12 tests passed; one existing pytest configuration warning**. This proves the landed runner and non-persisting panel mechanisms are internally green; it does not substitute for the missing operational population and gate authority.

Full log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T033934039486-n-benchmarkgaterun/runner-tests.log`
Log SHA-256: `57f440aa7397b275fba4856f81b1399feb2fe818ba659d67ad8297f74f134379`

## Census authorization

**The graph-wide advisory census is not authorized.** The locked decision permits it only after the benchmark gates pass. Five quality measures remain unmeasured, the three frozen run hashes do not exist, and only the independent cost ceiling check passes. Resume the same node only after the frozen population, thresholds, and repeatability protocol are committed or otherwise supplied as exact versioned authority.
