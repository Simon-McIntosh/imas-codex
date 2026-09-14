NEEDS-HELP: The required three-run benchmark did not begin: its graph-backed extraction cannot run on the compute node, and the checkout exposes 47 reference paths rather than the required 54.

tried: Read live `sn-release-readiness` version 135, accepted the cleared stability watch, verified the target in three endpoint listings, and launched the exact model/runs/physics benchmark twice. Job `1270343` exited 2 before Python because the compute environment duplicated uv's no-sync option. With that invocation-only duplication removed, job `1270344` reached candidate extraction and exited 1 because `localhost:17687` is the login-node-local Neo4j tunnel. The CLI receipt also printed `Reference paths: 47/47`. Neither launch reached generation or judging, no JSON report was written, and node spend is USD 0.000000.

options: Resume this node after the 54-versus-47 reference-set discrepancy is repaired or explicitly adjudicated, and run the benchmark on the login node under the declared tunnel exception; alternatively, provide a frozen 54-path extraction artifact that the benchmark can consume without Neo4j; separately, add a fail-on-first-provider-error mode because the current benchmark requests two attempts and records exhausted batch errors instead of refusing the sample immediately.

leaning: Repair or recover the seven missing reference entries first, then resume the exact CLI on the login node with a successful launch probe, zero observed retries or generation errors, and an immediate successful post-run probe. This is the only route that preserves the required 54-path population and live graph context without changing the experiment.

cost-if-wrong: Running the current 47-path population would spend judge budget on a different dataset and make any incumbent comparison invalid. Running on compute cannot pass extraction. Running before fail-first behavior is available risks silently replacing a failed endpoint sample with a retry. Any result from those routes would have to be discarded and all three runs repeated.

# Relaunched composer measurement

Date: 2026-09-14  
Source revision: `f66479e1186010009183902edd6888c3ac80e6db`  
Candidate requested: `hosted_vllm/deepseek-v4.1-flash`  
Required command: `imas-codex sn bench --models hosted_vllm/deepseek-v4.1-flash --runs 3 --physics`  
Required population: 54 reference paths  
Required verdict: `FAVOURABLE`, `UNFAVOURABLE`, or `INDETERMINATE` after a valid three-run comparison  
Operational status: **BLOCKED BEFORE MEASUREMENT**

## Outcome

No model-quality verdict is reported. Both launches failed before the first generation call, so there are no candidate outputs, reviewer scores, physics verdicts, run-to-run spreads, or valid dimensions to compare. The intended report path was:

`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260914T091748687306-n-srr-the-relaunched-composer-is-measured-before-it-is-seated/deepseek-v4.1-flash-54-path-physics-3runs.json`

An explicit filesystem receipt confirmed that this path does not exist after the failed launches. This is absence backed by the control-flow failure: `_extract_candidates()` raised before `_run_model()`, before any incremental report could be built or saved.

No configuration seat changed. The compose seat remains `hosted_vllm/deepseek-v4-flash`; the parent-enrich seat remains `openrouter/deepseek/deepseek-v4-flash`.

## Endpoint stability evidence

The supplied stability watch ran from 11:20:17 to 12:01:28 Europe/Paris. It recorded 32 failed probes during those 41 minutes, followed by ten consecutive successful probes at sixty-second spacing. The hold was therefore lifted on a settled final ten-minute window, while the preceding thrash remains part of this measurement's validity boundary.

The node then made three aimed listings. Each listing returned the known-present target rather than treating an HTTP response alone as availability:

| Observation | UTC | Europe/Paris | Returned model IDs | Target present |
|---|---|---|---|---:|
| Before first launch | 2026-09-14 10:05:56.694505 | 2026-09-14 12:05:56.694505 | `deepseek-v4.1-flash` | yes |
| Before corrected launch | 2026-09-14 10:08:40.565188 | 2026-09-14 12:08:40.565188 | `deepseek-v4.1-flash` | yes |
| Immediately after extraction failure | 2026-09-14 10:09:25.209613 | 2026-09-14 12:09:25.209613 | `deepseek-v4.1-flash` | yes |

The failed run therefore does not demonstrate another endpoint flap. It demonstrates a placement failure before endpoint use. A future completed run still requires an immediate post-benchmark listing; the successful post-failure listing is not a substitute.

## Launch receipts

### Job 1270343: invocation did not reach Python

The first SLURM launch failed with exit status 2 before the benchmark CLI ran:

```text
error: the argument '--no-sync' cannot be used multiple times
Usage: uv run [OPTIONS] [COMMAND]
srun: error: 98dci4-clu-3141: task 0: Exited with exit code 2
```

This launch made zero model and judge calls. The compute environment already supplied no-sync behavior, so removing only the duplicate option was an invocation correction rather than a change to the benchmark.

### Job 1270344: graph extraction cannot run on compute

The corrected launch entered the CLI and printed these run conditions before failing:

```text
SN Benchmark
  Models: hosted_vllm/deepseek-v4.1-flash
  Reference paths: 47/47
  Runs per model: 3
  Temperature: 0.0
  Reviewer(s): gemini-3.5-flash, gpt-5.5, grok-4.5,
    gemini-3.1-pro-preview, claude-sonnet-5, claude-opus-4.8
  Mode: names-only
```

Candidate extraction then attempted the login-local graph endpoint and exited 1:

```text
SSH tunnel start failed: ssh: connect to host iter port 22: Connection refused
Couldn't connect to localhost:17687
Failed to establish connection to 127.0.0.1:17687: Connection refused
```

The traceback ends in `benchmark.py:_extract_candidates`, `sources/dd.py:extract_dd_candidates`, and `graph/client.py:query`. This matches the repository's declared login-node exception: the Neo4j tunnel is local to the login node and cannot be established from a compute node.

## Reference-set blocker

The command's own help says the benchmark uses a fixed set of 54 curated DD paths and that omitting `--max-candidates` runs all 54. The actual runtime receipt printed `Reference paths: 47/47`. The exact done-when requires 54 paths, so 47 is not a smaller valid run and must not be compared with a 54-path incumbent.

This node did not diagnose or repair the seven-path discrepancy because its exclusive write scope contains only this evidence document and `pyproject.toml`. The fact required before resumption is a runtime receipt that says `Reference paths: 54/54`, or an explicit authority change redefining the frozen population and requiring both incumbent and candidate to be measured on that new population.

## Retry contamination guard

The current implementation cannot promise stop-on-first-generation-failure:

- `imas_codex/standard_names/benchmark.py:1647` calls `acall_llm_structured(..., max_retries=2)` for composition.
- `imas_codex/discovery/base/llm.py:2382` iterates those attempts and retries retryable failures.
- `imas_codex/standard_names/benchmark.py:1678` catches an exhausted batch exception, increments `batch_errors`, and continues.

No generation call occurred in this node, so no retry contaminated the absent measurement. Before a resumed campaign is accepted, either the benchmark needs a fail-on-first-error mode or the complete log and report must prove zero retries, zero batch errors, and complete row coverage. If any retry appears, the result is contaminated under the locked stability rule and cannot carry numbers or a model verdict.

## Incumbent comparison authority

No committed evidence in `docs/evidence/sn-benchmark-evolution/` provides the required incumbent comparison on this experiment's conditions. The nearest records are not substitutes:

| Record | Population and run conditions | Why it is not comparable |
|---|---|---|
| `benchmark-pinned-rerun.md`, 2026-08-25 | 40-row advisory paired corpus; two candidate passes; fresh three-seat panel | Different population, advisory scoring protocol, and no full 54-path physics run |
| `benchmark-calibration-run.md`, 2026-08-25 | 40-row advisory paired corpus; two calibration passes; candidate plus hosted panel | Different population and gate definitions |
| archived `sn_benchmark_20260717T162402.json`, 2026-07-17 | 20-path maximum, one run, Opus 4.8 reviewer, physics judge disabled | Different population, one run, and no physics judging |

Therefore no per-dimension margin can be founded yet. A resume needs either the path to a recorded `deepseek-v4-flash` report with the same frozen dataset and judge or a newly authorized incumbent rerun under exactly the candidate conditions. Without that, even a technically complete v4.1 report cannot support `FAVOURABLE` or `UNFAVOURABLE`.

## Spend receipt

| Scope | Before | Node spend | After |
|---|---:|---:|---:|
| Node judge and generation ceiling | USD 0.000000 | USD 0.000000 | USD 0.000000 of USD 30.000000 |
| Campaign against authorized ceiling | USD 101.638907 | USD 0.000000 | USD 101.638907 of USD 250.000000 |

Both failures preceded generation and judging. No provider request, graph mutation, signed manifest apply, pipeline drain, or seat update was attempted.

## Resume gate

A resumed measurement is actionable only when all of these are simultaneously true:

1. The CLI reports `Reference paths: 54/54`.
2. The launch runs on the login node under the declared Neo4j tunnel exception.
3. An immediate pre-run listing contains `deepseek-v4.1-flash`.
4. All three runs complete with zero generation failures, zero retries, zero missing batches, and spend at or below USD 30.000000.
5. An immediate post-run listing contains `deepseek-v4.1-flash`.
6. A same-population, same-judge incumbent record is available for per-dimension comparison.
7. Marked run disagreement is reported as endpoint/model confounding and yields `INDETERMINATE`; seats move only after a valid `FAVOURABLE` result.

## Corrected authority and resumed attempt

The earlier 54-path premise was corrected before resumption. `BenchmarkConfig.max_candidates = 54` is a cap, not a fixture size. The committed non-physics fixture is the 47-entry `REFERENCE_NAMES` mapping in `imas_codex/standard_names/benchmark_reference.py`; the physics fixture is the 15 hard-case paths in `research/physics_bench_paths.json`. Both fixtures were last changed on 2026-08-23. The earlier `47/47` output is therefore not a population shortfall.

The output remains defective under `--physics`: `imas_codex/cli/sn.py:3479` calculates `total_ref` from `REFERENCE_NAMES` unconditionally, while `_extract_candidates` selects the 15-path physics fixture. Consequently the banner cannot identify the physics population. This node records that defect as a follow-on and does not edit the CLI outside its scope.

Placement was also corrected: graph enrichment makes the benchmark eligible for the repository's login-node exception, and its workload is network-bound. No further SLURM launch was attempted.

### Comparable incumbent receipt

The most recent comparable report was read before the resumed run:

`/home/ITER/mcintos/.local/share/imas-codex/benchmarks/sn_benchmark_20260717T162402.json`

It was written at 2026-07-17T16:40:48.571305Z after resolver repair `617333bd3`, used `max_candidates=20`, one run, temperature 0.0, Opus 4.8 as the sole reviewer, and no physics judge. It extracted 17 items and recorded `dataset_hash=d66fef87c0f962f1`.

| Model | Reference precision | Reference recall | Overlap |
|---|---:|---:|---:|
| `hosted_vllm/deepseek-v4-flash` | 0.7059 | 0.2553 | 12/47 |
| `openrouter/openai/gpt-5.6-luna` | 0.6471 | 0.2340 | 11/47 |
| `openrouter/openai/gpt-5.5` | 0.6471 | 0.2340 | 11/47 |
| `openrouter/openai/gpt-5.6-terra` | 0.5882 | 0.2128 | 10/47 |

The incumbent row additionally records 17 candidates, 16 grammar-valid names, one grammar-invalid name, 16 field-consistent names, zero batch errors, Opus name-review mean 0.8139705882, and Opus description-review mean 0.9566176471. Its reviewer spend was USD 0.36551425 and local composition spend was USD 0.000000.

### Measurement A: INDETERMINATE before launch

Required command:

```text
imas-codex sn bench --models hosted_vllm/deepseek-v4.1-flash --max-candidates 20 --runs 3 --reviewer-model openrouter/anthropic/claude-opus-4.8
```

The immediate pre-run endpoint listing returned HTTP 503, so the command was not launched. A timestamped diagnostic listing immediately afterwards reproduced the same refusal:

```json
{"measurement":"A","position":"diagnostic-after-failed-preprobe","timestamp_utc":"2026-09-14T10:18:15.299653+00:00","http_status":503,"body":"{\"error\":{\"message\":\"no upstream catalogs are reachable\"}}"}
```

This is a direct recurrence of the earlier flap after the ten-success watch window. Under the locked validity rule, Measurement A is **INDETERMINATE** and no model numbers are reported. There is no emitted report, no candidate `dataset_hash`, no wall-clock benchmark span, and no post-run probe because no run began. The 503 receipt is the measurement outcome; a later successful probe cannot retroactively validate this attempt.

### Measurement B: not started

Measurement B must follow Measurement A and would run the same command with `--physics` added and `--max-candidates` omitted over the 15 hard-case paths. It was not launched after Measurement A's failed pre-run probe. No absolute physics figure exists from this attempt, and no incumbent delta is claimed.

### Seat decision after resumed attempt

Both necessary promotion conditions failed to become measurable:

1. Measurement A emitted no `dataset_hash`, so equality with `d66fef87c0f962f1` is not established.
2. Measurement A emitted no reference precision, so precision at or above 0.7059 is not established.

Accordingly both seats remain unchanged:

| Seat | Retained model | Reason |
|---|---|---|
| `[tool.imas-codex.sn-compose]` | `hosted_vllm/deepseek-v4-flash` | No matching-hash, threshold-clearing Measurement A result |
| `[tool.imas-codex.sn-parent-enrich]` | `openrouter/deepseek/deepseek-v4-flash` | Promotion is jointly gated on the same missing Measurement A result; remote v4.1 availability was not reached as a decision point |

### Spend after resumed attempt

No benchmark command was launched and no generation or judge request was made. Node spend remains USD 0.000000 of USD 30.000000. Campaign spend remains USD 101.638907 of the authorized USD 250.000000.

### Remaining follow-ons

- Correct the `--physics` reference-path banner so it reports the 15-path fixture selected by extraction.
- Expose fail-on-first-provider-error behavior or otherwise prove zero retries for a future benchmark; composition still uses `max_retries=2` and can replace a dropped request.
- Re-establish a stable endpoint window, then resume Measurement A on the login node before attempting Measurement B.
