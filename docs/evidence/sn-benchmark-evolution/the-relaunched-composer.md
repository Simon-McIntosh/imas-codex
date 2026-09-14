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

## Configured-route check after serve settlement

The lead subsequently confirmed that the serve had settled at `http://98dci4-gpu-0003:18802/v1`, that the router's literal model ID is `deepseek-v4.1-flash`, and that the `hosted_vllm/` prefix had not yet been proven end to end against the SGLang pass-through relay. A single minimal structured completion through the configured route was therefore required before either benchmark.

The check used the production seat identifier, the configured `ambix-local` route, the seat's reasoning effort, and `max_retries=1`. It sent one request and failed with exit status 1:

```text
LLM failed after 1 attempts: Error code: 400 - {'error': {'message':
'/chat/completions: Invalid model name passed in
model=openai/hosted_vllm/deepseek-v4.1-flash. Call `/v1/models` to view
available models for your key.'}}
```

The stack establishes that this was the direct local route, not a tunnel or OpenRouter fallback: `acall_llm_structured` selected `_acompletion_local`, which called the OpenAI-compatible client's `chat.completions.create` against the configured API base. The request reached the relay, but the forwarded model value was `openai/hosted_vllm/deepseek-v4.1-flash` rather than the served catalog ID `deepseek-v4.1-flash`.

This is the configured-route defect the pre-measurement check was designed to expose. The `hosted_vllm/` compatibility prefix does not currently resolve to the dotted SGLang model ID through this path. No retry occurred, and the failure was an HTTP 400 model-resolution refusal rather than CUDA OOM.

Both measurements remain unlaunched. Reporting either benchmark after bypassing the configured route would measure a different routing contract, so no candidate score, physics floor, dataset hash, wall-clock span, or seat decision is manufactured. Node spend remains USD 0.000000 of USD 30.000000, campaign spend remains USD 101.638907 of USD 250.000000, and both production seats remain unchanged.

The immediate next action belongs outside this node's exclusive write scope: repair the local OpenAI-compatible route so `hosted_vllm/deepseek-v4.1-flash` is translated to the catalog model ID `deepseek-v4.1-flash`, then repeat this same one-call check with retries disabled. Only a successful receipt can reopen Measurement A.

## Authorized registration update and second route check

The previous cause attribution was corrected by a peer reproduction: local endpoint registration occurs before proxy routing, and `_acompletion_local` strips `hosted_vllm/` for a registered model. The first HTTP 400 happened because only the old compose-seat ID was registered; the unregistered v4.1 identifier fell through to proxy shaping.

The lead authorized changing the compose seat before measurement because the prior local model no longer exists. Commit `ab36d6dca` changed exactly one value:

```toml
[tool.imas-codex.sn-compose]
model = "hosted_vllm/deepseek-v4.1-flash"
model-route = "ambix-local"
```

The prior value was `hosted_vllm/deepseek-v4-flash`. The route, API base, reasoning effort, and every other seat remained unchanged.

### Registered-route receipt: HTTP 404

The required minimal completion was then repeated with `max_retries=1` before either benchmark. Its resolved request facts were:

| Field | Value |
|---|---|
| Application model string | `hosted_vllm/deepseek-v4.1-flash` |
| Configured endpoint | `http://98dci4-gpu-0003:18802/v1` |
| Local client path | `_acompletion_local` |
| Wire model demonstrated by relay response | `deepseek-v4.1-flash` |
| Attempts | 1 |
| Response | HTTP 404 `unknown model id: deepseek-v4.1-flash` |

Exact response:

```text
LLM failed after 1 attempts: Error code: 404 - {'error': {'message':
'unknown model id: deepseek-v4.1-flash'}}
```

This result exonerates the registration and stripping paths: unlike the first refusal, the relay now reports the bare dotted ID rather than `openai/hosted_vllm/...`. It also proves that the configured endpoint did not accept that bare ID at the time of the request. The failure is a non-429 4xx, so the explicit stop condition applies. It is neither consumer-queue backpressure nor CUDA OOM, and it was not retried.

Measurement A and Measurement B were not launched. Their probe pairs, wall-clock benchmark spans, dataset hashes, precision, and physics floor therefore do not exist. No benchmark number is reported. The compose-seat registration change remains committed because the previous ID names the retired port-18800 model and restoring it would not recover a working configuration.

### Other stale local reviewer entries

No reviewer configuration was changed. Two name-review locations still carry the retired local ID:

| Location | Value | Active now? |
|---|---|---:|
| `pyproject.toml:583`, base `[tool.imas-codex.sn-review.names]` models list | `hosted_vllm/deepseek-v4-flash` | no |
| `pyproject.toml:612`, `[tool.imas-codex.sn-review.names.profiles.local-only]` | `hosted_vllm/deepseek-v4-flash` | no |

The resolved active profile is `default`, whose current name reviewers are `openrouter/x-ai/grok-4.5`, `openrouter/openai/gpt-5.6-luna`, and `openrouter/anthropic/claude-sonnet-5`. Therefore neither stale local reviewer entry participates in the active quorum. Changing either remains a separate reviewer-quorum decision.

### Final spend and seat state for this attempt

The rejected local route check returned before any billable provider result. Node spend remains USD 0.000000 of USD 30.000000. Campaign spend remains USD 102.872694 of the authorized USD 250.000000.

| Seat | Final value | Disposition |
|---|---|---|
| `[tool.imas-codex.sn-compose]` | `hosted_vllm/deepseek-v4.1-flash` | authorized registration update retained; measurement not reached |
| `[tool.imas-codex.sn-parent-enrich]` | `openrouter/deepseek/deepseek-v4-flash` | unchanged; still result-gated |
| Active name-review profile | `default` remote three-model quorum | unchanged |

The next external fact required is an endpoint catalog and completion route that agree on the accepted bare model ID. Once a one-attempt configured-route check returns a structured response, Measurement A may begin with its pre/post endpoint probes; Measurement B remains ordered after A.

## Measured serving envelope for the resumed benchmark

The serve operator supplied a measured capacity envelope after the registered-route refusal. The engine now enforces `--context-length 204800`, converting an over-length prefill from an endpoint-killing allocation into a distinguishable HTTP 400 refusal.

The positive and negative controls are both concrete:

| Request shape | Observation | Consequence |
|---|---|---|
| 180,961 prefill tokens with 16 tool definitions | Completed in 20.8 seconds while another worker was live | Demonstrated lower bound for accepted agent-shaped context |
| 281,421 prefill tokens | HTTP 400: `The input (281421 tokens) is longer than the model's context length (204800 tokens)` | Clean context refusal; serve remained reachable |
| Approximately 360,000 prefill tokens before the enforced cap | Serve died earlier in the day | Historical unsafe shape; not a valid probe |

The interval from 180,961 through 204,800 tokens remains untested. This node therefore adopts a strict request-size ceiling below 200,000 tokens rather than treating the configured 204,800 maximum as a demonstrated operating point. A future run must record any context-length HTTP 400 as a refused sample, never as endpoint loss and never as a score-bearing result.

Concurrency changes the safe context envelope because prefill workspace grows with both request length and simultaneous streams. Measured single-stream throughput is 33.5 tokens per second and remains 33.6 tokens per second at concurrency 4. The knee is 4; beyond concurrency 8, per-stream throughput falls approximately as `1/concurrency`.

The resumed benchmark constraints are therefore:

1. Keep every rendered request below 200,000 tokens.
2. Keep benchmark generation concurrency at or below 4.
3. When requests are long-context and concurrent, ramp toward 4 rather than opening at peak concurrency.
4. Treat HTTP 429 with a retry interval as queue backpressure.
5. Stop on CUDA OOM, connection failure, or non-429 4xx; a context-length HTTP 400 is a clean refusal and makes the affected measurement non-score-bearing.
6. Preserve the existing pre/post endpoint probes around each measurement.

These facts lower the risk of a future resumed campaign from endpoint death to an attributable refusal, but they do not clear the current blocker: the configured one-request check still received HTTP 404 for the bare model ID before any benchmark request could be sized or scheduled.

## Temporary direct route and successful configured completion

The router diagnosis dissolved the HTTP 404 as a transient replacement-window result and identified the persistent hang mechanism at port 18802: admission is keyed by client IP, so all login-node processes share two in-flight slots and four queued slots, while the queue waits on a condition with no timeout. The upstream model remained healthy during the observed router hangs.

Commit `b2c633c1c` temporarily changed only the `ambix-local` API base from the preferred router at `http://98dci4-gpu-0003:18802/v1` to the direct serve at `http://98dci4-gpu-0003:18810/v1`. It preserved the existing rotation warning and added the mechanism and revert condition: return to 18802 once the router bounds queued waits.

The first completion through the direct route returned an `LLMResult`, after which the receipt script raised `AttributeError` while trying to read a nonexistent `.value` attribute. That was an instrumentation failure after a successful provider response, not a provider error. The corrected receipt check then completed with one request and no retry:

```json
{"timestamp_utc":"2026-09-14T12:13:32.480782+00:00","request_model":"hosted_vllm/deepseek-v4.1-flash","served_model":"deepseek-v4.1-flash","api_base":"http://98dci4-gpu-0003:18810/v1"}
{"response":{"ok":true},"cost_usd":0.0,"tokens":69,"response_count":1}
```

The configured route is therefore proven end to end. The application-facing model is registered, `_acompletion_local` strips the compatibility prefix to the bare dotted ID, the direct endpoint accepts it, and the structured response returns through the project call layer. This receipt clears the route precondition for Measurement A while leaving the temporary-port rollback obligation open.

## Measurement A: completed on a changed population

Command:

```text
imas-codex sn bench --models hosted_vllm/deepseek-v4.1-flash --max-candidates 20 --runs 3 --reviewer-model openrouter/anthropic/claude-opus-4.8
```

Report:

`/home/ITER/mcintos/.local/share/imas-codex/benchmarks/sn_benchmark_20260914T121534.json`

The real-completion validity probes both passed against the direct serve:

| Position | UTC timestamp | Endpoint | Response | Attempts |
|---|---|---|---|---:|
| Immediately before | 2026-09-14T12:15:07.713791+00:00 | `http://98dci4-gpu-0003:18810/v1` | `{"ok":true}` | 1 |
| Immediately after | 2026-09-14T12:47:40.784678+00:00 | `http://98dci4-gpu-0003:18810/v1` | `{"ok":true}` | 1 |

The report records 1,845.3 seconds of benchmark wall-clock work: 1,660.15 seconds composing and 185.19 seconds reviewing. The output filename identifies command start at 12:15:34Z and the report timestamp is 12:47:14.368946Z, a 1,900-second outer span including extraction and setup. No batch error, compose error, review error, OOM, context refusal, connection failure, or non-429 4xx was recorded.

### Population identity: comparison refused

The incumbent and candidate populations are not identical:

| Report | Extracted items | Dataset hash |
|---|---:|---|
| Incumbent, 2026-07-17 | 17 | `d66fef87c0f962f1` |
| v4.1, 2026-09-14 | 12 | `636159b35205da41` |

Eight of the first twenty reference paths currently exist in the DD but are not admitted as Standard Name sources. The emitted hash therefore differs from the locked incumbent hash. The v4.1 result is **not appendable to the July table**, and no delta or better/worse claim is made against its four model rows.

The first seat condition fails: `636159b35205da41 != d66fef87c0f962f1`. Consequently the compose result cannot authorize the result-gated parent-enrich change, regardless of its score.

### Absolute three-run result

The aggregate report contains 36 candidates, 12 from each run. Its aggregate reference fields are overlap 9/47, precision 0.2500, and recall 0.1915; the precision denominator spans all 36 generated candidates and is therefore not the mean of the three per-run precisions.

| Dimension | Run 1 | Run 2 | Run 3 | Mean | Spread |
|---|---:|---:|---:|---:|---:|
| Reference precision | 0.8333 | 0.6667 | 0.7500 | 0.7500 | 0.1666 |
| Reference recall | 0.2128 | 0.1702 | 0.1915 | 0.1915 | 0.0426 |
| Reference overlap | 10/47 | 8/47 | 9/47 | 9/47 | 2 names |
| Grammar validity | 12/12 | 12/12 | 12/12 | 100% | 0 |
| Field consistency | 12/12 | 12/12 | 12/12 | 100% | 0 |
| Opus name-review mean | 0.915625 | 0.938542 | 0.939583 | 0.931250 | 0.023958 |
| Opus description-review mean | 0.985417 | 0.991667 | 0.973958 | 0.983681 | 0.017709 |

The precision spread is material, but the hash mismatch already prevents a comparative verdict. These values remain an absolute repeatability record only; endpoint variance and model variance cannot be separated from three runs.

### Measurement A spend

The JSON receipt carries USD 0.000000 composition cost, USD 0.60048775 name-review cost, and USD 0.17078325 description-review cost, for attributable spend of **USD 0.77127100**. The CLI cost headline reports USD 0.6005 because it omits `description_reviewer_cost`; that undercount is a follow-on defect rather than the spend authority used here.

Campaign spend after Measurement A is USD 103.643965 of USD 250.000000. Node spend is USD 0.771271 of USD 30.000000.

## Measurement B: INDETERMINATE at the time fence

Command:

```text
imas-codex sn bench --models hosted_vllm/deepseek-v4.1-flash --runs 3 --reviewer-model openrouter/anthropic/claude-opus-4.8 --physics
```

The immediate pre-run real completion passed:

```json
{"measurement":"B","position":"before","timestamp_utc":"2026-09-14T12:50:19.836207+00:00","endpoint":"http://98dci4-gpu-0003:18810/v1","request_model":"hosted_vllm/deepseek-v4.1-flash","served_model":"deepseek-v4.1-flash","response":{"ok":true},"cost_usd":0.0,"tokens":69,"response_count":1}
```

The command began at approximately 12:50:44Z and selected the 15-path physics fixture. Four of those paths existed in the DD but were not admitted as Standard Name sources:

- `equilibrium/time_slice/boundary_separatrix/x_point/r`
- `magnetics/ip/data`
- `magnetics/rogowski_coil/current/data`
- `pf_active/coil/b_field_max_timed/data`

The CLI printed `Reference paths: 47/47`, confirming the already-recorded banner defect: that line reports the non-physics fixture length even though extraction selected 15 physics paths.

The run remained at concurrency one and emitted no OOM, context-length overflow, connection failure, non-429 4xx, batch error, or retry. Its watchdog recorded several long but nonterminal calls. Three warning sequences stopped without an error after reaching 416, 186, and 474 seconds, indicating progress between calls. A later review or physics-judge call remained in flight through 652 seconds.

At that point the node had exceeded its hard 55-minute execution fence across Measurement A and Measurement B. The foreground command was interrupted rather than allowed to continue beyond the fence:

```text
WARNING: LLM possible stall: 1 call(s) in flight, no completion for 652s
^C
Aborted!
```

The outer Measurement B span was approximately 12:50:44Z to 13:23:5xZ, about 33 minutes. A timestamp taken immediately after cleanup was 2026-09-14T13:24:02.377312362Z.

The intended report path was `/home/ITER/mcintos/.local/share/imas-codex/benchmarks/sn_benchmark_20260914T125044.json`. It does not exist. Because the model unit did not complete, the benchmark's incremental save never landed. There is no post-run real completion, no complete validity bracket, no persisted cost receipt, and no physics verdict to recover.

Measurement B is therefore **INDETERMINATE**. No absolute physics figure is reported. The missing post-probe and absent report are evidence against reporting a number, not zero-valued measurements.

### Final seat decision

The result-gated seat condition did not pass:

- Measurement A hash equality failed: `636159b35205da41 != d66fef87c0f962f1`.
- Measurement B was indeterminate and supplies no compensating authority.

The authorized compose registration and temporary direct route remain because the old local model and port are retired. `[tool.imas-codex.sn-parent-enrich]` remains `openrouter/deepseek/deepseek-v4-flash`; it was not promoted to v4.1. The inactive local reviewer entries at lines 583 and 612 remain unchanged.

### Final spend boundary

Confirmed attributable node spend is USD 0.771271 from Measurement A. Campaign spend is therefore confirmed to at least USD 103.643965 from the supplied USD 102.872694 starting point. Measurement B produced no artifact-local receipt before interruption, so its exact provider spend is unknown and must not be reported as zero. The USD 30.000000 node ceiling was not knowingly exceeded; the missing partial-run cost receipt is a follow-on for the benchmark's durability contract.
