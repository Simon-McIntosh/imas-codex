# Candidate nondeterminism diagnosis

Date: 2026-08-25  
Source revision: `c03f849229eea4789fe0c206ea9490e807563e9e`  
Candidate: `hosted_vllm/deepseek-v4-flash`  
Candidate decoding: temperature `0.0`, request seed `20260825`, reasoning effort `medium`  
Corpus: `tests/standard_names/eval_sets/advisory_paired_corpus.json`  
Corpus SHA-256: `216e738a7ffbb7ae54b13e7e45f2fc5b1f99d34caba8b644621a931891fdae23`  
Spend ceiling: USD `6.00`

## Outcome

**Cause identified: provider-side inference nondeterminism.** The local vLLM
endpoint returns different structured judgments for byte-identical complete
requests even when those requests are issued serially. Neither a non-frozen
request element nor ordering/concurrency inside the benchmark runner is needed
to produce the variance.

The baseline remains exactly as measured: three concurrent frozen passes at
temperature `0.0` and seed `20260825` produced Brier errors `0.121516`,
`0.071531`, and `0.098832`, a spread of `0.049984`, around the fixed threshold
`0.103764`. The new candidate-only serial measurement produced Brier errors
`0.102465`, `0.146418`, and `0.112895`, a spread of **`0.043953`**. The first
serial pass is below the threshold and the other two are above it. Therefore,
removing concurrency from the benchmark does **not** turn the gate into a
single reproducible value.

| Candidate-only serial pass | Brier error | Accepted rows | Exact input hash | Four message hashes equal |
|---:|---:|---:|---:|---:|
| 1 | 0.102464844 | 17/40 | yes | yes |
| 2 | 0.146417969 | 14/40 | yes | yes |
| 3 | 0.112894531 | 14/40 | yes | yes |
| **Spread** | **0.043953125** | **3 rows** | — | — |

Thirty of the 40 row scores moved across the three serial full-corpus passes.
That is not a formatting or aggregation artifact: all three calls used the
same ordered input hash and the same ordered four-message hash vector, while
all three output hashes differed.

## Direct cause discrimination

### Provider-side execution: positive

The volatile known-good row `vertical_coordinate_of_shunt` was submitted three
times serially through the production-context scorer. All three calls had the
same input hash
`e60feb74d57cbd7f620a11af353f6e40418cac9e692ed2dce13a66b8db7c1858`
and rendered-message hash
`59babafc713dab19328e37bd5facf610df2b409696b716f1083592493d6523c1`.
The returned scores were `1.000`, `1.000`, and `0.975`; the complete structured
output hashes differed on all three calls. Three concurrent copies also
returned different complete outputs and scores `0.975`, `0.975`, and `1.000`,
but the serial result is decisive because it proves concurrency is not
required.

A second probe instrumented the final local-provider boundary, below the
prompt renderer and structured-call wrapper. It normalized and hashed every
semantic request field except the API-key value itself: endpoint, model,
messages, response format/schema, maximum tokens, timeout, temperature, seed,
reasoning `extra_body`, metadata, and extra headers; API-key presence was
included as a boolean. All three serial requests had the same complete request
hash:

`b8a301cc41aed078f088cac54e03e05c0edd22f460a27e840baddd812c86880c`

Despite that equality, the endpoint returned scores **`1.000`, `0.625`, and
`0.9875`** and three different structured-output hashes. This is direct
evidence that the variance begins after the complete request crosses the local
provider boundary.

At temperature zero the request seed cannot repair this class of residue: the
decoder is greedy, so sampling PRNG state is not the moving degree of freedom.
The observed behavior is more precisely provider-side numerical/execution
nondeterminism than stochastic sampling that merely ignores the seed.
Nevertheless, among the three candidate causes posed by the plan it belongs to
the first, provider-side cause: the endpoint does not honor the requested
reproducibility contract.

### Non-frozen request element: ruled out

The complete provider-bound semantic request hash was identical across three
serial calls. This strengthens the earlier four rendered-message hashes by
also covering the response schema, maximum-token limit, endpoint, timeout,
headers, metadata, reasoning controls, model, temperature, and seed. The
outputs still differed. No non-frozen caller-side field entering the request
accounts for the movement.

### Runner ordering or concurrency: ruled out as the cause

The prior three-pass stability program did launch its three full benchmark
passes concurrently with `asyncio.gather`; that was a real confound and could
have amplified continuous-batching effects. The new three-pass full-corpus
probe issued every candidate call serially and omitted the hosted panel
entirely. It still produced a `0.043953` Brier spread and 30 moving rows.
Runner concurrency is therefore not a necessary cause and serializing the
benchmark is not a remedy.

An original/reversed/original ten-row check also showed moving scores, but the
two original-order calls differed from one another, including different
complete output hashes. Four reversed-order scores lay outside both observed
original-order values, but that comparison is not identifiable against the
endpoint's larger repeat noise. More importantly, the three failing
full-corpus serial passes retained identical row order and message hashes.
Ordering can change a batch prompt, but it cannot explain variation when the
batch prompt and order are unchanged.

## Serving evidence and mechanism

The live endpoint was SLURM job `1252401`, vLLM `0.26.0`, on two H200 GPUs. Its
startup receipt records tensor parallelism `2`, asynchronous scheduling,
chunked prefill, prefix caching, breakable CUDA graphs, FP8 KV cache, FP4 MoE
experts, custom/PyNCCL all-reduce, and a maximum of 1,024 simultaneously
scheduled sequences. Those are execution conditions under which batch shape,
floating-point reductions, routing boundaries, and low-precision kernels can
change token logits even when greedy decoding and the request seed are fixed.

This configuration evidence explains why the seed is insufficient, but the
causal classification does not depend on choosing one kernel as the culprit:
the serial complete-request probe already locates the variance inside the
provider service. Isolating which serving option is necessary would require
restarting controlled server variants, which this measurement-only node was
not authorized to do.

## Removability and cost

**Residue: REMOVABLE in principle, but not by a benchmark-runner change.** A
deterministic inference contract would require a separately controlled serving
variant and remeasurement. The minimum credible experiment is a dedicated,
serialized two-GPU service with asynchronous scheduling and chunked batching
disabled, deterministic kernels where vLLM/PyTorch support them, and the
low-precision/cache choices varied one at a time. The current model service
uses two H200s; its measured cold start was about **26 minutes** from job start
at 18:09 to application readiness at 18:35. The cost is therefore reserved
two-H200 capacity, at least one cold-start cycle per serving variant, reduced
throughput, and potentially higher memory if FP8 KV or FP4 execution must be
removed. Bitwise determinism is not guaranteed until that experiment passes.

Simply serializing calls costs only throughput but is disproven as a fix by the
three serial Brier passes. Caching and replaying the first completion would be
cheap and exactly repeatable, but it would no longer measure repeated model
inference and is not a valid scientific remedy.

Consequently, the **57 further benchmark repeats are the wrong next purchase
before one controlled serving experiment**: the cause is inside a service
whose execution contract can in principle be changed, and the candidate itself
is free. If a controlled deterministic service remains variable or its
capacity cost is rejected, then the residue must be treated as irreducible for
this deployment and the benchmark gates must be read statistically; only in
that fallback does purchasing enough repetitions for mean separation become
the relevant route.

## Spend, authority, and durable evidence

| Spend class | Actual USD | Authorized ceiling |
|---|---:|---:|
| Local candidate probes | 0.000000 | — |
| Hosted panel calls | 0.000000 | — |
| **Total** | **0.000000** | **6.000000** |
| **Unused authority** | **6.000000** | — |

No threshold, gate definition, model, prompt, seat, graph state, service
configuration, or benchmark source was changed. No 57-repeat campaign was run.

Durable evidence:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T123634593665-n-benchnondet/nondeterminism_probe.json` — serial/concurrent single-row measurements, original/reversed/original check, three serial full-corpus measurements, hashes, scores, and zero-cost receipt;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T123634593665-n-benchnondet/request_envelope_probe.json` — complete normalized provider-bound request, three identical request hashes, and three different outputs;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T123634593665-n-benchnondet/serial_full_1.json`, `serial_full_2.json`, and `serial_full_3.json` — complete ordered candidate judgments for the serial 40-row passes;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T123634593665-n-benchnondet/nondeterminism-probe.log` and `request-envelope-probe.log` — complete execution logs;
- `/home/ITER/mcintos/Code/imas-ambix/deepseek-v4-flash-1252401.log` — live serving configuration and startup receipt.

## Disposition

The benchmark still measures a distribution under the current local serving
contract. The evidence does not support treating a near-threshold single run
as a stable value, and the graph-wide advisory census remains unauthorized by
this diagnosis. The next concrete action is a controlled deterministic-serving
experiment; not a runner serialization, prompt change, threshold change, or
57-repeat spend.
