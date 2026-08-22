NEEDS-HELP: The one permitted five-identity compose call completed, but its result was lost by the evidence wrapper before durable serialization, so four proposals and all five score verdicts cannot be recovered without an unauthorized second compose attempt.

tried: Ran one configured `sn-compose` structured call containing exactly the five required source-grounded identities, with compose retries disabled. The provider returned a valid `StandardNameComposeBatch`; post-processing then raised while rendering the first strict-grammar-invalid candidate. Searched the run logs, the LiteLLM service log, recent Langfuse observations, and local IMAS Codex logs for the completed structured response. None retained the response body.

options: (1) authorize one replacement compose call after fixing the wrapper to serialize the raw Pydantic response before any candidate rendering; (2) accept this as the single draw and close the node with four unrecoverable proposal rows and zero review draws; or (3) recover the original response from an operator-managed vLLM request trace if one exists outside the worker-visible logs.

leaning: Option 1, because it is the only route that can produce the required five proposed spellings and threshold verdicts through the ordinary review gate; the rerun must be explicitly authorized because the original provider call did occur.

cost-if-wrong: Authorizing option 1 spends one additional free local compose call and makes the five identities receive two compose attempts instead of the exact-one rule. Choosing option 2 permanently leaves four spellings and every score absent. Option 3 costs only trace-recovery effort if such a trace exists, but the worker-visible Langfuse and service logs already contain no response body.

# Five identity-gap compose and review batch

## Quantitative result

The node is **blocked** and did not meet its done-when measure.

| Measure | Required | Observed |
|---|---:|---:|
| Identity inputs in the sole compose request | 5 | **5** |
| Configured compose calls | exactly 1 per identity | **1 batch call containing each identity once** |
| Compose retries | 0 | **0** |
| Recoverable proposed spellings | 5 | **1 partial strict-grammar rejection; 4 unrecoverable** |
| Fresh quorum groups | at most 1 per identity | **0 total** |
| Identities drawn twice | 0 | **0** |
| Identities accepted | report result | **0** |
| Identities below threshold | report result | **0; no review was started** |
| Strict-grammar vocabulary-gap results | report exact rejected segment | **1 parser refusal, classifier `flat segment order`; no segment-level vocab-gap object survived** |
| Surviving `StandardName` or `StandardNameSource` claims | 0 | **0** |
| `LLMCost` nodes | before/after | **27,631 → 27,631 (delta 0)** |
| `StandardNameReview` nodes | before/after | **20,777 → 20,777 (delta 0)** |
| Attributable spend | at most USD 15 | **USD 0.00 / USD 15.00** |

The compose seat was the configured local `hosted_vllm/deepseek-v4-flash`
seat, so the completed call incurred USD 0.00. No graph staging, rename,
review, refinement, acceptance, attachment, detachment, or claim mutation was
performed.

## Per-identity record

| Required identity | Exact grounding supplied once | Proposed spelling | Strict grammar | Fresh score against 0.85 | Verdict |
|---|---|---|---|---:|---|
| Reflector center of curvature | `spectrometer_x_ray_crystal/channel/reflector/sphere_centre/phi`; explicitly distinguished from the reflector surface center/local-frame origin | **unrecoverable from completed response** | not recoverable | — | **no review draw** |
| State-resolved neutral momentum diffusion coefficient | `plasma_transport/model/profiles_1d/neutral/state/momentum/d_parallel`; parallel axis, neutral internal state, momentum owner, diffusion coefficient | **unrecoverable from completed response** | not recoverable | — | **no review draw** |
| Antenna-owned capacitance | `ic_antennas/antenna/module/matching_element/capacitance`; ICRH-antenna matching-element circuit ownership | **unrecoverable from completed response** | not recoverable | — | **no review draw** |
| Energy-tendency derivative | `summary/global_quantities/denergy_thermal_dt/value`; signed derivative of total thermal plasma internal energy | model rendering reached **`total_thermal_plasma_internal_energy`** before refusal | **rejected** by `_NonCanonicalParseError`: `name is not canonical: flat segment order renders as 'total_thermal_plasma_internal_energy'` | — | **vocabulary/grammar refusal; not reworded and no review draw** |
| Fusion neutron flux | `neutron_diagnostic/neutron_flux_total`; volume-integrated fusion-neutron production rate, unit `s^-1`, not power | **unrecoverable from completed response** | not recoverable | — | **no review draw** |

The parser exception does not expose a grammar segment identifier beyond
`flat segment order`, and the wrapper failed before the response's
`vocab_gaps` collection could be serialized. Reporting a guessed segment would
violate the exact-rejected-segment requirement, so the classifier and exact
exception are retained verbatim instead.

## Failure boundary

The failure occurred after the provider response had already parsed as
`StandardNameComposeBatch`, in the wrapper's loop over
`candidate.compose_name()`. That loop attempted to render every candidate
before writing `compose-once.json`; the first non-canonical candidate raised,
the process exited, and the in-memory response was lost. Re-running the script
would therefore be a second compose attempt, not a post-processing retry, and
was deliberately refused.

The direct call did not create `LLMCost` graph nodes for the free local seat.
Because no candidate was durably available to stage, the ordinary name-review
pool was never invoked and `StandardNameReview` remained unchanged.

## Runtime evidence

- Baseline graph census and exact source/name context:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T053323420721-identitybatch/preflight.log`
- Sole compose invocation and verbatim strict-parser traceback:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T053323420721-identitybatch/compose-once.log`
- The exact five-item wrapper used for the sole request:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T053323420721-identitybatch/compose_once.py`
- Langfuse recovery query, which returned no recent generation observations:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T053323420721-identitybatch/langfuse-recovery.log`
- Final graph counter and zero-claim census:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T053323420721-identitybatch/postflight.log`

No second provider request was made after the wrapper failure.
