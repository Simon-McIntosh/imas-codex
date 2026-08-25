# Brier calibration stability under pinned decoding

Date: 2026-08-25  
Source revision: `a8af5730`  
Candidate: `hosted_vllm/deepseek-v4-flash`  
Candidate decoding: temperature `0.0`, request seed `20260825`  
Corpus: `tests/standard_names/eval_sets/advisory_paired_corpus.json`  
Corpus SHA-256: `216e738a7ffbb7ae54b13e7e45f2fc5b1f99d34caba8b644621a931891fdae23`  
Brier threshold: at most `0.103763671875`  
Remaining authorised ceiling at entry: USD `9.0382496`

## Outcome

Three independent executions of the same pinned 40-row frozen batch produced
Brier errors of **0.121516**, **0.071531**, and **0.098832**. The observed
band is therefore **0.071531–0.121516**, with mean **0.097293** and spread
**0.049984**. The fixed threshold **0.103764 is INSIDE that band**.

**Verdict: gate is unresolvable at this sample size because the threshold lies inside the run-to-run band.**

This is neither a declared pass nor a declared fail. Under the lower-is-better
gate definition, a genuine pass would require every observed Brier value to be
at or below the threshold; a genuine fail would require every observed value
to be above it. One pass is above the threshold and two are below it.

| Independent pass | Brier error | Critical-defect recall | Clean-name false-positive rate | Good names accepted | Bad names rejected |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.121515625 | 1.000000 | 0.250000 | 15/20 | 20/20 |
| 2 | 0.071531250 | 1.000000 | 0.300000 | 14/20 | 20/20 |
| 3 | 0.098832031 | 1.000000 | 0.350000 | 13/20 | 20/20 |
| **Minimum** | **0.071531250** | **1.000000** | **0.250000** | — | — |
| **Maximum** | **0.121515625** | **1.000000** | **0.350000** | — | — |
| **Mean** | **0.097292969** | **1.000000** | **0.300000** | — | — |
| **Spread** | **0.049984375** | **0.000000** | **0.100000** | — | — |

Critical-defect recall did not move: every pass rejected all 20 corrupted
identities. Clean-name false-positive rate **did move**, from 0.25 through 0.30
to 0.35, contrary to the hoped-for stability check, although all three values
remain safely below the unchanged 0.50 ceiling. That movement is part of the
same candidate-score variability that widens the Brier band and must not be
worded into stability.

## Exact prompt and decoding identity

All three passes received the same ordered rows, candidate temperature, request
seed, target, reasoning effort, and rendered messages. Each pass produced the
same four SHA-256 values in the same batch order:

| Batch | Exact rendered-message SHA-256 | Passes equal |
|---:|---|---:|
| 1 | `bd790f4351c4681b410304619da9b2849ac4fe557ad8cb1b44441e615d2a3b25` | 1 = 2 = 3 |
| 2 | `96558fc6677206ec0f2bee426e80ede1ca61fdb3304add512e32f18184104b09` | 1 = 2 = 3 |
| 3 | `5a451458a9ccf7ea7d70264d1f3a24de756d8f3adffe05b9dbdb24cdcb24d0ca` | 1 = 2 = 3 |
| 4 | `30bfba44bab283ac0d542c24c9084b0563a798c728543affaa8f440752fe76d3` | 1 = 2 = 3 |

The three current passes therefore isolate run-to-run model execution under an
identical request envelope. The candidate outputs are not deterministic even
with temperature and seed pinned: identical prompts produced materially
different scores. The fresh hosted panels were also run on every pass to keep
the full frozen-run protocol unchanged, but they do **not** enter the Brier
formula, which compares the candidate score directly with the corpus's fixed
binary label. The observed Brier spread therefore cannot be attributed to the
hosted comparison panel; it is candidate advisory-judge variability under the
pinned request.

## Representative row-level movement

The aggregate spread is carried by physically concrete rows rather than a
rounding artifact. Representative identities, descriptions, exact DD bindings,
and candidate scores are:

| Frozen identity | Label and description | DD source binding | Pass 1 | Pass 2 | Pass 3 |
|---|---|---|---:|---:|---:|
| `vertical_coordinate_of_shunt` | Good — signed vertical coordinate of the shunt's second terminal point in the cylindrical frame. | `dd:magnetics/shunt/position/second_point/z` | 0.1875 | 1.0000 | 0.7000 |
| `ratio_of_coolant_mass_to_time` | Good — net coolant mass throughput per unit time across a flow boundary. | `dd:balance_of_plant/power_plant/system/component/port/mass_flow` | 0.5625 | 0.6000 | 0.3000 |
| `total_gas_source_rate_at_midplane_due_to_gas_injection` | Good — unweighted neutral-gas injection rate assigned to the vessel midplane. | `dd:summary/gas_injection_rates/midplane/value` | 1.0000 | 0.4750 | 0.6625 |
| `parallel_counter_passing_fast_particle_pressure` | Corrupted identity — source actually describes coherent-wave fast-ion power absorption per unit volume. | `dd:waves/coherent_wave/profiles_1d/ion/power_density_fast_n_phi` | 0.1875 | 0.6250 | 0.2500 |

For example, `vertical_coordinate_of_shunt` contributes `0.660156`, `0`, and
`0.090000` to the three per-row squared errors. That single known-good row's
movement is much larger than the full-run threshold margin, demonstrating why
the one-run 0.105455 near-miss was not a settled regression.

## Repeats needed to separate the mean from the threshold

The pilot mean is `0.097292969`, its sample standard deviation is
`0.025027704`, and its distance below the threshold is `0.006470703`. Assuming
future runs are independent and retain the observed variance, a two-sided 95%
Student-t interval around the mean first becomes narrower than that distance at
**60 total repeats**, or **57 additional repeats** beyond these three.

This is a mean-separation power estimate, not a guarantee that a future
min-to-max band will exclude the threshold: no finite repeat count can
guarantee an empirical range will not widen. The estimated 57 additional
repeats are not authorised by this node or by the remaining ceiling. Until a
60-run total is separately authorised and available, or the source of
candidate nondeterminism is removed, the only supported gate classification is
the exact unresolvable verdict above.

## Spend and authority

| Spend class | Actual USD | Remaining ceiling at entry |
|---|---:|---:|
| Three pinned local candidate passes | 0.000000 | — |
| Three fresh three-seat hosted panels | 2.652480 | — |
| **Total** | **2.652480** | **9.038250** |
| Remaining authority after measurement | 6.385769 | — |

The authorised amount was a ceiling, not a target. The protocol required
exactly three repeats, all completed without missing rows, and no further call
was justified after the band classification and repeat estimate were obtained.
No threshold, prompt, model, decoding control, gate formula, graph state,
review record, or budget node was changed.

Durable raw evidence:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093137986685-n-brierstability/brier_stability_receipt.json` — pass metrics, band statistics, hashes, verdict, repeat estimate and spend;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093137986685-n-brierstability/frozen_pass_1.json`, `frozen_pass_2.json`, and `frozen_pass_3.json` — complete ordered candidate and hosted-panel judgments;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093137986685-n-brierstability/score_checkpoints/` — three candidate and nine hosted-seat checkpoints;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093137986685-n-brierstability/row_variability.json` — all 40 identities with per-pass scores and squared-error contributions;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093137986685-n-brierstability/brier-stability.log` and `receipt-verification.log` — execution and independent receipt verification.

## Disposition

The 0.105455 Brier miss is not established as a real regression. It lies inside
a measured run-to-run band nearly thirty times wider than the original
0.001691 miss margin. This node does not authorize the graph-wide advisory
census and does not reinterpret the separate hosted-quorum failure.
