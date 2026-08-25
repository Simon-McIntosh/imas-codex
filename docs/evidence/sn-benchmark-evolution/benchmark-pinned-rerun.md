# Pinned-decoding advisory benchmark rerun

Date: 2026-08-25  
Source revision: `625a90cf` (includes decoding-policy commit `3990b9c6`)  
Candidate: `hosted_vllm/deepseek-v4-flash`  
Candidate decoding: temperature `0.0`, request seed `20260825`  
Corpus: `tests/standard_names/eval_sets/advisory_paired_corpus.json`  
Acceptance boundary: score at least `0.85`  
Hard cost ceiling: USD `10.000000`

## Outcome

Pinning candidate decoding removed three of the seven recorded label flips:
repeated-label agreement improved from **33/40 = 0.825** to
**36/40 = 0.900**, exactly meeting the threshold derived from the weakest
hosted production seat. That closes the sampler-confounding failure, but it
does not make the benchmark pass overall. Brier calibration error is now
slightly above its maximum and agreement with the fresh hosted quorum remains
below its minimum, so the overall verdict is **fail**.

| Gate | Direction | Derived threshold | Pinned measurement | Prior measurement | Verdict |
|---|---:|---:|---:|---:|---:|
| Critical-defect recall | at least | 1.000000 | 1.000000 | 1.000000 | **pass** |
| Clean-name false-positive rate | at most | 0.500000 | 0.300000 | 0.275000 | **pass** |
| Brier calibration error | at most | 0.103764 | 0.105455 | 0.097836 | **fail** |
| Repeated-label agreement | at least | 0.900000 | 0.900000 | 0.825000 | **pass** |
| Hosted-quorum agreement | at least | 0.909091 | 0.800000 | 0.800000 | **fail** |

**The graph-wide advisory census is not authorised because only three of the five gates pass.**

The thresholds were not tuned after seeing this rerun. They are the two-pass
weakest-hosted-seat envelope derived in the prior controlled calibration over
the same committed 40-row population: the minimum hosted value for recall,
repeatability and quorum agreement, and the maximum hosted value for false
positives and Brier error. Reusing that frozen threshold authority isolates the
candidate decoding intervention while the fresh three-seat panel re-measures
candidate-to-quorum agreement.

## Prompt-equality receipt

Both candidate passes received the same 40 rows in the same order and emitted
the same four SHA-256 digests for the exact rendered system/user message pairs.
The digest lists are byte-for-byte equal across passes:

| Batch | Primary rendered-message SHA-256 | Repeat rendered-message SHA-256 | Equal |
|---:|---|---|---:|
| 1 | `f06b06eed2bf454a86654b38f74415a62ff29ca31ae1bc4a78ed29263f3533ec` | `f06b06eed2bf454a86654b38f74415a62ff29ca31ae1bc4a78ed29263f3533ec` | yes |
| 2 | `eb3bb5f13da00cc42a8587bab2af3f3f7519b01b171b489cfd3b79bd2057afdc` | `eb3bb5f13da00cc42a8587bab2af3f3f7519b01b171b489cfd3b79bd2057afdc` | yes |
| 3 | `ba26ac61b79eb525ee45a40359bf97529523e8a4a8f5d460c4bf5f5b2b9491bc` | `ba26ac61b79eb525ee45a40359bf97529523e8a4a8f5d460c4bf5f5b2b9491bc` | yes |
| 4 | `3b2adf5bfff94f79feb2b03cf0ca751ad41698c91c6b68e3c282fdb53ff0e47b` | `3b2adf5bfff94f79feb2b03cf0ca751ad41698c91c6b68e3c282fdb53ff0e47b` | yes |

The frozen population hash remains
`216e738a7ffbb7ae54b13e7e45f2fc5b1f99d34caba8b644621a931891fdae23`,
and the fixed-seed ordered-input hash remains
`00fff70bb8e5ce66253ebc3ebdc48fbf02c3f90349e09de7fe62558c6b20a44c`.
The pinned primary result plus fresh panel has ordered-result hash
`a3b5610e57eea5e786b049ec726e4491ed0a03591847a198a797e465744ab978`.

This directly closes the provenance caveat in the earlier diagnosis: the
repeatability figure now measures two candidate generations over proven-equal
rendered messages under the same explicit decoding envelope, rather than two
generations under an unpinned sampler.

## Residual disagreements

Four known-good identities still crossed the acceptance boundary between the
two pinned passes. They are representative of the residual judgment variance,
not input drift; each retains its exact description and DD source binding in
the frozen report.

| Accepted identity | Description | DD source binding | Primary | Repeat | Fresh panel median |
|---|---|---|---:|---:|---:|
| `ratio_of_coolant_mass_to_time` | Net coolant mass throughput per unit time across a flow boundary. | `dd:balance_of_plant/power_plant/system/component/port/mass_flow` | 0.8750 | 0.7500 | 0.5500 |
| `accumulated_deuterated_methane_prefill_count` | Cumulative pre-breakdown deuterated-methane inventory weighted by equivalent electrons. | `dd:summary/gas_injection_prefill/methane_deuterated/value` | 0.8250 | 1.0000 | 0.7500 |
| `major_length_of_ferritic_element` | Euclidean alpha-side length of an oblique ferritic-element cross-section. | `dd:ferritic/object/axisymmetric/oblique/length_alpha` | 0.8000 | 1.0000 | 0.5750 |
| `normalized_radial_gyrocenter_energy_flux_at_flux_surface_due_to_perturbed_parallel_vector_potential` | Gyro-Bohm-normalized radial gyrocenter energy flux due to parallel vector-potential fluctuations. | `dd:gyrokinetics_local/non_linear/fluxes_1d_rotating_frame/energy_a_field_parallel` | 0.4500 | 0.9500 | 0.9000 |

The repeatability pass is exact but threshold-tight: one additional label flip
would fail it. The Brier miss is `0.105455078125 - 0.103763671875 =
0.001691406250`; it reflects probability calibration across both candidate
passes, including score movement that does not cross the binary boundary. The
quorum failure remains 32/40 agreement, unchanged from the prior run, and is
independent of the newly closed prompt-equality question.

## Spend and execution receipt

| Spend class | Actual USD | Authorised ceiling |
|---|---:|---:|
| Candidate, two local pinned passes | 0.000000 | — |
| Judging, one fresh three-seat comparison panel | 0.961750 | — |
| **Total** | **0.961750** | **10.000000** |
| Remaining authority, unused | 9.038250 | — |

The USD 10 amount was a hard ceiling, not a target. The already-derived hosted
thresholds were reused because the intervention changed only candidate
decoding; rerunning the two calibration panels would have added paid calls
without changing what was being tested. Actual spend therefore consists only
of the fresh comparison panel. Candidate cost was USD 0.00 at the local
endpoint, and the protocol completed without provider retries or missing rows.

Durable raw evidence:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083220762235-n-benchmarkrerun/pinned_benchmark_receipt.json` — five gates, prior measurements, decoding envelope, hashes and spend;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083220762235-n-benchmarkrerun/frozen_pinned_benchmark_report.json` — ordered 40-row primary candidate and fresh-panel judgments;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083220762235-n-benchmarkrerun/score_checkpoints/` — two candidate and three hosted-seat checkpoints;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083220762235-n-benchmarkrerun/pinned-benchmark.log` — complete resumed execution output;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T083220762235-n-benchmarkrerun/receipt-verification.log` — independent receipt assertions.

## Disposition

Pinned decoding materially improves repeatability and proves that the prior
0.825 result was partly sampler-driven, but the candidate is not yet qualified
for a graph-wide advisory census. No graph mutation, review persistence, budget
settlement or advisory census was performed by this node.
