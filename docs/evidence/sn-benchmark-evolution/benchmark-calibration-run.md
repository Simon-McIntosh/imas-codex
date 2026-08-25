# Frozen advisory benchmark calibration and candidate result

Run date: 2026-08-25  
Source revision: `bbd9b671`  
Candidate: `hosted_vllm/deepseek-v4-flash`  
Hosted reference panel: `openrouter/x-ai/grok-4.5`, `openrouter/openai/gpt-5.6-luna`, `openrouter/anthropic/claude-sonnet-5`  
Reasoning effort: `medium`  
Hard cost ceiling: USD 10.00

## Outcome

The frozen 40-row paired corpus was calibrated with two controlled passes of every hosted seat before the candidate result was opened. The candidate passed critical-defect recall, clean-name false-positive rate, and Brier calibration, but failed repeated-label agreement and hosted-quorum agreement. The result is therefore **fail** overall, not a qualified pass.

| Gate | Direction | Two-pass weakest-seat threshold | Candidate measurement | Verdict |
|---|---:|---:|---:|---:|
| Critical-defect recall | at least | 1.000000 | 1.000000 | **pass** |
| Clean-name false-positive rate | at most | 0.500000 | 0.275000 | **pass** |
| Brier calibration error | at most | 0.103764 | 0.097836 | **pass** |
| Repeated-label agreement | at least | 0.900000 | 0.825000 | **fail** |
| Hosted-quorum agreement | at least | 0.909091 | 0.800000 | **fail** |

**The graph-wide advisory census is not authorised by this result because two of the five locked gates failed.**

The apparent strength on defect recall does not cancel the two failures. DeepSeek rejected every deterministic bad identity in both candidate passes, but its accept/reject label changed on 7 of 40 rows and its primary label disagreed with the fresh hosted-panel median on 8 of 40 rows. All seven repeatability flips were known-good identities, so the instability is operationally relevant to a census that would rank accepted graph names for quality triage.

## Calibration authority

The threshold policy was fixed before candidate scoring: the candidate must be no worse than the weakest production hosted seat on the same independently labelled corpus. Each hosted seat reviewed the same 40 rows twice in the same order. Scores at or above the production boundary of 0.85 were treated as acceptance labels.

For recall and repeatability the threshold is the minimum hosted-seat value. For false-positive rate and Brier error it is the maximum hosted-seat value. For quorum agreement, each hosted seat was compared with the other two only on rows where those two agreed, and the minimum seat agreement became the threshold. The candidate's primary pass was then compared with the median label of a fresh full-panel pass.

| Hosted seat, two controlled passes | Critical recall | Clean FPR | Brier | Repeatability | Leave-one-seat-out quorum agreement | Comparable quorum rows |
|---|---:|---:|---:|---:|---:|---:|
| Grok 4.5 | 1.000000 | 0.400000 | 0.087072 | 1.000000 | 0.985915 | 71 |
| GPT-5.6 Luna | 1.000000 | 0.325000 | 0.103764 | 0.975000 | 0.972222 | 72 |
| Claude Sonnet 5 | 1.000000 | 0.500000 | 0.091387 | 0.900000 | 0.909091 | 77 |
| **Derived weakest-seat envelope** | **1.000000 minimum** | **0.500000 maximum** | **0.103764 maximum** | **0.900000 minimum** | **0.909091 minimum** | — |

These values derive from the committed controlled corpus, not from the circular graph-history bounds. The earlier graph-history figures were useful only to show that each formula was calculable; they were excluded because lifecycle outcomes came from the same review/refinement system, repeat coverage was opportunistic, prompts and grammar revisions varied, and the third seat was preferentially present on disputed rows. None of the provisional history bounds—recall 0.8114, FPR 0.2472, Brier 0.2441, repeatability 0.7143, quorum agreement 0.6945—was used as a threshold here.

## Frozen population and hashes

The controlled population is `tests/standard_names/eval_sets/advisory_paired_corpus.json`: 40 unique identities comprising 20 accepted-valid good rows and 20 deterministic critical-defect twins across 18 physics domains. Every row carries its DD path, proposed Standard Name identity, unit, description, documentation, binary label, defect class, and per-row provenance. The corpus declares DD 4.1.1, ISN 0.8.0rc67, seed 20260825, and a hashed read-only extraction query over 1,894 eligible graph identities.

The following are three distinct hashes with different meanings:

| Frozen object | SHA-256 |
|---|---|
| Population in committed corpus order | `216e738a7ffbb7ae54b13e7e45f2fc5b1f99d34caba8b644621a931891fdae23` |
| Ordered runner inputs after fixed-seed domain stratification | `00fff70bb8e5ce66253ebc3ebdc48fbf02c3f90349e09de7fe62558c6b20a44c` |
| Ordered primary candidate and fresh-panel results | `b229b0d26e0058c867676666e4a73864cdc625b9ab976d72eda6e4987cdbb3e2` |

All six calibration seat checkpoints cover exactly 40 judgments and carry the population hash. Both candidate checkpoints and all three fresh comparison-panel checkpoints cover exactly 40 judgments and carry the ordered-input hash. The raw frozen report contains the complete ordered inputs, candidate judgments, three per-seat judgments, panel median, spread, and contested state for every row.

## Representative failures

The failures are not abstract score differences. They affect accepted identities with exact source-path bindings:

| Accepted identity | DD source binding | Primary candidate score | Repeat score | Fresh panel median | Observed issue |
|---|---|---:|---:|---:|---|
| `vorticity` | `dd:mhd/ggd/vorticity/values` | 0.3750 | 1.0000 | 0.9000 | Candidate label flipped and primary disagreed with quorum. |
| `accumulated_deuterated_methane_prefill_count` | `dd:summary/gas_injection_prefill/methane_deuterated/value` | 1.0000 | 0.6875 | 0.7750 | Candidate label flipped; the fresh panel was also below the boundary. |
| `total_gas_source_rate_at_midplane_due_to_gas_injection` | `dd:summary/gas_injection_rates/midplane/value` | 0.6000 | 0.8625 | 0.9625 | Candidate label flipped and primary disagreed with quorum. |
| `normalized_radial_gyrocenter_energy_flux_at_flux_surface_due_to_perturbed_parallel_vector_potential` | `dd:gyrokinetics_local/non_linear/fluxes_1d_rotating_frame/energy_a_field_parallel` | 0.6875 | 0.9375 | 0.9125 | Candidate label flipped and primary disagreed with quorum. |
| `flux_surface_averaged_boron_density_at_plasma_boundary` | `dd:summary/local/separatrix_average/n_i/boron/value` | 1.0000 | 1.0000 | 0.7625 | Candidate was stable but disagreed with the fresh panel median. |

Seven identities crossed the 0.85 boundary between the two candidate passes: `vorticity`, `x_coordinate_of_diagnostic_aperture`, `accumulated_deuterated_methane_prefill_count`, `major_length_of_ferritic_element`, `total_gas_source_rate_at_midplane_due_to_gas_injection`, `vertical_outline_of_plasma_facing_component`, and `normalized_radial_gyrocenter_energy_flux_at_flux_surface_due_to_perturbed_parallel_vector_potential`.

## Spend and execution receipt

| Spend class | Actual USD | Authorised ceiling |
|---|---:|---:|
| Candidate, two local DSv4 passes | 0.000000 | — |
| Judging, two calibration panels plus one fresh comparison panel | 2.544742 | — |
| **Total** | **2.544742** | **10.000000** |
| Remaining authority, unused | 7.455258 | — |

The authorised amount was a ceiling, not a target. Only USD 2.544742 was used because the fixed protocol completed without provider retries or missing rows; there was no reason to consume the remaining USD 7.455258. Candidate and judging costs come from the scorer returns captured in the artifact-local checkpoints. No graph budget settlement or review persistence path was called.

Durable raw evidence is held in the run envelope:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070656329808-n-benchmarkcalibration/benchmark_calibration_receipt.json` — derived threshold, gate, hash, model, and spend matrix;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070656329808-n-benchmarkcalibration/frozen_benchmark_report.json` — ordered 40-row primary candidate and fresh-panel result;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070656329808-n-benchmarkcalibration/score_checkpoints/` — 11 exact model-phase checkpoints covering two calibration panels, two candidate passes, and one comparison panel;
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070656329808-n-benchmarkcalibration/benchmark-calibration.log` — complete execution log with exit marker recorded by the launch session.

## Disposition

The result is scientifically useful but does not unlock the next operational step. Critical-defect sensitivity and aggregate calibration are strong on this corpus, while candidate stability and agreement with the hosted reference panel are below the production-seat envelope. The graph-wide advisory census remains held; any later retry must be a newly identified benchmark run with its own ordered-result hash and spend, not a reinterpretation of these failed gates.
