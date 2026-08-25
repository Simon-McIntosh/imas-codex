# Frozen-row repeatability diagnosis

Date: 2026-08-25  
Candidate: `hosted_vllm/deepseek-v4-flash`  
Recorded source revision: `bbd9b671`  
Corpus: `tests/standard_names/eval_sets/advisory_paired_corpus.json`  
Acceptance boundary: score at least `0.85`  
Repeatability threshold: at least `0.900`

## Conclusion

The candidate changed its accept/reject label on **7 of the frozen 40 rows**,
so repeated-label agreement was **33/40 = 0.825**. All **7 unstable rows were
known-good identities**; **0 of 20 corrupted rows** were unstable. Instability
therefore concentrates entirely in the good half of the corpus: **7/20 = 35%**
of good rows flipped, against **0/20 = 0%** of corrupted rows.

The strongest mechanical cause is **uncontrolled sampling at the local vLLM
endpoint**. The review call supplies reasoning effort but neither temperature
nor a request seed. The installed vLLM stack consequently uses
`temperature=1.0` and `seed=None`. That is a direct mechanism for different
generations from the same prompt. It is strongly supported, but not proven by
a causal intervention: the original checkpoints preserve the ordered raw-row
hash, not the rendered messages or full request body, and this diagnosis did
not spend another model run to compare default sampling with greedy or seeded
sampling.

The gate requires **at least 36/40 repeated labels = 0.900**. Relative to the
recorded **33/40**, the candidate must make **at least 3 additional row pairs
agree**, reducing unstable rows from **7 to at most 4**. The graph-wide advisory
census remains unauthorized by the recorded result.

## Evidence inputs and method

The diagnosis compares the two durable candidate checkpoints from the same
benchmark run:

- primary:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070656329808-n-benchmarkcalibration/score_checkpoints/candidate_primary-3e98b50a1381.json`;
- repeat:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070656329808-n-benchmarkcalibration/score_checkpoints/candidate_repeat-3e98b50a1381.json`;
- frozen ordered inputs and source bindings:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070656329808-n-benchmarkcalibration/frozen_benchmark_report.json`;
- gate receipt:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T070656329808-n-benchmarkcalibration/benchmark_calibration_receipt.json`.

Both checkpoints contain exactly 40 judgments, use the same model, target and
`medium` reasoning effort, and carry the same ordered input hash:
`00fff70bb8e5ce66253ebc3ebdc48fbf02c3f90349e09de7fe62558c6b20a44c`.
Their row order is identical. A row disagreement is `1` when the two scores
fall on opposite sides of `0.85`, otherwise `0`; with only two recorded passes,
each row can contribute at most one pairwise disagreement.

The repeated scores changed more broadly than the binary gate exposes:

- only **12/40** numeric scores were exactly equal;
- only **9/40** four-dimension score tuples were exactly equal;
- **0/40** complete judgment objects were byte-equivalent after JSON parsing;
- the good-row mean moved from **0.7950** to **0.8825**;
- the corrupted-row mean moved from **0.1975** to **0.32375**;
- six label flips were reject-to-accept and one was accept-to-reject;
- the seven absolute score changes range from **0.1750** to **0.6250**, with a
  median of **0.3125**.

This is not merely boundary rounding. Across the seven flips, the nearer of the
two scores to `0.85` was between `0.0125` and `0.1500` away, with a median
distance of `0.0875`; several rows moved across most of the scoring range.

## Unstable rows

The dimension columns below are grammar / semantic / convention /
completeness, each out of 20. The failures span all four 10-row calls rather
than a single malformed batch: the batch disagreement counts are **1, 1, 2,
3**.

| Row / batch | Standard Name and DD source binding | Score and label change | Dimension totals |
|---|---|---|---|
| 10 / 1 | `vorticity`<br>`dd:mhd/ggd/vorticity/values` | 0.3750 reject -> 1.0000 accept | 0/10/10/10 -> 20/20/20/20 |
| 14 / 2 | `x_coordinate_of_diagnostic_aperture`<br>`dd:camera_ir/channel/camera/pinhole/x` | 0.5500 reject -> 0.9500 accept | 10/14/10/10 -> 20/18/20/18 |
| 24 / 3 | `accumulated_deuterated_methane_prefill_count`<br>`dd:summary/gas_injection_prefill/methane_deuterated/value` | 1.0000 accept -> 0.6875 reject | 20/20/20/20 -> 12/16/12/15 |
| 27 / 3 | `major_length_of_ferritic_element`<br>`dd:ferritic/object/axisymmetric/oblique/length_alpha` | 0.7500 reject -> 0.9250 accept | 18/12/18/12 -> 20/16/20/18 |
| 31 / 4 | `total_gas_source_rate_at_midplane_due_to_gas_injection`<br>`dd:summary/gas_injection_rates/midplane/value` | 0.6000 reject -> 0.8625 accept | 10/14/12/12 -> 15/18/18/18 |
| 34 / 4 | `vertical_outline_of_plasma_facing_component`<br>`dd:wall/description_2d/mobile/unit/outline/z` | 0.4750 reject -> 0.9375 accept | 12/8/10/8 -> 20/18/19/18 |
| 36 / 4 | `normalized_radial_gyrocenter_energy_flux_at_flux_surface_due_to_perturbed_parallel_vector_potential`<br>`dd:gyrokinetics_local/non_linear/fluxes_1d_rotating_frame/energy_a_field_parallel` | 0.6875 reject -> 0.9375 accept | 14/15/10/16 -> 20/18/19/18 |

All seven have `label=1` in the controlled corpus. The bad twins did change
scores in many cases, but none crossed the acceptance boundary; the candidate
rejected every corrupted row on both passes.

## Mechanical cause assessment

### Primary cause: default stochastic sampling

Evidence **for** this cause:

1. `score_with_reviewer()` calls `acall_llm_structured()` with the model,
   messages, response schema, service, reasoning effort and retry count, but no
   `temperature` (`imas_codex/standard_names/benchmark.py:698-711`). There is no
   request-seed argument on that path.
2. The shared request builder adds a temperature field only when its caller
   supplies a non-null value (`imas_codex/discovery/base/llm.py:2042-2048`). It
   has no request-seed parameter (`imas_codex/discovery/base/llm.py:1900-1911`).
3. The installed serving stack defines ordinary sampling as
   `temperature: float = 1.0` and `seed: int | None = None`
   (`vllm/sampling_params.py:236-250`). The Ambix vLLM launch path does not
   override generation temperature or request seeds
   (`imas_ambix/agent/slurm.py:164-218`). The active model profile selects the
   native PyTorch sampler path and does not declare generation defaults
   (`imas_ambix/agent/profiles/deepseek-v4-flash.toml:39-66`).
4. The two candidate checkpoints have the same ordered-input hash and identical
   row order, yet 28/40 numeric scores changed. The changes affect every scoring
   dimension and all four batches, matching sampling variation better than a
   single corrupt row, one malformed batch, or threshold quantization.
5. A no-provider reconstruction was run twice against the frozen ordered input.
   It produced byte-identical rendered-message hashes for all four batches:

   | Batch | Rendered-message SHA-256 | Serialized bytes |
   |---:|---|---:|
   | 1 | `f06b06eed2bf454a86654b38f74415a62ff29ca31ae1bc4a78ed29263f3533ec` | 79,019 |
   | 2 | `eb3bb5f13da00cc42a8587bab2af3f3f7519b01b171b489cfd3b79bd2057afdc` | 79,006 |
   | 3 | `ba26ac61b79eb525ee45a40359bf97529523e8a4a8f5d460c4bf5f5b2b9491bc` | 79,194 |
   | 4 | `3b2adf5bfff94f79feb2b03cf0ca751ad41698c91c6b68e3c282fdb53ff0e47b` | 79,978 |

Evidence **against, or not yet closed**:

1. The original checkpoints hash the raw ordered candidates at
   `run_benchmark_calibration.py:70-80`; they do **not** preserve the rendered
   messages, neighbor lists, scored examples, endpoint generation defaults or
   full request body. Exact prompt equality during the original two calls is
   therefore inferred, not directly recorded.
2. `score_with_reviewer()` reconstructs graph-backed examples and neighbors on
   each pass. A graph change between the primary and repeat calls could in
   principle alter the rendered prompt without changing the checkpoint's input
   hash. The immediate reconstruction above found no such nondeterminism, and
   the relevant source files have not changed since the recorded revision, but
   that cannot retroactively prove the graph context used by the original calls.
3. No controlled intervention compared default sampling against
   `temperature=0` or a fixed request seed. Even greedy decoding can retain
   hardware or kernel nondeterminism, so pinning sampling must be measured, not
   assumed to produce 1.000 repeatability.

The diagnosis is therefore: **default stochastic sampling is the primary,
high-confidence mechanical cause; live-context drift is a lower-confidence
alternative left open by incomplete request provenance.**

## Complete per-row disagreement ledger

| Row | Truth | Standard Name | Primary | Repeat | Disagreements |
|---:|---|---|---:|---:|---:|
| 1 | good | `radial_coordinate_of_neutral_beam_injector` | 1.0000 accept | 1.0000 accept | 0 |
| 2 | good | `helium_4_density_at_divertor_target` | 1.0000 accept | 1.0000 accept | 0 |
| 3 | good | `flux_surface_averaged_boron_density_at_plasma_boundary` | 1.0000 accept | 1.0000 accept | 0 |
| 4 | good | `tilt_angle_of_beam_tracing_beam` | 0.1875 reject | 0.1875 reject | 0 |
| 5 | corrupted | `of_flux_surface` | 0.0000 reject | 0.0000 reject | 0 |
| 6 | corrupted | `count_prefill_methane_deuterated_accumulated` | 0.0625 reject | 0.3250 reject | 0 |
| 7 | good | `breakdown_initial_time` | 1.0000 accept | 1.0000 accept | 0 |
| 8 | good | `vertical_coordinate_of_shunt` | 1.0000 accept | 0.9000 accept | 0 |
| 9 | corrupted | `element_ferritic_of_length_major` | 0.0250 reject | 0.0500 reject | 0 |
| 10 | good | `vorticity` | 0.3750 reject | 1.0000 accept | **1** |
| 11 | good | `ratio_of_coolant_mass_to_time` | 0.5750 reject | 0.5750 reject | 0 |
| 12 | corrupted | `number_atomic_ion_fast` | 0.0750 reject | 0.2500 reject | 0 |
| 13 | corrupted | `helium_4_density_at_internal_transport_barrier` | 0.1875 reject | 0.4625 reject | 0 |
| 14 | good | `x_coordinate_of_diagnostic_aperture` | 0.5500 reject | 0.9500 accept | **1** |
| 15 | good | `deuterium_deuterium_emissivity_due_to_fusion` | 0.9500 accept | 0.8875 accept | 0 |
| 16 | corrupted | `z_first_measurement_direction_unit_vector_of_strain_gauge` | 0.2625 reject | 0.5125 reject | 0 |
| 17 | good | `parallel_magnetic_field` | 0.9250 accept | 0.9500 accept | 0 |
| 18 | corrupted | `potential_vector_parallel_perturbed_to_due_surface_flux_at_flux_energy_gyrocenter_radial_normalized` | 0.0000 reject | 0.0000 reject | 0 |
| 19 | corrupted | `current_of_divertor_tile` | 0.2000 reject | 0.4625 reject | 0 |
| 20 | corrupted | `density_at_divertor_target` | 0.6250 reject | 0.6500 reject | 0 |
| 21 | corrupted | `boundary_plasma_at_density_boron_averaged_surface_flux` | 0.1000 reject | 0.1000 reject | 0 |
| 22 | corrupted | `neoclassical_tearing_mode_phase` | 0.1875 reject | 0.1875 reject | 0 |
| 23 | good | `elongation_of_flux_surface` | 0.9500 accept | 0.9500 accept | 0 |
| 24 | good | `accumulated_deuterated_methane_prefill_count` | 1.0000 accept | 0.6875 reject | **1** |
| 25 | corrupted | `thomson_scattering_laser_pulse_energy_at_outlet` | 0.0750 reject | 0.1750 reject | 0 |
| 26 | corrupted | `coordinate_of_shunt` | 0.6250 reject | 0.7000 reject | 0 |
| 27 | good | `major_length_of_ferritic_element` | 0.7500 reject | 0.9250 accept | **1** |
| 28 | corrupted | `upper_wavelength_of_filter` | 0.4000 reject | 0.5000 reject | 0 |
| 29 | corrupted | `of_coolant_mass_to_time` | 0.0000 reject | 0.0000 reject | 0 |
| 30 | good | `fast_ion_atomic_number` | 0.8750 accept | 1.0000 accept | 0 |
| 31 | good | `total_gas_source_rate_at_midplane_due_to_gas_injection` | 0.6000 reject | 0.8625 accept | **1** |
| 32 | corrupted | `coordinate_of_diagnostic_aperture` | 0.4500 reject | 0.6000 reject | 0 |
| 33 | corrupted | `fusion_to_due_emissivity_deuterium_deuterium` | 0.0500 reject | 0.2125 reject | 0 |
| 34 | good | `vertical_outline_of_plasma_facing_component` | 0.4750 reject | 0.9375 accept | **1** |
| 35 | corrupted | `magnetic_field` | 0.4000 reject | 0.5750 reject | 0 |
| 36 | good | `normalized_radial_gyrocenter_energy_flux_at_flux_surface_due_to_perturbed_parallel_vector_potential` | 0.6875 reject | 0.9375 accept | **1** |
| 37 | corrupted | `parallel_counter_passing_fast_particle_pressure` | 0.1750 reject | 0.6125 reject | 0 |
| 38 | good | `carbon_density_at_divertor_target` | 1.0000 accept | 0.9625 accept | 0 |
| 39 | good | `per_toroidal_mode_fast_ion_power_density` | 1.0000 accept | 0.9375 accept | 0 |
| 40 | corrupted | `4_density_at_divertor_target` | 0.0500 reject | 0.1000 reject | 0 |

Ledger sum: **7 disagreements**, **33 agreements**, **40 rows**.

## Next measurement

A causal confirmation should keep the same 40-row order and rendered-message
hashes, checkpoint the complete request envelope, and compare two passes under
an explicitly pinned decoding policy. The result, not the setting alone, must
show at least **36/40 = 0.900** agreement. If prompt hashes diverge, freeze the
graph-derived examples and neighbor context before attributing the residual to
the model. This requires changes and model execution outside this evidence-only
write scope.
