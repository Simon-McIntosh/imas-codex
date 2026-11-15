# One refinement round for the local model on signal enrichment

## Goal

Find out whether the locally served model can do the signals discovery
pipeline's enrichment reliably and well once the pipeline is fitted to it, and
measure the best configuration found against the paid model's existing results.

## Why

The first comparison ran the local model with the pipeline's defaults: batches
of ten signals, the non-thinking mode, and the prompt written for other models.
It delivered 130 of 200 descriptions, with none of its 20 raw responses parsing
as the pipeline's structured output without repair, and seven batches failing
outright. The lead believes one round of prompt and pipeline refinement, with
the correct context, will do much better. That round has not been run, so the
first result says the defaults do not suit the local model, not that the model
cannot do the work.

## What exists

- `scripts/benchmark_signal_enrichment.py` replays the production enrichment
  stage for a fixed cohort with frozen context and writes only to a private
  store. Its tests are in `tests/discovery/test_signal_enrichment_benchmark.py`.
- The first comparison's cohort (200 already-enriched signals of one other
  facility across 26 physics domains), its frozen prompts, the paid model's 200
  descriptions and the judge's verdicts are in the private store under
  `~/.cache/imas-codex-codac-uda/enrichment-benchmark/`. Reuse the cohort, the
  frozen context and the paid arm's outputs. Do not re-run the paid arm.
- The failure classes of the local arm are recorded in the first comparison's
  evidence fragment: invalid JSON, a string where an array was required, and a
  physics-domain value outside the enumeration.

## What to try

Change one thing at a time and record each variant; then combine the ones that
help.

1. Batch size: ten (the baseline), five, three and one signal per call.
2. Structured output: whatever the local route supports for constraining the
   response to the schema (a JSON-schema response format, guided or grammar
   decoding, or a JSON mode), established by reading the route's own
   capabilities, not assumed.
3. Thinking: the local model's reasoning mode on and off, with the token bound
   raised where reasoning needs it.
4. Prompt: a version of the enrichment prompt fitted to this model, with the
   enumerations stated in full next to the field that uses them, one worked
   example of a valid response, and the output schema restated last. Keep the
   context and the information content identical to the baseline prompt.
5. Repair: one bounded re-ask of a response that fails validation, quoting the
   validation error, counted separately from first-attempt success.

## Rules

- The local model only. No ITER signal and no ICP or IDM content is used.
- Each variant is run on the same 200 signals with the same frozen context.
- Nothing is written to the graph; the other facility's stored enrichment stays
  byte-identical, shown by the same digest check as before.
- A variant's parse rate is measured on the raw response before any repair.
- Changes to the production enrichment stage are made behind the existing
  per-seat configuration, so other seats and other models keep their current
  behaviour; the baseline arm must reproduce the first comparison's numbers
  within the run-to-run variation you measure by running it twice.
- Measurement runs execute on a compute partition:
  `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`, logs under
  `~/.cache/imas-codex-codac-uda/`.

## Done when

The evidence fragment gives, for the baseline and each variant and for the best
combination:

- the share of raw responses that parse as the structured output without
  repair, the share delivered after the one re-ask, and the count of failed
  signals by failure class;
- calls per minute, wall time and token counts;
- for the best combination only: the blind pairwise judgement against the paid
  model's existing descriptions on all 200 signals, with forty pairs read by
  hand and sealed before the judge's verdicts are read, the reversal control on
  forty pairs, and the count of descriptions wrong against the signal's
  context for each model;
- a plain statement of which changes mattered and by how much, and of what the
  best combination costs in wall time for one thousand signals.

One figure shows parse rate and delivery rate for every variant, saved through
`imas_codex.plotting`'s save function, rendered and looked at.

Tests cover each new configuration option and the re-ask path. Negative
control: a response that fails validation twice is counted as failed and never
as delivered; and removing the schema constraint from the best combination
lowers its measured raw parse rate, shown by the baseline arm.

## Scope

Write paths: `imas_codex/discovery/signals/`, `imas_codex/discovery/base/`,
`imas_codex/llm/` (the enrichment prompt templates), `pyproject.toml` (the
`discovery-describe` seat's options only), `scripts/`, `tests/discovery/`, the
evidence fragment
`docs/evidence/fragments/iter-signal-context-and-enrichment/refine-local-model-signal-enrichment.html`
and the figure directory
`docs/figures/iter-signal-context-and-enrichment/refine-local-model-signal-enrichment`.

Commit locally with a conventional subject and a body. Do not push or merge.
