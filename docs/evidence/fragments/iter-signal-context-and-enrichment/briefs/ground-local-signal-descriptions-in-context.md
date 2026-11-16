# A refinement round for correctness of the local model's signal descriptions

## Goal

Make the locally served model's signal descriptions correct against the context
they were given, and measure the result by hand against the paid model's
existing descriptions.

## Why

The previous round made the local model reliable: 197 of 200 raw responses
parse and all 200 descriptions are delivered. It did not make them right. In a
blind reading of forty pairs, sealed before any automated verdict was read, 26
of the local model's forty descriptions were wrong against the signal's context
and none of the paid model's were; the paid description was preferred in 27
pairs and the local one in none. The dominant defect was detail that the
signal's context does not support. The lead asked for another round aimed at
correctness before deciding which model writes ITER descriptions.

## What exists

- `scripts/refine_signal_enrichment.py` and
  `scripts/benchmark_signal_enrichment.py` replay the production enrichment
  stage for a fixed cohort with frozen context and write only to a private
  store. Per-seat enrichment options live under
  `[tool.imas-codex.discovery-describe.signal-enrichment]`.
- The cohort (200 already-enriched signals of one other facility), its frozen
  context, the paid model's 200 descriptions, the previous round's local
  descriptions and its forty sealed hand readings are in the private store
  under `~/.cache/imas-codex-codac-uda/enrichment-benchmark/` and the previous
  run's private directory. Reuse them. Do not re-run the paid arm.

## What to try

Start from the landed best configuration (one signal per call, fitted prompt,
one re-ask, no schema constraint). Change one thing at a time, record each
variant, then combine what helps.

1. Grounding in the prompt: the description may state only what the supplied
   context supports; where the context does not say, the description says less,
   and never fills the gap from the signal's name or general knowledge.
2. Evidence with each claim: the model returns, beside the description, the
   context item each statement rests on; code drops any statement with no item
   or with an item that does not exist in the context, and counts the drops.
3. A second local pass: a separate call that receives the context and the
   draft description, and removes or corrects any statement the context does
   not support. Measure it with reasoning on and off.
4. Sampling: a lower temperature for the drafting call.
5. Length: a shorter description limit, since unsupported detail tends to
   arrive at the end.

## How correctness is measured

- The instrument is a hand reading against the signal's context, done blind to
  which variant wrote the description. The automated judge agreed with the
  previous hand reading on 17 of 40 pairs; report its figures, but do not rank
  variants by them.
- Read the same forty signals as the previous round, so the result compares
  directly with 26 wrong in 40, and a second forty chosen by a fixed hash
  before any variant is run, so the result is not fitted to the first forty.
- For screening variants, read a fixed twenty of the first forty for each
  variant; read all eighty for the best combination only.
- Seal every reading before any automated verdict for that variant is read.
- State, for the best combination: wrong, partly right and right in each
  forty; the blind preference against the paid description on both forties;
  the count of statements dropped by code or by the second pass; calls per
  signal, wall time and tokens, and the wall time for one thousand signals.
- The bar the plan proposes for the ITER pilot is at most four wrong in forty.
  Say plainly whether the best combination meets it on each forty.

## Rules

- The local model only for generation and for the second pass. No ITER signal
  and no ICP or IDM content is used.
- Each variant runs on the same signals with the same frozen context. Nothing
  is written to the graph; the other facility's stored enrichment stays
  byte-identical, shown by the same digest check as before.
- New behaviour sits behind the existing per-seat enrichment options, so other
  seats keep their behaviour. Set the seat's options to the best combination
  only if it meets the bar on both forties; otherwise leave the landed options
  as they are and report the best combination as a proposal.
- Reliability must not regress: report raw parse rate and delivery for every
  variant beside its correctness.
- Measurement runs execute on a compute partition:
  `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`, logs under
  `~/.cache/imas-codex-codac-uda/`.
- Write the manifest with what you have before any long wait.

## Done when

The evidence fragment gives the figures above for the baseline and each
variant and for the best combination, with one figure showing wrong-by-hand
counts per variant beside delivery rate, saved through `imas_codex.plotting`'s
save function, rendered and looked at. Tests cover each new option, the
dropping of an unsupported statement and the second pass.

Negative control: a statement whose cited context item does not exist is
dropped and counted, never delivered; and with the grounding instruction
removed from the best combination, the hand-read wrong count on the screening
twenty does not fall below the best combination's.

## Scope

Write paths: `imas_codex/discovery/signals/`, `imas_codex/discovery/base/`,
`imas_codex/llm/` (the enrichment prompt templates), `pyproject.toml` (the
`discovery-describe` seat's signal-enrichment options only), `scripts/`,
`tests/discovery/`, the evidence fragment
`docs/evidence/fragments/iter-signal-context-and-enrichment/ground-local-signal-descriptions-in-context.html`
and the figure directory
`docs/figures/iter-signal-context-and-enrichment/ground-local-signal-descriptions-in-context`.

Commit locally with a conventional subject and a body. Do not push or merge.
