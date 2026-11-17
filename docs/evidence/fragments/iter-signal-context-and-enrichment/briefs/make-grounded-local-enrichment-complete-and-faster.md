# One more round: no empty descriptions, and faster

## Goal

Make the locally served model's grounded signal enrichment deliver every
description and run faster, without losing the correctness the last round
gained, and switch the seat's settings on if it holds.

## Why

The correctness round made the local model's descriptions right: a blind hand
reading found 2 wrong of forty on the original sample and 1 of forty on a
fresh one, against 26 before. Two things stopped it being switched on. On the
full 200 signals it delivered 196, and 194 raw responses parsed, where the
settings in use deliver all 200 with 197 parsing; two of the three wrong cases
are descriptions that came back empty. And it makes a second model call per
signal, so it is slower. The lead chose to keep the local model and asked for
one more round aimed at exactly these two things before anything is switched
on, and set the quality bar for the ITER pilot at no more than two wrong and
at least thirty-two right of forty, read by hand.

## What exists

- The grounding options and the local review pass are on the primary branch
  behind the per-seat options under
  `[tool.imas-codex.discovery-describe.signal-enrichment]`, switched off.
  `scripts/assess_signal_enrichment_grounding.py` and
  `scripts/refine_signal_enrichment.py` replay the stage on the fixed cohort.
- The cohort, frozen context, the paid model's descriptions, both sealed
  forties and every earlier arm are in the archived run directories of the two
  earlier rounds under `~/.config/reckon/crew/archive/runs/` (search there for
  the run names ending `refine-local-model-signal-enrichment` and
  `ground-local-signal-descriptions-in-context`). Treat them as read-only.
- The evidence fragments
  `docs/evidence/fragments/iter-signal-context-and-enrichment/refine-local-model-signal-enrichment.html`
  and `ground-local-signal-descriptions-in-context.html` record what each
  earlier variant did.

## What to do

1. Empty and unparsed responses. Find, from the stored raw responses of the
   last round, why four descriptions were not delivered and why six raw
   responses did not parse. Then make the stage recover them: a bounded retry
   of the failed call, and where the review pass removes every statement, a
   fall-back that delivers the shortest supported statement of what the signal
   is, never an empty description and never unsupported text. Count each
   recovery path separately.
2. Speed. Measure calls per signal, wall time and tokens for the last round's
   configuration, then try, one at a time: running the review pass only when
   the draft makes a statement the code cannot tie to a context item; reviewing
   several drafts in one call; reasoning effort for the review pass; and
   concurrency against the local route within its published limits. Report
   wall time per thousand signals for each.
3. Correctness must hold. For the best combination, read by hand, blind, the
   same two forties as the last round, sealed before any automated verdict is
   read. State wrong, partly right and right on each, and say whether each
   meets at most two wrong and at least thirty-two right.
4. Switch on, or do not. If the best combination delivers 200 of 200, parses
   at least 197 of 200 raw responses, and meets the bar on both forties, set
   the seat's options to it. Otherwise leave the options as they are and report
   the best combination as a proposal with what it missed.

## Rules

- The local model only. No ITER signal and no ICP or IDM content.
- Nothing is written to the graph; the other facility's stored enrichment
  stays byte-identical, shown by the same digest check as before.
- New behaviour sits behind the existing per-seat options. Other seats keep
  their behaviour.
- A description delivered by the fall-back is marked as such in the stage's
  result, so a reader can tell it from a full description.
- Measurement runs execute on a compute partition: `env -u SLURM_JOB_ID -u
  SLURM_JOBID sbatch --partition=rigel`, logs under
  `~/.cache/imas-codex-codac-uda/`.
- Write the manifest with what you have before any long wait, with `commits`
  as a JSON list of full shas.

## Done when

The evidence fragment gives, for the last round's configuration and each
variant and the best combination: descriptions delivered of 200, raw responses
parsed of 200, the count recovered by each path, calls per signal and wall
time per thousand signals; for the best combination, the hand reading on both
forties against the bar; and a plain statement of whether the seat's options
were switched on. One figure shows delivery and wall time per thousand for
each variant, saved through `imas_codex.plotting`'s save function, rendered
and looked at, with the bar and the settings in use marked. Tests cover the
retry, the fall-back and each new option.

Negative control: a signal whose every drafted statement is unsupported is
delivered with the fall-back statement and marked, never empty and never with
the unsupported text; and with the retry removed, the count delivered on the
full 200 falls below 200.

## Scope

Write paths: `imas_codex/discovery/signals/`, `imas_codex/discovery/base/`,
`imas_codex/llm/` (the enrichment prompt templates), `pyproject.toml` (the
`discovery-describe` seat's signal-enrichment options only), `scripts/`,
`tests/discovery/`, the evidence fragment
`docs/evidence/fragments/iter-signal-context-and-enrichment/make-grounded-local-enrichment-complete-and-faster.html`
and the figure directory
`docs/figures/iter-signal-context-and-enrichment/make-grounded-local-enrichment-complete-and-faster`.

Commit locally with a conventional subject and a body. Do not push or merge.
