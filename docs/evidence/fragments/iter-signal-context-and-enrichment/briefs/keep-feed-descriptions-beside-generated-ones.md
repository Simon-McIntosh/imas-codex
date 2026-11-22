# Keep the controls-feed descriptions beside the generated ones

## Goal

Restore the controls-feed descriptions that pilot group enrichment overwrote,
and make generated descriptions live beside the source text instead of
replacing it, so no enrichment can erase a source again.

## Why

Pilot group enrichment (landed 2026-11-20) wrote each composed description
into the signal's `description` property, which is where the controls
metadata feed stores its operator label, and left `description_source` empty.
On 2026-11-22 the coordinator compared the 190 described pilot signals with a
snapshot taken just before that run: every one of them had a feed description
(for example an absolute-pressure label naming the supply line), and every
one is now the generated paragraph. The feed text survives only in that
snapshot and in the feed itself. The plant-wide role assignment then read the
damaged signals as having no source and undecided their pilot values, so it is
held until this is repaired.

## Inputs

- The pre-enrichment values, saved privately by the coordinator:
  `~/.cache/imas-codex-codac-uda/pilot-enrichment-by-group/overwritten-feed-descriptions.json`
  (id, feed description, feed source for each of the 190 signals).
- The original snapshot it was taken from:
  `~/.cache/imas-codex-codac-uda/pilot-vocabulary/signals.json`.
- The controls metadata import (`iter_codex/discovery/signals/controls_metadata.py`)
  is repeatable and writes `description`, `unit`, `data_type`,
  `description_source` and `description_source_updated_at`.
- The group enrichment writer (`iter_codex/discovery/signals/group_enrichment.py`,
  through `write_signal_descriptions` in `iter_codex/discovery/signals/parallel.py`).

## What must be true afterwards

1. A signal's source description (feed text, or the archive's own) and a
   generated description are held in separate properties declared in the
   schema. The generated one records that it is generated and by which
   mechanism. Writing a generated description never changes the source
   description or its source marker.
2. The 190 overwritten signals carry their feed description and feed source
   again, equal to the snapshot, and keep their generated description in the
   new property.
3. Every reader that means "what the sources say" (the signal vocabulary, the
   plant search, the server tools) reads the source description; anything
   that shows a description to a user or embeds it states which one it uses.
   Say in the evidence which readers you changed and why.
4. The embedding of each of the 190 signals is consistent with the text the
   search now indexes, and the evidence says which text that is.

## Done when

- For the 190 signals, the source description and its source marker equal the
  snapshot for all 190, and the generated description equals the text the
  pilot run wrote, with the counts in the landing record.
- Across all ITER signals, the number carrying a feed source marker is at
  least what it was before the pilot run (the snapshot gives the pilot share;
  state the graph total before and after).
- A test writes a generated description onto a signal that has a feed
  description and shows the feed description and marker unchanged; with the
  separation removed, the test fails.
- The signal vocabulary run on the pilot system after the repair decides at
  least as many quantity and role values as it did on 2026-11-20 (742
  quantities, 202 roles), with the per-axis counts recorded.
- Graph-marked statements pass on `rigel`; the default tier for the touched
  packages adds no failure, both arms recorded as single-line JSON objects
  with whole test ids.

Negative control: a generated description written through the old path is
caught by the test above; and a feed description changed by hand in a scratch
graph run is reported by the snapshot comparison.

## Rules

- Change only the 190 signals' description properties in the live graph, plus
  any schema migration the separation needs; take a private before-image of
  every property you change first, and record how to restore it.
- No real signal name or description text in a tracked file; tests use
  invented names.
- Heavy work as its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch
  --partition=rigel`, logs on GPFS.
- Private data you write goes under `~/.cache/imas-codex-codac-uda/`, never
  only inside your worktree's ignored files: the worktree is removed when the
  run lands.
- Rebuild the generated models in your worktree after any schema change, and
  say in the manifest that the main checkout needs the same.
- Run `ruff check --fix` and `ruff format` on exactly the files you change and
  add no lint finding against the base.
- Commit locally with a conventional subject and a body, and give `commits`
  as a JSON list of full shas. Do not push or merge.

## Scope

Write paths: `iter_codex/schemas/facility.yaml`, `iter_codex/discovery/signals`,
`iter_codex/tools/plant_search.py`, `iter_codex/llm`, `tests/discovery`,
`tests/graph`, `tests/llm`, and
`docs/evidence/fragments/iter-signal-context-and-enrichment/keep-feed-descriptions-beside-generated-ones.html`.
