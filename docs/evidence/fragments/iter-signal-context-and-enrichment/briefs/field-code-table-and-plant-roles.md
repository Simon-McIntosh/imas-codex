# Coordinator instructions for the field-code table and the plant-wide roles

The plan sections (12, the table; 13, the plant-wide assignment) are the
authority for what is delivered. These instructions add the reuse map, the
inputs and the operating rules. Each section is its own node.

## Reuse map

- The signal vocabulary is in `iter_codex/discovery/signals/` (the vocabulary
  module and its graph module) and `iter_codex/remote/signal_vocabulary.py`.
  It already credits a field-pattern source; the table becomes that source.
  Extend it; do not write a second assignment path.
- Field codes and their patterns come from the archive grouping already in the
  graph (`ArchiveFieldPattern` nodes and each signal's `field_code`).
- Sources for a code's meaning: the pilot signal-list rows already imported
  (`PlantSignalRow`), the controls metadata the signals carry (`description`,
  `unit`, `data_type` from the feed), and the functional specification text
  under `~/.cache/imas-codex-codac-uda/pilot-documents/guarded-download/`.
- The sibling-relation work is pinned at `refs/keep/relate-sibling-signals`;
  read it for how roles were counted, but do not merge it.

## Starting facts

- The vocabulary decided a role for 202 signals, all in the pilot system:
  alarm 42, command 80, measurement 3, state 77, setpoint 0.
- The graph holds 97,525 ITER signals; that is the population for section 13,
  not the 995,006-variable private catalogue.
- The package is now `iter_codex` and the command `iter-codex`; the checkout
  directory keeps its old name.

## Rules

- Nothing is inferred from the letters of a code. A code no source explains
  has no entry and is counted.
- Hand readings: one written verdict and reason per item, read by you against
  the source. A file whose verdicts come from a script is not a reading.
- No real signal name, code meaning taken from a private document, or document
  identity in a tracked file. Private table rows live in private
  configuration; tests use invented codes.
- Heavy work, including the plant-wide assignment and graph-marked tests, runs
  as its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch
  --partition=rigel`, logs on GPFS, each pytest log ending with pytest's
  summary and one EXIT line.
- Record suite arms in the manifest as single-line JSON objects with whole
  test ids; the `tests` field is one literal command that parses.
- Run `ruff check --fix` and `ruff format` on exactly the files you change and
  add no lint finding against the base.
- Commit locally with a conventional subject and a body, and give `commits` as
  a JSON list of full shas. Do not push or merge.
