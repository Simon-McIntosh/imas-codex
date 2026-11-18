# Coordinator instructions for enriching the pilot system by group

The plan section is the authority for what is delivered. These instructions add
the reuse map, the starting facts and the operating rules.

## Reuse map

- The discovery pipeline's group enrichment is the one path by which a group
  description reaches its members. Extend it; add no second propagation path.
- Groups come from the archive grouping already on the primary branch
  (`imas_codex/discovery/signals/`), by component and by field pattern.
- Context comes from the pilot plant context import
  (`imas_codex/discovery/signals/plant_context.py` and its graph module): the
  signal-list rows, component records, function-category meanings and document
  references linked to each pilot signal.
- The model is the `discovery-describe` seat with the grounded settings on the
  primary branch. Change neither the seat nor its settings.
- Calls go through the project's LLM call layer, which records cost and usage.

## Starting facts, measured 2026-11-18

- Of 1,766 pilot signals, 190 have a derived link to context through a
  component and 1,576 have none. None has an exact link.
- The section enriches only signals with linked context. The 1,576 are marked
  as awaiting context and counted; they are not described from their names.
- The functional specification and the other pilot documents are on disk under
  `~/.cache/imas-codex-codac-uda/pilot-documents/guarded-download/`. The hand
  comparison reads the specification there.
- The plant context import has no production entry point; its live run was
  driven from a run-local script. If this node needs the import as a pipeline
  step, give it one in the existing discovery command surface and say so.

## The hand comparison

- Choose the forty member signals before the run, write them to a private
  file, and record its digest. If fewer than forty signals are enriched, read
  all of them and say so.
- Read each description against the functional specification yourself, by the
  plan's three definitions. A judge model's score is not the bar.
- If the bar is missed, deliver the section as failed on quality with the
  forty readings. Do not retune the prompt to pass.

## Rules

- Heavy work runs as a job of its own:
  `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`, logs on GPFS.
  Graph-marked tests run there too.
- The local model lane is shared. Keep concurrency at or below eight requests
  and report the wall-clock time and calls made.
- No real variable name, component number, document identity, title or
  description text goes into a tracked file, test, report or commit message.
  Tests use invented names. The forty readings stay under
  `~/.cache/imas-codex-codac-uda/`; the landing record carries counts only.
- No archive data read, no ICP or document-host request.
- Write the manifest with what is done before starting any long wait.
- Commit locally with a conventional subject and a body, and give `commits` in
  the manifest as a JSON list of full shas. Do not push or merge.
