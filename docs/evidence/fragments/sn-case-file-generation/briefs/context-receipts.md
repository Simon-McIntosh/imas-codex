# Brief: context receipts for every model-calling stage

## Authority

Plan `sn-case-file-generation`, section 8 ("Every stage records the context it
was given, and omissions are counted"). Read that section in your worktree
first: its done-when is the measure. This brief adds only the reuse map and the
constraints; it does not restate the endpoint.

## Reuse map

- `imas_codex/standard_names/case_file.py` already builds a per-name case file
  and owns `completeness_verdict`, `assert_read_only` and `case_file_digest`. A
  code review of 2026-11-21 found it has no importer in the pipeline. The
  receipt's notion of an expected channel and of completeness is this module's:
  extend it, do not write a second one. After this node the module has a
  production importer.
- `imas_codex/standard_names/context.py` assembles the context the compose path
  uses today. Each stage's prompt is rendered from its own assembly code; find
  that code per stage and record what it produced. Do not change what any stage
  sends.
- The stages are the ones the section lists. Their worker code is under
  `imas_codex/standard_names/worker_tasks/` and
  `imas_codex/standard_names/review/`, with prompts under
  `imas_codex/llm/prompts/sn/`.
- The held-out members and their adjudicated defect classes are in
  `docs/evidence/fragments/sn-case-file-generation/benchmark-second-round-incumbent-defects.json`;
  the reference split is the one `benchmark.py` in that directory uses. The
  source census is `case-file-source-census.json` beside it.
- The equilibrium group's membership is defined by
  `docs/evidence/fragments/sn-audited-group-release/domain-groups.json`.

## Constraints

- No prompt text changes and no model call. Render each stage's prompt for a
  name without sending it. Metered spend is zero.
- Read-only on the graph. Bound every query to the named cohort (the
  equilibrium group and the 87 held-out members). No whole-graph scan.
- Several core modules are far larger than one context window
  (`worker_extraction_and_enrichment.py`, `review/jev_review.py`, `context.py`,
  `case_file.py`). Read them by function, not whole.
- Write the census and its evidence under
  `docs/evidence/fragments/sn-case-file-generation/` as
  `context-receipts.py`, `context-receipts.json` and `context-receipts.html`.
  The script carries a `--check` that verifies the committed JSON and fails on
  the negative control the section names.
- The census is heavy enough to leave the fleet node: run it with
  `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug
  --cpus-per-task=4 --mem=16G --time=00:55:00 --output=<a log under
  ~/.cache/imas-codex-context-receipts/> --wrap '<command>'`, and read the log
  afterwards. Unit tests for the receipt may run in place.
- Environment: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv
  PYTHONPATH="$PWD" UV_NO_SYNC=1 uv run --no-sync …`. Never `pip install`. The
  generated model files and the `.env` link are placed in your worktree for you.
- Gate: `tests/standard_names` for the modules you touch, by delta against the
  base revision, plus the new tests. The default markers exclude `graph`.
- Commit locally with a conventional subject and a body, staging explicit
  paths. Do not push, merge, rebase or stash. Never stage the five generated
  model files.
- Write the manifest with what you have before any long wait. Recognised
  status words are `complete`, `blocked` and `failed`.
- No plan labels, section numbers, version tags or ticket numbers in file
  names, symbols, comments or commit messages.

## What the lead will read

The ranked list of omissions: per stage and channel, how many names lacked it,
and which of the 33 adjudicated-defective held-out names coincide with each
omission. Lead with that table in the HTML.
