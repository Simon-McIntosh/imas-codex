# Brief: the blind audit (section 1 of sn-audited-group-release)

The section states the endpoint. This brief is the reuse map from the plan's design review: extend these owners; do not build a parallel harness beside them.

## Reuse
- **Sampling.** `imas_codex/standard_names/benchmark_roles.py` `_stratified_sample(rows, sample, seed, key="physics_domain")` already draws a seeded round-robin sample balanced across physics domains; `judgment_goldens.py` `_sample` does the same across domain and pattern family. Use or extend one; record the seed.
- **Readers.** `advisory_panel.py` (`PanelSeatJudgment`, `PanelAdjudication`) keeps each seat's judgment and derives panel statistics; `review/panel_audit.py` (`audit_selected`, `panel_audit_rate`, `panel_audit_report`) reports drift between two judges over a fixed sample. The two blind readers are seats of that machinery with models the pipeline does not compose or review with. Name the two models and why each qualifies.
- **Planted names.** `judgment_goldens.py` holds four committed golden sheets with independent verdicts (`_west_verdicts`, `_expert_verdicts`), each row labelled. Draw the twenty defective and twenty sound planted names from them, and say how you keep them unseen by the readers as planted. A golden label must never come from a model that is one of the two readers.
- **Record shape.** `review/panel_audit.py` `_audit_record` builds a non-deciding review record with model, cost, axis and role. Each reader's verdict is stored in that shape, so cost and usage are read from the record and the later sections read one store. The committed evidence file is a summary of those records, not a second store.
- **Source text shown to a reader.** The enclosing-structure read the name review now has (see the repair on the definitions plan) is the one to use; do not write another ancestor query.

## Method points the section leaves to you
- Fix the defect classes in a committed file before any reading and do not change them after.
- The readers see no stage, score or earlier verdict. Check this by printing one full reader prompt into the evidence.
- No stored name, description or documentation changes; audit records are review records and do not decide anything.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Heavy work runs as jobs: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, or chained on `rigel`; a bare `srun --partition` hangs where you start. Suite arms are extracted from a git archive into the job's own `/tmp`, never under `~/.config`. Run scratch and progress files under `~/.cache/imas-codex-live-rewrite/` are yours to write; your declared write paths bound only what you commit.
- Manifest lists are comma-separated; commits are bare hashes; `baseline_suite` and `after_suite` are JSON objects naming the exact revisions run. Status is `complete`, `blocked` or `failed`, nothing else; commit before you write `complete`. Never kill by pattern. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
- A figure only where it shows something a table cannot; if you draw one, embed it in the page, save it through the repository's style module and look at the render before committing.
