# Brief: an adjudicated label set for the name-against-definition check

Lead ruling, 2026-11-15: the 53 labels both fresh readers rejected are corrected, and the 40 names the readers split on go to a third reader of another model family. The result is kept beside the original labels, which are not edited.

## Inputs
- Original labels: `docs/evidence/fragments/sn-name-definition-graph-reading/reader-verdicts.json` (2,044 names; do not modify this file).
- The adjudication: `docs/evidence/fragments/sn-definitions-by-construction/name-check-prototype.json` and its page `sdbc-name-check-prototype.html` (322 names read by two fresh readers; 161 disputes: 53 label errors, 68 upheld, 40 splits), with the script `name-check-prototype.py` whose loaders and reader prompt you reuse.

## Deliver
1. A third blind reading of the 40 split names by a reader of a different model family from both earlier readers and from the label readers. Use `openrouter/x-ai/grok-4.7` at high reasoning effort if the project's seat catalog prices it; otherwise the newest priced model of a family none of the earlier readers used, named in the manifest with why. The reader sees name, definition, unit and the class sheet only, never a label or any earlier verdict, in shuffled order, mixed with 40 names the earlier readers agreed on (20 each way) as a control. Report its agreement with the earlier readers on the control names.
2. `docs/evidence/fragments/sn-name-definition-graph-reading/reader-verdicts-adjudicated.json`: every one of the 2,044 names with its original verdict, its adjudicated verdict, and how the adjudicated verdict was reached (`unchanged`, `corrected: both fresh readers rejected the label`, `settled by third reader`, `contested: no majority`), with each reader's verdict where one was taken. A name with no majority stays `contested` and is excluded from scoring; list them.
3. A page, `sdbc-adjudicated-labels.html`: the counts moved in each direction (flagged to stands, stands to flagged), by reader class; the twenty-five corrected labels with the largest reader agreement shown by name with the definition's opening sentence, so the lead can spot-check; and the adopted question re-scored against the adjudicated set over all names that have a reading (state how many of the 2,044 that is and why the rest have none).
4. Say plainly what the set is and is not: only disputed names were re-read, so a label that is wrong where the check agreed with it is still wrong. Give the count of names never re-read.

## Limits and conditions
- Spend ceiling USD 5 from the calls' recorded usage.
- No graph write, no change under `imas_codex/` or `tests/`. No catalog YAML file is an input.
- Model calls run as a job, not where you start: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel --cpus-per-task=2 --mem=8G --time=02:00:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, resumable from a progress file.
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Never print `.env`. Never kill by pattern.
- Write the manifest before any wait; if your turn is ending with a job still running, report `blocked` with a `wait_condition` naming it. Status is `complete`, `blocked` or `failed`, nothing else. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
