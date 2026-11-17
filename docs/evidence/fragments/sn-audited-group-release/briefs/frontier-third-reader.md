# Brief: a frontier third reader for the blind audit

Lead ruling, 2026-11-17: the two audit readers are too weak to fix a release bar from, and a frontier third reader is added. On the 40 planted names the two readers (gemini-3.8-flash, kimi-k3.1-pro) caught 45 % and 55 % of the known-defective names and each flagged 25 % of the names ruled sound. Over the 339-name sample at least one reader reads 174 names as defective and both read 84. This node re-reads a subset with one stronger reader and reports whether that reader is good enough to decide group audits.

## Inputs (on main)
- `docs/evidence/fragments/sn-audited-group-release/blind-audit.json` and `blind-audit.py`: the sample, the planted names with their independent verdicts, both readers' readings, the fixed class list and the exact reader prompt. Reuse the script: its sampler, reader call, record shape, guard against pipeline models, and `--check`. Add a third seat to it; do not write a second harness.

## What to read
- All 40 planted names.
- All 174 sampled names that either reader read as defective.
- A control of 60 sampled names that both readers read as sound, drawn by a recorded seed.
The third reader sees exactly what the first two saw (name, definition, documentation, unit, the Data Dictionary text of the source and its enclosing structures), never a stage, score, earlier verdict or which set a name is in; the three sets are shuffled together. It records every class of the fixed list that applies, as the first two did.

## The reader
One frontier model of a family that is neither of the first two readers' families and is not a model the pipeline composes or reviews with (the script's guard decides; show its pass). Use the strongest such model the project's seat catalog prices, at high reasoning effort. Name it and say why it qualifies. If the catalog prices two candidates, prefer the one with the larger measured context window and say so.

## Deliver, under `docs/evidence/fragments/sn-audited-group-release/`
`frontier-third-reader.json` and `sagr-frontier-third-reader.html`, with the script changes:
- The third reader's result on the planted names beside the first two: recall on the known-defective, false alarms on the known-sound, each with its interval at n=20, and the planted names it gets wrong listed by name with its stated reason, so the lead can judge whether the planted verdict or the reader is at fault.
- On the 174 flagged names: how many it confirms as defective, by class, split by whether one or both earlier readers flagged them.
- On the 60 control names: how many it reads as defective (the earlier readers' misses), by class.
- From those, a re-estimate of the catalog's defect rate over the 339-name sample with its interval, stating every assumption used to combine the three sets, and beside it the two earlier figures (24.8 % both, 51.3 % either). Planted names stay out of every sample count.
- One plain sentence: is this reader good enough to be the deciding reader for group audits, and on what evidence.
- Twenty names where the third reader disagrees with both earlier readers, with all three readings side by side, for the lead.

## Limits and conditions
- Spend ceiling USD 20 from the calls' recorded usage; a failed call's cost is unknown, not free. Stop and report if the first 30 names project past the ceiling.
- No stored name, description or documentation changes. Readings are stored as non-deciding audit review rows, as the first two readers' were.
- Model calls run as a job, not where you start: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel --cpus-per-task=2 --mem=8G --time=03:00:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, resumable from a progress file there. Run scratch under `~/.cache/imas-codex-live-rewrite/` is yours to write; your declared write paths bound only what you commit.
- The evidence header names the committed revision the reported run used; run the final report from a clean committed tree.
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`. Never print `.env`. Never kill by pattern.
- Manifest lists are comma-separated; commits are bare hashes. Write the manifest before any wait; if your turn is ending with a job still running, report `blocked` with a `wait_condition` naming it. Status is `complete`, `blocked` or `failed`, nothing else; commit before you write `complete`. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
