# Brief: each further group (section 5 of sn-audited-group-release)

## Reuse
- Everything the first group used: the blind audit's sampler, reader seats and record shape (see `blind-audit.md`), the domain audit and family owners (`domain-groups.md`), the release gates with the bar's causes (`release-bar.md`), and `sn release --domain` for the cut.
- The repair method is whichever arm the first group's comparison showed better, per defect class; read that result and say which you use for which class. Recurring causes go back to the coordinator as pipeline defects, as in the first group's brief.

## Method points
- One group at a time, in the ruled order; one graph writer at a time. Confirm no other writer is running before the first write and say what you checked.
- Each group has its own spend ceiling, ruled by the lead; if none is recorded for the next group, end `blocked` naming it after committing what is done.
- A group that cannot meet the bar is reported and not cut. Never lower the bar or weaken a refusal to cut one.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Heavy work runs as jobs: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, or chained on `rigel`; a bare `srun --partition` hangs where you start. Suite arms are extracted from a git archive into the job's own `/tmp`, never under `~/.config`. Run scratch and progress files under `~/.cache/imas-codex-live-rewrite/` are yours to write; your declared write paths bound only what you commit.
- Manifest lists are comma-separated; commits are bare hashes; `baseline_suite` and `after_suite` are JSON objects naming the exact revisions run. Status is `complete`, `blocked` or `failed`, nothing else; commit before you write `complete`. Never kill by pattern. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
- A figure only where it shows something a table cannot; if you draw one, embed it in the page, save it through the repository's style module and look at the render before committing.
