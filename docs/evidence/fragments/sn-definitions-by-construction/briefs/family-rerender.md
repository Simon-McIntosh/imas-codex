# Brief: the 316 family members re-rendered from their stored sentences (section 10, second stage)

Stage one is on main: the 23 species-family sentences are stored, the documentation writer renders a family member from its sentence, and guard tests cover both. This node does the live re-render the section's done-when requires. It is resumable from a progress file.

## Lead rulings (recorded on the plan; do not re-decide)
- The 23 sentences are approved as written.
- Placement: the writer rewrites a member's opening so the family sentence comes first and any paraphrase of the counting rule is gone. The renderer appending the sentence at the end is not used.
- A bare species name counts ions summed over all charge states, neutrals excluded. The catalog is COCOS 17 throughout; definitions never name a COCOS.

## Preconditions (check first; stop and report if one fails)
- The equivalent-electron renames have settled: six of the families are keyed on names those renames change. Re-read the family membership from the live graph; do not use the stage-one member list.
- You are the only graph writer: no `sn run`, `sn edit` or signed apply of this account in `squeue` or the process table. Say what you checked.

## Write path
Stored text changes only through `sn edit` and the pipeline it feeds; read `.claude/skills/sn-edit` first and dry-run once. Never write name, description or documentation through Cypher. Run edits serially.

## How to run it
- Probe one small family first; stop and report if it averages over USD 0.05 a member or if a member's documentation score falls below 0.85.
- Then serial batches as jobs: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, or chained on `rigel`.
- Progress file `~/.cache/imas-codex-live-rewrite/family-rerender-progress.jsonl`: one line per member when submitted and one when it settles, with description, documentation, unit, source bindings and review scores before and after, the outcome and the spend from the pipeline's own cost records.
- A member whose score would fall below 0.85 keeps its earlier text and is listed.
- At the end run the closing census unchanged (`rewrite-closing-census.py`; do not re-implement it) and report families at one wording beside 0 of 23 before.

## Limits
- Spend ceiling USD 30 from `LLMCost` rows joined to each edit's own run id.
- Stop and report at once if a documentation review fails with an error rather than a verdict, if a signed apply is refused for drift, or if live names without a source rise above 10.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Run scratch and progress files under `~/.cache/imas-codex-live-rewrite/` are yours to write; your declared write paths bound only what you commit. Suite arms run from a git archive in the job's own `/tmp`.
- Manifest lists are comma-separated; commits are bare hashes; `baseline_suite` and `after_suite` are JSON objects. Status is `complete`, `blocked` or `failed`, nothing else; commit before you write `complete`. Never kill by pattern. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
