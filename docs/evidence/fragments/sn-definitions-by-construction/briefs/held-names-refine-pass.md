# Brief: one more refine pass on the names the first rewrite left unaccepted

Lead ruling, 2026-11-15: the 119 names the first rewrite left reviewed and not accepted and the 13 it left exhausted get one more refine pass, with a USD 20 ceiling. This node runs it. It is resumable from a progress file.

## Precondition (check first; stop and report if it fails)
You are the only graph writer. The family wording pass must have ended: no `sdbc-family` job or `sn edit` / `sn run` process of this account in `squeue` or the process table. Say what you checked.

## Population
`docs/evidence/fragments/sn-definitions-by-construction/live-rewrite-bulk.json` and `live-rewrite-manifest.json`: the names whose stage reads `reviewed` (119), `exhausted` (13) or `drafted` (4), each with its class, its hold reason and its review reason. Re-read each name's stage from the live graph before working it: a name the family wording pass has since settled is recorded as settled by that pass and skipped.

## What a pass may and may not do
- Read each name's hold first and sort the population by what holds it, recorded in the manifest: (a) the name check says the definition states something the name does not (a qualifier, an operator, a restriction to one object); (b) the documentation review score is below 0.85 for a prose reason; (c) the definition contradicts its Data Dictionary source; (d) refine attempts used up.
- For (a): the remedy is the one the readers' class implies and the rulings in `briefs/live-rewrite-bulk.md` state: the name gains the qualifier where the source is the qualified quantity, the definition loses it where the source is general. A rename goes through the signed rename path; where no existing grammar spelling parses, list the name as waiting on grammar with the word it needs. Do not regenerate text a third time hoping the check passes.
- For (b) and (d): one `sn edit` with the review's own reason as the steer; the pipeline and review settle it. A name that exhausts again is held, with both reviews' reasons side by side.
- For (c): read the source documentation; if the definition is wrong, steer it to the source; if the binding is wrong, list the name for a rebind and leave its text alone.
- A bare species name counts ions summed over all charge states, neutrals excluded. The catalog is COCOS 17 throughout; definitions never name a COCOS.
- Never weaken a hold or a refusal to make a name pass.

## Write paths
Stored text changes only through `sn edit` and the pipeline it feeds; read `.claude/skills/sn-edit` first and dry-run each kind once. Renames through the signed rename path, one at a time. Never write name, description or documentation through Cypher.

## How to run it
- Probe first: ten names spread over the four hold kinds; stop and report if they average over USD 0.20 a name or if fewer than three end accepted.
- Then serial batches of about forty names, each its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, or chained on `rigel`. A bare `srun --partition` hangs where you start.
- Progress file `~/.cache/imas-codex-live-rewrite/held-names-progress.jsonl`: one line per name when submitted and one when it settles, with name, description, documentation, unit, source bindings and review scores before and after, the hold kind, the outcome and the spend from the pipeline's own cost record.
- Write the manifest before each wait and after each batch; if your turn is ending with batches left, report `blocked` with `wait_condition` naming the jobs.

## Limits
- Spend ceiling USD 20 from `LLMCost` rows joined to each edit's own run id.
- Stop and report at once if a documentation review fails with an error rather than a verdict, if a signed apply is refused for drift, or if the count of live names without a source (the repository's own definition in `imas_codex/standard_names/source_invariant.py`) rises.
- Never kill by pattern; list pids and signal those.

## Deliver, under `docs/evidence/fragments/sn-definitions-by-construction/`
`held-names-refine-pass.json` and `held-names-refine-pass.html`: counts by hold kind and outcome; the before-and-after table with scores; every name still held with its reason; names renamed with old and new identity and the semantic distinction in a sentence; names waiting on grammar with the word each needs; names needing a rebind; spend per hold kind beside the measured rate and the total beside the ceiling.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- A pipeline run has a slow cold start; read its log before judging it stuck.
- Status is `complete`, `blocked` or `failed`, nothing else. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
