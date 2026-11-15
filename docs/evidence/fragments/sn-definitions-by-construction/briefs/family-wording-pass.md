# Brief: the species families and operator names brought to the one stored wording (sn-definitions-by-construction §6, second pass)

The first rewrite worked the 485 names the readers flagged. The closing census over every live name shows the section's wording endpoint is still unmet outside that set. This node works the rest. It is resumable: every resume continues from a progress file.

## What the census found (read it first; it is the population)
`docs/evidence/fragments/sn-definitions-by-construction/rewrite-closing-census.json`, phase `live`:
- `families`: 23 species families, 316 members, none at one wording (`divergent_families` lists each family's members and templates).
- `population`: 91 species names; 76 depart from the ruled statement "The density counts ions summed over all charge states; neutral atoms are excluded." (`departing` gives each name with its reason); 64 mention neutrals; 12 are silent on the sum.
- `operators`: the flux-surface average carries 3 forms over 73 names, the line average 2 over 28, the volume average 2 over 66; the names in each `<omitted>` bucket paraphrase instead of carrying the stored form.

## Rulings (recorded on the plan; do not re-decide)
- A bare species name counts ions summed over all charge states, neutrals excluded. Where a name's Data Dictionary source really includes neutrals or a single charge state, the name is wrong, not the wording: record it as needing a rename with the evidence from the source, leave its text alone, and list it. Do not rename in this node.
- One stored wording per operator and per species family, rendered by the pipeline, never rewritten freehand. Re-render through the route the first pass measured for shared wording (the wording-only regeneration, about USD 0.012 a name); read how the first pass did it in `docs/evidence/fragments/sn-definitions-by-construction/live-rewrite-bulk.json` and `sdbc-live-rewrite-bulk.html` and its cost diagnosis (`sdbc-live-rewrite-cost-diagnosis.html`).
- The catalog is COCOS 17 throughout; definitions never name a COCOS.
- The other rulings of the first pass stand as written in `briefs/live-rewrite-bulk.md`.

## Write paths to the graph
Stored text changes only through `sn edit` and the pipeline it feeds; read `.claude/skills/sn-edit` first and dry-run each kind once. Never write name, description or documentation through Cypher. You are the only graph writer: before the first write, confirm no other writer is running (no `sn run`, `sn edit` or signed apply in `squeue` or the process table) and say what you checked. Run edits serially.

## How to run it
- Build the work list from the census: every member of a divergent family whose text is not already the family's stored wording, every departing species name whose source supports the ruled population, and every operator name in an `<omitted>` bucket. A name already settled by the first pass at the stored wording is skipped and counted.
- Order: one whole small family first as the cost probe (stop and report if it averages over USD 0.10 a name), then the remaining families, then the operator names.
- Batches of about forty names, each its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, or chained on `rigel`. A bare `srun --partition` hangs where you start.
- Progress file on GPFS (`~/.cache/imas-codex-live-rewrite/family-pass-progress.jsonl`): one line per name when submitted and one when it settles, with description, documentation, unit, source bindings and review scores before and after, the outcome, and the spend from the pipeline's own cost record. Every start reads it and skips what is settled.
- Write the manifest before each wait and after each batch. If your turn is ending with batches left, report `blocked` with `wait_condition` naming the jobs.

## Limits
- Spend ceiling USD 30 from the pipeline's cost records (`LLMCost` rows joined to each edit's own run id). Stop and report if the probe family averages over USD 0.10 a name, or if the projection of the remainder passes the ceiling.
- Stop and report at once if a documentation review fails with an error rather than a verdict, if a write is refused for drift, or if the count of live names with no producer edge rises (count by `PRODUCED_NAME` edges over names whose status and name stage are not superseded; it is 10 now).
- A name whose documentation score would fall below 0.85 keeps its earlier text or is listed as held; record the score before and after for every name, so "as a result of the rewrite" can be settled this time.
- Never kill by pattern; list pids and signal those.

## Deliver, under `docs/evidence/fragments/sn-definitions-by-construction/`
`family-wording-pass.json` and `family-wording-pass.html`: per family and per operator, the forms before and after; every name worked with before and after text; every name left with its reason (source includes neutrals, held, declined, exhausted); spend per family beside the measured rate and the total beside the ceiling; and the census's own family and operator counts re-read from the live graph at the end (reuse `rewrite-closing-census.py`, do not re-implement it). A figure helps: families at one wording before and after, one SVG under `docs/figures/sn-family-wording-pass/`, drawn with the repository's plotting conventions and looked at before it is committed.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- A pipeline run has a slow cold start; read its log before judging it stuck.
- Status is `complete`, `blocked` or `failed`, nothing else. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
