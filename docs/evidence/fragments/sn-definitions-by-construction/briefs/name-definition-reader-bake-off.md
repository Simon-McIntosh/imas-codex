# Brief: which reader should check a name against its definition

## The question
The review pipeline asks "does this definition state the quantity the name says" with one chat call to the first documentation-review model, one reading per name. That seat scored 57 % on freshly labelled names. The instrument that has worked is several independent blind readings decided two-of-three. The lead asks whether a cheap model (Haiku) should do this, and whether a hybrid is justified. This node measures it. It changes no product behaviour.

## Labels (the reference; do not relabel)
`docs/evidence/fragments/sn-name-definition-graph-reading/reader-verdicts.json`: 2,044 export candidates, each `flagged` (485) or `stands` (1,559), settled by blind readers. Each name is scored on the name, definition and unit **as the readers read them** (the reading sheets under `docs/evidence/fragments/sn-name-definition-graph-reading/` and `docs/evidence/fragments/sn-name-definition-catalog-reading/`), not on the live graph, because about a hundred names have been rewritten since. A name whose read text cannot be recovered is listed and left out. The graph is not read for text and no catalog YAML file is an input.

## Arms (same prompt and the same text for every arm; the production seat's prompt)
1. the current seat as it runs: the first documentation-review model, one reading;
2. `openrouter/anthropic/claude-haiku-5.5`, three independent readings: scored as a single reading and as two-of-three;
3. `openrouter/openai/gpt-6-luna`, three readings, same two scorings;
4. `local/deepseek-v4.1-flash` on the local route, three readings, same two scorings;
5. the mixed panel: one reading each from arms 2, 3 and 4, two-of-three.

Each reading is its own call with no other reading in context. Record every verdict, its class and its cost from the call's own usage.

## What to report, per arm
- flagged-recall, flagged-precision and overall agreement against the labels, each with a 95 % interval;
- agreement by the readers' class (operator, single object, family parent, unit misfit, equivalent-electron, shared wording), so a reader that misses one whole class is visible;
- cost per name and for the 2,044, and wall time per name;
- agreement between repeat readings of one model (how much two-of-three buys over one).

## The labels are not independent of every arm
The labels came from blind readers on the local lane, so arm 4 is graded against its own family and its score is an upper bound, not a measurement. Say so beside its number. To measure the others fairly: draw 80 names where the Haiku two-of-three verdict disagrees with the label (40 each way, or all if fewer) and 40 where they agree, and have one fresh reader of a different family from every arm and from the label readers (`openrouter/openai/gpt-6.1-sol`, high effort) judge each blind: it sees the name, definition and unit only, never a label or an arm's verdict, in shuffled order. Report who that reader sides with. Do not print any verdict to that reader.

## Deliver
`docs/evidence/fragments/sn-definitions-by-construction/name-definition-reader-bake-off.json` (every reading), `sdbc-name-definition-reader-bake-off.html` (the tables, the fresh-reader result, and a recommendation stated as: which arm, at what cost, with what measured miss rate, and what it would have caught of the 485), and one figure, `docs/figures/sn-name-definition-readers/recall-precision-by-arm.svg`: flagged-recall against flagged-precision, one labelled point per arm with its interval, drawn with the repository's plotting conventions, rendered and looked at before it is committed.

## Limits and conditions
- Spend ceiling USD 15, from the calls' recorded usage; stop and report if the projection passes it.
- No graph write. No change to the review pipeline or its prompts.
- The run is a job, not run where you start: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel --cpus-per-task=4 --mem=16G --time=04:00:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`. Make it resumable from a progress file. A bare `srun --partition` hangs where you start.
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Never print `.env`. Never kill by pattern.
- Write the manifest before any wait; if your turn is ending with the job still running, report `blocked` with a `wait_condition` naming the job. Status is `complete`, `blocked` or `failed`, nothing else. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
