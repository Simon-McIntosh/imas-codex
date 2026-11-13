# Brief: the live rewrite of the flagged names (sn-definitions-by-construction §6)

The first slice proved the write paths and measured the cost. This node runs the rest. It will not finish in one turn: it is built to be resumed, and every resume continues from a progress file.

## Precondition (check first; stop if either fails)
- `docs/evidence/fragments/sn-full-dd-continuation/sfdc-graph-checkpoint.json` exists on main.
- Your tree is at or after the main revision that carries the repaired documentation review (`tests/standard_names` passes there). Take main's product source; do not run the pipeline from an older tree.

## Population (the graph is the only source; no catalog YAML file is read)
`docs/evidence/fragments/sn-name-definition-graph-reading/reader-verdicts.json`: every export candidate with a verdict. The rewrite takes the 485 rows whose verdict is `flagged`, each with its readers' classes and quoted evidence. A name that is no longer live under that id when you reach it (already renamed by the slice or by an earlier batch) is recorded as settled with what settled it, not attempted again.

## Rulings (do not re-decide; they are recorded on the plan as lead rulings)
Classify each flagged name from the readers' quoted evidence, not from the class word alone, and record the class in the manifest:
- **equivalent-electron** (the evidence says the count or rate is in equivalent electrons): rename with the word `equivalent_electron` next to the base, after `prefill` (`accumulated_argon_equivalent_electron_count`, `neon_prefill_equivalent_electron_count`, `lithium_equivalent_electron_source_rate_due_to_gas_injection`). The grammar word is being released. **Do not attempt these until the coordinator tells you the pin is raised**; list them as waiting.
- **operator** (an average, integral, weighting or normalisation the name omits): the name gains the operator where the DD source is the averaged or weighted quantity; the definition is rewritten only where the source is local.
- **single object or mechanism** (a generic name defined for one object, part or mechanism): `sn edit` with the readers' phrase as the reason; the pipeline and review settle the spelling and may decline with a recorded reason.
- **family parent** (reader class `abstract-parent-as-quantity`): regenerate the definition to state one quantity where one exists; otherwise record the name as held from the cut.
- **unit misfit** (reader class `unit-does-not-fit-name`): the name is regenerated to match the definition and unit. `electron_number_density`, defined as a path integral, becomes `line_integrated_electron_number_density` (lead ruling; the operator exists in the grammar). Apply the same reading to other unit misfits: prefer an operator or qualifier the grammar already has over a new word, and list a name as waiting on grammar only when no existing spelling parses, with the word it would need.
- **shared wording**: the species families and the flux-surface-average names whose definitions use a non-settled form are re-rendered to the one stored wording. A bare species name counts ions summed over charge states, neutrals excluded.

A flag you find false on reading the name, its definition and its DD source is declined with its reason. That is an expected outcome.

## Write paths
Stored text changes only through `sn edit` and the pipeline it feeds; read `.claude/skills/sn-edit` first and dry-run each kind once. Renames go through the signed rename path so sources and paid history are kept. Never write name, description or documentation through Cypher. You are the only graph writer; run signed applies one at a time.

## How to run it
- Rebuild `live-rewrite-manifest.json` from the population above with the class of each name, and order the work: shared wording first (cheapest), then family parents, operators, unit misfits, single-object names.
- Run in serial batches of about forty names, each batch its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`. A bare `srun --partition` hangs where you start.
- Keep a progress file on GPFS (`~/.cache/imas-codex-live-rewrite/bulk-progress.jsonl`): one line per name when its edit is submitted and one when it settles, with the stored name, description, documentation, unit, source bindings and review scores before and after, the outcome (accepted, renamed to what, declined with reason, held, still refining) and the spend from the pipeline's own cost record. Every start reads it and skips what is settled.
- Write your run manifest before each wait and after each batch. If your turn is ending with batches left, report `blocked` with `wait_condition` naming the batches left; the coordinator resumes you.

## Limits
- Spend ceiling for this node: USD 60, from the pipeline's cost records. Stop and report if any kind of edit averages over USD 0.60 a name across twenty names, or if the projection of the remainder passes the ceiling.
- Stop and report at once if a documentation review fails with an error rather than a verdict, if a signed apply is refused for drift, or if the count of live names without a source rises.
- Never kill by pattern (`pkill -f`, `killall`); list pids and signal those.

## Deliver, under `docs/evidence/fragments/sn-definitions-by-construction/`
`live-rewrite-manifest.json` (rebuilt), `live-rewrite-bulk.json` and `sdbc-live-rewrite-bulk.html`: totals by class and outcome, the before-and-after table, every decline and hold with its reason, names waiting on grammar with the word each needs, spend per class beside the measured rate and the total beside the ceiling, and the wording and unit censuses of §6 re-read at the end.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- A pipeline run has a slow cold start; read its log before judging it stuck.
- Status is `complete`, `blocked` or `failed`, nothing else. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
