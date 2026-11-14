# Brief: why a live rewrite edit costs USD 0.39 and does not settle

The bulk rewrite ran five shared-wording names and stopped itself: USD 0.39 a name on average, none settled cleanly. The lead ruled that nothing more is spent until the cause is known. You find it, repair what is a defect, and re-measure on a small set. You do not run the bulk.

## What happened (read these first)
- Progress file: `/home/ITER/mcintos/.cache/imas-codex-live-rewrite/bulk-progress.jsonl`. The five edits: `volume_averaged_neutral_species_density`, `volume_averaged_ion_charge`, `flux_surface_averaged_parallel_current_density` and `flux_surface_averaged_toroidal_current_density` are "not settled: wrapper exit 124 after 280s"; `volume_averaged_oxygen_density` was refused by the documentation writer's own population check because the regenerated text counted neutral atoms under a bare species name (the same refusal the first slice saw on `volume_averaged_boron_density`).
- The bulk run's tree, scripts and logs: worktree `sdbc-live-rewrite-bulk`, logs under `/home/ITER/mcintos/.cache/imas-codex-live-rewrite/`. An earlier measurement put a wording-only regeneration at USD 0.012 a name and a full regeneration at USD 0.152 (`docs/evidence/fragments/sn-definitions-by-construction/rendered-wording-census.json`, cost block).

## Questions to answer, each from records, not from reasoning
1. **Where the USD 0.39 goes.** For each of the five names, every model call the edit made (generate, review, refine, judgment, embedding), with its model seat, tokens and cost, from the pipeline's own cost records in the graph and logs. State which calls a wording-only re-render needs and which it made anyway. State how many generate and refine attempts each name burned.
2. **Why three edits did not settle in 280 s.** Whether the pipeline was still working (cold start, queue, retries) or stuck, what state each name is in now in the graph, and whether the 280 s bound is the batch wrapper's own or the pipeline's.
3. **Why the writer counts neutrals under a bare species name.** The ruling is that a bare species name counts ions summed over charge states, neutrals excluded, and the writer is given that wording. Find what in the writer's inputs for these names says otherwise (the DD source documentation, a parent's or sibling's stored text, a linked name's description) and say whether the refusal is the right outcome for that name (its source really includes neutrals, so the name is wrong and the ruling for it is a rename) or a defect (the writer was fed stale text it should not trust).
4. **What the cheapest correct route is** for each class of the manifest (`live-rewrite-manifest.json` in the bulk tree: shared wording 11, family parent 144, operator 77, unit misfit 26, single object 171, equivalent-electron 56), with the projected cost of the pass at that route.

## Repair and re-measure
- Fix what is a defect in the pipeline or in the batch wrapper, with a test shown failing at the base for each product change. Do not weaken the population check or any other refusal to make an edit pass.
- Then re-measure on ten names spread over the classes (two shared wording, two family parents, two operators, two unit misfits, two single object), through the same write paths, as the only graph writer. Ceiling for this node USD 6 in all; stop a kind after three names if it averages over USD 0.40.
- Report the measured rate per class beside the route, and the projection for the whole pass against USD 150.

## Deliver, under `docs/evidence/fragments/sn-definitions-by-construction/`
`live-rewrite-cost-diagnosis.json` and `sdbc-live-rewrite-cost-diagnosis.html`: the per-call cost table for the five names, the answer to each question, the repairs made, the ten re-measured names with before and after, and the projection. A figure helps: cost per name by call type for the five names, as stacked bars, one SVG under `docs/figures/sn-live-rewrite-cost/`, drawn with the repository's plotting conventions and looked at before it is committed.

## Working conditions
- You may write the live graph only through `sn edit` and the signed rename path, only for the ten re-measured names; read `.claude/skills/sn-edit` first. No other graph writer is running. Never write stored text through Cypher.
- Every pipeline run and every test file that opens the graph is its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`. A bare `srun --partition` hangs where you start.
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Never kill by pattern (`pkill -f`, `killall`); list pids and signal those.
- If a file you must edit is outside your write paths, edit it and record it in `changed_paths` with a `scope_note`.
- Write the manifest before any wait. Status is `complete`, `blocked` or `failed`, nothing else. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
