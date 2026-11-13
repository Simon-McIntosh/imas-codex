# Brief: the first slice of the live rewrite, and a costed manifest of the rest (sn-definitions-by-construction §6)

§6 of your plan states the endpoint of the whole rewrite. This node does not finish §6. It proves every write path on a small slice of the live graph, measures the real cost per name, and writes the manifest the remaining passes will run from. Stop there.

## Precondition (check before your first write, and stop if it fails)
`docs/evidence/fragments/sn-full-dd-continuation/sfdc-graph-checkpoint.json` exists in the main checkout at `/home/ITER/mcintos/Code/imas-codex` and records a verified archive. No checkpoint, no write: report `blocked`.

## What is settled (do not re-decide)
- Shared wording: the operators and species families of §2 are rendered from one stored wording; the docs writer already refuses an altered shared part. A bare species name counts ions summed over charge states, neutrals excluded.
- Unit spelling: §3's registry.
- The lead's rulings on the name check, comment `c-s5-lead-ruling-dispositions` on the plan, by class of `docs/evidence/fragments/sn-definitions-by-construction/name-definition-seat-catalog.json`:
  - equivalent-electron counts and rates: rename; the definition stays and the name is regenerated to say so;
  - an average, integral, weighting or normalisation the name omits: the name gains the operator where the DD source is the averaged or weighted quantity; the definition is rewritten only where the source is local;
  - a generic name defined for one object or mechanism: `sn edit` with the seat's phrase as the reason and its proposed spelling as a hint only; the pipeline and review settle the spelling and may decline with a recorded reason;
  - a parent defined as a family of quantities: regenerate the definition to state one quantity where one exists; otherwise the name is held from the cut.
- About a quarter of the seat's flags are expected to be false. A flag you find false is declined with its reason; that is a correct outcome.

## Write paths (never anything else)
Stored standard-name text changes only through `sn edit` and the pipeline it feeds; read `.claude/skills/sn-edit` (or `agents/` if absent) before the first edit, and run every edit as a dry run first. Renames go through the signed rename path so sources and paid history are kept. Never write name, description or documentation through Cypher. The catalog is COCOS 17; documentation never names a COCOS.

## The slice
1. One species family of at least five members that the wording census counts divergent, brought to the one wording.
2. One operator family (the flux-surface average) of at least five names whose definitions use a non-settled form.
3. Five names from each of the four ruled classes, chosen to span their sub-patterns (for the operator class: at least two flux-surface averages and one charge-state weighting), taken through the ruled path to an accepted name and documentation, or declined with a reason.
Before and after each: the stored name, description, documentation, unit, source bindings and review scores.

## Measure and project
- Spend per name for each kind of edit (wording-only regeneration, rename with regeneration, parent definition), from the pipeline's own cost records, with run ids.
- Ceiling for this node USD 10. If the first five edits of any kind average over USD 0.60 a name, stop that kind and report.
- A manifest `live-rewrite-manifest.json`: every remaining live name the full pass would touch, with its kind, the ruling it falls under, the reason text the edit will carry and the hint if any; totals by kind; and the projected spend of the whole pass against §6's USD 150, with the measured rates beside it.

## You may not settle
- A spelling the pipeline will not accept, a ruled class that does not fit a name, or a name whose DD source is ambiguous between local and averaged: list each with both readings. Do not force it.
- Any name outside the slice. The manifest lists them; it does not edit them.

## Deliver, under `docs/evidence/fragments/sn-definitions-by-construction/`
`live-rewrite-first-slice.json` and `sdbc-live-rewrite-first-slice.html` (the before/after table, declines with reasons, spend per kind, the censuses of §6 re-read over the slice's families), and `live-rewrite-manifest.json`. A gate log ending `EXIT=<n>` for the tests you ran.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Every pipeline run, every pass over more than a few dozen names, and every test file that opens the graph is its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`. A bare `srun --partition` hangs where you start. A pipeline run has a slow cold start; read its log before judging it stuck.
- You are the only graph writer. Run signed applies one at a time.
- Write the manifest file of your run with what you have before any long wait, and evidence as results arrive. Status is `complete`, `blocked` or `failed`, nothing else.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
