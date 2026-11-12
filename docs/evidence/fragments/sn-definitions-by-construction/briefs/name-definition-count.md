# Brief: every published definition counted against its name (sn-definitions-by-construction §4)

Read-only labelling over the published catalog files in `/home/ITER/mcintos/Code/imas-standard-names-catalog/standard_names/*.yml` (tag v0.4.0rc8, commit 4a48c6c). No graph access is needed and none is to be used.

## What to deliver
The committed record §4 describes, as one JSON file plus one HTML fragment, under `docs/evidence/fragments/sn-definitions-by-construction/`: `name-definition-labels.json` and `name-definition-labels.html`, with the sampling script beside them.

## Rules that protect the labels
- Draw the sample with a fixed seed and commit the seed. Stratify by domain file as §4 states.
- Label by reading name, description, documentation and unit together. Do not call any model-based judge, review check or `decide_batch` to produce a label: these labels become the reference a later check is measured against, and a label produced by the thing under test is void.
- The 60-name second reading must be made without the first labels on screen. If you cannot provide an independent second reader, say so in the manifest and leave that done-when unmet; the coordinator dispatches a separate reader. Do not label twice yourself and call it agreement.
- The ten names cited in §1's last three rows must be in the sample (add them outside the random draw and flag them as added).

## Classes
qualifier-missing-from-name, definition-narrower-than-name, abstract-parent-as-quantity, unit-does-not-fit-name, agrees. Add a class only if at least five names need it, and say why.

## Standing rules for this node
- Read your plan section from your worktree; it is the authority for the endpoint. This brief adds the reuse map and the questions you may not settle.
- Export `UV_NO_SYNC=1`, use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent.
- Heavy gates (any test file opening the graph, any pass over the whole catalog) run as their own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-<node>/>`. A bare `srun --partition` from this node hangs. Never run them on the node you start on.
- Do not write the live graph. Use a disposable graph or fixtures for every control.
- Write your manifest with what you have before any long wait. Status is `complete`, `blocked` or `failed`.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.

