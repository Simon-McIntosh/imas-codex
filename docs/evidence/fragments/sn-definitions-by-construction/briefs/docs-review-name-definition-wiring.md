# Brief: the documentation review resolves its name-against-definition note

## The defect
On main, `imas_codex/standard_names/worker_tasks/worker_review_docs.py` resolves `_name_definition_note` through the worker dependency lookup (near line 891), but the dependency table in `imas_codex/standard_names/workers.py` carries no such entry; the function is defined in the same module (near line 699) and takes the resolver as its first positional argument. Every documentation review item therefore raises `KeyError: '_name_definition_note'` before review, and the pool retries without end. No test caught it because no test drives the documentation review batch with a name-against-definition hold; the existing test covers only the pure function. A reproduction is recorded in `docs/evidence/fragments/sn-definitions-by-construction/` at commit `9750aaa87` of the first live-rewrite run (worktree `sdbc-live-rewrite-first-slice`); read it there if useful.

## Deliver
- The documentation review batch runs with the note resolved the way its sibling collaborators in that module are resolved. Pick the one form the module already uses for a collaborator defined in it; do not add a second lookup mechanism.
- A test that drives the documentation review batch function itself (not the pure helper) on one item whose definition adds a qualifier its name omits, with model calls stubbed, and asserts the persisted hold. Shown failing at the base revision with the `KeyError`, and shown failing again when the call that applies the hold is removed.
- A test, or an assertion in an existing one, that every name the worker modules resolve through the dependency lookup has an entry in the table, shown failing with one entry removed.

## Gate
The test files that cover the two modules, run as one `all_debug` job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:30:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, the log ending `EXIT=<n>`, with the base-revision failure in a second log. Judge by added failures against the base; add none.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- No live graph write. No model spend.
- Write the manifest before any wait. Status is `complete`, `blocked` or `failed`, nothing else.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
