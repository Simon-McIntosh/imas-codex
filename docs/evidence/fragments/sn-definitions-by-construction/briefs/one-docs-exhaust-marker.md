# Brief: one docs-exhausted marker for both documentation writers (sn-definitions-by-construction §2)

Your worktree starts from a revision that already holds an unfinished attempt at this change (its head commit says so). Finish it; do not start again.

## What is wrong on main
- `imas_codex/standard_names/worker_tasks/worker_generate_docs.py` carries `_mark_generate_docs_exhausted`, a copy of `_mark_refine_docs_exhausted` in `imas_codex/standard_names/graph_store/graph_claim_release.py`. The copy differs only in the stage it fences on and the tag it writes, and it omits the clause that closes an open steering edit, so an edit open on a pending name stays open when its first draft is refused.
- The generate-path test replaces the marker with a mock, so no test fails when the marker's query is wrong.
- The recorded failing run for `test_contradicting_answer_reaches_terminal_state_within_the_bound` reddens by the 30 s test timeout, not by its assertion.

## What the unfinished attempt already did
- Renamed the owner to `_mark_docs_exhausted` with `eligible_stage` and `tag` parameters and removed the generate worker's copy.
- Known gap: the generate call site does not pass the `services` handle the owner requires. Nothing has been run at that revision.

## Done when
- One function marks a name docs-exhausted; both writers call it; no second copy of its query exists in the package (show the search).
- A test runs the marker's query path for each stage with the graph client replaced at the client boundary, not at the marker, and reads the statement and parameters it sent: the stage fence, the claim token, the tag and the steering-edit clause. Each assertion is shown failing with the corresponding part of the query changed.
- The terminal-state test for the refine writer is shown failing by its own assertion message with the repair removed (raise that test's timeout if the harness needs it; say what you set).
- The focused gate passes, with its revision and a final `EXIT=<n>` line in the log: `tests/standard_names/test_generate_docs.py tests/standard_names/test_docs_gates.py tests/standard_names/test_refine_docs_settled_wording.py tests/standard_names/test_refine_docs_chain.py tests/standard_names/test_pool_batch_processors.py`, plus any test file that imports the renamed function.

## Standing rules for this node
- Export `UV_NO_SYNC=1`; use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent.
- The focused gate above is light enough to run where you start, in the foreground, with `-n 4` at most. Never start a whole test directory there. Anything that opens the graph runs as its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-<node>/>`.
- Do not write the live graph. No metered model calls are needed.
- Write your manifest with what you have before any long wait. Status is `complete`, `blocked` or `failed`, nothing else.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
