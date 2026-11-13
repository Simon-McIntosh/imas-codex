# Brief: fifteen documentation-review tests fail on main

## The defect
On main at merge `64db0f1fa`, fifteen tests that drive the documentation review fail. They failed at the base of the last repair too, so that repair did not cause them and did not fix them. In the gate log their failures carry `Jev cascade judgment unavailable for <name>` followed by `review_docs failed for <name>`. The failing ids are the `FAILED` lines of `/home/ITER/mcintos/.cache/imas-codex-live-rewrite/head-gate.log`, in these files: `tests/standard_names/test_dd_gap_automated_writers.py`, `test_docs_conformance_gate.py`, `test_docs_contradiction_veto.py`, `test_pool_batch_processors.py`, `test_prose_hold_before_accept.py`, `test_review_docs_stages.py`.

Two readings are possible and you establish which holds, per test, before changing anything:
1. the tests reach a real judgment call they used to stub, because the name-against-definition check added a judgment the tests' stubs do not cover; the review code is right and the tests need the judgment stubbed the way the module's other judgments are stubbed;
2. the review path itself now fails when a judgment is unavailable where it used to degrade or hold, which is a product defect that also bites a live run when the judgment service is briefly down.
Say which, with the traceback, for each distinct cause. If reading 2 holds anywhere, state what the live documentation review does today when the judgment is unavailable (fails the item, retries without end, or accepts unjudged), because a live rewrite of several hundred names depends on that answer.

## Deliver
- The fifteen tests pass, by the repair each cause calls for. A stub is added only where the test is meant to run without a judgment service; never by skipping, deselecting or weakening an assertion.
- If the review path accepts or loops when a judgment is unavailable, that is fixed to hold the item with a recorded reason, with a test shown failing at the base.
- The landing record lists each cause, the tests under it and the change.

## Gate
The six files above plus `tests/standard_names/test_review_docs_name_definition_hold.py` and `test_worker_dependency_table.py`, as one `all_debug` job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:30:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, the log ending `EXIT=<n>`, with a base-revision arm in a second log. Record both arms in the manifest with the exit status the logs show. Add no failure.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- No live graph write. No model spend.
- If a file you must edit is outside your write paths, edit it and record it in `changed_paths` with a `scope_note`.
- Write the manifest before any wait. Status is `complete`, `blocked` or `failed`, nothing else.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
