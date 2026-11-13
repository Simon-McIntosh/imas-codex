# Brief: the standard-names suite passes on main

## The state
The whole `tests/standard_names` directory was run from an archive of main and of the revision before the name-against-definition check landed, on a debug partition:

- main (`e8c1bc02a`): 19 failed, 8,730 passed. Log: `/home/ITER/mcintos/.cache/imas-codex-live-rewrite/merged-head-standard-names-e8c1bc02a.log`.
- before the name check (`5085c54e9`): 25 failed, 8,707 passed. Log: `/home/ITER/mcintos/.cache/imas-codex-live-rewrite/base-before-name-check-standard-names.log`.

Six failures are on main and not at the earlier revision, so the name check and its repairs added them: five in `tests/standard_names/test_jev_docs_review.py` and `tests/standard_names/test_refine_failed_checks.py::test_shipped_choice_conforming_forms_are_explicit`. The other thirteen are older and fail at both revisions; their ids are the `FAILED` lines common to both logs (`test_atomic_refined_name_persistence`, `test_catalog_layout_hierarchy`, `test_docs_opening_insertion`, `test_edit_prompt_injection`, `test_house_style_checks`, two further `test_jev_docs_review` tests, `test_manifest_generability`, `test_package_import_boundaries`).

## Deliver
- Every one of the nineteen passes, by the repair its cause calls for. Establish the cause per test from its traceback first and record it: a product defect is fixed in the product; a test that encodes a superseded expectation is updated to the behaviour the code and its plan records now hold, with the record cited; a golden file is regenerated only when the diff is read and is the intended change. Never skip, deselect or weaken an assertion to reach green.
- For the six added failures, say which landing caused each and why its gate did not catch it.
- Where a failure shows the product is wrong in a way a live documentation or name rewrite would hit, say so at the top of your manifest: a live rewrite of several hundred names is waiting on this.

## Gate
The whole `tests/standard_names` directory at your head, from an archive, as one `all_debug` job with no parallel workers (about fifteen minutes): `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:50:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, the log ending `EXIT=<n>`. The base arm is the main log above. Zero failures at your head, or each remaining one listed with why it is not yours to fix. Record both arms in the manifest with the exit status the logs show.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- No live graph write. No model spend.
- Your write paths name the test files and the modules most likely involved. If the cause lies in another file, edit it and record it in `changed_paths` with a `scope_note`.
- Never kill by pattern (`pkill -f`, `killall`); list pids and signal those.
- Write the manifest before any wait. Status is `complete`, `blocked` or `failed`, nothing else.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
