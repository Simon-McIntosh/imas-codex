# Brief: the last refine rotation and the escalation seat

## The state
`tests/standard_names/test_refine_attempt_budget.py::TestEscalationFiresOnTheLastRotation::test_last_rotation_uses_the_escalation_seat[3-True]` fails on main when the whole `tests/standard_names` directory is run from an archive: the call on attempt 3 of 3 is made with the refine seat's model where the test expects the escalation seat (`assert 'refine-seat' == 'openrouter/anthropic/claude-sonnet-5.5'`). Two logs show the same failure at two revisions: `/home/ITER/mcintos/.cache/imas-codex-live-rewrite/sdbc-gate/base-gate.log` and `head-gate.log`.

## Deliver
- Establish the cause from the code and the traceback, and say which it is: the product no longer escalates on the last rotation under the conditions the test sets (a product defect: a name's last refine attempt runs on the cheaper seat), the test encodes a superseded expectation, or the test passes alone and fails only in the directory run (an isolation defect; name the test that leaks state).
- Repair it by what the cause calls for. A product defect is fixed in the product with the test left as it is. Never skip, deselect or weaken the assertion.
- If it is a product defect, say at the top of the manifest since which commit, so the live rewrite's exhausted names can be judged: thirteen names ended `exhausted` in a pass that ran on main's product source.

## Gate
The test file alone, then the whole `tests/standard_names` directory at your head from an archive as one job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:50:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, the log ending `EXIT=<n>`. Zero failures at your head. The base arm is the head-gate log above.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- No live graph write. No model spend.
- If the cause lies in a file outside your write paths, edit it and record it in `changed_paths` with a `scope_note`.
- Never kill by pattern. Write the manifest before any wait. Status is `complete`, `blocked` or `failed`, nothing else. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
