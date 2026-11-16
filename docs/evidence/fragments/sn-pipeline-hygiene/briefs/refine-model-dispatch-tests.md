# Brief: the two red refine model dispatch tests

## The state
`tests/standard_names/test_refine_model_dispatch.py::test_refine_env_override_independent_of_compose` and `::test_refine_does_not_alias_compose` fail on main when the whole `tests/standard_names` directory is run from an archive. `settings.get_model("sn-refine")` raises `ValueError: Model seat 'sn-refine' has no recognised route for 'test/refine-only'` (and `'test/refine-canary'`) from `_validate_model_identifier` in `imas_codex/settings.py`. The route check arrived on main on 2026-11-15 (`ae10df18f`, `03733ae27`); the tests set placeholder model identifiers that predate it. Log: `/home/ITER/mcintos/.cache/imas-codex-live-rewrite/merged-head/standard-names-bd3f019a0.log` (2 failed, 8766 passed).

## Deliver
- Establish from the code which is right, and say which: the tests encode what they mean to assert (the refine seat reads its own setting and never falls back to the compose seat's) with identifiers the route check now legitimately refuses, or the route check refuses identifiers a seat should accept.
- Repair by what the cause calls for. If the tests are at fault, keep both assertions and give them identifiers with a recognised route, without weakening the route check and without a test-only escape in `settings.py`. If the check is at fault, fix the check and leave the tests.
- Look for the same placeholder pattern in the other test files under `tests/` that set a model seat, and say what you found.

## Gate
The test file alone, then the whole `tests/standard_names` directory at your head from an archive as one job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:50:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, the log ending `EXIT=<n>`. Extract the archive into the job's own `/tmp`, not under `~/.config`, so the catalog checkout resolves. Zero failures at your head. The base arm is the log above.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- No live graph write. No model spend.
- Your declared write paths bound what you commit in the repository; job logs under `~/.cache/imas-codex-live-rewrite/` are run scratch and are yours to write. If the cause lies in a file outside your write paths, edit it and record it in `changed_paths` with a `scope_note`.
- Manifest lists are comma-separated; commits are bare hashes; `baseline_suite` and `after_suite` are JSON objects. Never kill by pattern. Write the manifest before any wait. Status is `complete`, `blocked` or `failed`, nothing else; commit before you write `complete`. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
