# Brief: close what the review of the model-seat change left open

The change that put every model seat on its provider's latest release landed on main. Its review (`~/.config/reckon/crew/runs/r-20261114T122906089714-sdbc-model-seats-latest/review-records/`, the record at head `1bad1af53`) raised no blocking finding and eight smaller ones. Read the record first; it cites file and line for each.

## Deliver
- Every model identifier a Python default names resolves to a cataloged price: the benchmark candidate default in `imas_codex/settings.py` names `openrouter/deepseek/deepseek-v4-pro`, which has no price entry and whose proxy route was renamed. A test enumerates the defaults and fails on an unpriced one; show it failing at the base.
- `imas_codex/llm/README.md` and any other document under `imas_codex/` name the models the code now uses.
- The escalation test fails for the right reason: with the escalation seat pointing at a different priced model, the refine worker's last rotation resolves that model, and the test fails if the rotation ignores the seat. The declared negative control is that mutation, not a deleted key.
- The refine-default guard states the capability it protects as an allow-list of the seated lines, not a deny-list of substrings.
- The fetched OpenRouter model record the latestness claims rest on is preserved beside `model-seats-latest.json` as a compressed file, or the evidence says it was not kept and cites the release dates it used.
- The seat-resolution listing runs inside the gate, at the head revision.

## Gate
The seat, settings and refine-dispatch tests plus every file under `tests/` that imports `settings` model accessors, as one `all_debug` job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:50:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, the log ending `EXIT=<n>`. A bare `srun --partition` hangs where you start.

## Working conditions
- No graph write, no pipeline run, no change to which model any seat names. Spend at most USD 0.10 if a probe is needed.
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- If a file you must edit is outside your write paths, edit it and record it in `changed_paths` with a `scope_note`. Never kill by pattern.
- Write the manifest before any wait. Status is `complete`, `blocked` or `failed`, nothing else. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
