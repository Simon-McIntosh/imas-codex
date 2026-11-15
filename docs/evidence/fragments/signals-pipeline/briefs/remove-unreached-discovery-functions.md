# Remove three unreached signal discovery functions

## Goal

Delete the three functions in the signals discovery package that no caller
reaches, leaving the TDI scanner's path as the only TDI discovery
implementation.

## What is known

A code review of `imas_codex/discovery/signals` found, and a search of
`imas_codex/`, `tests/` and `scripts/` confirmed, that these have no caller:

- `discover_mdsplus_signals` in `imas_codex/discovery/signals/parallel.py`.
  It embeds its own remote MDSplus tree walk as an inline script, a capability
  the repository already owns in `imas_codex/remote/scripts/extract_tree.py`
  and `imas_codex/discovery/mdsplus/pipeline.py`.
- the synchronous `discover_tdi_signals` in
  `imas_codex/discovery/signals/parallel.py`. Every reference to that name
  resolves to the asynchronous function of the same name in
  `imas_codex/discovery/signals/tdi.py`, which the TDI scanner calls.
- `run_tdi_discovery` in `imas_codex/discovery/signals/tdi.py`.

Together they are about 250 lines.

## Constraints

- Confirm each function is still unreached before deleting it. If one has
  gained a caller, leave it and say so in the manifest.
- Delete only these three functions and any import, helper or constant that
  becomes unused solely because of the deletion. Do not restructure anything
  else in `parallel.py`.
- Keep `discover_tdi_signals` in `tdi.py`; it is live. Correct the `tdi.py`
  module docstring if it presents `run_tdi_discovery` as the entry point.
- Tests run from the worktree against the shared environment:
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider <targets>`,
  submitted to a compute partition with
  `env -u SLURM_JOB_ID -u SLURM_JOBID srun --partition=all_debug --time=00:59:00 --cpus-per-task=4 --mem=16G`.
- Commit locally with a conventional subject and a body. Do not push or merge.

## Done when

- `grep -rn "def discover_mdsplus_signals\|def discover_tdi_signals\|def run_tdi_discovery" imas_codex/discovery/signals/`
  returns only the definition in `tdi.py`.
- `tests/discovery/` and `tests/cli/test_discover_signals_stage.py` show no
  added failure against the base revision, with the base and after failure
  ids recorded.
- The net change in lines is negative and is stated in the manifest.
- Negative control: `import imas_codex.discovery.signals.parallel as p; p.discover_tdi_signals`
  raises `AttributeError` after the deletion and resolves before it; both
  outcomes are recorded.
