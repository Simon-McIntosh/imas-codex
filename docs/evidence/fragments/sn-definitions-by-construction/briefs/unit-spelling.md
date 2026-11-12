# Brief: one unit spelling per quantity (sn-definitions-by-construction §3)

## Reuse map (extend these; do not add a parallel module)
- Dimension equality of two unit strings: `imas_codex/standard_names/audits.py` `_dimension_sets_equivalent`, `unit_dimension_check`, `name_unit_consistency` family. The census extends these.
- Rewriting a Data Dictionary unit at extraction: `imas_codex/standard_names/unit_overrides.py` reading `config/unit_overrides.yaml`. The per-quantity spelling record is a table that mechanism reads, placed beside it.
- `imas_codex/standard_names/unit_audit.py` `check_unit_sanity` covers five special cases; fold new name-versus-unit rules there or into the audits owner, one place.
- Unit normalisation: `imas_codex/units/__init__.py` `normalize_unit_symbol`.

## Starting evidence
`docs/evidence/fragments/sn-definitions-by-construction/published-unit-census.txt` and its script list the published outliers: torque density N.m^-2 (51) vs kg.m^-1.s^-2 (8); torque N.m (18) vs kg.m^2.s^-2 (2); six source rates in Hz vs 45 in s^-1; particle fluxes in s^-1.

## You may not settle
- Whether a wall-integrated rate named `*_flux_*` with unit s^-1 is renamed or re-united: list it for §5, change nothing.
- eV versus K for a temperature, rad.s^-1 versus Hz for a frequency: these carry meaning; list with reason, do not unify.
- Any change of a live name's unit in the graph: this node delivers the rule, the census and the corrected write path on a disposable graph. The live rewrite is §6.

## Standing rules for this node
- Read your plan section from your worktree; it is the authority for the endpoint. This brief adds the reuse map and the questions you may not settle.
- Export `UV_NO_SYNC=1`, use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent.
- Heavy gates (any test file opening the graph, any pass over the whole catalog) run as their own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-<node>/>`. A bare `srun --partition` from this node hangs. Never run them on the node you start on.
- Do not write the live graph. Use a disposable graph or fixtures for every control.
- Write your manifest with what you have before any long wait. Status is `complete`, `blocked` or `failed`.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.

