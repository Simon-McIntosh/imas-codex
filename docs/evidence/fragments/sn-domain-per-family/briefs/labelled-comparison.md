# Brief: the per-family domain decision measured against blind readers (sn-domain-per-family §2)

The mechanism landed (`decide_family_domain`, `store_family_decision`, `fan_out_family_decision` in `imas_codex/standard_names/domain_decision.py`; `fetch_domain_families` in `families.py`). Two done-when clauses of the section are unmeasured: the labelled comparison and the calibration regression. This node measures them and changes no product behaviour.

## Reference labels (never produce your own)
- `docs/evidence/fragments/sn-domain-family-labels/blind-sheet.json`: 40 families (pattern, three example members) and 73 single names.
- `labels.json` and `labels-second-reader.json` in that directory: two readers who saw no stored domain. Use only items where the two agree as the reference (37 of 40 families, 61 of 70 drawn single names). List the items they disagree on; resolve none.
- The sheet groups families by species or component token; the decision groups by the repository's pattern family with the locus fixed. Map each sheet family to the decision's family through its example member names, and report any sheet family that maps to more than one decision family.
- Calibration labels: `docs/evidence/fragments/sn-catalog-grouping/scg-domain-label-sheet.json` with its incumbent reading in `scg-domain-incumbent.json`.

## What to measure
- For the agreed families: the share of member names the per-family decision files under the reference domain, beside the share the stored per-name domains reach. Stored domains are read from the live graph read-only.
- For the agreed single names: the same two shares, with `safety_factor`, `greenwald_density` and `voltage_of_valve` reported apart.
- The calibration regression reading the section asks for.
- Families the decision reports undecided, with their members and two leading domains.
- Judgments paid and their cost.

## You may not settle
- Whether a reference label is right: where the decision and the agreed reference differ, list the family with both domains for the lead.
- The option set: this node measures the decision as landed. Widening the options is another section.

## Spend
Domain judgments run on the configured judgment seat; keep under USD 2 and record it.

## Standing rules for this node
- Read your plan section from your worktree; it is the authority for the endpoint. This brief adds the reuse map and the questions you may not settle.
- Export `UV_NO_SYNC=1`, use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent, then run `build-models --force` in your worktree if the schema differs.
- Heavy gates (any test file opening the graph, any pass over the whole catalog) run as their own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-<node>/>`. A bare `srun --partition` from this node hangs. Never run them on the node you start on.
- Do not write the live graph. Use the disposable_neo4j fixtures in `tests/standard_names/conftest.py` for every control.
- Write your manifest with what you have before any long wait. Status is `complete`, `blocked` or `failed`.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
