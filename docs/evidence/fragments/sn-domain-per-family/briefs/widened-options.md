# Brief: a family's domain options are not bounded by its sources (sn-domain-per-family §5)

Your plan section states the endpoint and every bar. This brief adds the reuse map, the reference data and the questions you may not settle.

## Reuse map (extend these; add no second decision)
- The one family decision is `decide_family_domain` in `imas_codex/standard_names/domain_decision.py`. Its options come from `family_candidate_domains` (and `candidate_domains` for the per-path form), its evidence from `build_family_state` and `family_path_counts`, its question from `build_family_question`. The widening is a change to what options that decision is offered and how the sources' domains are presented as evidence. `store_family_decision` and `fan_out_family_decision` are not changed.
- Domain grouping already exists: `band_of`, `band_umbrella`, `option_text` and the guidance they read. A wider option set built from those bands is one candidate; the flat choice over all 32 domains was measured at 0.560 and refused, so do not re-run it as the proposal.
- Families come from `fetch_domain_families` in `imas_codex/standard_names/families.py`. A single name is a family of one.
- The measurement harness exists: `docs/evidence/fragments/sn-domain-per-family/labelled-comparison-run.py` and its output `labelled-comparison.json` already map sheet families to decision families, apply the agreed-reader reference, and read the calibration sheet. Extend it to run both option sets in one run; do not write a second harness.

## Reference labels (never produce your own)
- `docs/evidence/fragments/sn-domain-family-labels/blind-sheet.json`, `labels.json`, `labels-second-reader.json`, `disputed-items.json`. Only items where the two readers agree are the reference. Resolve no disputed item.
- Calibration: `docs/evidence/fragments/sn-catalog-grouping/scg-domain-label-sheet.json` with `scg-domain-incumbent.json`.
- Readings to beat or hold, from `labelled-comparison.json`: 35 families at 48.6 % of member names; 61 single names at 63.9 %; calibration 60.4 % with undecided counted wrong; 42 undecided families.

## You may not settle
- Whether a reference label is right. Where the decision and the agreed reference differ, list both domains.
- The bars. If no option set you try clears them, report `blocked` with each set's readings; do not weaken a bar or change which names it is taken over.
- For a family left below the floor in the member-share comparison, apply one rule to both arms and state it.

## Spend
Judgments run on the configured judgment seat. Keep the node under USD 5 and record judgments paid, cost per decision and the projected cost of a live pass.

## Standing rules for this node
- Read your plan section from your worktree; it is the authority for the endpoint.
- Export `UV_NO_SYNC=1`, use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent, then run `build-models --force` in your worktree if the schema differs.
- Heavy work (any test file opening the graph, any pass over the whole catalog, the whole-catalog decision in a disposable copy) runs as its own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-sdpf-widened-options/>`. A bare `srun --partition` from the node you start on hangs. Never run heavy work where you start, and never start a whole test directory there.
- Read the live graph read-only. Never write it. Use the disposable_neo4j fixtures in `tests/standard_names/conftest.py` for every control and for the whole-catalog pass.
- Write your manifest with what you have before any long wait, and write evidence files as results arrive. Status is `complete`, `blocked` or `failed`, nothing else.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
