# Brief: one domain decision per family (sn-domain-per-family §2)

## Reuse map (extend these; do not add a parallel module)
- The family: `imas_codex/standard_names/families.py` `pattern_signature` over `PATTERN_INVARIANT_SEGMENTS` / `PATTERN_VARIATION_AXES`, assembled by `fetch_pattern_families` / `fetch_pattern_family`; `_family_member_collect` already returns each member's `physics_domain`. The plan's family is this one with position, geometry and region held fixed. Do not write a new grouping; do not use the parent-edge route in `harmonize.assemble_family`.
- The decision and its only writer: `imas_codex/standard_names/domain_decision.py` `candidate_domains`, `build_state`, `build_question`, `decide_domain`, `write_name_domain`. The guard `tests/standard_names/test_name_domain_single_writer.py` and the four writer-site tests must pass unchanged.
- One function takes a family and answers for every member: option union, one judgment, one stored record, fan-out through the one writer. A loop calling the per-name decision once per member is the wrong design: it pays per member and can disagree.
- The decision's stored fields are schema scalars; a family-level record is a schema change in `imas_codex/schemas/` followed by `uv run build-models --force` in your worktree. Never stage generated models.
- Calibration labels and accuracy harness: the `sn-catalog-grouping` evidence under `docs/evidence/fragments/sn-catalog-grouping/`.

## You may not settle
- The labelled multi-member family set: you assemble the 30 or more families and their member lists and commit them unlabelled; you do not assign the reference domains. The coordinator dispatches an independent reader. Report the done-when that needs them as waiting, with everything else measured.
- Any write to the live graph: that is §3.
- Spend: Jev judgments for the measurements here are capped at USD 3; record what you spend.

## Standing rules for this node
- Read your plan section from your worktree; it is the authority for the endpoint. This brief adds the reuse map and the questions you may not settle.
- Export `UV_NO_SYNC=1`, use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent.
- Heavy gates (any test file opening the graph, any pass over the whole catalog) run as their own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-<node>/>`. A bare `srun --partition` from this node hangs. Never run them on the node you start on.
- Do not write the live graph. Use a disposable graph or fixtures for every control.
- Write your manifest with what you have before any long wait. Status is `complete`, `blocked` or `failed`.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.

