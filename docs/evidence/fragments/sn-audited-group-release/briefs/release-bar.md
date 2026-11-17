# Brief: the release bar (section 3 of sn-audited-group-release)

## Reuse: the bar is new causes on the existing gates, not a second layer
The plan's design review found an owner for almost every refusal the section lists. Add to them; add no verb, no wrapper and no new module that forwards to them.
- `imas_codex/standard_names/export.py`: the eligibility filter already refuses `name_not_accepted`, `missing_physics_domain`, `outside_requested_domain` and `invalid_validation_status`; the gate registry (the four gates, exclusion accounting, `all_gates_passed`) is where a cut is refused. `sn release --domain` already scopes a cut to one physics domain.
- No source: `source_invariant.find_sourceless_live_name_rows`.
- Family membership: `families.detect_families`. A family with a member in another group has no wired owner yet; its home is the cross-field consistency gate (`_run_gate_b`), which already censuses cross-domain edges.
- Shared wording not the rendering of its stored sentence: `docs_gates.refuse_altered_shared_parts` / `shared_part_findings`.
- New: documentation not accepted, and the group's audited defect rate above the ruled bar or never measured. The rate is read from the audit records of section 1 and its bar from the lead's ruling recorded on the plan.

## Interface budget
State in the evidence how many refusal causes the change adds and which existing cause each folds into or sits beside. No new command-line option is expected: `--domain` already selects the group.

## Method points
- The section's recorded refusals are gate tests: one planted group per fault, each refused with its cause named, and an unplanted group cut.
- The bar itself is the lead's ruling. If it is not recorded on the plan when you reach that clause, build the refusal reading the bar from configuration, test it with a stated test value, and end `blocked` with `wait_condition: the lead has ruled the release bar` after committing.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Heavy work runs as jobs: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, or chained on `rigel`; a bare `srun --partition` hangs where you start. Suite arms are extracted from a git archive into the job's own `/tmp`, never under `~/.config`. Run scratch and progress files under `~/.cache/imas-codex-live-rewrite/` are yours to write; your declared write paths bound only what you commit.
- Manifest lists are comma-separated; commits are bare hashes; `baseline_suite` and `after_suite` are JSON objects naming the exact revisions run. Status is `complete`, `blocked` or `failed`, nothing else; commit before you write `complete`. Never kill by pattern. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
- A figure only where it shows something a table cannot; if you draw one, embed it in the page, save it through the repository's style module and look at the render before committing.
