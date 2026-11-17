# Brief: the domain groups measured (section 2 of sn-audited-group-release)

## Reuse
- `imas_codex/standard_names/domain_audit.py` `audit_domains` already produces the per-name domain state (`DomainAuditResult`, `NameDomainState`, `DomainCheck`): live count per domain, whether the domain is among the sources' domains, agreement. Extend its result with the ranking and the split proposal; do not re-measure with new queries.
- `imas_codex/standard_names/families.py` `detect_families` owns family membership; the families whose members sit in more than one group come from it.
- The wrong-domain counts come from the audit records of section 1, read from the review-record store.

## Method points
- State the measure you rank by and show the ranking's sensitivity to it (a second reasonable measure, same table).
- The release order and any split of `transport` are the lead's to rule. Deliver them as a proposal with the evidence beside each option, and end `blocked` with `wait_condition: the lead has ruled the group order` after committing; do not write `complete` before the ruling is recorded on the plan.
- No stored domain changes.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Heavy work runs as jobs: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, or chained on `rigel`; a bare `srun --partition` hangs where you start. Suite arms are extracted from a git archive into the job's own `/tmp`, never under `~/.config`. Run scratch and progress files under `~/.cache/imas-codex-live-rewrite/` are yours to write; your declared write paths bound only what you commit.
- Manifest lists are comma-separated; commits are bare hashes; `baseline_suite` and `after_suite` are JSON objects naming the exact revisions run. Status is `complete`, `blocked` or `failed`, nothing else; commit before you write `complete`. Never kill by pattern. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
- A figure only where it shows something a table cannot; if you draw one, embed it in the page, save it through the repository's style module and look at the render before committing.
