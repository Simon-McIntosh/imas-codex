# Brief: the first group's defective members, repaired two ways and compared

Lead ruling, 2026-11-16: for the members of the first release group that the audit reads as defective, plain regeneration and agent-proposed repair are run side by side and compared by the audited result. This brief fixes how that comparison is run. It serves section 4 of `sn-audited-group-release` and changes nothing in that section's done-when.

## Reuse
- The audit is the one built for section 1: its sampler, reader seats, planted goldens and review-record shape (see `blind-audit.md`). The group is read in full with the same instructions.
- The cut is `sn release --domain <group>` against the release gates carrying the bar's causes (see `release-bar.md`); add no cut path of your own.
- Regeneration uses the pipeline's own reset and stages; agent proposals use `sn edit` and the signed rename path.

## Inputs
- The first group and its full audit: every member read by the two blind readers of section 1, with one defect class or none each. Use that record as it stands; do not re-read to pick the population.
- The release bar ruled in section 3.
- The name review's source-context repair must be on main before any name is regenerated (it is a declared wait of section 4).

## Step one: sort the defective members by cause before repairing any
For every member both readers call defective, state the cause in one line and put it in one of two bins, recorded in the evidence with the count in each:
- **Recurring.** The same cause holds for three or more members (for example, a qualifier the source's enclosing structure states and the name omits). A recurring cause is a pipeline defect. Do not repair these names one at a time in either arm: report the cause with its members and stop for the coordinator, who decides whether the pipeline is repaired first.
- **Singular.** Everything else. These are the population of the comparison.
Members only one reader calls defective are listed separately and are not repaired here.

## Step two: assign arms without choosing
Split the singular members into two arms by a seeded shuffle within each defect class (record the seed), so each class is spread across both arms. An odd member goes to arm A.

## Arm A: regeneration
Each member is reset and regenerated from its source through the pipeline's own stages, with the enclosing-structure context, and reviewed by the pipeline. No steer beyond what the pipeline itself supplies.

## Arm B: agent-proposed repair
A worker on the local lane works one member at a time, with read access to the graph and the Data Dictionary and no write access to stored text.
- It may read the source and every enclosing structure, sibling names of the same family and the group's other names, and the two audit readings for the member.
- It may **propose** exactly one of: a steer for the name or the documentation; a rename; a rebind to another source; a withdrawal. Each proposal is submitted through the signed edit path (`sn edit`, or the signed rename path) and carries the source text it rests on. Read `.claude/skills/sn-edit` first and dry-run each kind once.
- It never writes a name, description or documentation by Cypher or by any path other than those, and never marks a member accepted. Acceptance is the pipeline's review and then the audit.
- It gets one proposal per member. A member whose proposal the pipeline refuses is recorded as not repaired, with the refusal.

## Judging: the same blind audit for both arms
After both arms settle, every member of both arms is read again by the two blind readers of section 1, under the same instructions, with the arm hidden and the members shuffled together with twenty unchanged sound members of the group as a control. Report, per arm and per defect class: members repaired (both readers now read no defect), members still defective, members made worse (a new defect class), and the control members' readings. State plainly which arm did better and on which classes, and where the counts are too small to say.

## Limits
- One graph writer at a time. Before the first write, confirm no other writer is running (no `sn run`, `sn edit` or signed apply of this account in `squeue` or the process table) and say what you checked. The two arms write in turn, never together.
- Spend: arm A at most USD 15, arm B at most USD 5 of metered spend (the local lane is not metered), the re-audit at most USD 10; all inside section 4's USD 40. Read spend from the pipeline's cost records and the calls' recorded usage.
- Stop and report at once if a signed apply is refused for drift, if a review fails with an error rather than a verdict, or if live names without a source rise.
- Heavy work runs as jobs: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-live-rewrite/>`, or chained on `rigel`. Progress file under `~/.cache/imas-codex-live-rewrite/`, one line per member when submitted and one when it settles. Run scratch there is yours to write; your declared write paths bound only what you commit.

## Deliver, under `docs/evidence/fragments/sn-audited-group-release/`
A page and a data file: the two bins with causes; the arm assignment with its seed; per member, the before text, the proposal or regeneration, the after text, the pipeline's review and both blind readings before and after; the per-arm, per-class table; spend per arm beside its ceiling. A table, not a figure, unless a figure shows something the table cannot.

## Working conditions
- Copy (never symlink) the five gitignored generated files from the main checkout with `cp -n`; never stage them. `UV_NO_SYNC=1`; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync ...`.
- Manifest lists are comma-separated; commits are bare hashes. Status is `complete`, `blocked` or `failed`, nothing else; commit before you write `complete`. Never kill by pattern. Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.
