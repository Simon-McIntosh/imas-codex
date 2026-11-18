# Coordinator instructions for rehearsing the rename

The plan section is the authority for what is delivered. These instructions add
the inputs, the peer conditions and the operating rules.

## Inputs

- The consumer inventory, with a count and a command per surface and the list
  of surfaces that keep the imas name:
  `docs/evidence/fragments/codac-uda-signals/current-name-consumer-inventory.html`.
- The three locked decisions in the plan: both repositories are renamed, the
  import package becomes `iter_codex`, the full server advertises `iter-cx`,
  and the data-dictionary server keeps `imas-dd`.

## Surfaces that keep the imas name

The standard-names catalog and grammar library (`imas-standard-names`,
`imas_standard_names`, the catalog repository and its fork), the
data-dictionary server and its graph labels, the IMAS data-dictionary version
settings, `imas-python` usage, and the data-dictionary-only server name. A
change to any of these is a defect in the rename.

## Conditions the cutover sequence must cover

The session coordinating standard names gave these on 2026-11-18. Each needs a
step, a check and an undo in the cutover sequence:

- Reckon's project key, `docs/state/<project>`, run records, reviews, the crew
  file and the worktree root that carries the project name.
- Worktrees retained with unmerged commits: each is resolved or carried across
  before paths move. List them from `git worktree list` at the time; do not
  touch them.
- Saved gate scripts under `~/.cache/imas-codex-live-rewrite/` and node briefs
  on the primary branch that write the checkout path or the package name: say
  for each whether the rename rewrites it or its owner regenerates it.
- State, cache and config roots under the home directory, service and timer
  units, scheduler job names, registry images, MCP client configurations and
  the two stored graph templates that import the old package name.

## Rules

- Work only in scratch copies made with `git worktree add` under node-local
  `/tmp` inside a compute job, and in an environment built for that copy alone
  under the same directory. Never sync, repoint or write the shared
  environment at the main checkout. Never rename, move or edit anything under
  the home directory, any service, any repository setting, the graph or
  reckon's state.
- The rehearsal reads the surfaces outside the tracked tree; it changes none.
- Run the rename and both test tiers as jobs of their own:
  `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`, logs on GPFS.
  An environment build is heavy work and belongs in the job.
- A copy's `.env` is linked from the main checkout; generated models are built
  inside the copy, never linked.
- Removal of scratch copies and the scratch environment prints what it removes
  and uses `git worktree remove` without force for the copies.
- No real CODAC host, variable, pulse or document identity goes into a tracked
  file.
- Commit locally with a conventional subject and a body, and give `commits` in
  the manifest as a JSON list of full shas. Do not push or merge.
