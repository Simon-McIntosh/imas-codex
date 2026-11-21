# Keep the paths of directories that do not move out of the rename

## Goal

Stop the rename tool rewriting filesystem paths to three directories that
stay where they are at the cutover, and test the shipped kept contexts
themselves.

## Why

The session coordinating standard names ran the tool's rules on sample
strings at the primary branch head on 2026-11-21 and found that it rewrites
paths the cutover sequence says do not move. After the tracked rename, the
repository's own instructions would point every worker's environment variable
at a path that does not exist, and `uv` would then build a second environment
in the worktree, which is the failure those instructions exist to prevent.

## The three directories that do not move at this cutover

1. The checkout on this workstation, in its absolute, `~` and `$HOME`
   spellings (the directory named after the old project under the user's code
   directory).
2. Reckon's worktree root for the project, whose directory name starts with
   the old project name followed by a hyphen and a digest.
3. The saved gate-script cache directory whose name is the old project name
   followed by `-live-rewrite`.

The peer measured, over files the rename rewrites: 63 files name the first,
17 the second and 5 the third. They include the worktree recipe in the root
agent instructions and one line in the compute CLI.

## What must still be renamed

The state, cache and config roots that the cutover does move: the project's
directories directly under the user's share, config and cache directories.
The pattern must tell the cache root that moves (the old name as a whole path
component) from the gate-script directory that stays (the old name followed
by `-live-rewrite`), and the checkout directory from a repository URL or a
package directory of the same spelling.

## Done when

- The shipped kept list holds a kept context for each of the three
  directories, each with a floor, and the reason recorded.
- A test reads the kept contexts from the shipped kept-name file itself, not
  from a copy the test writes, and applies them with the rename to literal
  sample strings for each directory in each spelling: every one is unchanged.
  The same test shows the three roots that move are renamed, a repository
  URL is renamed, and an import of the package is renamed. Deleting any
  shipped context, or lowering its floor to zero, fails a test.
- Applied to a scratch copy of the current primary branch head, the tool
  leaves every occurrence of the three directories' paths unchanged, with the
  count of occurrences and of files for each in the evidence, and the worktree
  recipe in the root agent instructions is byte-identical in its path lines
  before and after; a second application changes nothing; the closing search
  reconciles with zero leftover.
- The default tier adds no failure after the rename, both arms recorded in
  the manifest as single-line JSON objects with whole test ids; the rehearsal
  counts are stored in a log the manifest names, not only in prose.
- `tests/scripts` passes. Scratch copies and environment are removed, with
  removals printed.

Negative control: with the checkout context removed from the shipped list,
the sample-string test fails on the environment-variable path; with the
gate-script context removed, the cache root and the gate-script directory are
both renamed and the test fails. Each is a pytest run in a scratch copy with a
log that shows a non-zero exit and a failing test id.

## Rules

- Scratch copies only, made with `git worktree add` under node-local `/tmp`
  inside a compute job, with their own environment there. Never sync or
  repoint the shared environment. Change nothing under the home directory, no
  service, no repository setting, no graph content and no reckon state.
- Do not write the user's real home path into a new tracked file: tests use
  an invented home directory and the same directory names.
- Heavy work as its own job:
  `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`, logs on GPFS,
  each pytest log ending with pytest's summary and one EXIT line.
- The manifest's `tests` field is one literal command that parses; failure
  ids are whole (everything between `FAILED ` and ` - `).
- Run `ruff check --fix` and `ruff format` on exactly the files you change and
  add no lint finding against the base.
- Commit locally with a conventional subject and a body, and give `commits`
  as a JSON list of full shas. Do not push or merge.

## Scope

Write paths: `scripts/rename`, `tests/scripts`, and
`docs/evidence/fragments/codac-uda-signals/keep-unmoved-directories-in-the-rename.html`.
