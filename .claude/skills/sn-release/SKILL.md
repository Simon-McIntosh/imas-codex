---
name: sn-release
description: >-
  Route and gate an IMAS Standard Names catalog release. Use when cutting,
  reviewing, finalizing, approving, or discussing a catalog batch; use the
  separate repository runbooks for grammar-library and iter-codex package
  releases.
allowed-tools: Bash(*)
---

# sn-release — route catalog releases to the authoritative recipe

## Authority

Read `iter_codex/standard_names/AGENTS.md`, section **Release recipe**, in the current
`iter-codex` checkout before acting. It owns every operational fact, command, gate, stop
condition, and ordering decision; do not restate or infer them here.

If this skill and the release recipe disagree, the recipe wins and this skill is at fault.

Before any write, run `iter-codex sn release --help` and `iter-codex sn release status`.
If status cannot be shown, refuse to proceed: an unresolvable target is unknown, and
unknown is not permission.

## Repository routing

- `iter-codex` holds the graph pipeline and frozen batch roster artifacts; start here.
- `IMAS-Standard-Names` owns the grammar and controlled vocabulary and must never receive
  a catalog batch. Its package-index release is in that repository's `AGENTS.md`, section
  **Release Workflow**, and is outside this skill.
- `imas-standard-names-catalog` owns the catalog definitions and alone receives batches.
- The `iter-codex` package release is a different workflow governed by
  `iter_codex/cli/AGENTS.md`, section **Release Workflow**.

## Target and series gates

The operator must read the resolved `Path` and both remotes printed by status. Anything
other than the catalog checkout is a stop. Repair `[tool.iter-codex.sn].isnc-dir` in
`pyproject.toml`, resolved by `iter_codex/settings.py:get_sn_isnc_dir`, and rerun status;
never hide a bad resolution for one invocation with `--isnc` or `ITER_CODEX_SN_ISNC`.

The candidate must continue the target's own series. Otherwise, treat it as a target error,
not a numbering surprise, and stop. In a mid-candidate state, a version-bump flag is wrong;
the commands printed by status are authoritative.

## Pull-request target gate

State the pull-request target whenever a catalog release is discussed or authorized. The
release's pull request targets the upstream organization; a personal-fork review request is
a separate state. Say **no pull request** only when neither exists; say **no upstream pull
request** when a fork review pull request may still exist.

After these gates pass, follow the release recipe exactly and take every next action from it.
