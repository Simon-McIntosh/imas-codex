# Every graph push that carries facility content lands in a private package

## Goal

A graph push whose archive holds any facility node fails unless its registry
package reads back as private.

## Why

The lead's standing rule: ITER graph content (built from ICP and IDM records)
must never be public, and is protected by a private backup. The push command
already refuses a public package, but only on the facility-only path
(`_report_plant_package_visibility` in `iter_codex/cli/graph/registry.py` returns
early unless `facilities and without_dd`). The weekly timer runs
`iter-codex graph push --cycle`, a full-graph push that includes the ITER
facility, and that path never checks visibility. On 2026-11-23 the coordinator
read the existing packages: all three are private, the last full push was
2026-11-11, before the package rename. After the rename the full graph pushes
to a package name that does not exist yet, so the next weekly cycle creates a
new package whose visibility nothing verifies.

## What must be true afterwards

- Every push path (full graph, cycle, facility-filtered, remote) that writes an
  archive containing any facility's nodes reads the package visibility back
  from the registry after the write and fails, loudly and with a nonzero exit,
  if it is anything but private. A Data-Dictionary-only push is unaffected.
- The check runs before any tag is recorded as the latest good push, so a
  failed check never becomes the reference backup.
- Whether a dump holds facility content is decided from the dump itself or its
  stated scope, not assumed from the flags; say in the evidence which and why.
- The evidence states where each path pushes (registry and package name) after
  the rename, and that no push is made by this node.

## Done when

- Tests cover each push path: with the registry faked to report public for a
  package that holds facility content, the push exits nonzero and records no
  latest tag; with private it passes; a DD-only push never queries visibility.
- The negative control: with the new check removed, the full-graph public test
  passes the push, and is caught.
- The tests covering the graph CLI add no failure against the base, both arms
  recorded in the manifest as single-line JSON objects with whole test ids.

## Rules

- Do not run a real push, change any package's visibility, or touch the push
  timer. Use faked registry calls in tests.
- No token, registry credential or private host in a tracked file.
- Run `ruff check --fix` and `ruff format` on exactly the files you change.
- Commit locally with a conventional subject and a body, and give `commits` as
  a JSON list of full shas. Do not push or merge.

## Scope

Write paths: `iter_codex/cli/graph/registry.py`, `iter_codex/graph/ghcr.py`,
`iter_codex/graph/remote.py`, `tests/cli`, `tests/graph`, and
`docs/evidence/fragments/iter-signal-context-and-enrichment/full-graph-push-stays-private.html`.
