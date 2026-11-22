# Make the server's tools survive its module reload

## Goal

A tool call never fails because the server is reloading its own modules.

## Why

The lead is about to try the plant search tools from a fresh session. On
2026-11-22 the coordinator started the server in-process, called the signal
search and then the pulse search, and the pulse search failed with
`KeyError: 'iter_codex.search'`. Called again in a fresh process it passed
three times in a row. The server has a reload step
(`iter_codex/llm/server.py` around line 1787) that deletes every
`iter_codex` module from `sys.modules`; a tool that imports a module while
that runs finds a half-removed entry. The same symptom was noted on
2026-11-17 under the old package name and never fixed.

## What must be true afterwards

- No tool call fails with a missing-module error, whatever order tools are
  called in or however close together.
- The reload step, if it is kept, cannot run while a tool call is importing,
  or it no longer deletes modules other code may be importing; say in the
  evidence which, and why the reload exists at all. If nothing needs it,
  removing it is acceptable and preferred.

## Done when

- A test reproduces the failure at the base revision: calling the signal
  search then the pulse search (and the reverse) in one process fails at
  base, and passes at head across at least 50 repetitions in both orders.
- Every plant tool (`search_pulses`, `search_signals`, the signal walk and
  context tools, `get_signal_load_recipe`) is called once in each of several
  orders in one process with no error.
- The tests covering the server (`tests/llm`) add no failure against the base,
  both arms recorded in the manifest as single-line JSON objects with whole
  test ids.

Negative control: with the fix reverted, the reproduction test fails.

## Rules

- Change only the server's reload or import handling; no tool's behaviour.
- No real signal or pulse names in tracked tests; use the faked graph the
  existing server tests use.
- Run `ruff check --fix` and `ruff format` on exactly the files you change.
- Commit locally with a conventional subject and a body, and give `commits` as
  a JSON list of full shas. Do not push or merge.

## Scope

Write paths: `iter_codex/llm/server.py`, `iter_codex/tools`, `tests/llm`, and
`docs/evidence/fragments/codac-uda-signals/server-tools-survive-module-reload.html`.
