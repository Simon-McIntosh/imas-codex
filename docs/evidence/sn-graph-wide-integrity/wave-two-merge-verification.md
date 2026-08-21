# Integrated source reconciliation and geometry projection verification

Date: 2026-08-21

Tested HEAD: `aca063a7bfbb093d9fd709ca19ba106cdc3de0f0`

Both implementation commits are ancestors of the tested HEAD:

- `f093f57c` — signed source-target reconciliation
- `25c4c35e` — geometry-base projection correction

## Result

PASS, with the graph-test availability qualification below. The integrated tree
held both changes with zero test failures, zero test errors, and both ruff gates
at exit 0.

| Gate | Result | Exit |
|---|---:|---:|
| `tests/standard_names/` | 6,554 passed, 8 skipped, 279 deselected, **0 failed, 0 errors** | 0 |
| `test_source_target_reconciliation.py` | 1 passed, 3 skipped, **0 failed, 0 errors** | 0 |
| `test_geometry_base_projection.py` | 2 passed, **0 failed, 0 errors** | 0 |
| `ruff check --no-cache .` | All checks passed | 0 |
| `ruff format --check --no-cache .` | 1,172 files already formatted | 0 |

The three skipped source-target reconciliation cases require a disposable graph
through `IMAS_CODEX_TEST_NEO4J_URI`; that variable was not configured in this
merge-verification worktree, and the fixture skipped before graph access. The
non-graph registry case passed. The full Standard Names command uses the
repository's default marker policy, which deselects graph-marked cases.

## Commands and retained logs

All Python commands reused `/home/ITER/mcintos/Code/imas-codex/.venv` through
`UV_PROJECT_ENVIRONMENT` and ran with `PYTHONPATH` set to this integrated
worktree. No dependency sync or local worktree environment was created.

- `uv run --no-sync pytest -p no:cacheprovider tests/standard_names/`
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T203051135237-mergeverify/logs/standard-names-full.log`
  - SHA-256: `ed01a915f9e1808db62e23091c5efb8eea550b39014084d17997d1317fa4e8ec`
- `uv run --no-sync pytest -p no:cacheprovider -m 'not slow' tests/standard_names/test_source_target_reconciliation.py`
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T203051135237-mergeverify/logs/source-target-reconciliation-all-markers.log`
  - SHA-256: `ac8643f1ec652348d420ec68b6e45003b507cafd500da31912f82c808e2c4166`
- `uv run --no-sync pytest -p no:cacheprovider tests/standard_names/test_geometry_base_projection.py`
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T203051135237-mergeverify/logs/geometry-base-projection.log`
  - SHA-256: `4d383132cc0a224872127b02855aab4f47086f663c00b35361ba57d351d5ceed`
- `uv run --no-sync ruff check --no-cache .`
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T203051135237-mergeverify/logs/ruff-check.log`
  - SHA-256: `e5ad9d73aa430a41bdfdfd607cfef4ad105edafd658987e1d7757c5cdbf24301`
- `uv run --no-sync ruff format --check --no-cache .`
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T203051135237-mergeverify/logs/ruff-format-check.log`
  - SHA-256: `d4919da03808c8859b68f7df18597ab16ba73987ef2107fb1d73eebf0f2eb604`

The tested HEAD remained unchanged for every command. The only repository write
after verification was this evidence record.
