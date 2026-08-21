# Integrated closed-branch and peel-unit verification

Date: 2026-08-21

Tested HEAD: `1e618531fc55a248386337628556a44eccaaeae8`

The integrated tree contains all four implementation commits exercised by this
verification:

- `f093f57c` — signed source-target reconciliation
- `25c4c35e` — geometry-base projection correction
- `1d50f3f8` — signed structural-source revival
- `89f84942` — normalization-peel unit-authority correction

Each commit is an ancestor of the tested HEAD. The HEAD remained unchanged for
every command below.

## Result

PASS. The full Standard Names suite completed with 6,555 passed, zero failed,
and zero errors. All four focused suites ran against the disposable endpoint
where applicable and completed with zero failed and zero skipped. Both ruff
gates exited 0.

| Gate | Result | Exit |
|---|---:|---:|
| `tests/standard_names/` | 6,555 passed, 8 skipped, 281 deselected, **0 failed, 0 errors** | 0 |
| `test_source_target_reconciliation.py` | 4 passed, **0 failed, 0 skipped** | 0 |
| `test_geometry_base_projection.py` | 2 passed, **0 failed, 0 skipped** | 0 |
| `test_structural_source_revival.py` | 3 passed, **0 failed, 0 skipped** | 0 |
| `test_normalization_peel_unit_repair_graph.py` | 7 passed, **0 failed, 0 skipped** | 0 |
| `ruff check --no-cache .` | All checks passed | 0 |
| `ruff format --check --no-cache .` | 1,173 files already formatted | 0 |

The full suite's eight skips are existing environment-marked cases under the
repository's default marker policy. They are separate from the four explicitly
required focused suites, all of which ran with the repository addopts cleared
and recorded zero skips.

## Disposable graph isolation

The graph-backed suites used an authentication-disabled Neo4j Community
2026.01.4 instance bound only to `bolt://127.0.0.1:48687` and
`http://127.0.0.1:48688`. Every focused command set:

```text
IMAS_CODEX_TEST_NEO4J_URI=bolt://127.0.0.1:48687
IMAS_CODEX_TEST_NEO4J_EPHEMERAL=1
IMAS_CODEX_TEST_NEO4J_PASSWORD=
IMAS_CODEX_TEST_PROJECT_NEO4J_URI=bolt://production-endpoint.invalid:7687
```

The invalid production override makes unintended project-endpoint access fail
by construction. The retained console log records the loopback Bolt endpoint,
startup, and clean shutdown. Both disposable ports were closed after the run.

## Commands and retained logs

All commands reused `/home/ITER/mcintos/Code/imas-codex/.venv` through
`UV_PROJECT_ENVIRONMENT`, set `PYTHONPATH` to this integrated worktree, removed
an inherited `VIRTUAL_ENV`, and used `uv run --no-sync`. No dependency sync or
worktree-local environment was created.

- `uv run --no-sync pytest -p no:cacheprovider tests/standard_names/`
  - 6,555 passed, 8 skipped, 281 deselected, 0 failed, 0 errors in 297.35 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T210110705496-mergeverify2/logs/standard-names-full.log`
  - SHA-256: `81460ff2e7c4430baf827f5f4fc8fe0253005fdce1f16dfe6608c60c47c6850d`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/standard_names/test_source_target_reconciliation.py`
  - 4 passed, 0 failed, 0 skipped in 15.58 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T210110705496-mergeverify2/logs/source-target-reconciliation.log`
  - SHA-256: `672e0e31a8dce1ad0d879329008760145941ec624651137d669e08f422d0d840`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/standard_names/test_geometry_base_projection.py`
  - 2 passed, 0 failed, 0 skipped in 5.67 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T210110705496-mergeverify2/logs/geometry-base-projection.log`
  - SHA-256: `0f815cc8331d6d57d75804ed64a36a02c0302e7c326e6d38f75be69721ef8737`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/standard_names/test_structural_source_revival.py`
  - 3 passed, 0 failed, 0 skipped in 9.37 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T210110705496-mergeverify2/logs/structural-source-revival.log`
  - SHA-256: `486e05fe1a1643f089fcbf17a7359f3c71bcb2aaf0be12ab9e1fe111fbddd9f2`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/standard_names/test_normalization_peel_unit_repair_graph.py`
  - 7 passed, 0 failed, 0 skipped in 9.81 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T210110705496-mergeverify2/logs/normalization-peel-unit-repair-graph.log`
  - SHA-256: `41bc6b9ebdb6b8d85e8392a68d046cd6d49d2cff1a9e972e6e1f11d53b36dd98`
- `uv run --no-sync ruff check --no-cache .`
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T210110705496-mergeverify2/logs/ruff-check.log`
  - SHA-256: `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`
- `uv run --no-sync ruff format --check --no-cache .`
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T210110705496-mergeverify2/logs/ruff-format-check.log`
  - SHA-256: `54c0f1c90fa0b115290c0837a8fefc1f5af6cbe110a4abb8e50bba65b473057e`
- Disposable Neo4j console:
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260821T210110705496-mergeverify2/logs/neo4j-console.log`
  - SHA-256: `d1450593f0a816b893dadeee2caf3bad2f9e746b6da0770fa1b017ad26453cd4`

No production graph endpoint was contacted by the focused verification.
