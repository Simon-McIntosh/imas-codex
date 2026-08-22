# Final integrated merge verification

Date: 2026-08-22

Tested HEAD: `a492a64648e326310143d0b506161767505ca20f`

Verdict: **PASS**. The fully integrated tree contains the closed signed
source-target reconciliation, structural-source revival, and ordinary-source
migration programs. Their focused graph behavior, the shared authority builder,
the geometry projection correction, and the normalization-peel unit repair all
pass together. The full Standard Names suite, live-graph ratchets, and both Ruff
gates are green at the same HEAD.

The tested HEAD remained unchanged for every command. Commits `f093f57c`,
`1d50f3f8`, `58d165f4`, `508e4e14`, `25c4c35e`, and `89f84942` are all
ancestors of that HEAD.

## Results

| Gate | Result | Exit |
|---|---:|---:|
| `tests/standard_names/` | **6,570 passed**, 8 skipped, 286 deselected, **0 failed, 0 errors** | 0 |
| `test_source_target_reconciliation.py` | **4 passed**, 0 failed, 0 skipped | 0 |
| `test_structural_source_revival.py` | **3 passed**, 0 failed, 0 skipped | 0 |
| `test_ordinary_source_migration.py` | **6 passed**, 0 failed, 0 skipped | 0 |
| `test_repair_authority_builder.py` | **14 passed**, 0 failed, 0 skipped | 0 |
| `test_geometry_base_projection.py` | **2 passed**, 0 failed, 0 skipped | 0 |
| `test_normalization_peel_unit_repair_graph.py` | **7 passed**, 0 failed, 0 skipped | 0 |
| `tests/graph/test_sn_integrity_ratchets.py` against the live graph | **4 passed**, 0 failed, 0 skipped | 0 |
| `ruff check --no-cache .` | All checks passed | 0 |
| `ruff format --check --no-cache .` | 1,176 files already formatted | 0 |

The full suite's eight skips are the existing environment-marked cases under
the repository's default marker policy. They are distinct from the six
explicitly required focused files, which ran with repository addopts cleared
and recorded zero skips in every file.

## Disposable graph isolation

The six focused files ran serially against an authentication-disabled Neo4j
Community 2026.01.4 instance bound only to `bolt://127.0.0.1:48687` and
`http://127.0.0.1:48688`. Every focused command set:

```text
IMAS_CODEX_TEST_NEO4J_URI=bolt://127.0.0.1:48687
IMAS_CODEX_TEST_NEO4J_EPHEMERAL=1
IMAS_CODEX_TEST_NEO4J_PASSWORD=
IMAS_CODEX_TEST_PROJECT_NEO4J_URI=bolt://production-endpoint.invalid:7687
```

The invalid project-endpoint override makes unintended production access fail
by construction. The disposable console log records loopback startup and clean
shutdown, and both disposable ports were closed after the run. The separately
required integrity-ratchet file ran without those overrides against the live
graph and passed all four ratchets.

## Commands and retained logs

All Python commands reused `/home/ITER/mcintos/Code/imas-codex/.venv` through
`UV_PROJECT_ENVIRONMENT`, set `PYTHONPATH` to this worktree, removed inherited
`VIRTUAL_ENV`, and used `uv run --no-sync`. No dependency sync or worktree-local
environment was created.

- `uv run --no-sync pytest -p no:cacheprovider tests/standard_names/`
  - 6,570 passed, 8 skipped, 286 deselected, 0 failed, 0 errors in 200.60 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/standard-names-full.log`
  - SHA-256: `21a18a75bf4b3cdf4d154f20c0d5e0d235377a21da34c81ed551cde2d65748a6`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/standard_names/test_source_target_reconciliation.py`
  - 4 passed, 0 failed, 0 skipped in 11.98 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/source-target-reconciliation.log`
  - SHA-256: `61df030b698e00cc4ac02e91e7933ca5fb79cad7520ec92091b0a504888dadfd`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/standard_names/test_structural_source_revival.py`
  - 3 passed, 0 failed, 0 skipped in 6.95 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/structural-source-revival.log`
  - SHA-256: `8228135ee936e71e441a7bd29ff9613028a34dd9aa898c20bd7a5548e4fb4b13`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/standard_names/test_ordinary_source_migration.py`
  - 6 passed, 0 failed, 0 skipped in 8.07 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/ordinary-source-migration.log`
  - SHA-256: `7b87b47b9dd0403d62b441c53d2be7b44a64e21e64cd12b519ae6ae10d6f7884`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/standard_names/test_repair_authority_builder.py`
  - 14 passed, 0 failed, 0 skipped in 6.00 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/repair-authority-builder.log`
  - SHA-256: `01b8fbb9abc361208182dbba3972fc04a1dd627691dcb49d843d9afdb192ffe8`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/standard_names/test_geometry_base_projection.py`
  - 2 passed, 0 failed, 0 skipped in 5.72 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/geometry-base-projection.log`
  - SHA-256: `fb5fdc08bf81bffa5f4c83192490ce61b626c28de685cd563202143ea2b0e8d4`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/standard_names/test_normalization_peel_unit_repair_graph.py`
  - 7 passed, 0 failed, 0 skipped in 6.12 s
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/normalization-peel-unit-repair-graph.log`
  - SHA-256: `eea4bbfac2895e5cf46078555bbf0f899942400781ec0fa4032242111caad981`
- `uv run --no-sync pytest -o addopts='' -p no:cacheprovider -q -s tests/graph/test_sn_integrity_ratchets.py`
  - 4 passed, 0 failed, 0 skipped in 5.53 s against the live graph
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/sn-integrity-ratchets.log`
  - SHA-256: `f9c3e65397ea8df3f412ea7168c96590eaef6a08b5d23db60a641b7b8d645517`
- `uv run --no-sync ruff check --no-cache .`
  - all checks passed, exit 0
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/ruff-check.log`
  - SHA-256: `82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`
- `uv run --no-sync ruff format --check --no-cache .`
  - 1,176 files already formatted, exit 0
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/ruff-format-check.log`
  - SHA-256: `7df1f01dd3b87cc110937275adbf7009a8cbd2c7591085c567d9d0f919677c67`
- Disposable Neo4j console:
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/neo4j-console.log`
  - SHA-256: `3c4b9e9f2d60f6b2f6230a386f3f31c7e2937b61e9ebdd612962d55d79b86939`
- Combined quantitative capture:
  - log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T005033549104-mergeverify3/logs/verification-summary.log`
  - SHA-256: `7a69b63c06ef309eab1d90f8b794c7fc6e9d8e9df3d74801cfb28b7a75e17a92`

No production graph endpoint was contacted by the six disposable suites. The
live graph was contacted only by the explicitly required read-only ratchet
suite.
