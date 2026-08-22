# Integrated Standard Names verification

Date: 2026-08-22

Tested HEAD: `cd6b6e26e82a4c1f40c12ce302a7b93054b5b1df`

Verdict: **PASS**. The integrated Standard Names and units tree, the retained
focused checks, all three newly integrated signed-operator suites, generated
model import, committed authority bytes, and both repository-wide Ruff gates
are green at the stated HEAD.

## Quantitative gates

| Gate | Result | Exit |
|---|---:|---:|
| Full `tests/standard_names tests/units` run | **6,922 collected** = 6,615 passed + 8 skipped + 299 deselected; **0 failed, 0 errors** | 0 |
| Freshly generated `imas_codex.graph.models` import | **0** warnings matching `shadows an attribute in parent` | 0 |
| Committed authority artifacts | **4/4 intact**, **0 mismatches** | 0 |
| `test_repair_authority_field_naming.py` | **3 passed**, **0 failed, 0 errors** | 0 |
| `test_dd_resolution_version_window.py` | **7 passed**, **0 failed, 0 errors** | 0 |
| `ruff check --no-cache .` | **0 findings** (`All checks passed!`) | 0 |
| `ruff format --check --no-cache .` | **1,184 files already formatted**, **0 files needing reformatting** | 0 |

The full suite used the repository's normal marker policy, which is why its
collection includes skipped and deselected cases. The signed-operator graph
contracts below were then run explicitly against a disposable Neo4j 2026.01.4
instance with `-m 'not slow'`, so the default graph deselection cannot be
mistaken for coverage. The fixture listened only on local port 52787 and was
stopped after the runs.

## Newly integrated suites

| Suite | Result | Verdict |
|---|---:|---:|
| Parent release: `tests/standard_names/test_structural_release.py` | **3 passed**, 0 skipped, **0 failed, 0 errors** | **PASS** |
| Unbound source attachment: `tests/standard_names/test_unbound_source_attachment.py` | **8 passed**, 0 skipped, **0 failed, 0 errors** | **PASS** |
| Supersede successor: `tests/standard_names/test_supersede_successor_scalar.py` | **1 passed**, 0 skipped, **0 failed, 0 errors** | **PASS** |

An initial per-file invocation under the default marker policy was not counted:
it ran only the non-graph parent-release case and deselected all unbound-source
cases. The final results above come from the corrected explicit-marker runs and
cover every case in each named file.

## Generated model and authority integrity

`uv run --no-sync build-models --force` completed with exit 0. A separate new
Python process then imported `imas_codex.graph.models` with warnings enabled;
the import completed with exit 0 and emitted zero field-shadow warnings.
Generation produced no tracked changes, and the Git index remained empty.

The four committed authority files match their complete pinned SHA-256 values:

| Authority artifact | Verified SHA-256 |
|---|---|
| `catalog-edit-dual-binding-adjudication.json` | `5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e` |
| `refused-target-orphan-adjudication.json` | `2c2d38f3241ec3057d24a5d05c27840f5e4ffe99520063059ab31c1e9d4bca36` |
| `owner-geometry-rc66-partition.json` | `dbb37f7be12ba99d7e85bf13b9d63e6c19cb6c20bd35fe687e590f798e2dc85b` |
| `stale-source-lifecycle.json` | `f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad` |

## Exact commands

Every project command reused `/home/ITER/mcintos/Code/imas-codex/.venv` through
`UV_PROJECT_ENVIRONMENT`, set `PYTHONPATH` to this worktree, and used
`uv run --no-sync`.

```text
uv run --no-sync build-models --force
uv run --no-sync python -W always -c 'import imas_codex.graph.models'
uv run --no-sync pytest -p no:cacheprovider tests/standard_names tests/units
uv run --no-sync pytest -p no:cacheprovider tests/standard_names/test_repair_authority_field_naming.py
uv run --no-sync pytest -p no:cacheprovider tests/standard_names/test_dd_resolution_version_window.py
IMAS_CODEX_TEST_NEO4J_URI=bolt://127.0.0.1:52787 IMAS_CODEX_TEST_NEO4J_EPHEMERAL=1 uv run --no-sync pytest -p no:cacheprovider -m 'not slow' tests/standard_names/test_structural_release.py
IMAS_CODEX_TEST_NEO4J_URI=bolt://127.0.0.1:52787 IMAS_CODEX_TEST_NEO4J_EPHEMERAL=1 uv run --no-sync pytest -p no:cacheprovider -m 'not slow' tests/standard_names/test_unbound_source_attachment.py
IMAS_CODEX_TEST_NEO4J_URI=bolt://127.0.0.1:52787 IMAS_CODEX_TEST_NEO4J_EPHEMERAL=1 uv run --no-sync pytest -p no:cacheprovider -m 'not slow' tests/standard_names/test_supersede_successor_scalar.py
uv run --no-sync ruff check --no-cache .
uv run --no-sync ruff format --check --no-cache .
```

## Retained logs

Complete output is retained under
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T214543535626-n-verify/logs/`:

- `build-models.log`
- `model-import-warnings.log` and `model-import-check.log`
- `authority-digests.log`
- `pytest-standard-names-and-units.log`
- `pytest-field-naming.log`
- `pytest-dd-window.log`
- `pytest-parent-release-complete.log`
- `pytest-unbound-source-attachment-complete.log`
- `pytest-supersede-successor-complete.log`
- `ruff-check.log`
- `ruff-format-check.log`
- `neo4j-console.log`

The combined pytest run reported 34 non-failing warnings. They remain visible
in the retained log; the separately required fresh generated-model import is
the measurement that emitted zero field-shadow warnings.
