NEEDS-HELP: the integrated tree passes every functional gate but fails the required full-tree formatting gate on a concurrent node's exclusive path.

tried: Built generated models, imported them under an always-on warnings filter, ran the complete Standard Names and units suites, ran both focused regressions, checked the four pinned authority artifacts, measured the live DD-resolution window, and ran both full-tree Ruff gates at HEAD `770b399859ba0ca8ecab1f385ecf35c079c0b649`.

options: Have the concurrent owner format `imas_codex/standard_names/signed_manifest.py` and integrate that commit, or dispatch a new verification node after that owner lands; do not weaken the gate or exclude the field-shadow adapter file.

leaning: Let the existing `releaseprog2` owner correct and land its exclusive path, then rerun this exact verification wave on the resulting integrated HEAD.

cost-if-wrong: Excluding the file would certify a tree that fails its stated formatter gate; taking the path from its concurrent owner risks conflicting edits and invalidates the exclusive-write fence.

# Integrated adapter and resolution-window verification

## Verdict

The integrated tree is **not releasable under this node's done-when clause**.
All functional, compatibility, artifact-integrity, warning, and lint checks pass,
but full-tree `ruff format --check --no-cache .` exits **1** because
`imas_codex/standard_names/signed_manifest.py` would be reformatted. That path
is assigned exclusively to the concurrent `releaseprog2` node and was not
modified here.

The run was taken at HEAD
`770b399859ba0ca8ecab1f385ecf35c079c0b649`, the merge of the field-shadow
adapter and DD resolution retirement-window changes.

## Quantitative verification

| Gate | Result |
|---|---|
| `uv run --no-sync build-models --force` | exit 0 |
| Import `imas_codex.graph.models` with `warnings.simplefilter("always")` | exit 0; **0** warnings matching `shadows an attribute in parent` |
| `pytest tests/standard_names/` | **6,587 passed, 8 skipped, 288 deselected, 0 failed, 0 errors** in 184.79 s |
| `pytest tests/units/` | **27 passed, 0 failed, 0 errors** in 5.80 s |
| `test_repair_authority_field_naming.py` | **3 passed, 0 failed, 0 errors** in 7.27 s |
| `test_dd_resolution_version_window.py` | **7 passed, 0 failed, 0 errors** in 6.00 s |
| `ruff check --no-cache .` | exit 0; all checks passed |
| `ruff format --check --no-cache .` | **exit 1**; 1 file would be reformatted, 1,180 already formatted |

The complete Standard Names suite emitted 34 warnings, none test failures or
errors. The units and focused suites each emitted the repository's single
pytest configuration warning; their exit status and failure/error counts are
unaffected.

## Field-shadow and wire-compatibility result

The generated model now exposes `RepairAuthorityArtifact.schema_id` and does
not expose a `schema` model field. After a forced model rebuild, a fresh import
of `imas_codex.graph.models` emitted **0** matching Pydantic field-shadow
warnings. The committed JSON authority wire key remains `schema`; the adapter
loads it without changing the original bytes.

All four committed authority artifacts retain their full pinned SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| `catalog-edit-dual-binding-adjudication.json` | `5ca7761a7b022ac7889387d7bf63a027114a168cc3785ed4fdc8d31c08417b6e` |
| `refused-target-orphan-adjudication.json` | `2c2d38f3241ec3057d24a5d05c27840f5e4ffe99520063059ab31c1e9d4bca36` |
| `owner-geometry-rc66-partition.json` | `dbb37f7be12ba99d7e85bf13b9d63e6c19cb6c20bd35fe687e590f798e2dc85b` |
| `stale-source-lifecycle.json` | `f2da3ff78d5427fe4477bc46c57a7dc33c8c2d6659d4a48e52f94a4014ae90ad` |

The required prefixes are therefore unchanged at `5ca7761a`, `2c2d38f3`,
`dbb37f7b`, and `f2da3ff7`.

## DD resolution retirement-window census

The live configured request version was **4.1.1**, with **49** effective active
resolution records. Excluding
`equilibrium/time_slice/constraints/n_e/reconstructed`, the number of active
resolution records whose recorded `dd_version` differs from the requested
version while that request remains inside `retiring_release` is **0**, spanning
**0 distinct DD paths**.

The focused seven-case regression separately proves the window behavior for a
reviewed 4.1.1 resolution requested as 4.1.0 and 4.1.9, refusal at and after
4.2.0 retirement, open-ended `none-yet`, content mismatch, and conflicting
windowed remedies.

## Generated-file and scope audit

Immediately after the forced build and all verification commands, both
`git status --short --untracked-files=all` and `git diff --cached --name-only`
were empty. The generated models and schema reference were therefore left
unstaged; this node later staged only this assigned evidence report. No source,
test, generated model, plan, index, or concurrent-node path was modified.

## Captured logs

All commands used the main checkout's single project environment through
`UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv`, with this
worktree first on `PYTHONPATH` and `uv run --no-sync`.

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/build-models.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/model-import-warnings.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/model-import-status.txt`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/pytest-standard-names.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/pytest-units.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/pytest-field-naming.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/pytest-dd-window.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/authority-hashes.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/dd-resolution-window-census.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/ruff-check.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260822T164448031713-verifywave/logs/ruff-format-check.log`

