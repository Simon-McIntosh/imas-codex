# Residual whole-tree failures

All three residual failure IDs are closed at root cause. Four persistent Cypher
properties are now LinkML-owned, their obsolete runtime adjudications have been
removed, and the two shared-environment tests resolve files relative to the
project module actually imported rather than the checkout that supplied the
test file.

## Reproduction census

The detached worktree was at `6f36a423f3d932cd9ab0c2c6e5aa4c102a57952a`; the main checkout was at `6e2869e5dd2524c721f5b1b76a3f36e86545e1d1`. Each baseline test was run once per layout with the shared root environment, `PYTHONPATH` aimed at that layout, bytecode disabled, and pytest's cache provider disabled.

| Failure id | Worktree command and result | Main-checkout command and result | Verdict and fix |
|---|---|---|---|
| `test_repository_cypher_literals_have_declared_properties` | `PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=<worktree> uv run --no-sync pytest -p no:cacheprovider tests/graph/test_cypher_property_check.py::test_repository_cypher_literals_have_declared_properties` — exit 0 | Same command with `PYTHONPATH=/home/ITER/mcintos/Code/imas-codex` from the main checkout — exit 0 | The earlier full-suite failure was a mixed-checkout artifact: its newer shared-environment checker inventory did not align with the detached source census. The named properties are nevertheless genuine persistent writers, so they are now declared and their obsolete runtime allowances were removed. |
| `test_archive_uses_packaged_sha_and_source_date_epoch` | `PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=<worktree> uv run --no-sync pytest -p no:cacheprovider tests/standard_names/test_export_determinism.py::TestCommitTimestamp::test_archive_uses_packaged_sha_and_source_date_epoch` — exit 0 | Same command with main-checkout `PYTHONPATH` — exit 0 | Shared-environment artifact. The failing whole-tree run collected the test from the worktree but imported `export.py` from the main checkout, so the mocked packaged path and imported module path differed. The test now derives the packaged path from `export_module.__file__`. |
| `test_pipeline_hash_changes_when_prompt_changes` | `PYTHONDONTWRITEBYTECODE=1 UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=<worktree> uv run --no-sync pytest -p no:cacheprovider tests/standard_names/test_pipeline_version.py::test_pipeline_hash_changes_when_prompt_changes` — exit 0 | Same command with main-checkout `PYTHONPATH` — exit 0 | Shared-environment artifact. The failing whole-tree run edited a worktree prompt while the imported hash module read main-checkout prompts. The test now derives its project root from `pipeline_version.__file__`. |

Baseline logs are under `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T205715946062-n-suite-residual-three/logs/baseline-{worktree,main}-{cypher,packaged_sha,prompt_hash}.log`.

## Final gate

- `build-models --force`: exit 0; log `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T205715946062-n-suite-residual-three/logs/build-models.log`; generated files remain untracked/ignored and will not be committed.
- Property test after the schema and inventory repair: exit 0, 1 passed; log `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T205715946062-n-suite-residual-three/logs/pytest-property-after.log`.
- Combined three-test gate after formatting: exit 0, 3 passed; log `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T205715946062-n-suite-residual-three/logs/pytest-after-ruff.log`.
- Ruff check and format: both exit 0; logs `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T205715946062-n-suite-residual-three/logs/ruff-check.log` and `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T205715946062-n-suite-residual-three/logs/ruff-format.log`.

The generated model outputs remained ignored and are not part of the commit.
