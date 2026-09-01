# Graph operational-safety integrated verification

Verified at `2026-09-01T18:57:23+00:00` from detached integration commit
`37be1554e001161ab3ef9c3c81881af5118f0e74`. At the verification boundary,
`origin/main` was `c019e480f5dba14b8bc775700fe4a489b541ff35`; the only intervening change was
the cumulative plan-evidence HTML, and `git diff --name-only HEAD..origin/main
-- imas_codex tests` returned no paths. The source and test tree exercised here
therefore matched current main while the worker remained isolated.

## Verdict

The integrated graph-safety result passes its focused source and test tiers:
`tests/core` reported **899 passed, 2 skipped, 0 failed**; `tests/cli` reported
**174 passed, 0 failed**; and ruff reported **15 checked paths, 0 findings**.
The live graph was running as `codex` on `titan`, and `graph status` reported
the required backup-currency line: newest backup
`imas-codex-graph-dev-6fc745c-20260901T125109Z.tar.gz`, newest live file
`checkpoint.0`, **11,847 seconds behind live data (`stale`)**.

The repository-wide default pytest command is not green: it stopped during
collection with **0 passed, 0 failed, 1 collection error**, plus 2 skipped and
764 deselected. The sole error is named below and is preserved rather than
filtered out.

| Tier | Exit | Exact result | Verdict |
|---|---:|---|---|
| Default pytest suite | 2 | 0 passed, 0 failed, 1 collection error; 2 skipped; 764 deselected; 34 warnings | Not caused by this plan. `tests/docs/test_documentation_layout.py` imports `reckon.resources.TYPE_ROOTS`, but `reckon` is absent from the project environment and is not declared in `pyproject.toml`. That test was introduced by commit `651d2367`, outside every graph-safety landing commit and outside the touched-path set. Collection stopped before graph-safety tests ran. |
| `tests/core` | 0 | 899 passed, 2 skipped, 0 failed; 1 warning | Pass. Includes the backup-currency, graph CLI package, and Neo4j backup coverage touched by this plan. |
| `tests/cli` | 0 | 174 passed, 0 failed; 1 warning | Pass. Includes destructive backup/target, archive naming, and manifest-scope coverage touched by this plan. |
| Scoped ruff | 0 | 15 Python paths checked; 0 findings | Pass. |
| Live graph status | 0 | Graph `codex` running on `titan`; SLURM job `1260760` RUNNING; backup lag 11,847 s (`stale`) | Pass. The command emitted the required backup-currency line and named both compared artifacts. |

## Commands and exit codes

The three pytest commands were executed once each, with complete output written
to the durable run directory. No suite was rerun for different formatting.

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider
EXIT=2

UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/core
EXIT=0

UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/cli
EXIT=0

UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ruff check --no-cache imas_codex/cli/graph/__init__.py imas_codex/cli/graph/data.py imas_codex/cli/graph/registry.py imas_codex/cli/graph/server.py imas_codex/cli/release.py imas_codex/cli/services.py imas_codex/graph/neo4j_ops.py imas_codex/graph/remote.py tests/cli/test_graph_archive_naming.py tests/cli/test_graph_destructive_backup.py tests/cli/test_graph_destructive_target.py tests/cli/test_graph_manifest_scope.py tests/core/test_backup_currency.py tests/core/test_graph_cli_package.py tests/core/test_neo4j_ops.py
EXIT=0

UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync imas-codex graph status
EXIT=0

git rev-parse HEAD origin/main
EXIT=0

git diff --name-only HEAD..origin/main -- imas_codex tests
EXIT=0 (no paths)
```

## Live status evidence

```text
Backup currency:
  Newest backup: /home/ITER/mcintos/.local/share/imas-codex/backups/imas-codex-graph-dev-6fc745c-20260901T125109Z.tar.gz (2026-09-01T12:52:53.778724+00:00)
  Newest live file: /home/ITER/mcintos/.local/share/imas-codex/neo4j/data/transactions/neo4j/checkpoint.0 (2026-09-01T16:10:20.669936+00:00)
  Behind live data: 11847 s (stale)

SLURM:
  neo4j: job 1260760 RUNNING on 98dci4-gpu-0002

Neo4j: running
  Graph: codex
  Location: titan
```

## Durable logs

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T185207704152-n-integratedsuite/pytest-default.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T185207704152-n-integratedsuite/pytest-core.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T185207704152-n-integratedsuite/pytest-cli.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T185207704152-n-integratedsuite/ruff.log`
