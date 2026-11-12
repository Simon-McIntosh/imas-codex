# Brief: first answer in under ten seconds (serve-answers-inside-the-connect-limit §2)

## Reuse map (extend these; do not add a parallel mechanism)
- The cost is one eager call: `register_commands()` in `imas_codex/cli/__init__.py` imports all fifteen command modules at import time. `imas_codex.cli.serve` alone costs milliseconds; `imas_codex.cli.discover` costs most of the time (it reaches `discovery/paths/scorer.py` -> `litellm`, and `graph/__init__.py` -> `ensure_models_fresh`, `graph.schema`, `linkml_runtime`).
- The design is lazy resolution inside the existing click group (a deferred command table resolved in `get_command`, with `list_commands` and `--help` produced from the table). Not a second entry point, not a `--fast` flag, not per-library deferral inside `serve`.
- A deferral mechanism already exists and stays the only one: `imas_codex/llm/_warmup.py`, started at `imas_codex/llm/server.py` from `serve()`. Make sure it is no longer racing a cost already paid.
- Import-boundary guard: extend `tests/discovery/test_engine_import_order.py` (`_run_fresh_interpreter`). Tool-name parity: extend `tests/llm/test_tool_schemas.py` (`_tool_names`, the `AgentsServer(dd_only=True)` fixture needs no graph).
- The evidence script reads its spawn recipe from the committed `.mcp.json`; do not hand-copy the argv.

## Measurements
- Write the timing script under the evidence directory: it spawns the server, sends `initialize`, and times the first response line.
- Base readings taken 2026-11-12 on the fleet node at load 73: 35.1 s to first answer, 27.5 s for `serve --help`.
- The five fleet-node starts are light (one process each) and may run where you start; the debug-partition median runs as a job.

## Standing rules for this node
- Read your plan section from your worktree; it is the authority for the endpoint. This brief adds the reuse map and the questions you may not settle.
- Export `UV_NO_SYNC=1`, use `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync ...`. Copy (never symlink) the five gitignored generated files from the main checkout if absent.
- Heavy gates (any test file opening the graph, any pass over the whole catalog) run as their own job: `env -u SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=all_debug --time=00:59:00 --output=<a GPFS log under ~/.cache/imas-codex-<node>/>`. A bare `srun --partition` from this node hangs. Never run them on the node you start on.
- Do not write the live graph. Use a disposable graph or fixtures for every control.
- Write your manifest with what you have before any long wait. Status is `complete`, `blocked` or `failed`.
- Commit locally with a conventional subject and a body; stage explicit paths; no version, plan or ticket labels in source names or comments.

