# Graph service partition evidence

## Implementation

- Changed file: `imas_codex/cli/services.py`
- Changed symbol: `_submit_service_job()`
- Before: the CPU-only branch called `_general_partition_name()`, which resolved the general `sun` partition for Neo4j even though the graph location was `titan`.
- After: the CPU-only branch resolves `get_graph_location()` through `resolve_location()`, requires a SLURM compute target with a declared partition, and uses that target's partition. For the live graph configuration the value changed from `sun` to `titan`.
- Commit: `68e79fb2`

## Generated submission

The isolated submission probe exercised `_submit_service_job("codex-neo4j", ..., gpus=0)` with graph location `titan` and asserted the generated command contained:

```text
#SBATCH --partition=titan
```

The same probe asserted `#SBATCH --partition=all` was absent. Result: `generated_sbatch_partition=titan`, `general_partition_present=false`.

## Live restart and Bolt read

- Previous job: `1260741`, `PENDING` on partition `sun`, reason `Priority`.
- Restart result: job `1260760`, state `RUNNING`, partition `titan`, node `98dci4-gpu-0002`.
- Service health: the launcher reported `codex-neo4j healthy`, Bolt at `bolt://98dci4-gpu-0002:7687`, and HTTP at `http://98dci4-gpu-0002:7474`.
- Read-only Bolt verification: `MATCH (n) RETURN count(n) AS node_count` answered `node_count=1612957`.

The live job therefore moved from the general `sun` partition to the configured `titan` partition and answered a real read-only graph query on its allocated node.

## Verification

Command:

```text
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider --timeout=120 tests/cli/test_services_remote.py tests/core/test_graph_profiles.py
```

Result: **31 passed, 0 failed, 1 warning in 10.49 seconds**. The warning is the pre-existing pytest configuration warning for unknown option `cache_dir`.

An earlier run using the repository's default 30-second timeout recorded 30 passed and one timeout while importing the CLI under transient shared-filesystem latency. The timed-out test passed independently with the explicit 120-second ceiling (`1 passed, 0 failed in 56.60 seconds`) before the complete focused suites passed as reported above.

## Evidence files

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T132019046878-n-graphpartition/generated-sbatch.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T132019046878-n-graphpartition/live-restart-retry.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T132019046878-n-graphpartition/bolt-node-count.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T132019046878-n-graphpartition/targeted-tests-green.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T132019046878-n-graphpartition/timeout-recheck.log`
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260901T132019046878-n-graphpartition/targeted-tests.log`
