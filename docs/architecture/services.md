# Service Management

ITER Codex manages two independent services that can be started, stopped, and
scaled independently. Each service has its own CLI group with a unified
`start` command that auto-detects the deployment mode.

## Services Overview

| Service | CLI | Scheduler | Resources | Purpose |
|---------|-----|-----------|-----------|---------|
| **Neo4j** | `iter-codex graph` | SLURM (compute) | 4 CPUs, 32 GB, no GPU | Knowledge graph database |
| **Embed** | `iter-codex embed` | SLURM (compute) | 2 CPUs, 16 GB, N GPUs | GPU-accelerated embedding |

## Quick Start

```bash
# Start all services
iter-codex graph start           # Neo4j on SLURM
iter-codex embed start           # Embed server on SLURM (4 GPUs default)

# Check status
iter-codex graph status          # Neo4j health + SLURM resource usage
iter-codex embed status          # Embed health + GPU info

# Stop all
iter-codex embed stop
iter-codex graph stop
```

## Deployment Architecture

### Location Resolution

The deploy target is derived from configuration, not hardcoded:

1. `pyproject.toml` sets `[embedding].location = "titan"`
2. `resolve_location("titan")` searches facility YAMLs
3. `iter.yaml` defines `compute_locations.titan: {scheduler: slurm, partition: titan}`
4. Result: `LocationInfo(scheduler="slurm", partition="titan", is_compute=True)`

This means the same `start` command works across different facilities —
only the YAML configuration differs.

### SLURM Services (Neo4j, Embed)

When the location maps to a SLURM partition, `start` submits a batch job.
The command owns deployment — there is no separate submission step to
perform by hand:

```
Usage: iter-codex embed start [OPTIONS]
Usage: iter-codex graph start [OPTIONS]
```

The first line is the embedding server; the second is Neo4j. Both usage
lines are quoted from `iter-codex embed start --help` and `iter-codex graph
start --help`.

```
User: iter-codex embed start
  → _is_compute_target() → True (scheduler=slurm)
  → _submit_service_job("codex-embed", ...)
    → generates the sbatch script
    → writes it to ~/.local/share/iter-codex/services/codex-embed.sh
    → sbatch ~/.local/share/iter-codex/services/codex-embed.sh
  → _wait_for_job("codex-embed")  ← rich spinner + countdown
    → polls squeue until RUNNING
    → health check: curl /health
  → ✓ codex-embed healthy
```

The generated script is written beside the service's log rather than to a
temporary path, so it survives the submission. Read it to see exactly what
was submitted, and re-submit it verbatim (`sbatch
~/.local/share/iter-codex/services/codex-embed.sh`) to reproduce a launch
without going through `start` again. Re-running `start` overwrites it under
the same job name. `iter-codex graph start` works the same way, writing
`codex-neo4j.sh` beside `codex-neo4j.log`.

Inside the SLURM batch script, the process is the last command (`exec`),
so SLURM manages it directly — `scancel` stops it, cgroup enforcement
is automatic.

### Language-model endpoints

Language-model seats call their serving endpoints directly: the configured
local model router, OpenRouter chat completions, or the provider decisions
endpoint. The local router's lifecycle belongs to its hosting service, and
external providers manage their own availability. This CLI manages graph and
embedding services; model selection and credentials live in the per-seat
configuration.

Discovery health checks query a local seat's models listing or the provider's
key-info endpoint without generating tokens. Chat and decisions calls sharing
an origin and service credential share one key-info probe per poll. A healthy
key-info response establishes reachability and authentication, while the
opt-in live tests prove that a model actually answers.

### Foreground Mode

Use `--foreground` / `-f` to run a service directly in the terminal
(for debugging or when running inside a SLURM allocation):

```bash
iter-codex embed start -f                    # Run embed server directly
iter-codex embed start -f --gpu 1            # On specific GPU
```

When `SLURM_JOB_ID` is set (i.e., already inside a batch script), the
embed `start` command auto-detects foreground mode.

## GPU Management

The embed server supports dynamic GPU reallocation without manual SLURM
interaction. Reallocation cycle time is ~18 seconds.

```bash
# Start with specific GPU count
iter-codex embed start -g 2      # 2 GPUs

# Reallocate GPUs (stop + restart)
iter-codex embed restart -g 8    # Scale to 8 GPUs (~18s)
iter-codex embed restart -g 2    # Scale back to 2 GPUs (~18s)
iter-codex embed restart         # Reset to default (4 GPUs)
```

### Resource Budget (Titan Partition)

Single node: 20 CPUs, 250 GB RAM, 8× Tesla P100-16GB

| Config | Neo4j | Embed (4 GPU) | Available |
|--------|-------|---------------|-----------|
| CPUs | 4 | 2 | 14 |
| Memory | 32 GB | 16 GB | 202 GB |
| GPUs | 0 | 4 | 4 |

## Status & Monitoring

### Graph Status

```bash
$ iter-codex graph status
SLURM:
  neo4j: job 1017391 RUNNING on 98dci4-gpu-0002
    Allocated: CPUs: 4/20, Time: 44:09
    CPU:  [██░░░░░░░░░░░░░░░░░░] 10%  41% of 4 cores
    Mem:  [░░░░░░░░░░░░░░░░░░░░] 2%  6084 MB / 250 GB

Neo4j: running
  Graph: codex
  Location: titan
  Bolt: 7687, HTTP: 7474
```

### Embed Status

```bash
$ iter-codex embed status
  embed: job 1017494 RUNNING on 98dci4-gpu-0002
    Allocated: CPUs: 2/20, GPUs: 4/8, Time: 0:29
Server (http://98dci4-gpu-0002.iter.org:18765):
  ✓ Healthy
  Model: Qwen/Qwen3-Embedding-0.6B
  Device: Tesla P100-PCIE-16GB
  Dimension: 256
  GPU: Tesla P100-PCIE-16GB (16276 MB)
  Uptime: 136.9h
```

### Logs

```bash
iter-codex embed logs            # Last 50 lines
iter-codex embed logs -f         # Follow live
iter-codex embed logs -n 100     # Last 100 lines
```

## Systemd Service Management

For persistent services that survive user logout:

```bash
# Embed (login node fallback)
iter-codex embed service install
iter-codex embed service start
iter-codex embed service status

```

## Troubleshooting

### Embed server won't start

1. **SLURM resources**: Check `squeue -u $USER` — are other jobs consuming GPUs?
2. **stale jobs**: `scancel <job_id>` any stuck jobs
3. **Script errors**: `cat ~/.local/share/iter-codex/services/codex-embed.log`
4. **Health check**: `curl http://<compute-node>:18765/health`

### Model endpoint is unavailable

1. Resolve the seat with `get_model_config()` to identify its serving endpoint.
2. For a local seat, check the hosting router and the key named by `api-key-env`.
3. For an OpenRouter seat, check its service-specific `OPENROUTER_API_KEY_<SERVICE>`
   or the fallback `OPENROUTER_API_KEY_ITER_CODEX` credential.
4. Use the opt-in live seat tests for completion-level verification; they can
   incur provider charges.

### Neo4j won't start

1. **Lock files**: Check `graph status` for lock warnings
2. **Memory**: JVM heap defaults to 8-16 GB, check `neo4j.conf`
3. **Logs**: `cat ~/.local/share/iter-codex/services/codex-neo4j.log`

### Port conflicts

Default ports (configurable in `pyproject.toml`). These are the base ports; a
facility with a non-zero index in `locations` is offset by that index.

| Service | Port |
|---------|------|
| Neo4j Bolt | 7687 |
| Neo4j HTTP | 7474 |
| Embed | 18765 |

### Rich output disabled

The spinner/countdown requires a TTY. In pipes, CI, or redirected
output it falls back to plain text. Force rich with `ITER_CODEX_RICH=1`.
