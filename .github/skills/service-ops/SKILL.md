---
name: service-ops
description: How to manage iter-codex services (Neo4j graph, embedding server, model endpoints, SSH tunnels). Use when starting, stopping, checking status, or troubleshooting services.
---

# Service Operations

## Neo4j Graph Database

```bash
uv run iter-codex graph start       # Start (auto-detects SLURM/SSH/local)
uv run iter-codex graph stop        # Stop
uv run iter-codex graph status      # Check status
uv run iter-codex graph shell       # Interactive Cypher shell
uv run iter-codex graph profiles    # List all profiles and ports
uv run iter-codex graph export      # Export graph archive
uv run iter-codex graph load <f>    # Load archive
uv run iter-codex graph pull        # Pull latest from GHCR
uv run iter-codex graph backup      # Create neo4j-admin dump
uv run iter-codex graph clear       # Clear graph (auto-backup first)
```

### Ports by Location

| Location | Bolt | HTTP |
|----------|------|------|
| iter | 7687 | 7474 |
| tcv | 7688 | 7475 |
| jt-60sa | 7689 | 7476 |

### Neo4j Safety Rules

- Never delete Lucene `write.lock` files — corrupts vector indexes
- Always backup before destructive operations
- Never use `DETACH DELETE` without confirmation
- Use `NOT (x IN [...])` instead of `x NOT IN [...]` (Cypher 5 syntax)

## Embedding Server

```bash
uv run iter-codex embed start         # Start via SLURM
uv run iter-codex embed start -g 2    # Start with 2 GPUs
uv run iter-codex embed start -f      # Foreground (debugging/inside SLURM batch)
uv run iter-codex embed stop          # Stop SLURM job + cleanup rogues
uv run iter-codex embed restart       # Stop + start
uv run iter-codex embed status        # Health + SLURM job + node state
uv run iter-codex embed logs          # View SLURM logs
uv run iter-codex embed service install  # Install systemd service
```

### Embedding Configuration

- Model: `Qwen/Qwen3-Embedding-0.6B` (dimension: 256)
- Default: 4× P100 GPUs on titan partition
- Port: 18765 (HTTP)
- SLURM script: `slurm/codex-embed.sh`

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `embed status` shows node draining | Ask admin to resume node |
| Server healthy but no SLURM job | `iter-codex embed stop` kills rogues |
| Embedding calls timeout | Check tunnel: `lsof -i :18765` |
| Rapid FAILED jobs | Check `iter-codex embed logs`; report a missing or stale node/root environment instead of rebuilding it |

## Language-model endpoints

Seats resolve directly to the local model router, OpenRouter chat completions,
or the provider decisions endpoint. `get_model_config()` resolves each seat's
model, endpoint and credential name; an unrecognised route raises an error
naming the seat. The `litellm` dependency is the client for provider requests.

Local seats use the configured `api-base` or named `model-route` and the key
named by `api-key-env`. Provider calls use `OPENROUTER_API_KEY_<SERVICE>`,
falling back to `OPENROUTER_API_KEY_ITER_CODEX`. Keep the `openrouter/` model
prefix for client prompt caching and response-cost accounting.

Discovery checks the local router's models listing or the provider's key-info
endpoint without generating tokens. Chat and decisions routes sharing a
provider origin and service key share one probe per poll. These checks verify
reachability and authentication; opt-in live seat tests verify an actual reply
and can incur charges. The router's hosting service owns its lifecycle.

## SSH Tunnels

```bash
uv run iter-codex tunnel start iter   # Start SSH tunnel to remote host
uv run iter-codex tunnel stop         # Stop tunnel
uv run iter-codex tunnel status       # Show active tunnels
```

## HPC Job Management

```bash
uv run iter-codex hpc status          # Active SLURM allocations
uv run iter-codex hpc shell           # Interactive compute shell
uv run iter-codex hpc run <cmd>       # Execute on compute node
uv run iter-codex hpc submit <script> # Submit batch job
uv run iter-codex hpc attach <jobid>  # Attach to running job
uv run iter-codex hpc cancel <jobid>  # Cancel job
uv run iter-codex hpc info <jobid>    # Job details
```

## MCP Server

```bash
uv run iter-codex serve                                  # Full mode (all tools)
uv run iter-codex serve --read-only                      # Search/read only
uv run iter-codex serve --dd-only --transport streamable-http  # DD-only container
uv run iter-codex serve --transport stdio                 # For MCP clients
```

## Critical Rules

1. **All services MUST run as SLURM jobs** — never nohup, screen, tmux, or ssh &
2. **Check SLURM allocation** before starting services: `squeue -u $USER`
3. **Never start services directly on compute nodes via SSH** — use SLURM
4. **If a node is draining**, the fix is to get the node resumed — not work around SLURM
