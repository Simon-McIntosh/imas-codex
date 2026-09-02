This file governs the `imas_codex/embeddings/` subtree — the embedding server, its SLURM
lifecycle, and the local embedding clients. The embedding server MUST run as a SLURM job,
never bypassed with nohup/screen/tmux (SLURM provides cgroup isolation, clean lifecycle,
accounting and drain cleanup); the shared services rule is in `imas_codex/graph/AGENTS.md`.



### Embedding server

Config: `[tool.imas-codex.embedding]`. `get_embedding_location()` returns the facility or `"local"`. Port = `18765 + offset` in the shared `locations` list.

```bash
imas-codex embed start [-g 2]    # Start (optionally with N GPUs)
imas-codex embed status          # Health + SLURM job + node state
imas-codex embed restart -g 8    # Restart with 8 GPUs (~18s cycle)
imas-codex embed stop            # Stop SLURM job + cleanup rogue processes
imas-codex embed logs            # View SLURM logs
imas-codex embed service install # Install systemd service (login node only)
```

Troubleshooting: `embed status` shows node state. Common: node draining → ask admin to RESUME; rogue process → `embed stop` kills it; package issue → check `embed logs` and report the node/root environment as a blocker rather than rebuilding it; timeouts → check tunnel (`lsof -i :18765`).
