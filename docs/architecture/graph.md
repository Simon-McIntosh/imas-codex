# Knowledge Graph Architecture

> **Module**: `imas_codex.graph`

## Overview

The imas-codex knowledge graph is a Neo4j-based store that unifies:
- **Facility knowledge**: DataNodes, CodeChunks, Diagnostics, Analysis Codes, Wiki content
- **IMAS Data Dictionary**: IMASNode nodes with version tracking and embeddings
- **Cross-facility data**: Shared IMAS mappings, semantic clusters

All schema definitions live in **LinkML** (`schemas/*.yaml`) as the single source of truth.

## Graph Data Identity and Server Location

The active `~/.local/share/imas-codex/neo4j` symlink selects the graph data
directory. Each graph lives under
`~/.local/share/imas-codex/.neo4j/<name>/`, and `imas-codex graph switch NAME`
stops Neo4j when necessary, repoints the symlink, and restarts it. This is the
only supported procedure for changing graph data identity.

Server location is independent of data identity. The configured
`[tool.imas-codex.graph].location`, overridden by
`IMAS_CODEX_GRAPH_LOCATION`, selects the host, scheduler placement, and Bolt
and HTTP ports. It does not select a data directory. `IMAS_CODEX_GRAPH` is not
consumed by graph profile resolution and must not be used to aim a graph
command.

### Port Convention

| Location | Bolt | HTTP |
|----------|------|------|
| iter | 7687 | 7474 |
| tcv | 7688 | 7475 |
| jt-60sa | 7689 | 7476 |

### Location-Aware Resolution

The `host` field on each profile records where Neo4j physically runs:

- `host="iter"` — Neo4j runs on the ITER login node
- `host=None` — Neo4j runs locally
- At connection time, `is_local_host(host)` determines direct vs tunnel access

**From the configured facility** (location is directly reachable):
```
resolve_neo4j() → direct local or SLURM service URI
```

**From a remote workstation**:
```
resolve_neo4j() → localhost tunnel URI
```

**Dual-instance** (local + tunneled, conflicting ports):
```bash
# .env
IMAS_CODEX_TUNNEL_BOLT_ITER=17687
# Then: ssh -f -N -L 17687:localhost:7687 iter
```

### Connection Resolution Priority

1. `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` env vars (escape hatch)
2. `IMAS_CODEX_GRAPH_LOCATION` for the server location
3. `[tool.imas-codex.graph].location` in `pyproject.toml`
4. Convention-based port mapping for the resolved location

Data identity is resolved separately from the active symlink, before a
`Neo4jProfile` is returned.

### Configuration

```toml
# pyproject.toml
[tool.imas-codex.graph]
location = "titan"      # Where it runs (override: IMAS_CODEX_GRAPH_LOCATION=local)
username = "neo4j"
password = "imas-codex"
```

## Graph Client

```python
from imas_codex.graph import GraphClient

# Use the active symlink and configured location
with GraphClient() as client:
    result = client.query("MATCH (n:Facility) RETURN n.id")

# Resolve the active symlink and location explicitly
with GraphClient.from_profile() as client:
    print(client.get_stats())
```

To change data identity, run `imas-codex graph switch NAME`; do not set an
environment variable in Python.

## Graph Management CLI

### Server Operations

```bash
# Start/stop/status (under 'graph')
imas-codex graph start                 # Start the configured Neo4j service
imas-codex graph stop                  # Stop the configured Neo4j service
imas-codex graph status                # Show service, manifest, backup currency, and SLURM status
imas-codex graph profiles              # List all profiles
imas-codex graph shell                 # Interactive Cypher shell
```

### Graph Lifecycle

```bash
# Export and load
imas-codex graph export                # Full graph export
imas-codex graph export --facility tcv # Per-facility export (filtered)
imas-codex graph load archive.tar.gz codex  # TARGET must match the active symlink

# GHCR registry
imas-codex graph push                  # Push release to GHCR
imas-codex graph push --dev            # Push dev build
imas-codex graph push --facility tcv   # Push per-facility graph
imas-codex graph pull                  # Pull latest from GHCR
imas-codex graph pull --facility tcv   # Pull per-facility graph
imas-codex graph tags                  # List GHCR versions
imas-codex graph tags --facility tcv   # List per-facility versions

# Destructive operations name the active target
imas-codex graph load archive.tar.gz codex  # Load only when codex is active
imas-codex graph clear codex           # Clear codex (auto-backup first)

# Registry cleanup (dry-run by default in this example)
imas-codex graph prune --dev-only --keep 5 --dry-run
```

Both `load ARCHIVE TARGET` and `clear TARGET` refuse to proceed unless
`TARGET` is the name selected by the active symlink. To operate on another
graph, first run `imas-codex graph switch NAME`, then pass the same `NAME` as
the destructive command's target.

`graph status` also reports backup currency: the newest non-empty restorable
backup, the newest file under the active graph's `data/` directory, and the
measured number of seconds the backup is behind live data (`current`, `stale`,
or unavailable when no backup exists).

### SSH Tunnels

```bash
imas-codex tunnel start iter           # Start tunnel to specific host
imas-codex tunnel stop iter
imas-codex tunnel status               # Show active tunnels
```

## Per-Facility Federation

Full graph contains all facilities. Per-facility graphs are extracted via dump-and-clean:

1. Dump the full graph via `neo4j-admin database dump`
2. Load into a temporary Neo4j instance
3. Delete nodes with `facility_id != target_facility`
4. Delete orphaned non-DD nodes (no relationships)
5. Re-dump the cleaned graph

This preserves the full IMAS Data Dictionary (shared across facilities) while isolating facility-specific data.

```bash
# Create and push per-facility graph
imas-codex graph export --facility tcv
imas-codex graph push --facility tcv --dev

# End user activates a named data directory, then pulls into it
imas-codex graph switch tcv
imas-codex graph pull --facility tcv
imas-codex graph start
```

## GHCR Package Naming

| Package | Content |
|---------|---------|
| `imas-codex-graph` | Full unified graph (all facilities) |
| `imas-codex-graph-tcv` | TCV-only graph + IMAS DD |
| `imas-codex-graph-jt-60sa` | JT-60SA-only graph + IMAS DD |

## Schema Management

### LinkML as Single Source of Truth

```
imas_codex/schemas/
├── common.yaml      # Shared enums, PhysicsDomain
├── facility.yaml    # Facility nodes (SignalNode, CodeChunk, etc.)
└── imas_dd.yaml     # IMAS DD nodes (IMASNode, DDVersion, etc.)
```

Models auto-generated during `uv sync` via build hook. Regenerate manually:
```bash
uv run build-models --force
```

## Vector Indexes

| Index | Content |
|-------|---------|
| `imas_node_embedding` | IMASNode nodes |
| `cluster_embedding` | IMASSemanticCluster embeddings |
| `code_chunk_embedding` | CodeChunk nodes |
| `wiki_chunk_embedding` | WikiChunk nodes |
| `facility_signal_desc_embedding` | FacilitySignal descriptions |
| `facility_path_desc_embedding` | FacilityPath descriptions |
| `signal_node_desc_embedding` | SignalNode descriptions |
| `wiki_artifact_desc_embedding` | WikiArtifact descriptions |

## Docker Compose

```bash
# Default ports (iter convention)
docker compose --profile graph up

# Custom ports for a different facility
BOLT_PORT=7688 HTTP_PORT=7475 docker compose --profile graph up
```
