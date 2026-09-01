# Graph Profile Configuration

Graph operation has two independent selectors. They must not be confused:

| Concept | Authority | How to change it | Example |
|---------|-----------|------------------|---------|
| **Data identity** | Active `neo4j/` symlink | `imas-codex graph switch NAME` | `codex` (all facilities + IMAS DD) |
| **Server location** | `location` config | `IMAS_CODEX_GRAPH_LOCATION` | `titan` (SLURM compute location) |

The graph name comes from the symlink target, not from an environment
variable. `IMAS_CODEX_GRAPH_LOCATION` selects where Neo4j runs and the port
slot it uses; it never selects or repoints the data directory.

## Configuration

All graph settings live in `pyproject.toml`:

```toml
[tool.imas-codex.graph]
location = "titan"      # Where it runs (override: IMAS_CODEX_GRAPH_LOCATION=local)
username = "neo4j"
password = "imas-codex"

# Port slots — position = port offset
# iter=0 (7687/7474), tcv=1 (7688/7475), jt-60sa=2 (7689/7476), ...
locations = ["iter", "tcv", "jt-60sa", "jet", "west", "mast-u", "asdex-u", "east", "diii-d", "kstar"]
```

## Key Concepts

### Data identity (what data)

Each graph data directory lives at
`~/.local/share/imas-codex/.neo4j/<name>/`. The active
`~/.local/share/imas-codex/neo4j` symlink points to exactly one of them:

- `"codex"` — the main graph with all facilities + IMAS data dictionary
- `"tcv"` — a single-facility graph
- `"sandbox"` — any arbitrary name for experimentation

Use `imas-codex graph list` to inspect the available directories and
`imas-codex graph switch NAME` to stop Neo4j when needed, repoint the symlink,
and restart it. `load ARCHIVE TARGET` and `clear TARGET` require `TARGET` to
match this active name and refuse before side effects when it does not.

### Location (where it runs)

The **location** determines where Neo4j physically runs. Each location maps
to an SSH alias and a port slot:

| Location | Bolt Port | HTTP Port | SSH alias |
|----------|-----------|-----------|-----------|
| iter | 7687 | 7474 | `iter` |
| tcv | 7688 | 7475 | `tcv` |
| jt-60sa | 7689 | 7476 | `jt-60sa` |
| jet | 7690 | 7477 | `jet` |

Port formula: `bolt = 7687 + index`, `http = 7474 + index` (index from the
`locations` list).

### How data identity and location interact

The two selectors meet only when `resolve_neo4j()` assembles a connection:
the active symlink supplies `name` and `data_dir`, while the configured
location supplies `host`, `uri`, `bolt_port`, and `http_port`. Changing a data
name does not change ports. Changing a location does not change data identity.

### SSH hosts

By default, each location's name doubles as its SSH alias (e.g. location
`"tcv"` → `ssh tcv`). Only add explicit entries in `[graph.hosts]` when
the SSH alias differs from the location name:

```toml
[tool.imas-codex.graph.hosts]
# Only needed when SSH alias ≠ location name:
# custom-location = "my-ssh-alias"
```

## URI Resolution

```
Location → is_local_host(location) → URI
                    ↓ (remote)
              auto-tunnel → bolt://localhost:{port+10000}
```

1. A facility or compute location resolves to its facility's port slot.
2. `is_local_host("iter")` checks facility private YAML:
   - On ITER login node: True → `bolt://localhost:7687`
   - Elsewhere: False → auto-tunnel → `bolt://localhost:17687`

## Auto-Tunneling

When connecting to a **remote** location, the profile resolver automatically
establishes an SSH tunnel with a +10000 offset:

```
Direct (on the host):    bolt = 7687 + offset
Tunneled (remote):       bolt = 17687 + offset
```

Override with env var: `IMAS_CODEX_TUNNEL_BOLT_ITER=17687`

Manual tunnel management:
```bash
imas-codex tunnel start iter         # Start tunnel
imas-codex tunnel status             # Show active tunnels
imas-codex tunnel stop iter          # Stop tunnel
```

## Configuration Scenarios

### 1. Configured service location

```toml
[tool.imas-codex.graph]
location = "titan"   # Runs as a SLURM service on the configured compute location
```

Compute locations inherit their parent facility's port slot. On the facility,
the client discovers the service allocation and connects directly; from
elsewhere, it establishes a tunnel.

### 2. Select a per-facility graph

Switching changes the symlink, not the location or ports:

```bash
imas-codex graph switch tcv
imas-codex graph status
```

### 3. Local Development

```toml
[tool.imas-codex.graph]
name = "codex"
location = "local"    # No SSH, direct localhost access
```

Or via env var:
```bash
export IMAS_CODEX_GRAPH_LOCATION=local
```

### 4. Multiple locations

Use the location override to connect to a different deployment. This changes
the host and port slot only:

```bash
# Each location gets its own tunnel port:
#   iter:   17687/17474
#   tcv:    17688/17475
#   jt-60sa: 17689/17476

IMAS_CODEX_GRAPH_LOCATION=iter imas-codex graph status  # iter ports
IMAS_CODEX_GRAPH_LOCATION=tcv  imas-codex graph status  # tcv ports
```

## Data Directory Convention

| Object | Directory |
|--------|-----------|
| `codex` graph | `~/.local/share/imas-codex/.neo4j/codex/` |
| `tcv` graph | `~/.local/share/imas-codex/.neo4j/tcv/` |
| `sandbox` graph | `~/.local/share/imas-codex/.neo4j/sandbox/` |
| Active selector | `~/.local/share/imas-codex/neo4j` symlink |

## Quick Reference

| Env Var | Purpose |
|---------|---------|
| `IMAS_CODEX_GRAPH_LOCATION` | Override where Neo4j runs |
| `IMAS_CODEX_TUNNEL_BOLT_ITER` | Override tunnel port |
| `NEO4J_URI` | Override URI completely |

| CLI Command | Purpose |
|-------------|---------|
| `graph profiles` | List all profiles and status |
| `graph status` | Show active identity, service state, and backup currency |
| `graph list` | List graph data directories and the active one |
| `graph switch NAME` | Repoint the active data symlink |
| `graph start` | Start active graph |
| `graph stop` | Stop active graph |
| `graph shell` | Interactive Cypher shell |
| `graph secure` | Rotate Neo4j password |
| `graph tags` | List GHCR tags |
| `graph prune` | Prune old GHCR tags |
| `tunnel start iter` | Manual tunnel to iter |
| `tunnel status` | Show active tunnels |
| `graph push` | Push graph to GHCR |
| `graph pull` | Fetch + load from GHCR |

`IMAS_CODEX_GRAPH` is not a data selector and is not read by
`resolve_neo4j()`. Use `graph switch NAME` whenever the active data identity
must change.
