This file governs the `imas_codex/discovery/` subtree — facility path, wiki, signal, and
code discovery, the claim-based worker coordination state machine, and discovery persistence.



## Graph State Machine

Status enums represent **durable states only**. No transient states like `scanning`, `scoring`, or `ingesting`.

**Worker coordination:** Claim via `claimed_at = datetime()` (status unchanged), complete by updating status and clearing `claimed_at = null`. Orphan recovery is automatic via timeout check in claim queries.

### Claim Patterns — Deadlock Avoidance

All claim functions **must** use three anti-deadlock patterns. Reference implementations: `discovery/wiki/graph_ops.py`, `discovery/code/graph_ops.py`. Shared infrastructure: `discovery/base/claims.py`.

1. **`@retry_on_deadlock()`** — decorator from `claims.py`. Retries on `TransientError` with exponential backoff + jitter. Apply to every function that writes `claimed_at`.
2. **`ORDER BY rand()`** — randomize lock acquisition order. Deterministic ordering (`ORDER BY v.version ASC`, `ORDER BY score DESC`) causes lock convoys where concurrent workers deadlock on the same rows.
3. **`claim_token` two-step verify** — SET a UUID token in step 1, then read back by token in step 2. Prevents double-claiming race conditions.

```python
from imas_codex.discovery.base.claims import retry_on_deadlock

@retry_on_deadlock()
def claim_items(facility: str, limit: int = 10) -> list[dict]:
    token = str(uuid.uuid4())
    with GraphClient() as gc:
        gc.query("""
            MATCH (n:MyNode {facility_id: $facility})
            WHERE n.status = 'discovered' AND n.claimed_at IS NULL
            WITH n ORDER BY rand() LIMIT $limit
            SET n.claimed_at = datetime(), n.claim_token = $token
        """, facility=facility, limit=limit, token=token)
        return list(gc.query("""
            MATCH (n:MyNode {claim_token: $token})
            RETURN n.id AS id, n.path AS path
        """, token=token))
```

**Never** use deterministic `ORDER BY` in claim queries. **Never** write a manual retry loop for deadlocks — use `@retry_on_deadlock()`. See `imas_codex/discovery/README.md` for detailed rationale.

### FacilityPath Lifecycle

```
discovered → explored | skipped | stale
```

| Score | Use Case |
|-------|----------|
| 0.9+ | IMAS integration, IDS read/write |
| 0.7+ | MDSplus access, equilibrium codes |
| 0.5+ | General analysis codes |
| <0.3 | Config files, documentation |

## Exploration

Before disk-intensive operations, **check facility excludes** to avoid repeating known timeouts:

```python
info = get_facility('tcv')
excludes = info.get('excludes', {})
print(excludes.get('large_dirs', []))
print(excludes.get('depth_limits', {}))
print(info.get('exploration_notes', [])[-3:])
```

When a command times out, **persist the constraint immediately** via `update_infrastructure()` in the repl. Never repeat a timeout.

### Persistence

| Discovery Type | Destination |
|----------------|-------------|
| Source files, paths, codes, trees | `add_to_graph()` (public graph) |
| Public facility config, wiki sites, discovery roots | `update_facility_config()` |
| Infrastructure notes (hostnames, tool versions) | `repl("update_infrastructure('tcv', {...})")` |

### Data Classification

- **Graph (public):** MDSplus tree names, analysis code names/versions/paths, TDI functions, diagnostic names
- **Infrastructure (private):** Hostnames, IPs, NFS mounts, OS versions, tool availability, user directories
