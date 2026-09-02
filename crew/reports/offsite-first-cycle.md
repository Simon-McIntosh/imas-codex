# First weekly offsite push cycle

The repaired login-node schedule was installed and its first immediate cycle completed successfully on `98dci4-srv-1006.iter.org`. The accepted receipt records `counts_match: true`, a measured wall time of **991.462436 seconds**, and **2,461,580,565 archive bytes**. The GitHub Packages API lists the new tag, `graph status` reports the offsite copy **current at 0 seconds**, Neo4j is back, and a fresh `GraphClient` census matches the receipt exactly.

## Measured outcome

Observed from revision `198ec8250173f4d15bfa26a3f2c20ae18e2d824e` on 2026-09-02.

| Evidence | Observed result | Verdict |
|---|---|---|
| Schedule replacement | `graph push --remove-schedule` removed the earlier unit; `graph push --schedule` installed and enabled the repaired service and timer | **pass** |
| Repaired service environment | Installed `PATH=/home/ITER/mcintos/bin:/home/ITER/mcintos/.local/bin:/usr/local/bin:/usr/bin`; `oras` resolves to `/home/ITER/mcintos/bin/oras` | **pass** |
| Immediate service | Started `2026-09-02 10:58:16 CEST`, exited `11:14:59 CEST`; `Result=success`, `ExecMainStatus=0`, inactive/dead after the successful oneshot | **pass** |
| Accepted receipt | `/home/ITER/mcintos/.local/share/imas-codex/offsite-push/receipts/dev-198ec82-20260902T085827Z.json`; `outcome=pushed`, `counts_match=true`, `error=null` | **pass** |
| Measured cycle cost | `wall_time_seconds=991.462436`; `archive_bytes=2461580565` (2.462 GB decimal, 2.293 GiB) | **pass** |
| Registry publication | GitHub Packages API version count increased from 47 to 48 and lists `dev-198ec82-20260902T085827Z-r1` as version id `1199464046` | **pass** |
| Offsite currency | `imas-codex graph status`: new tag is newest offsite copy; `Offsite behind live data: 0 s (current)` | **pass** |
| Neo4j recovery | `graph status`: Neo4j running in SLURM job `1261130` on `98dci4-gpu-0002` | **pass** |
| Post-restart census | `GraphClient`: 1,614,780 nodes, 4,259,356 relationships, 70 labels; exact equality with the receipt live census | **pass** |
| Timer next elapse | Timer enabled, active/waiting; next `2026-09-07 00:00:00 CEST` (391,384 s / 4.529918 d from the final snapshot) | **pass** |

The cycle spent 552 seconds queued for the archive-scope census after `sun_debug` filled, then job `1261131` completed its census in 65 seconds. That real scheduler delay is included in the receipt wall-time measurement.

## systemd evidence

The earlier unit was removed before reinstalling the repaired schedule:

```text
$ imas-codex graph push --remove-schedule
Weekly login-node graph push timer removed

$ imas-codex graph push --schedule
Weekly login-node graph push timer installed and enabled
Service: /home/ITER/mcintos/.config/systemd/user/imas-codex-graph-push.service
Timer:   /home/ITER/mcintos/.config/systemd/user/imas-codex-graph-push.timer
```

The installed service now carries the uploader directory and remains pinned to the login host:

```ini
ConditionHost=98dci4-srv-1006.iter.org
WorkingDirectory=/home/ITER/mcintos/Code/imas-codex
EnvironmentFile=-/home/ITER/mcintos/Code/imas-codex/.env
Environment="PATH=/home/ITER/mcintos/bin:/home/ITER/mcintos/.local/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/ITER/mcintos/bin/uv run --no-sync --project /home/ITER/mcintos/Code/imas-codex imas-codex graph push --cycle
```

Final timer status:

```text
imas-codex-graph-push.timer
Loaded: loaded; enabled
ActiveState=active
SubState=waiting
UnitFileState=enabled
Result=success
NextElapseUSecRealtime=Mon 2026-09-07 00:00:00 CEST
```

Final service status:

```text
imas-codex-graph-push.service
Loaded: loaded; static
ActiveState=inactive
SubState=dead
Result=success
ExecMainStartTimestamp=Wed 2026-09-02 10:58:16 CEST
ExecMainExitTimestamp=Wed 2026-09-02 11:14:59 CEST
ExecMainCode=1
ExecMainStatus=0
```

`ExecMainCode=1` is systemd's `CLD_EXITED` code; `ExecMainStatus=0` and `Result=success` are the successful process result.

## Receipt and archive evidence

Receipt path:

```text
/home/ITER/mcintos/.local/share/imas-codex/offsite-push/receipts/dev-198ec82-20260902T085827Z.json
```

Acceptance fields:

```json
{
  "outcome": "pushed",
  "archive_stamp": "dev-198ec82-20260902T085827Z",
  "archive_ref": "ghcr.io/simon-mcintosh/imas-codex-graph:dev-198ec82-20260902T085827Z-r1",
  "archive_path": "/home/ITER/mcintos/.local/share/imas-codex/exports/imas-codex-graph-dev-198ec82-20260902T085827Z.tar.gz",
  "archive_bytes": 2461580565,
  "wall_time_seconds": 991.462436,
  "counts_match": true,
  "error": null,
  "started_at": "2026-09-02T08:58:27.336560+00:00",
  "completed_at": "2026-09-02T09:14:58.798993+00:00"
}
```

The archive exists at the receipt path with the same measured size:

```text
/home/ITER/mcintos/.local/share/imas-codex/exports/imas-codex-graph-dev-198ec82-20260902T085827Z.tar.gz
2461580565 bytes
```

## Registry and currency evidence

Before the cycle, the GitHub Packages API returned 47 versions and the newest full-graph package remained `v5.3.0-rc6`. After the cycle, the same API returned 48 versions and the requested tag as its newest record:

```json
{
  "api_version_count": 48,
  "id": 1199464046,
  "name": "sha256:99fb55826b165ceef190595a21e444be4a2b222eb8d6feb6323712d9f7b7b6c8",
  "created_at": "2026-09-02T09:14:57Z",
  "updated_at": "2026-09-02T09:14:57Z",
  "tags": ["dev-198ec82-20260902T085827Z-r1"]
}
```

`imas-codex graph status` after publication and restart reported:

```text
Graph manifest:
  Pushed: True
  Pushed as: dev-198ec82-20260902T085827Z-r1

Backup currency:
  Newest live file: .../id-buffer.tmp.0 (2026-09-02T09:01:50.521640+00:00)
  Newest offsite copy: ghcr.io/simon-mcintosh/imas-codex-graph:dev-198ec82-20260902T085827Z-r1 (2026-09-02T09:14:57+00:00)
  Offsite behind live data: 0 s (current)

SLURM:
  neo4j: job 1261130 RUNNING on 98dci4-gpu-0002

Neo4j: running
```

## GraphClient census evidence

A fresh `live_graph_census()` call through `GraphClient` after Neo4j restarted matched the receipt's live census exactly, including every label count:

```json
{
  "receipt_counts_match": true,
  "receipt_live": {
    "node_count": 1614780,
    "relationship_count": 4259356,
    "label_count": 70,
    "label_counts_sha256": "1e1c11a2642999adb26fd4fe4b8c6eccf2299db0f836ecbb6af79a1ffa7b14d6"
  },
  "graphclient_after_restart": {
    "node_count": 1614780,
    "relationship_count": 4259356,
    "label_count": 70,
    "label_counts_sha256": "1e1c11a2642999adb26fd4fe4b8c6eccf2299db0f836ecbb6af79a1ffa7b14d6"
  },
  "matches_receipt_live_census": true
}
```

## Operational state left in place

- The repaired timer remains enabled and active, with its next calendar-week elapse at `2026-09-07 00:00:00 CEST`.
- The immediate oneshot has finished successfully and no duplicate cycle is running.
- Neo4j is running and queryable; its post-restart census matches the accepted receipt.
- The accepted archive remains on disk and the new private GHCR package version is listed by the GitHub Packages API.
- The user journal is not readable by this account, so durable evidence comes from systemd status/show, the receipt JSON, the GitHub Packages API, `graph status`, and `GraphClient`.
