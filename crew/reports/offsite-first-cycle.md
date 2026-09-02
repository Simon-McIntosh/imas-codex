NEEDS-HELP: the login-node timer is installed, but its oneshot omits the directory containing `oras` from `PATH` and failed before upload, so the cycle produced no receipt or recovery point

tried: Installed the schedule with `imas-codex graph push --schedule`, verified the user timer, and immediately started `imas-codex-graph-push.service`. The service exported a count-valid 2,461,476,711-byte archive and restored Neo4j, then exited 1 after 203 seconds. Reproducing the unit's exact `PATH` resolves `oras` to `None` and raises `ClickException: oras not found in PATH`; the uploader is installed at `/home/ITER/mcintos/bin/oras`, while the generated unit includes only `/home/ITER/mcintos/.local/bin:/usr/local/bin:/usr/bin`.

options: (1) extend the implementation scope to include `/home/ITER/mcintos/bin` in the generated service `PATH`, retain a regression for the resolved uploader, reinstall the timer, and start the service once; (2) resolve the uploader path during schedule installation and use that durable absolute path in the service environment; (3) hot-patch the installed unit and rerun, but treat that only as recovery evidence because the next installation would reproduce the defect. The timer semantics also need adjudication: `OnCalendar=weekly` means the next calendar Monday, measured 4.561492 days away, not seven days after installation.

leaning: Option 1. It is the smallest durable change, matches how the unit already uses the absolute `uv` path, and lets the exact operator command remain the installation authority. The timer should also use elapsed-interval semantics if “seven days out” is literal acceptance rather than “once per calendar week.”

cost-if-wrong: Each unchanged retry stops and restarts Neo4j, writes another roughly 2.46 GB archive, consumes about 203 seconds before failing, creates no receipt, and leaves the offsite recovery point stale. Leaving the active timer untouched without landing a repair schedules the same failure for 2026-09-07 00:00:00 CEST.

## Measured outcome

Observed on login node `98dci4-srv-1006.iter.org` from revision `7fa960e7b698434dc585e537cc488b29fe43e31a`.

| Evidence | Observed result | Required result | Verdict |
|---|---|---|---|
| Timer installation | `imas-codex-graph-push.timer` is loaded, enabled, active, and waiting | Installed and active user timer | **met** |
| Immediate service | Started at `2026-09-02 10:25:30 CEST`; exited at `10:28:53`; `Result=exit-code`, `ExecMainStatus=1`; wall clock 203 s | Successful oneshot | **unmet** |
| Exported archive | `/home/ITER/mcintos/.local/share/imas-codex/exports/imas-codex-graph-dev-7fa960e-20260902T082540Z.tar.gz`; 2,461,476,711 B | Archive bytes recorded by a successful receipt | **partial**; artifact exists, receipt does not |
| Cycle receipt | No receipt directory or JSON was created under `/home/ITER/mcintos/.local/share/imas-codex/offsite-push/receipts` | Receipt path with `counts_match: true`, `wall_time_seconds`, and `archive_bytes` | **unmet** |
| GitHub Packages API | 47 versions; intended tag `dev-7fa960e-20260902T082540Z-r1` absent; newest remains version id `1008109688`, created `2026-07-07T12:27:23Z`, tags `v5.3.0-rc6` and `latest` | Newly listed scheduled tag | **unmet** |
| Offsite status | `stale`; `4,910,474` s behind live data; ref `ghcr.io/simon-mcintosh/imas-codex-graph:v5.3.0-rc6` | `current`, with age in seconds | **unmet** |
| Neo4j recovery | `graph status` reports Neo4j running in SLURM job `1261105` on `98dci4-gpu-0002`; GraphClient census succeeds | Neo4j back and queryable | **met** |
| Census equality | Archive manifest and post-restart GraphClient both contain 1,614,780 nodes, 4,259,356 relationships, and the same 70-label census | Receipt `counts_match: true` and a post-restart census equal to it | **partial**; archive/live equality is true, but no receipt exists |
| Next timer elapse | At `2026-09-02T10:31:27.057074+02:00`, next elapse was `2026-09-07T00:00:00+02:00`: 394,113 s or 4.561492 days away | Seven days out | **unmet** |

The requested quantitative done-when is therefore blocked: 2 gates are met, 2 are independently demonstrated but cannot be promoted without a receipt, and 4 are unmet.

## systemd evidence

Timer status after installation and after the failed immediate service:

```text
imas-codex-graph-push.timer
Loaded: loaded; enabled
ActiveState=active
SubState=waiting
UnitFileState=enabled
NextElapseUSecRealtime=Mon 2026-09-07 00:00:00 CEST
```

Service status after the immediate start:

```text
imas-codex-graph-push.service
Loaded: loaded; static
Active: failed (Result: exit-code)
ExecMainStartTimestamp=Wed 2026-09-02 10:25:30 CEST
ExecMainExitTimestamp=Wed 2026-09-02 10:28:53 CEST
ExecMainCode=1
ExecMainStatus=1
```

The installed service is host-pinned and does run from the main checkout:

```ini
ConditionHost=98dci4-srv-1006.iter.org
WorkingDirectory=/home/ITER/mcintos/Code/imas-codex
EnvironmentFile=-/home/ITER/mcintos/Code/imas-codex/.env
Environment="PATH=/home/ITER/mcintos/.local/bin:/usr/local/bin:/usr/bin"
ExecStart=/home/ITER/mcintos/bin/uv run --no-sync --project /home/ITER/mcintos/Code/imas-codex imas-codex graph push --cycle
```

The installed timer uses calendar-week semantics:

```ini
OnCalendar=weekly
Persistent=true
Unit=imas-codex-graph-push.service
```

## Failure localization

The service PATH check reproduces the prerequisite failure without replaying the export or upload:

```text
resolved_oras=None
click.exceptions.ClickException: oras not found in PATH. Install from: https://github.com/oras-project/oras/releases
```

The installed uploader is `/home/ITER/mcintos/bin/oras`. The outer cycle reaches the count gate first, so the archive is valid; the nested `graph push --source-dump` then calls `require_oras()` before upload. That exception occurs before `run_offsite_push_cycle()` can write its success receipt. The GitHub Packages API independently confirms that no tag was created.

The current systemd journal is not readable by this user (`journalctl --user` reports insufficient permissions), so the unit status, deterministic PATH reproduction, absent receipt, and absent API tag are the durable failure evidence. The older log at `/home/ITER/mcintos/.local/share/imas-codex/services/codex-graph-push.log` belongs to the superseded compute-node attempt and is not evidence for this login-node failure.

## Census and archive evidence

The exported archive manifest is readable and contains a valid graph census. A fresh `GraphClient` census after Neo4j restarted matched it exactly:

```json
{
  "archive_bytes": 2461476711,
  "archive_manifest": {
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
  "matches_archive_manifest": true,
  "receipt_path": null,
  "receipt_counts_match": null
}
```

## Registry and status evidence

The GitHub Packages REST API returned the following newest version and explicit absence check:

```json
{
  "api_version_count": 47,
  "wanted_tag": "dev-7fa960e-20260902T082540Z-r1",
  "wanted_present": false,
  "newest": {
    "id": 1008109688,
    "created_at": "2026-07-07T12:27:23Z",
    "updated_at": "2026-07-07T12:27:23Z",
    "tags": ["v5.3.0-rc6", "latest"]
  }
}
```

After Neo4j returned, `imas-codex graph status` reported:

```text
Backup currency:
  Newest live file: .../id-buffer.tmp.0 (2026-09-02T08:28:37.032406+00:00)
  Newest offsite copy: ghcr.io/simon-mcintosh/imas-codex-graph:v5.3.0-rc6 (2026-07-07T12:27:23+00:00)
  Offsite behind live data: 4910474 s (stale)

SLURM:
  neo4j: job 1261105 RUNNING on 98dci4-gpu-0002

Neo4j: running
```

## Operational state left in place

- The timer remains enabled and active because installation was explicitly requested.
- The failed service is inactive with a retained failed result; no duplicate cycle is running.
- Neo4j is running and its post-restart census matches the archive.
- The 2,461,476,711-byte archive remains available for diagnosis or a repaired cycle.
- No registry package version, receipt, or offsite-current claim was created.
