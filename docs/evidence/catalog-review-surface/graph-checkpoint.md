# Full-graph checkpoint, 2026-09-05

A whole-graph recovery archive was taken from the live `codex` store and the
database was proved to come back afterwards. Before this run the newest archive
in the recovery directory was
`imas-codex-graph-dev-6fc745c-20260901T125109Z.tar.gz` (2,458,442,456 bytes,
written 2026-09-01), so the graph had been unbacked for four days while the
store was being mutated.

The operation was a single `imas-codex graph export` with no facility filter,
leaving the server restart at its default. That command stops Neo4j, dumps,
archives, and resubmits the service. Neo4j runs as SLURM job `codex-neo4j` on
`98dci4-gpu-0002`, so the restart depends on the scheduler and is the part that
can fail.

Command, exit `0`:

```
uv run imas-codex graph export -o \
  ~/.local/share/imas-codex/backups/imas-codex-graph-dev-5b5faf1-20260905T121249Z.tar.gz
```

`-o` is the only departure from a bare `graph export`, and it is load-bearing
rather than cosmetic. The default destination is the exports directory
(`imas_codex/cli/graph/data.py`), while `get_backup_currency` reads the backups
directory (`BACKUPS_DIR` in `imas_codex/graph/profiles.py`, used at
`imas_codex/graph/neo4j_ops.py`). Written to the default location the archive
lands where the currency instrument does not look, and the checkpoint would not
register as a backup at all. The prior receipt in this directory used the same
`--output` placement for the same reason. No facility filter was passed, so the
whole graph is captured.

## 1. The archive

- Path: `/home/ITER/mcintos/.local/share/imas-codex/backups/imas-codex-graph-dev-5b5faf1-20260905T121249Z.tar.gz`
- Exact byte size: `2,451,894,300`
- Modification time: `2026-09-05 14:15:36.791694815 +0200` (epoch `1788610536`)
- SHA-256: `4e6d38c9a9923825c3b169a84bff8fb3435533929ff755bed6257280a89a89af`
- `gzip -t <archive>` exit `0`.
- `tar -tzvf <archive>` exit `0`, carrying a **non-empty** dump member:

```
drwxr-xr-x  imas-codex-graph-dev-5b5faf1-20260905T121257Z/
-rw-r--r--  2454058276  imas-codex-graph-dev-5b5faf1-20260905T121257Z/graph.dump
-rw-r--r--        2162  imas-codex-graph-dev-5b5faf1-20260905T121257Z/manifest.json
```

The dump member is 2,454,058,276 bytes, which is what makes this a full
recovery archive under `_is_full_recovery_archive`: that predicate is
content-based, matching any gzip tar holding a non-empty `graph.dump`, so
recognition does not depend on the filename.

## 2. Backup currency reports `stale`, not `current`

This is the one measure that was not met, and it was not met for a structural
reason rather than a fault in the checkpoint. Verbatim, after the export:

```
BackupCurrency(status='stale', backup_path=PosixPath('/home/ITER/mcintos/.local/share/imas-codex/backups/imas-codex-graph-dev-5b5faf1-20260905T121249Z.tar.gz'), backup_modified_at=datetime.datetime(2026, 9, 5, 12, 15, 36, 791695, tzinfo=datetime.timezone.utc), live_path=PosixPath('/home/ITER/mcintos/.local/share/imas-codex/neo4j/data/databases/neo4j/id-buffer.tmp.0'), live_modified_at=datetime.datetime(2026, 9, 5, 12, 15, 59, 748477, tzinfo=datetime.timezone.utc), age_seconds=22.956782, backup_size_bytes=2451894300)
```

For contrast, the same call before the export, against the four-day-old
archive:

```
BackupCurrency(status='stale', backup_path=PosixPath('/home/ITER/mcintos/.local/share/imas-codex/backups/imas-codex-graph-dev-6fc745c-20260901T125109Z.tar.gz'), backup_modified_at=datetime.datetime(2026, 9, 1, 12, 52, 53, 778724, tzinfo=datetime.timezone.utc), live_path=PosixPath('/home/ITER/mcintos/.local/share/imas-codex/neo4j/data/transactions/neo4j/checkpoint.0'), live_modified_at=datetime.datetime(2026, 9, 5, 9, 1, 47, 950868, tzinfo=datetime.timezone.utc), age_seconds=331734.172144, backup_size_bytes=2458442456)
```

So the checkpoint moved the measured lag from 331,734 s to 22.96 s and
repointed `backup_path` at the new archive. It did not reach `current`.

**An export that restarts the server can never report `current`.** The status
is `current` only when `age_seconds` is exactly 0, and `age_seconds` is
`max(0, newest-live-file-mtime - archive-mtime)` over everything under the live
`data/` tree. The export seals the archive *before* resubmitting the service,
and the restart then writes into `data/`. Every one of the newest live files
shares a single mtime of `12:15:59.74Z` — the startup burst, 23 s after the
archive was sealed at `12:15:36.79Z`:

```
1788610559.7484770 .../databases/neo4j/id-buffer.tmp.0
1788610559.7460424 .../databases/neo4j/schema/index/range-1.0/376/index-376
1788610559.7413562 .../databases/neo4j/schema/index/range-1.0/374/index-374
1788610559.7380848 .../databases/neo4j/schema/index/range-1.0/372/index-372
```

These are startup scratch and index files, not graph mutations — no write
reached the store between the dump and the reading. The 23 s is the restart
itself being counted as live-data divergence. This is why the 2026-09-01
receipt in this directory took its currency proof with `--no-restart`, at the
stopped-snapshot boundary, where the archive is necessarily the newest thing.

The two are mutually exclusive as the instrument is currently defined: a
checkpoint may restart the service, or it may report `current`, not both.
Nothing was done to force the reading — advancing the archive mtime by touch or
by copy would make the timestamp assert a property the artifact does not have,
which is the same failure the 2026-09-01 receipt reasoned through when it
refused to let a filename claim currency.

Resolving it is a decision, not a repair, and there are two clean options:
measure currency against the newest live file that is not written by startup
(so the comparison tracks data divergence rather than service lifecycle), or
capture the currency reading at the stopped boundary before the restart and
carry it forward as the checkpoint's proof.

## 3. The server is serving again

`imas-codex graph status` after the export:

```
SLURM:
  neo4j: job 1262883 RUNNING on 98dci4-gpu-0002
    Allocated: CPUs: 4/20, Time: 2:18

Neo4j: running
  Graph: codex
  Location: titan
  Data: /home/ITER/mcintos/.local/share/imas-codex/neo4j
  Bolt: 7687, HTTP: 7474
```

The export stopped job `1262259` and resubmitted the service as job `1262883`
on the same node. The store the worker path reaches and the store that was
dumped are one and the same: `data_dir` resolves to
`/home/ITER/mcintos/.local/share/imas-codex/neo4j`, the active symlink into
`.neo4j/codex`, served at `bolt://98dci4-gpu-0002:7687`.

## 4. The store came back intact

Read through an ordinary `GraphClient` after the restart:

| Measure | Before export | After restart | Peer figure, 08:31Z |
|---|---|---|---|
| `StandardName` identities | 4937 | **4937** | 4937 |
| Current `DDVersion` | 4.1.1 | **4.1.1** | 4.1.1 |
| Total nodes | not taken | 1,628,593 | — |

Both figures were also taken *before* the export in this same run, against the
same connection, and are unchanged across the stop/dump/restart cycle. An equal
count is the evidence that the store returned whole rather than partially.

One correction to the reading method, since it would otherwise mislead the next
reader: `StandardName` identity is the `id` property, not `name`. Counting
`DISTINCT n.name` returns 0, and `DDVersion` has no `version` property — the
current version is the node with `is_current: true`, read as `v.id`. A first
attempt using the wrong property names produced a `0` that looked like an empty
store and was not one.

## What was not touched

No `graph load`, `switch`, `clear`, `init`, `push` or `pull` was run. No archive
was deleted or pruned; the four-day-old
`imas-codex-graph-dev-6fc745c-20260901T125109Z.tar.gz` is still in place beside
the new one. No source file was changed.
