# Graph substrate audit for the catalog fold-back rehearsal

Audit basis: read-only inspection of `imas-codex` commit
`7cd09ff39b87c5059b118fd0fb28e0ab0339d618` on 2026-09-01. The live plan's
constraint is decisive: ordinary approval undo does not erase an accepted human
edit from graph history, so the rehearsal must not run against the live graph.

## Measured result

- Requested command coverage: **5/5** (`graph init`, `switch`, `export`, `load`,
  and `clear`), including both local and remote branches. The shared command
  registration and profile resolver were also traced.
- Current active on-disk identity: the symlink
  `/home/ITER/mcintos/.local/share/imas-codex/neo4j` resolves to
  `/home/ITER/mcintos/.local/share/imas-codex/.neo4j/codex`.
- Local graph directories: **3** (`codex`, `imas`, `titan-test`); there is no
  pre-existing rehearsal directory.
- Live `codex` directory size: **15,058,189,845 bytes**.
- Current backup: **none**. The real-dump directory
  `/home/ITER/mcintos/.local/share/imas-codex/backups` contains no dump. The
  recovery directory contains zero-byte `graph_data_existed.marker` files only;
  `backup_existing_data()` explicitly creates markers, not recoverable data.
- Newest archive, but **not current**:
  `/home/ITER/mcintos/.local/share/imas-codex/exports/imas-codex-graph-dev-d4c6ac3.tar.gz`,
  **2,457,088,786 bytes**, modified **2026-08-27 12:49:09.391121934 CEST**.
  Live database files were modified later, including
  `data/transactions/neo4j/checkpoint.0` at
  **2026-09-01 14:16:59.710037449 CEST**, so this archive cannot be called a
  current backup.

The packaged local status command succeeded and reported the active local graph
manifest as having been loaded from the 2026-03-26 archive. A direct Bolt
read-only count/`GraphMeta` query could not be executed in this worker sandbox:
socket creation to `localhost:7687` was denied with `PermissionError: [Errno 1]
Operation not permitted`. This does not weaken the filesystem selector or backup
findings, but it means the operator run must capture the logical counts/identity
receipt itself.

## Selector and registration path

All five commands are registered in
`imas_codex/cli/graph/__init__.py:47` (`_register_graph_commands`). They all
ultimately select storage through
`imas_codex/graph/profiles.py:389` (`resolve_neo4j`): location chooses the host
and ports, while data identity comes from the single active symlink. There is no
per-command destination profile for `load` or `clear`.

This distinction is the main safety boundary: `IMAS_CODEX_GRAPH` is described in
some user-facing documentation, but current `resolve_neo4j()` does not use it to
select a named directory. The active `neo4j -> .neo4j/<name>` symlink is the
actual data selector. Therefore an environment variable alone does **not**
isolate a destructive command.

## Complete command-path walk

### `imas-codex graph init NAME`

Entry point: `imas_codex/cli/graph/data.py:963` (`graph_init`).

Local path:

1. `imas_codex/graph/dirs.py:105` (`create_graph_dir`) creates or reuses
   `.neo4j/NAME`; `--force` permits reuse and rewrites configuration but does not
   clear the data directory.
2. `imas_codex/graph/dirs.py:216` (`switch_active_graph`) unlinks and repoints
   the single active symlink.
3. `imas_codex/cli/graph/data.py:873` (`_start_neo4j_after_switch`) starts the
   server only if the HTTP port is not already responding.
4. `imas_codex/graph/meta.py:23` (`init_graph_meta`) executes a `MERGE` and, on
   both create and match, writes `name`, `facilities`, `imas`, and `updated_at`.

**Live-write/destruction assessment:** it writes graph content through
`init_graph_meta`. With `NAME=codex`, especially with `--force`, it updates the
live `GraphMeta`. More seriously, on the local branch a different `NAME` is
switched into the symlink **without first stopping a running Neo4j**. If live
codex is still serving the port, the command skips startup and its Bolt client
can update `GraphMeta` on the still-running live codex process even though the
symlink now names the new directory. It does not bulk-delete nodes, but direct
use while codex is running is not safe enough for isolation.

Remote path:

1. `imas_codex/graph/remote.py:110` (`remote_create_graph_dir`) creates/reuses
   the remote directory.
2. `graph_init` stops the service if the shared HTTP port responds.
3. `imas_codex/graph/remote.py:188` (`remote_switch_active_graph`) repoints the
   remote symlink, then the target service starts.
4. `init_graph_meta` writes through the tunnel.

**Live-write/destruction assessment:** a new name normally protects codex data
because the service is stopped before the switch. `NAME=codex --force` still
writes live `GraphMeta`; a wrong target/service selection can also expose live
state. No backup is created.

### `imas-codex graph switch NAME`

Entry point: `imas_codex/cli/graph/data.py:717` (`graph_switch`).

Local path: if `NAME` does not exist, `create_graph_dir` silently creates it;
the command stops the responding server using
`imas_codex/cli/graph/data.py:846` (`_stop_neo4j_for_switch`), repoints via
`switch_active_graph`, and restarts via `_start_neo4j_after_switch` only if it
observed a running server before the switch.

Remote path: it similarly auto-creates through `remote_create_graph_dir`, stops
the service, repoints through `remote_switch_active_graph`, and starts the target
service.

**Live-write/destruction assessment:** `switch` does not itself modify graph
nodes or overwrite database files, but it changes the global selector and causes
a live-service interruption. A newly auto-created graph is empty and has no
`GraphMeta`. Every later unqualified graph command now targets the new active
directory. Conversely, switching to `codex` makes every later `load`/`clear`
capable of destroying live state. There is no backup or confirmation prompt.

### `imas-codex graph export`

Entry point: `imas_codex/cli/graph/data.py:277` (`graph_export`).

Local full-export path: `imas_codex/graph/neo4j_ops.py:49`
(`Neo4jOperation`) acquires an operation lock, stops the active service,
`imas_codex/graph/neo4j_ops.py:680` (`run_neo4j_dump`) dumps the selected active
directory, and the context restarts the service unless `--no-restart` is set.
It writes a dump file and archive, not live nodes.

Local filtered path: `imas_codex/graph/temp_neo4j.py:757`
(`create_dd_only_dump`) and `:792` (`create_facility_dump`) load the dump into a
temporary Neo4j instance, delete/filter nodes there, and update only the
temporary `GraphMeta`. `--source-dump` also operates on the supplied dump rather
than the live graph.

Remote path: `imas_codex/graph/remote.py:983`
(`build_remote_export_script`) stops the active remote service, dumps the active
symlink target, archives it, and restarts. Remote facility filtering is refused.
The remote script always restarts, so the CLI's `--no-restart` does not provide
the same guarantee remotely.

**Live-write/destruction assessment:** no graph-content mutation in the normal
path; it can stop/restart the live service and writes into its `dumps/`
directory. This is the correct public mechanism for making the missing current
recovery archive, using a full unfiltered export. It must complete and the
archive must be statted before any switch.

### `imas-codex graph load ARCHIVE`

Entry point: `imas_codex/cli/graph/data.py:496` (`graph_load`). It has no graph
name/destination argument; `resolve_neo4j` binds it to the currently active
symlink.

Local path: `Neo4jOperation` stops the selected server;
`imas_codex/graph/neo4j_ops.py:426` (`backup_existing_data`) creates only a
zero-byte recovery marker; then `neo4j-admin database load ...
--overwrite-destination=true` replaces the active database. Authentication is
reset. The server restarts unless `--no-restart` is set.

Remote path: `imas_codex/graph/remote.py:491` (`remote_load_archive`) stops the
selected service and runs the same `neo4j-admin ...
--overwrite-destination=true`, resets authentication, and restarts. It creates
neither a real dump nor even the local marker.

**Live-write/destruction assessment:** this is destructive to the active
directory. If the symlink resolves to `codex`, it overwrites the live graph.
The accepted `--force` parameter is unused in both branches; overwrite happens
regardless. The remote branch also ignores `--no-restart`. A positive symlink
assertion immediately before `load` is mandatory.

### `imas-codex graph clear [--force]`

Entry point: `imas_codex/cli/graph/data.py:1231` (`graph_clear`). It resolves the
active profile, requires a running server, shows counts when possible, asks for
confirmation unless `--force` is supplied, and calls
`imas_codex/graph/client.py:437` (`GraphClient.drop_all`). `drop_all` executes
`MATCH (n) DETACH DELETE n RETURN count(n)`.

**Live-write/destruction assessment:** total destruction of the selected active
graph on local or remote connections. `--force` means only “skip the prompt”; it
does not select or protect a profile. If the active symlink is `codex`, all live
nodes and relationships are deleted. There is no automatic backup. It is safe
for unwind only after an exact assertion that the active link is the rehearsal
directory.

## Exact isolated-substrate runbook

These commands deliberately avoid `graph init`; the unsafe local init ordering
is unnecessary. They first create the missing current archive, leave codex
stopped, fingerprint the stopped live data directory, switch to an empty named
directory, assert the selector before overwrite, load the clone, and start it.
They must be run from the canonical checkout by an operator authorised to stop
and restart the graph service.

```bash
cd /home/ITER/mcintos/Code/imas-codex

LIVE_GRAPH_DIR=/home/ITER/mcintos/.local/share/imas-codex/.neo4j/codex
REHEARSAL_GRAPH_DIR=/home/ITER/mcintos/.local/share/imas-codex/.neo4j/catalog-foldback-rehearsal
ACTIVE_LINK=/home/ITER/mcintos/.local/share/imas-codex/neo4j
BACKUP_ARCHIVE=/home/ITER/mcintos/.local/share/imas-codex/backups/codex-pre-foldback-rehearsal-20260901T123636Z.tar.gz
BASELINE_DIGEST=/home/ITER/mcintos/.local/share/imas-codex/backups/codex-pre-foldback-rehearsal-20260901T123636Z.sha256
AFTER_DIGEST=/home/ITER/mcintos/.local/share/imas-codex/backups/codex-post-foldback-rehearsal-20260901T123636Z.sha256

test "$(readlink -f "$ACTIVE_LINK")" = "$LIVE_GRAPH_DIR"
test ! -e "$BACKUP_ARCHIVE"
mkdir -p /home/ITER/mcintos/.local/share/imas-codex/backups
uv run imas-codex graph export --no-restart --output "$BACKUP_ARCHIVE"
test -s "$BACKUP_ARCHIVE"
stat -Lc '%n %s bytes %y' "$BACKUP_ARCHIVE"

find "$LIVE_GRAPH_DIR/data" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum > "$BASELINE_DIGEST"

uv run imas-codex graph switch catalog-foldback-rehearsal
test "$(readlink -f "$ACTIVE_LINK")" = "$REHEARSAL_GRAPH_DIR"
uv run imas-codex graph load "$BACKUP_ARCHIVE" --no-restart
test "$(readlink -f "$ACTIVE_LINK")" = "$REHEARSAL_GRAPH_DIR"
uv run imas-codex graph list
uv run imas-codex graph start
```

At this point the fold-back commands may run only after their own read-only
receipt confirms the active directory is `catalog-foldback-rehearsal`. The
loaded clone retains the archive's `GraphMeta` content, including the old name;
directory/symlink identity is what current connection resolution actually uses.
Do not run `graph init --force` merely to rename `GraphMeta`, because it adds an
unnecessary write and re-enters the unsafe init path.

After capturing the accepted/refused/traceability receipts, unwind and prove the
live directory was byte-for-byte untouched while it was offline:

```bash
cd /home/ITER/mcintos/Code/imas-codex

LIVE_GRAPH_DIR=/home/ITER/mcintos/.local/share/imas-codex/.neo4j/codex
REHEARSAL_GRAPH_DIR=/home/ITER/mcintos/.local/share/imas-codex/.neo4j/catalog-foldback-rehearsal
ACTIVE_LINK=/home/ITER/mcintos/.local/share/imas-codex/neo4j
BASELINE_DIGEST=/home/ITER/mcintos/.local/share/imas-codex/backups/codex-pre-foldback-rehearsal-20260901T123636Z.sha256
AFTER_DIGEST=/home/ITER/mcintos/.local/share/imas-codex/backups/codex-post-foldback-rehearsal-20260901T123636Z.sha256

test "$(readlink -f "$ACTIVE_LINK")" = "$REHEARSAL_GRAPH_DIR"
uv run imas-codex graph clear --force
uv run imas-codex graph stop
test "$(readlink -f "$ACTIVE_LINK")" = "$REHEARSAL_GRAPH_DIR"
uv run imas-codex graph switch codex
test "$(readlink -f "$ACTIVE_LINK")" = "$LIVE_GRAPH_DIR"

find "$LIVE_GRAPH_DIR/data" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum > "$AFTER_DIGEST"
cmp "$BASELINE_DIGEST" "$AFTER_DIGEST"
uv run imas-codex graph start
```

`cmp` exit 0 is the hard live-untouched gate. Capture in addition the pre/post
logical `GraphMeta`, node, relationship, and affected `StandardName` receipts
once Bolt access is available, but do not substitute counts for the stopped
directory digest: equal counts can hide changed properties.

## Verdict

**NO-GO — do not run the fold-back rehearsal now.** An isolated substrate is
mechanically achievable with the stop/export/fingerprint/switch/assert/load
sequence above, but the current state fails the recovery prerequisite: **0
current backups exist**, and the newest 2.457 GB archive predates live writes by
more than five days. Direct `graph init` is also unsafe on the local running
service, while `graph load` and `graph clear` can destroy live codex whenever the
single active symlink is wrong. The verdict may change to go only after the
full export exists, its exact path/size/mtime are recorded, the stopped codex
digest is captured, and the active-link assertions succeed before every load or
clear.
