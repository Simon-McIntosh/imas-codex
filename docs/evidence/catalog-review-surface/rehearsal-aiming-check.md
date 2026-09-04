# Fold-back rehearsal aiming check

Measured against commit `abd5be253452db300936d229a96eaaa02a9c36d8` on
2026-09-04 at 23:56 CEST. This was a read-only inspection except for an
in-process refusal probe whose downstream service and graph-client calls were
replaced with mocks. No graph was stopped, restarted, loaded, cleared, pulled,
or switched.

## Verdict

**NO-GO.** The current tree has firing target guards for direct `load` and
`clear`, but the isolated-substrate go-condition is not met:

1. there is no current verified full export and no stopped-live-tree digest;
   the verified 2,458,442,456-byte archive predates the newest live write by
   271,912.874 seconds; and
2. `graph pull` does not take a destination argument. Its local path invokes
   the now-targeted `graph_load` command without the required target, while its
   remote path derives a name from the active profile rather than comparing an
   operator-named destination with the active symlink.

The stale backup alone is sufficient to hold the rehearsal. A direct
fetch-then-`load` runbook can avoid the `pull` defect, but that does not make the
missing current backup or stopped-tree digest true.

## Aiming mechanism in the current tree

The resolved identity chain is explicit:

- `imas_codex/graph/dirs.py:197-213` resolves the active
  `~/.local/share/imas-codex/neo4j` symlink and returns its target directory
  name.
- `imas_codex/graph/profiles.py:235-249` exposes that name, and
  `imas_codex/graph/profiles.py:389-408` places it in the resolved
  `Neo4jProfile`.
- `imas_codex/cli/graph/data.py:155-167` compares the command's `target` with
  `profile.name` and raises `ClickException` with a refusal and switch
  instruction when they differ.

Command-by-command findings:

- **Load is aimed.** `imas_codex/cli/graph/data.py:638-650` declares
  `graph load ARCHIVE TARGET`; lines 662-663 resolve the symlink-backed profile
  and call the target guard before password lookup, remote dispatch, service
  stop, extraction, or replacement.
- **Clear is aimed.** `imas_codex/cli/graph/data.py:1392-1405` declares
  `graph clear TARGET` and calls the same guard before the service-status check
  or graph-client construction.
- **Pull is not aimed, and its signature differs from the requested design.**
  `imas_codex/cli/graph/registry.py:1101-1133` declares options only and has no
  target argument. On the remote path, lines 1283-1290 pass `profile.name` to a
  generated load script without accepting or comparing an operator-named
  destination. On the local path, lines 1371-1378 invoke `graph_load` with only
  the archive and `--force`, omitting the target that `graph_load` now requires;
  this path will stop at Click argument parsing rather than load.

The real-backup and fail-closed implementation is present on the two local
archive-replacement call sites. `imas_codex/graph/neo4j_ops.py:655-688`
implements `backup_graph_dump` and refuses an absent or zero-byte dump. Direct
load calls it at `imas_codex/cli/graph/data.py:726-735`; local pull calls it at
`imas_codex/cli/graph/registry.py:1347-1349`. The call is not wrapped in a
swallow, so its exception aborts the replacement path. The old misleading
marker identity has also been corrected: the marker-only helper is named
`write_data_presence_marker` at
`imas_codex/graph/neo4j_ops.py:639-650`.

## Refusal firing probe

The probe resolved the real active profile with tunnelling disabled, generated
the guaranteed-different target by prefixing the resolved name, then invoked
the `graph_clear` Click callback through `CliRunner`. Both the service-status
function and graph-client constructor were mocked and asserted uncalled, so a
guard regression could not clear a graph by accident; it would instead fail the
probe assertions.

Quantitative result:

- resolved active graph: `codex`
- deliberately named target: `refusal-probe-not-codex`
- command exit status: **1**
- message: `Error: Refusing to clear graph 'refusal-probe-not-codex': active graph is 'codex'. Switch to 'refusal-probe-not-codex' before retrying.`
- service-status calls: **0**
- graph-client calls: **0**
- probe harness exit status: **0**
- log: `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T215001801849-n-crs-can-the-rehearsal-be-aimed-today/refusal-probe.log`

This proves the command-boundary refusal fires; it is not only dead source or a
help-text promise.

## Backup and service position

The newest **verified full recovery archive** remains:

- path: `/home/ITER/mcintos/.local/share/imas-codex/backups/imas-codex-graph-dev-6fc745c-20260901T125109Z.tar.gz`
- size: **2,458,442,456 bytes**
- mtime: **2026-09-01 14:52:53.778724 CEST**
- previously verified SHA-256: `e13a90541db6628ffb6cee1142e51368a5e586c44b1cf5da91910f7f7184a71b`

The newest live database file is
`/home/ITER/mcintos/.local/share/imas-codex/.neo4j/codex/data/transactions/neo4j/checkpoint.0`,
size **972,672 bytes**, mtime **2026-09-04 18:24:46.652301 CEST**. It is
271,912.874 seconds newer than the verified full archive. The archive is
therefore not current under the audit's stopped-snapshot standard.

There is a newer filesystem entry in `backups`,
`offsite-trial-dev-c987057-20260902T101502Z.dump`, but it is only **4,748
bytes**, mtime **2026-09-02 12:15:06.473524 CEST**. It is an offsite trial
artifact, not a verified full recovery archive for the 15 GB-class live graph,
and does not satisfy the go-condition. The backup inventory contains these two
non-empty files and no stopped-live-tree digest.

A current full dump is **mechanically producible today, but has not been
produced**. The read-only `graph status` check at 23:55 CEST reported:

- active graph `codex`, active link resolving to
  `/home/ITER/mcintos/.local/share/imas-codex/.neo4j/codex`;
- Neo4j running at location `titan` under SLURM job `1262259` on
  `98dci4-gpu-0002`; and
- Bolt 7687 and HTTP 7474 responding.

A full `graph export` enters `Neo4jOperation(require_stopped=True)` before the
dump (`imas_codex/cli/graph/data.py:603-623`), so producing the required
stopped-snapshot archive would stop the current service and cancel its running
SLURM allocation. With `--no-restart`, it deliberately remains stopped while
the live-tree digest is captured. Restoring service then depends on
`graph start` obtaining a SLURM allocation and the HTTP/Bolt listeners becoming
ready (`imas_codex/cli/graph/server.py:43-83` and `:96-142`). The existing
allocation cannot be assumed to survive the stop. The prior export's restart
submission, job `1260741`, remained queued for scheduler reason `Priority` and
port 7687 refused connections; current job `1262259` being RUNNING proves
service availability now, not availability of a replacement allocation after
it is cancelled.

## Verification

Baseline at `abd5be253452db300936d229a96eaaa02a9c36d8`:

`UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH=$PWD uv run --no-sync pytest -p no:cacheprovider tests/cli/test_graph_destructive_target.py tests/cli/test_graph_destructive_backup.py`

Result: **7 passed, 0 failed, 1 warning**, exit **0**. Log:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T215001801849-n-crs-can-the-rehearsal-be-aimed-today/baseline-graph-safety-tests.log`.

After the report was written, the same command completed with **7 passed, 0
failed, 1 warning**, exit **0**. Log:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260904T215001801849-n-crs-can-the-rehearsal-be-aimed-today/after-graph-safety-tests.log`.
There were **0 added failures** and therefore no failure attribution. This node
changes documentation only; merged-head verification remains the separately
dispatched test node's responsibility.
