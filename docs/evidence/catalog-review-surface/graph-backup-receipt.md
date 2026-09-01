# Live codex graph backup receipt

Backup operation started on 2026-09-01 from the active live `codex` graph. The
active selector resolved as follows before export:

`/home/ITER/mcintos/.local/share/imas-codex/neo4j` -> `/home/ITER/mcintos/.local/share/imas-codex/.neo4j/codex`

## Recovery archive

- Absolute path: `/home/ITER/mcintos/.local/share/imas-codex/backups/codex-current-20260901T125005Z.tar.gz`
- Exact byte size: `2,458,442,456`
- Modification time: `2026-09-01 14:52:53.778723686 +0200` (epoch `1788267173`)
- SHA-256: `e13a90541db6628ffb6cee1142e51368a5e586c44b1cf5da91910f7f7184a71b`
- Export command: `uv run imas-codex graph export --no-restart --output /home/ITER/mcintos/.local/share/imas-codex/backups/codex-current-20260901T125005Z.tar.gz` (exit `0`)
- Compression check: `gzip -t /home/ITER/mcintos/.local/share/imas-codex/backups/codex-current-20260901T125005Z.tar.gz` (exit `0`)
- Recovery-content check: `tar -tzf /home/ITER/mcintos/.local/share/imas-codex/backups/codex-current-20260901T125005Z.tar.gz` (exit `0`), containing `graph.dump` and `manifest.json` under `imas-codex-graph-dev-6fc745c/`.

## Currency proof at the stopped snapshot boundary

Immediately after the `--no-restart` export completed and before any restart
was attempted, the newest file in the stopped live database tree was:

`/home/ITER/mcintos/.local/share/imas-codex/.neo4j/codex/data/databases/system/neostore.relationshipgroupstore.db.id`

Its mtime was epoch `1788267020.1517509830` (2026-09-01
14:50:20.151750983 +0200). The archive mtime was epoch
`1788267173.778723686`, **153.626972703 seconds later**. The archive is
therefore later than the newest live database file in the stopped source
snapshot it was taken from, rather than merely later than the prior 2026-08-27
archive.

## Restart and read-only verification

Restart submission created SLURM job `1260741`. Verification is pending while
that job remains queued with scheduler reason `Priority`.

The active selector still resolves as follows after the restart submission:

`/home/ITER/mcintos/.local/share/imas-codex/neo4j` -> `/home/ITER/mcintos/.local/share/imas-codex/.neo4j/codex`

The receipt is not complete until job `1260741` reaches RUNNING and a read-only
node count succeeds.
