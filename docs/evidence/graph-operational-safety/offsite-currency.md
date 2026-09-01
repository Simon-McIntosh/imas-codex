# Offsite graph currency

Measured 2026-09-01 against the exact registry destination recorded in the
local graph manifest:
`ghcr.io/simon-mcintosh/imas-codex-graph-tcv`.

## Result

The newest package version actually present in that registry package is
`5.3.0rc6.dev91-g2479801d7.d20260706-r1`. GitHub Packages reports version ID
`1008212308`, created and last updated at `2026-07-07T12:57:57Z`. The OCI
manifest is fetchable, carries the same version annotation, and references a
2,810,090,518-byte layer with digest
`sha256:0cb92406fe43953249922b2f32abdf6daa38eb365133a397d105930545208f41`.
This proves that the tag resolves to an artifact rather than existing only in a
local record.

The newest local restore point under
`/home/ITER/mcintos/.local/share/imas-codex/backups` is:

| Field | Value |
|---|---|
| Path | `/home/ITER/mcintos/.local/share/imas-codex/backups/imas-codex-graph-dev-6fc745c-20260901T125109Z.tar.gz` |
| Size | 2,458,442,456 bytes |
| Modification time | `2026-09-01T14:52:53.778723686+02:00` |
| Modification time, UTC | `2026-09-01T12:52:53.778723686Z` |

Measured from the registry package's actual creation time to that local
restore point, the offsite exposure window is **4,838,096.778723 seconds**, or
**55.996490494 days** (`55d 23:54:56.778723`). A loss of the shared filesystem
at the time of the newest local restore point would therefore fall back nearly
56 days in time to the registry copy.

This is a currency measure, not a content-equivalence claim. The confirmed
offsite package is explicitly TCV-scoped (`imas-codex-graph-tcv`, described as
“TCV + DD read-access share”), while the newest local archive uses the full
graph package name. The registry artifact is therefore not evidence by itself
that every part of the local archive exists offsite.

## Manifest verdict

**AGREES.** The local manifest claims
`pushed_version=5.3.0rc6.dev91-g2479801d7.d20260706-r1` and
`pushed_to=ghcr.io/simon-mcintosh/imas-codex-graph-tcv:5.3.0rc6.dev91-g2479801d7.d20260706-r1`.
The registry returns exactly that one tag for exactly that package, and the
GitHub Packages version record reports the same `2026-07-07T12:57:57Z` time as
the local manifest's `pushed_at` value. The independently fetched OCI manifest
also repeats the same version.

The OCI publisher annotation says `2026-07-07T12:56:16Z`, 101 seconds before
the GitHub Packages version creation time. The exposure calculation uses the
registry-owned package `created_at` value, not the client-supplied OCI
annotation.

## Command evidence

All measurement commands completed successfully. Commands that inspect source
to identify the canonical read-only interfaces are not measurement inputs and
are not reproduced here.

1. Enumerate every candidate restore point and its modification time:

   ```text
   fd -H -t f . /home/ITER/mcintos/.local/share/imas-codex/backups -x stat --printf='%y\t%s\t%n\n'
   exit: 0
   output: 2026-09-01 14:52:53.778723686 +0200  2458442456  /home/ITER/mcintos/.local/share/imas-codex/backups/imas-codex-graph-dev-6fc745c-20260901T125109Z.tar.gz
   ```

2. Read the local graph manifest and the selected restore point's filesystem
   metadata:

   ```text
   stat --printf='%y\t%s\t%n\n' /home/ITER/mcintos/.config/imas-codex/graph-manifest.json /home/ITER/mcintos/.local/share/imas-codex/backups/imas-codex-graph-dev-6fc745c-20260901T125109Z.tar.gz; sed -n '1,220p' /home/ITER/mcintos/.config/imas-codex/graph-manifest.json; git remote -v
   exit: 0
   relevant output: pushed_version=5.3.0rc6.dev91-g2479801d7.d20260706-r1; pushed_to=ghcr.io/simon-mcintosh/imas-codex-graph-tcv:5.3.0rc6.dev91-g2479801d7.d20260706-r1; pushed_at=2026-07-07T12:57:57.405557+00:00
   ```

3. Query the exact registry package through the project's read-only graph CLI:

   ```text
   UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync imas-codex graph tags --facility tcv --registry ghcr.io/simon-mcintosh
   exit: 0
   output: one tag — 5.3.0rc6.dev91-g2479801d7.d20260706-r1 — TCV + DD read-access share
   ```

4. Query GitHub Packages for the registry-owned package-version timestamp:

   ```text
   gh api --paginate /users/simon-mcintosh/packages/container/imas-codex-graph-tcv/versions --jq '.[] | {id, created_at, updated_at, tags: .metadata.container.tags}'
   exit: 0
   output: {"created_at":"2026-07-07T12:57:57Z","id":1008212308,"tags":["5.3.0rc6.dev91-g2479801d7.d20260706-r1"],"updated_at":"2026-07-07T12:57:57Z"}
   ```

5. Fetch the tagged OCI manifest independently:

   ```text
   oras manifest fetch ghcr.io/simon-mcintosh/imas-codex-graph-tcv:5.3.0rc6.dev91-g2479801d7.d20260706-r1
   exit: 0
   relevant output: version=5.3.0rc6.dev91-g2479801d7.d20260706-r1; created=2026-07-07T12:56:16Z; layer size=2810090518; layer digest=sha256:0cb92406fe43953249922b2f32abdf6daa38eb365133a397d105930545208f41
   ```

6. Compute the exposure interval from the two UTC timestamps:

   ```text
   UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync python -c 'from datetime import datetime; registry=datetime.fromisoformat("2026-07-07T12:57:57+00:00"); local=datetime.fromisoformat("2026-09-01T12:52:53.778723+00:00"); delta=local-registry; print(f"seconds={delta.total_seconds():.6f}"); print(f"days={delta.total_seconds()/86400:.9f}"); print(f"duration={delta.days}d {delta.seconds//3600:02d}:{(delta.seconds%3600)//60:02d}:{delta.seconds%60:02d}.{delta.microseconds:06d}")'
   exit: 0
   output: seconds=4838096.778723; days=55.996490494; duration=55d 23:54:56.778723
   ```
