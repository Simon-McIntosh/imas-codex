NEEDS-HELP: The registry archive loaded into the scratch graph, but the generated scratch configuration binds Neo4j to compute-node loopback and both allowed client access attempts failed, so the required census comparison and scratch removal gate remain unproven.

tried: Fetched the exact registry tag with `graph fetch`, created `offsite-trial` with `graph init`, loaded it with `graph load`, started its SLURM-owned Neo4j service, then queried first from the login node and once from inside the assigned compute node. The login-node `GraphClient` received connection refused because the scratch listener is bound to `localhost`; direct SSH to the compute node was closed before the command ran. I then stopped the scratch service, switched back to `codex`, started it, and proved a `GraphClient` query succeeds.

options: (1) Authorize a narrowly scoped correction so new graph instances bind Bolt/HTTP on the SLURM node's reachable interface, then rerun the census against the preserved loaded scratch directory. (2) Restore or authorize an approved compute-node execution channel (`srun --jobid` or equivalent) and run the canonical `GraphClient` census with `NEO4J_URI=bolt://localhost:7687`. (3) If neither runtime route is allowed, treat the drill as failed, remove the scratch directory through the graph lifecycle tooling, and schedule a new trial after the graph-init networking defect lands.

leaning: Option 1, because the existing `codex` graph already binds to `0.0.0.0` and is reachable through the normal SLURM discovery path, while the newly generated scratch config explicitly binds both listeners to `127.0.0.1`. Making initialized instances follow the supported serving topology fixes the recovery path rather than adding a one-off access route.

cost-if-wrong: The preserved 9,998,934,641-byte scratch instance may need to be discarded and the 41.38-second load repeated; the 2,461,580,657-byte fetched archive is already local, so a repeated registry download is avoidable unless artifact identity is questioned.

# Offsite trial restore report

## Outcome

Status: **blocked**. The registry fetch and archive load succeeded, and the production graph was returned to a healthy serving state. The required scratch census was not observed, so no equality claim is made and the scratch graph was deliberately not removed.

The push receipt requires:

- nodes: `1,614,780`
- relationships: `4,259,356`
- distinct labels: `70`
- canonical sorted label-count SHA-256: `1e1c11a2642999adb26fd4fe4b8c6eccf2299db0f836ecbb6af79a1ffa7b14d6` (required prefix `1e1c11a2`)

Observed from the restored scratch graph: **absent evidence**. Neither aggregate counts nor the 70-label census were returned by a successful query.

## Registry artifact

- registry reference: `ghcr.io/simon-mcintosh/imas-codex-graph:dev-198ec82-20260902T085827Z-r1`
- fetched path: `/home/ITER/mcintos/.local/share/imas-codex/exports/offsite-trial-dev-198ec82-20260902T085827Z-r1.tar.gz`
- fetched bytes: `2,461,580,657`
- fetched SHA-256: `ecc8dc292eeaa988be828eb2cbba39575fbea55bd68c63fa9126865498d5e2d7`
- registry layer digest reported by `graph fetch`: `sha256:99fb55826b165ceef190595a21e444be4a2b222eb8d6feb6323712d9f7b7b6c8`
- push-receipt source archive bytes: `2,461,580,565`
- byte delta after registry round trip: `+92` bytes; content equivalence therefore must be established from the loaded graph census, not archive byte identity

## Command record

All project commands used the shared root environment with `env -u VIRTUAL_ENV UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" uv run --no-sync`. No project command was piped or redirected. Wall times below are those emitted by `/usr/bin/time`.

| Command | Exit | Wall time | Material result |
|---|---:|---:|---|
| `imas-codex graph list` | 0 | 7.55 s | Baseline active graph was `codex`; `offsite-trial` absent. |
| `imas-codex graph status` | 0 | 14.74 s | Baseline `codex` active; SLURM job running; Neo4j running. |
| `test ! -e <fetched-archive>; test ! -e <scratch-dir>; df -B1 <data-root>` | 0 | 0.10 s | Both targets absent; 388,764,379,643,904 bytes available. |
| `imas-codex graph fetch --version dev-198ec82-20260902T085827Z-r1 --registry ghcr.io/simon-mcintosh --output <fetched-archive>` | 0 | 97.93 s | Exact tag fetched; CLI reported 2,347.5 MiB and the registry layer digest above. |
| `stat --printf=... <fetched-archive>` | 0 | 0.00 s | Fetched path contains 2,461,580,657 bytes. |
| `sha256sum <fetched-archive>` | 0 | 1.71 s | Fetched SHA-256 is `ecc8dc292eeaa988be828eb2cbba39575fbea55bd68c63fa9126865498d5e2d7`. |
| `imas-codex graph stop` while `codex` active | 0 | 7.43 s | Safely stopped the production service before changing the active symlink. |
| `imas-codex graph init offsite-trial` | 1 | 19.88 s | Created the scratch directory and selected it; GraphMeta write raced service readiness and failed to connect to the prior job address. |
| `imas-codex graph list` | 0 | 7.17 s | Confirmed `offsite-trial` exists and is active. |
| `imas-codex graph status` | 0 | 8.35 s | Confirmed active identity `offsite-trial`; service subsequently visible as running. |
| `imas-codex graph load <fetched-archive> offsite-trial` | 0 | 41.38 s | Loaded manifest version `5.3.0rc7.dev2213+g0c37fdae6.d20260901`, commit `198ec82`; load and password reset succeeded. |
| `imas-codex graph status` after load | 0 | 13.35 s | Archive manifest recorded; Neo4j deliberately remained stopped. |
| `imas-codex graph start` for scratch | 1 | 40.63 s | Submitted SLURM job `1261199`; readiness command timed out although the service log later recorded Bolt/HTTP started. |
| `imas-codex graph status` for scratch | 0 | 15.33 s | SLURM job running, but login-node endpoint probe still reported stopped. |
| login-node `GraphClient` full census query | 1 | 6.99 s | `98dci4-gpu-0002:7687` refused the connection. |
| compute-node SSH `GraphClient` census with `NEO4J_URI=bolt://localhost:7687` | 255 | 0.18 s | SSH connection was immediately closed; command did not execute. |
| `imas-codex graph stop` while scratch active | 0 | 7.07 s | Stopped scratch SLURM service before switching away. |
| `imas-codex graph switch codex` | 0 | 7.11 s | Active symlink restored to `codex`. |
| `imas-codex graph start` for `codex` | 1 | 40.18 s | Submitted SLURM job `1261202`; CLI readiness window expired shortly before Neo4j completed startup. |
| `imas-codex graph status` after startup settled | 0 | 16.85 s | `codex` active, SLURM job `1261202` running, Neo4j running, GraphMeta name `codex`, facilities `tcv, jet, iter`. |
| `GraphClient` query for `GraphMeta` | 0 | 7.65 s | Returned `{"facilities":["tcv","jet","iter"],"name":"codex"}`. |
| `du -sb <scratch-dir>` | 0 | 0.59 s | Preserved scratch directory consumes 9,998,934,641 bytes. |
| `readlink -f <active-graph-link>` | 0 | 0.03 s | Resolves to `/home/ITER/mcintos/.local/share/imas-codex/.neo4j/codex`. |
| `test -d <scratch-dir>` | 0 | 0.00 s | Scratch directory remains present by design. |

The explicitly timed fetch-to-safe-baseline commands total 339.19 seconds. This is an attempt duration, not the required successful restore-path elapsed time: the census and deletion gates never completed, and diagnosis between commands is excluded.

## Failure evidence

The scratch service log at `/home/ITER/mcintos/.local/share/imas-codex/services/codex-neo4j.log` recorded:

- `Bolt enabled on localhost:7687`
- `HTTP enabled on 127.0.0.1:7474`
- Neo4j `2026.01.4` reached `Started.`

The preserved scratch configuration was generated by `graph init`; no manual config edit was made because the worker's exclusive repository write scope is only this report. The production graph's subsequent startup log instead recorded Bolt and HTTP on `0.0.0.0`, and the normal `GraphClient` query succeeded.

## Cleanup and recovery state

- active graph: `codex`
- Neo4j: running under SLURM
- production client query: succeeded
- scratch directory: `/home/ITER/mcintos/.local/share/imas-codex/.neo4j/offsite-trial`
- scratch bytes retained: `9,998,934,641`
- fetched archive bytes retained: `2,461,580,657`
- space reclaimed: `0` bytes

Scratch removal is intentionally withheld. Removing it would erase the successfully loaded recovery candidate before a corrected, authorized census route can inspect it. After a successful census comparison, the concrete final action is to stop Neo4j, ensure `codex` remains active, and remove `offsite-trial` through the supported graph lifecycle path, then measure reclaimed bytes.

