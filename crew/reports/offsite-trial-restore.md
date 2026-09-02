COMPLETE: The exact offsite archive was fetched and loaded into `offsite-trial`; a canonical `GraphClient` census executed on its SLURM node matched all receipt counts exactly; `codex` was restored and queried successfully; the inactive scratch directory was removed with measured space reclamation.

# Offsite trial restore report

## Outcome

Status: **complete**. The registry fetch, archive load, scratch census, production recovery, and scratch removal gates all succeeded.

The push receipt requires:

- nodes: `1,614,780`
- relationships: `4,259,356`
- distinct labels: `70`
- canonical sorted label-count SHA-256: `1e1c11a2642999adb26fd4fe4b8c6eccf2299db0f836ecbb6af79a1ffa7b14d6` (required prefix `1e1c11a2`)

Observed from the restored scratch graph on `98dci4-gpu-0002` through `run_python_script`:

- nodes: `1,614,780` — exact match
- relationships: `4,259,356` — exact match
- distinct labels: `70` — all 70 receipt labels compared
- label differences: `0`
- canonical sorted label-count SHA-256: `1e1c11a2642999adb26fd4fe4b8c6eccf2299db0f836ecbb6af79a1ffa7b14d6` — exact match

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
| resumed `imas-codex graph status` | 0 | 13.81 s | `codex` remained active and serving before the authorized continuation. |
| `du -sb <scratch-dir>` | 0 | 0.41 s | Scratch apparent size was 9,998,934,641 bytes. |
| `du -s -B1 <scratch-dir>` | 0 | 0.42 s | Scratch allocated size was 9,999,472,640 bytes. |
| `imas-codex graph stop` while `codex` active | 0 | 7.32 s | Stopped production cleanly before reactivating scratch. |
| `imas-codex graph switch offsite-trial` | 0 | 7.48 s | Selected the preserved loaded scratch instance. |
| `imas-codex graph start` for scratch | 1 | 40.77 s | Submitted job `1261206`; the login-node readiness probe could not reach the loopback-only listener, while the job remained running. |
| repository `_get_neo4j_job()` query | 0 | 7.69 s | Derived job `1261206`, node `98dci4-gpu-0002`, state `RUNNING`, exactly as graph status does. |
| direct-node `run_python_script` census | 1 / remote 255 | 7.73 s | Executor reported site SSH closing direct compute-node connections before script execution. |
| direct-node executor hostname probe using FQDN | 1 / remote 255 | 7.88 s | Confirmed the same direct SSH policy refusal. |
| executor-over-login plus allocation census using system Python | 1 | 8.54 s | Reached the allocation but lacked the Neo4j Python dependency. |
| executor-over-login plus allocation hostname probe | 0 | 8.05 s | Executed on `98dci4-gpu-0002.iter.org`, proving the sanctioned allocation route. |
| executor-over-login census with captured failure detail | 1 | 7.93 s | Confirmed `ModuleNotFoundError: neo4j` from system Python; no graph query executed. |
| `run_python_script` census via overlapping allocation step and shared project Python | 0 | 25.16 s | Compared 70/70 labels with zero differences; exact node, relationship, label, and SHA-256 matches. |
| `imas-codex graph stop` while scratch active | 0 | 8.52 s | Stopped job `1261206` before switching away. |
| `imas-codex graph switch codex` | 0 | 7.53 s | Restored the production active symlink. |
| `imas-codex graph start` for `codex` | 0 | 36.07 s | Submitted job `1261209`; Bolt and HTTP became reachable on `98dci4-gpu-0002`. |
| final `imas-codex graph status` | 0 | 15.90 s | `codex` active, Neo4j running, GraphMeta name `codex`, facilities `tcv, jet, iter`. |
| final pre-delete `GraphClient` GraphMeta query | 0 | 6.86 s | Returned `{"facilities":["tcv","jet","iter"],"name":"codex"}`. |
| resolved-target deletion guards | 0 | 0.10 s | Active target was `.neo4j/codex`; scratch resolved exactly to inactive `.neo4j/offsite-trial` and was not a symlink. |
| `rm -rf -- /home/ITER/mcintos/.local/share/imas-codex/.neo4j/offsite-trial` | 0 | 2.51 s | Removed only the explicit inactive scratch directory. |
| post-delete absence and filesystem-space checks | 0 | 0.10 s | Scratch path absent; filesystem available bytes increased from 388,671,421,284,352 to 388,680,397,094,912. |
| post-delete `imas-codex graph list` | 0 | 7.09 s | Lists only active `codex`, `imas`, and `titan-test`; `offsite-trial` is absent. |
| post-delete `GraphClient` GraphMeta query | 0 | 7.15 s | Production graph still serves the expected identity after scratch deletion. |

The explicitly timed commands across the complete restore drill total **574.01 seconds** of active command wall time, including failed readiness and access attempts but excluding the human approval pause and untimed read-only source inspection. The registry fetch was 97.93 seconds, archive load 41.38 seconds, successful scratch census 25.16 seconds, final production start 36.07 seconds, and scratch deletion 2.51 seconds.

## Recovered access and follow-on

The scratch service log at `/home/ITER/mcintos/.local/share/imas-codex/services/codex-neo4j.log` recorded:

- `Bolt enabled on localhost:7687`
- `HTTP enabled on 127.0.0.1:7474`
- Neo4j `2026.01.4` reached `Started.`

The scratch configuration was generated by `graph init`; no manual config edit was made because the worker's exclusive repository write scope is only this report. Direct compute-node SSH is also closed by site policy. The successful sanctioned route used the repository's `run_python_script` executor to transmit an in-memory Python 3.12 script, with an overlapping step inside the already-owned Neo4j SLURM allocation and `NEO4J_URI=bolt://localhost:7687`. Both executor and allocation carried explicit timeouts.

Follow-on: align new graph-instance listener configuration with the supported SLURM topology. Production `codex` binds Bolt and HTTP on `0.0.0.0`, while `graph init` currently emits `127.0.0.1`; this makes ordinary graph-status readiness and login-node `GraphClient` discovery fail for a new instance even when Neo4j is healthy. This node records the defect but does not alter source or configuration.

## Cleanup and recovery state

- active graph: `codex`
- Neo4j: running under SLURM
- production client query before deletion: succeeded
- production client query after deletion: succeeded
- scratch path removed: `/home/ITER/mcintos/.local/share/imas-codex/.neo4j/offsite-trial`
- scratch apparent bytes removed: `9,998,934,641`
- scratch allocated bytes removed: `9,999,472,640`
- measured filesystem available-space increase: `8,975,810,560` bytes
- fetched archive bytes retained: `2,461,580,657`
- post-delete local graph count: `3` (`codex`, `imas`, `titan-test`)
