# Shared-infrastructure coupling in imas-codex

Where imas-codex reimplements or binds tightly to infrastructure a shared
layer already owns, instead of calling a thin interface. Five surfaces, one
heading each. Every finding names the imas-codex site, the shared facility it
duplicates or binds to, the concrete failure the coupling causes or risks, and
a one-sentence refactor to a thin interface.

This is a survey of the code as it stands at the dispatch base; it changes no
behaviour. Line numbers are from
`69a2fb2df0f6fbaf158f327ca5d54a690254c5b8`.

![Every coupling site sits below the thin-interface boundary, reimplementing a capability the shared facility already owns](/imas-codex/figures/sn-catalog-audit-instrument/shared-infrastructure-coupling.svg)

| Surface | Sites | Coupling class |
|---|---|---|
| reckon crew and plan layer | 2 | records a transient worktree root; reaches plan state through a foreign static server |
| local lane and router | 2 | re-derives lane selection and admission locally; hardcodes endpoint constants |
| fleet and SLURM placement | 3 | submits and cancels jobs outside the placement ledger; re-probes the fleet; per-node tool cache |
| imas-python data access | 2 | opens IDS without a pinned DD version; duplicates the shared IMAS pattern table |
| GPFS paths or state files | 3 | hand-rolled lock on GPFS; absolute machine paths in generated artifacts and path rules |

## 1. The reckon crew and plan layer

### 1.1 A committed review artifact records an absolute reckon worktree path

- **Site:** `imas_codex/standard_names/catalog_release.py:2068`, `:2209`,
  `:2219` — `minted_from=str(focus_file)` — written into the frozen batch
  record at `:1615` (`"minted_from": minted_from`).
- **Shared facility:** the reckon worktree, which is transient by contract (a
  worker's tree is removed at session end) and the crew/plan layer, which is
  the authority on durable, repo-relative paths.
- **Observed:** twelve committed manifests carry the captured root, e.g.
  `imas_codex/standard_names/manifests/reviews/v0.4.0rc7+west-task-2e.sn_names.yaml:6`
  → `/home/ITER/mcintos/Code/.reckon-worktrees/imas-codex-c994bf55fb01/ship-s10-20260908/n-swcr-cut-the-candidate-now-the-guard-reads-it-right/imas_codex/standard_names/manifests/west_production_dd_paths.yaml`
  (`rg -l 'minted_from:.*\.reckon-worktrees'` → 12 files).
- **Failure:** the artifact's own docstring calls it "the reproducible batch
  identity carried through export → PR → merge", but its source pointer
  resolves to a directory that no longer exists once the worktree is cleaned,
  and it embeds a session-scoped worktree id (`ship-s10-20260908`) that is
  meaningless in any other checkout. The record is reproducible only on the
  machine and session that minted it.
- **Refactor:** record the manifest's repo-relative path together with the
  commit sha at mint time, and resolve the source through the crew/plan
  interface rather than a captured filesystem root.

### 1.2 Plan state is reached through a foreign static server's port and path

- **Site:** `imas_codex/settings.py:737-760` — `get_docs_server_location()`
  documents `~/docs-server/serve.py` as "a host-wide static HTML/JSON state
  server that fronts every project's `docs/plans-html/` under one port and
  exposes `/state/...` endpoints", with `location` defaulting to `"iter"` and
  port defaulting to `8765`; `imas_codex/cli/tunnel.py:242` tunnels it.
- **Shared facility:** the reckon plan/state layer, which already exposes plan
  state to agents (MCP `read_plan` / status surface) as the interface of
  record. imas-codex instead binds to a host-local static server it does not
  own, behind a port constant and a facility-name default.
- **Failure:** on any host without `docs-server`, plan state reads as empty
  rather than as unavailable; the human browser path and the agent filesystem
  path share one port whose disagreement the code cannot detect; and the
  default `"iter"` silently assumes one facility for a repo that configures
  facilities by name.
- **Refactor:** read plan status through the plan interface (reckon MCP /
  plan-state API) and treat the docs-server as an optional rendering surface,
  not the state authority.

## 2. The local lane and router

### 2.1 Lane selection and admission are re-derived locally

- **Site:** `imas_codex/discovery/base/llm.py:1956-2007`
  (`_build_completion_kwargs`) chooses between a direct endpoint, the LiteLLM
  proxy, and OpenRouter-direct by reading `LITELLM_PROXY_URL`,
  `OPENROUTER_API_KEY_IMAS_CODEX` and `_supports_cache_control(model)`; the
  direct-endpoint branch (`:1956`) logs "proxy/OpenRouter bypassed". The retry
  and backoff loop lives in the same module at `:2129-2239`
  (`max_retries`, `_is_retryable`).
- **Shared facility:** the local lane's router, which owns lane choice and the
  request admission gate.
- **Failure:** the direct-endpoint branch bypasses the router entirely, so N
  concurrent workers each decide for themselves that the lane has room and
  over-subscribe it — the degradation signature measured on this workstation
  (no error record, stalls at a fixed ceiling). The retry loop then multiplies
  load against an already-saturated lane, which is exactly the traffic the
  admission gate exists to hold back. Lane policy is re-derived from two env
  vars rather than asked of the router, so a lane's own budget is never
  consulted.
- **Refactor:** route every call through one thin `route(model, service)` that
  returns an endpoint and a permit, so lane choice and admission stay with the
  router.

### 2.2 Endpoint and service-job constants stand in for the live lane topology

- **Site:** `imas_codex/settings.py:704-731` (`LLM_BASE_PORT = 18400`,
  `VLLM_PORT = 18800`, `get_vllm_port()`), and
  `imas_codex/cli/services.py:41-42` (`_NEO4J_JOB = "codex-neo4j"`,
  `_EMBED_JOB = "codex-embed"`). Node discovery for the embedding endpoint is
  re-implemented at `imas_codex/cli/services.py:260-331` and
  `imas_codex/embeddings/readiness.py:39`.
- **Shared facility:** the lane/router's published endpoint document (the
  `lane.json` the dispatch guidance reads for `running`, `concurrent_requests`
  and `waiting`), which is the live topology.
- **Failure:** a module constant is a snapshot of a topology that the router
  changes at runtime, so a client can bind to a port or job name that no
  longer serves (the tunnel/discovery failures recorded in this workspace) with
  no signal that the constant is stale. The failure reads as a connection
  fault rather than as a stale constant.
- **Refactor:** resolve endpoints from the published lane document at call
  time; keep constants only as overridable fallbacks.

## 3. Fleet and SLURM placement

### 3.1 Jobs are submitted and cancelled outside the placement ledger

- **Site:** `imas_codex/cli/compute.py:251-289` (`srun` via `os.execvp`),
  `:427-447` (`#SBATCH` script + `sbatch`), `:509-519` (`scancel`);
  `imas_codex/cli/services.py:565,593` (`#SBATCH --partition=` + `sbatch`),
  `:620,663,703` (`scancel`), `:434` (`squeue -n <job> -u $USER`).
- **Shared facility:** the fleet/placement layer that tracks runs (the crew
  ledger) and the site's SLURM launch templates.
- **Failure:** imas-codex owns submission and cancellation outside the layer
  that records runs, so a service it cancels or relaunches is invisible to the
  fleet's accounting — a job can vanish from under a peer with nothing in the
  ledger to recover from. The `scancel`/`squeue` here select by name or user
  rather than by an enumerated id, the destructive-selection class the shared
  rules ban; and a stale job-name constant (§2.2) makes `squeue -n` match the
  wrong job or nothing.
- **Refactor:** submit through a thin `submit(kind, resources)` /
  `cancel(job_id)` placement interface that owns the ledger, and cancel only
  by an id the caller enumerated.

### 3.2 The fleet is re-probed by SSH rather than read from its dashboard

- **Site:** `imas_codex/cli/host.py:357-487`:
  `_discover_login_nodes_direct` (SSH), `_parse_etc_hosts`, `_discover_via_gateway`
  (parallel `ProxyJump` probes), `_query_node`, `_gather_survey_data`;
  process and load survey at `:147-241`.
- **Shared facility:** the host dashboard that already covers every agent
  process on this workstation, and the perspective that sees every login node.
- **Failure:** the client keeps its own node list, discovered by probing known
  hosts and `/etc/hosts`, so a node missing from that list reads as inactive
  while the dashboard sees it — the display-vs-truth divergence the workspace
  has already paid for twice. Two survey implementations disagree about which
  nodes exist without either being able to detect the other.
- **Refactor:** read process and node state from the fleet dashboard, and keep
  the SSH probe only as a fallback for a host the dashboard does not cover.

### 3.3 A per-node tool cache is managed outside provisioning

- **Site:** `imas_codex/remote/ssh_worker.py:539-546` — `_cache_remote_tools`
  copies `rg`/`fd`/`tokei` from `$HOME/bin` to `/tmp/imas-codex-tools/` on each
  remote host; `:199-201` prepends that path to `PATH`.
- **Shared facility:** the fleet's node provisioning, which owns what tools
  exist on a node.
- **Failure:** the cache is a per-node, untracked artifact keyed to an
  ephemeral `/tmp`; it is invisible to provisioning, can serve a stale binary
  after `$HOME/bin` is updated (the copy is skipped when the destination
  exists), and on a shared login node adds a directory outside every policy
  that governs node state.
- **Refactor:** resolve tools through the fleet's provisioned PATH, and drop
  the self-managed `/tmp` copy.

## 4. imas-python data access

### 4.1 IDS is opened with a literal mode and no pinned DD version

- **Site:** `imas_codex/ids/assembler.py:433-443` —
  `uri = f"imas:{backend}?path={output_path}"; entry = imas.DBEntry(uri, "x")`
  then `entry.put(ids)`. The `"x"` is a literal create mode and no
  `dd_version` is passed.
- **Shared facility:** imas-python (`imas.DBEntry` opened with the DD version
  the data is written in) and the repo's configured `get_dd_version()`
  (`imas_codex/settings.py:1112`).
- **Failure:** the file is written in whatever Data Dictionary the installed
  imas-python defaults to, which need not equal the DD version the graph
  resolves against. A later reader that opens with the configured version then
  resolves paths under a different dictionary than the writer used, and the
  mismatch is silent — the binding rule in the shared IMAS data-access
  reference is "open with the DD version the data was written in".
- **Refactor:** open and write through one thin
  `open_ids(path, backend, dd_version=get_dd_version())` helper shared by every
  IMAS site.

### 4.2 The remote classifier redefines the shared IMAS pattern table

- **Site:** `imas_codex/remote/scripts/enrich_directories.py:73-95` —
  `DEFAULT_PATTERN_CATEGORIES` re-declares `mdsplus`, `hdf5`, `imas`, `cocos`,
  `sign_convention`, `unit_conversion` regexes. The shared table already
  exists at `imas_codex/discovery/base/imas_patterns.py` and is imported by
  `imas_codex/ingestion/extractors/ids.py:7` and
  `imas_codex/discovery/wiki/entity_extraction.py:25`.
- **Shared facility:** `imas_codex.discovery.base.imas_patterns`, the one
  source of truth for these patterns.
- **Failure:** the remote script's copy drifts from the shared module, so a
  source classified as IMAS (or hdf5) by one and not the other — the
  classifier-parity class the workspace has already measured. The duplication
  also means a fix to the shared pattern set does not reach the remote scan.
- **Refactor:** import the shared pattern module in the remote script instead
  of redefining the categories.

**No h5py-on-IMAS-data access found.** `rg 'h5py\.'` finds no use of h5py to
read IMAS data; every `h5py` hit (`enrich_directories.py:79`,
`discovery/paths/enrichment.py:78`, `graph/models.py` documentation) is a
detection regex or a docstring example. The binding rule (imas-python, never
h5py, on IMAS data) holds today; the risk is only that the duplicated pattern
table above keeps the two classifiers apart.

## 5. GPFS paths or state files

### 5.1 A hand-rolled lock on GPFS with an alarm around a known-hanging call

- **Site:** `imas_codex/graph/neo4j_ops.py:36` (`NEO4J_LOCK_FILE` at
  `~/.config/imas-codex/neo4j-operation.lock`), `:138-166` (write, stale-check,
  manual `rm` recovery hint), and `:828-856` (`check_database_lock`) whose own
  docstring says "On GPFS, POSIX locks survive across nodes, so a lock may
  appear held even after Neo4j on the compute node has exited" and
  "`fcntl.lockf` with `LOCK_NB` can hang indefinitely on stale cross-node
  locks despite being non-blocking", worked around with a 5-second
  `signal.alarm`.
- **Shared facility:** the shared claim/lock layer (crew claims) and Neo4j's
  own `database_lock`, plus the liveness probe that is the real authority.
- **Failure:** on GPFS the lock's state and the database's state can disagree
  in both directions — a stale cross-node POSIX lock reads as held while the
  database is up, and the recovery path is a documented manual `rm` of a file
  whose deletion is itself a blind step. The `signal.alarm` timeout converts a
  hang into an exception, which the caller can read either way; a guard that
  cannot distinguish "held" from "unreadable" is the fail-open shape the
  workspace's "absence is never permission" rule targets.
- **Refactor:** take the claim through the shared claim interface and decide
  liveness from a probe that only the healthy database can answer (a real
  query), never from the lock file's presence.

### 5.2 Absolute machine paths are baked into generated artifacts and path rules

- **Site:** `imas_codex/config/models.py:93` and `:967`,
  `imas_codex/graph/models.py:91`, `imas_codex/graph/dd_models.py:88` — a
  generated model carries `'source_file': '/home/ITER/mcintos/Code/imas-codex/...'`;
  `imas_codex/discovery/paths/users.py:41-42` hardcodes the rule
  `r"^/home/ITER/([a-z0-9_-]+)(?:/|$)"` to extract a username from a home path.
- **Shared facility:** the repo-relative path convention and the fleet's
  path-site configuration (username extraction is a per-site rule, not a
  constant).
- **Failure:** the generated files are gitignored and rebuilt per checkout, so
  the baked `source_file` is correct only for the checkout that generated it
  and wrong for every other clone and worktree — a value that reads as
  provenance but is a snapshot of one machine. The `/home/ITER/` username rule
  silently fails to extract a username at any other site, and the resulting
  `None` is the fail-open shape again.
- **Refactor:** emit repo-relative paths in generated provenance, and make the
  home-path rule a configured per-site pattern rather than a constant.

**Cross-reference:** §1.1 records an absolute `.reckon-worktrees/...` path in a
committed manifest — the same class of defect as the baked `source_file`
above, reached from a different layer.

## What the survey does not establish

- It names coupling sites and their failure modes; it changes no code and runs
  no gate. The refactors are one-sentence directions, not landed repairs.
- Each site is a *coupling*, not necessarily a present bug: several are latent
  until the shared layer moves (a stale port, a removed worktree). The two
  with a directly observable artifact are §1.1 (12 committed manifests) and
  §4.1 (an unpinned write), both verified by the commands quoted above.
- The census is grep-derived over `imas_codex/`, `docs/` and `agents/`; a
  coupling expressed only in a YAML, a shell template or a docs tool was not
  swept.