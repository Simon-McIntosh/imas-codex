# CODAC UDA loader prototype and design assessment

The isolated reader works for live variable metadata and pulse listing, and all
30 focused fake-client tests pass. A successful live signal read and plot are
**still missing**. On the resumed attempt, one absolute-time envelope request
for the previously listed variable over the last 24 hours failed after 3.042 s
with `ReadError: ERROR udaPlugin: Requested variable does not exist`. Live
requests stopped on that error as instructed. The earlier historical pulse
selector is no longer present on the archive; no pulse selector was used in the
resumed attempt. A metadata entry alone does not establish that the variable is
readable over a particular interval. The prototype is not ready to land under
the added successful-read-and-plot requirement.

No operator endpoint was contacted. No real archive address, variable name,
variable prefix, pulse name, or live data is included here or in the prototype.

The implementation lives entirely in
[`scripts/codac_uda_prototype/`](../../../../scripts/codac_uda_prototype/).
It is an experiment, with no scanner registration, graph mutation, dependency
change, server change, or rename applied.

## Live outcomes and evidence

The requests ran from `98dci4-clu-2001` on 2026-11-14. Each started a fresh
module-loaded reader from the project's Python 3.12 loader. The dependency
module supplied Python 3.13.5, NumPy and `uda_client_reader`; no build or install
was needed. Module load and execution occurred in one shell invocation:
`module load m-uda-client/7.4-iimkl-2025b >&2 && exec python3 reader.py`.
Timings include shell, module load, interpreter startup, connection and JSON
serialization; these are not isolated server latency measurements.

| Request | Outcome | Elapsed [s] | Result JSON [bytes] | Signal samples / returned span | Unit |
|---|---|---:|---:|---|---|
| Variables, one exact known variable | 1 entry; nonempty UDA description; no metadata warning | 2.414 | 169 | not a signal read | present in catalogue |
| Pulses, historical selector | 0 pulses; 743 categories | 2.359 | 25,896 | not a signal read | not applicable |
| Signal, historical pulse, relative 2–4 s, 200 envelope bins | Exact error: `ReadError: Incorrect time or pulse arguments` | 2.093 | no successful result | no array or span returned | not returned |
| Pulse positive control, one server-declared category | 1 pulse; same 743-category catalogue | 2.135 | 25,990 | not a signal read | not applicable |
| Resumed absolute-time read, last 24 hours, 200 envelope bins, previously listed variable | Exact error: `ReadError: ERROR udaPlugin: Requested variable does not exist` | 3.042 | no successful result | no array or span returned | not returned |

The pulse control establishes that the pulse-listing path can return a pulse in
the initial session. It does not establish that the control pulse contains the
requested signal. The lead subsequently confirmed that the historical pulse
selector no longer exists; the resumed attempt used only `tsFormat=absolute`
and integer nanosecond `startTime`/`endTime`, without a pulse argument.

The resumed attempt used **1 of at most 6 live requests**. Its requested span
was 86,400 s, with a target of 200 envelope bins, on the same non-operator
archive role used for the successful variable listing. It returned an error,
not a successful empty interval. Under the instruction to record errors and
stop if reads fail, no 30-day widening, alternative-variable listing, short
follow-up read, or replacement-host request was made. Five request slots remain
unused because the error stop condition fired. No sample count, actual data
span, unit, min, max, mean or cadence can be measured from a failed read.

Private receipts are under `~/.cache/imas-codex-codac-uda/`:
`live-variables.json`, `live-pulses.json`, `live-read.json`,
`live-summary.json`, `pulse-control.json`, and the resumed
`absolute-live-summary.json`. The last receipt records the exact absolute
bounds, 200-bin request, completed error and request count. These receipts carry
actual selectors and server-returned metadata and must remain internal. There
is no live plot or successful live signal-data file. `fake-signal.svg` and
`fake-signal-preview.png` contain invented test data only; that earlier preview
was rendered and visually inspected. No synthetic plot is counted as live
validation.

The remaining evidence needs a currently readable variable/archive binding and
a populated absolute interval. A subsequent authorized attempt can establish
that binding with a bounded listing on the intended archive role and then
exercise the short-read and plot path. This one refusal does not establish that
all variables or the whole archive are unavailable.

The focused gate is
`uv run --no-sync pytest -p no:cacheprovider scripts/codac_uda_prototype/test_reader.py`,
with the shared project environment and this worktree on `PYTHONPATH`.
`~/.cache/imas-codex-codac-uda/prototype-tests.log` contains the full command,
revision and tree header, `30 passed, 1 warning in 1.30s`, and final `EXIT=0`.
The warning concerns the existing `cache_dir` pytest setting when its cache
plugin is disabled; it does not skip a test.

The mctb research document's paging defect was reproduced with a fake client:
`ddContinueFrom=0,ddSamples=4` without the explicit raw flag returns timestamps
`[0, 2, 4, 6]`; the fake `isDecimated()` returns false, and the cadence guard
refuses it with `raw sample spacing differs from expected cadence`. In a scratch
copy, removing `decSamples=-1` from the prototype's paging request makes
`test_raw_pages_retain_explicit_no_decimation` fail with that refusal, ending
`1 failed` and `EXIT=1`. This is a deterministic reproduction of the documented
mechanism, not a new measurement of silent decimation on the live archive.
The refusal log is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20261114T222330036277-codac-uda-loader-prototype/negative-control.log`.

## 1. Placement and reuse

Implement a `codac_uda` scanner beside the existing scanners in
[`imas_codex/discovery/signals/scanners/`](../../../../imas_codex/discovery/signals/scanners/).
It should implement `DataSourceScanner.scan()` and `.check()` from
[`base.py`](../../../../imas_codex/discovery/signals/scanners/base.py), return
`ScanResult` with `FacilitySignal` and `DataAccess` objects, and register through
`register_scanner()` and `_auto_register()`. `DataSourceScanner` is a protocol,
not a superclass to inherit implementation from. Catalogue discovery belongs
here; array reading should be a reusable data-access service used by both
scanner checks and the MCP tool.

[`jpf.py`](../../../../imas_codex/discovery/signals/scanners/jpf.py) is the useful
precedent: batch enumeration in a facility environment, a `DataAccess` record,
and translation into shared signal discovery and enrichment. Reuse that flow,
the facility loader/private overlay, and
[`remote/environment.py`](../../../../imas_codex/remote/environment.py).
Put the standalone installed reader in `imas_codex/remote/scripts/` and execute
it through the shared executor after addressing its local setup behavior.
Do not duplicate SSH management, facility resolution, graph writes, embeddings,
signal enrichment, unit normalization, cost accounting or model routing.

The project has no general CODAC array-loading service today. I would add a
small data-access module owning the typed request/result contract and backend
dispatch; the CODAC backend uses the shared remote executor. It must not execute
arbitrary graph-stored Python templates or accept a host from an MCP caller.
Resolve a graph signal identity to its accessor and configured archive role,
then accept only a pulse or time window, sampling mode and explicit bounds.

MCP registration belongs in `AgentsServer._register_tools()` in
[`llm/server.py`](../../../../imas_codex/llm/server.py), next to existing facility
search tools. Add a facility-only read tool and hide it in DD-only mode. A
restricted read-only deployment still needs an explicit facility-network policy:
"read-only" does not itself authorize access to plant data. Return small
metadata, statistics and an internal artifact reference for larger arrays, with
an inline-array option only under a strict byte cap. Preserve the existing
bounded MCP-output behavior.

[`reader.py`](../../../../scripts/codac_uda_prototype/reader.py) reads all
endpoints from `CODAC_UDA_HOST`, `CODAC_UDA_PORT` and mandatory JSON
`CODAC_UDA_OPERATOR_HOSTS`. Operator refusal normalizes case, short DNS label
and trailing dot before constructing the client; tests prove that construction
never happens on refusal. Production should resolve permitted archive roles
from `iter_private.yaml`, maintain an approved endpoint allowlist as well as
operator exclusions, and test aliases explicitly. It should never fall back to
the plotting client's default archive.

Reuse mctb's query-keyed cache concept, but key a production cache on archive
identity, exact variable/field, absolute bounds or resolved pulse bounds,
sampling settings, reader protocol and metadata provenance. Cache complete
receipts atomically. Keep errors, genuinely empty intervals and incomplete
scans distinct. Do not cache a transient refusal as an empty signal.

## 2. Invoking the MINT module environment

There is no client build problem to solve. The experiment proves that the UDA
dependency module directly provides its Python client. The project remains on
Python 3.12; the subprocess runs the module's Python 3.13. A JSON request/result
contract keeps the project environment independent of the module's native ABI.
The prototype's
[`Loader`](../../../../scripts/codac_uda_prototype/loader.py) accepts a trusted
argument vector or `CODAC_UDA_READER_COMMAND` containing a JSON argument vector.
Its command starts a shell that loads the module and immediately executes the
reader. Neither module loading in a previous call nor adding the client to
`pyproject.toml` is appropriate.

For a **batch scan**, resolve `data_systems.codac_uda.setup_commands` and
`python_command`, load the module once inside one bounded invocation, and
enumerate a configured cohort of variable patterns with resumable batches.
Return explicit counts, source metadata, warnings and completion bounds. A full
catalogue/enrichment population runs on a compute partition; it must not become
one subprocess or one SSH connection per variable.

For an **interactive MCP read**, reuse
`async_run_python_script()` in
[`remote/executor.py`](../../../../imas_codex/remote/executor.py), its cancellation,
subprocess timeout and clean structured stdout. Resolve the facility/data-system
environment centrally, apply setup and execution in the same shell, and keep a
short response deadline plus sample and byte limits. Start with one process per
read; the measured startup-inclusive operations took roughly two seconds. A
persistent module-loaded worker is a later, measured optimization and would
need ownership, cancellation and bounded lifetime rather than an unmanaged
background service.

A concrete integration gap matters: `run_python_script()` and its async peer
apply `setup_commands` only in the SSH branch. Their local branch directly
executes `python_command`. The CODAC route is local on SDCC, so simply passing
the module setup to today's executor silently omits it. Extend the executor's
local setup path once, with focused tests, while keeping plain local invocation
unchanged. The prototype uses a configured local subprocess instead because
existing source is outside its write scope. With a custom interpreter command,
retain site initialization: the executor's generic `python3 -S` optimization
must not hide the module's NumPy or native client packages.

For larger reads, use typed binary arrays plus a small JSON receipt. JSON here
is intentionally small and preserves integer epoch timestamps and dtype names;
it is not an efficient bulk transport. Do not mix module diagnostics into the
receipt. Python stdout redirection in the prototype does not guarantee capture
of every native library write to the OS stdout file descriptor; the loader
fails visibly on an invalid receipt instead of interpreting it as an empty one.

## 3. Port-or-depend assessment dropped

The revised instruction drops the `iter-uda` port/dependency question. It is not
a dependency of this prototype. The active prior art is mctb's
`docs/figures/uda_servers.py`, `htscl-thermal/htscl_data.py`,
`discharge-fit/discharge_data.py`, `uda-access/discharge_records.py`, and
`docs/research/mctb-uda-access.html`. Those sources were read in place and were
not modified. An earlier read of the superseded repository occurred before the
correction arrived; it is not a proposed dependency or source to carry forward.

The implementation uses the mctb client calls, timestamp-type branch, explicit
raw query flag, field selectors, handle release, and documented paging/cadence
rules. Archive metadata lookup and the bounded protocol are the experiment's
additional code. External metadata-service requests are deliberately excluded
from the reader: it refuses a caller-supplied metadata URL. Any future controls
metadata enrichment should resolve its endpoint and field precedence through
private deployment configuration. No study-specific channel lists or private archive defaults
were brought into this repository.

## 4. Schema fit and plant naming

| Concept | Existing fit | Missing or unsuitable today |
|---|---|---|
| Archive catalogue | `DataSource`, `source_type=uda`, facility ownership | `server_hostname` has range `MDSplusServer`; UDA needs a generic typed endpoint/role and private resolution |
| Native variable or record field | `SignalNode` for source structure, `FacilitySignal` for semantics, `data_source_path` and `accessor` for the exact selector | Record-field structure, metadata source per field, native timestamp base and sampling policy need explicit representation |
| Fetch recipe | `DataAccess`: method, library, setup, connection/data/time/cleanup templates, output format | A typed bounded read contract, endpoint role, interpreter policy, request validation and sampling receipt should accompany or replace executable templates for this backend |
| Pulse | No first-class pulse entity | String external identity, archive namespace, absolute start/end in ns, catalogue provenance, and category relation; integer `example_shot` cannot store it |
| Pulse category | No matching first-class class | A separate category entity and pulse membership; it is neither a diagnostic nor a numeric shot |
| Plant component and control function | `signal_role` captures measured/commanded/setpoint/status/configuration semantics; `SignalNode.same_component` links source nodes | No plant breakdown identity, hierarchy or verified source-to-component relation |

These observations come from
[`schemas/facility.yaml`](../../../../imas_codex/schemas/facility.yaml), especially
`DataSource`, `MDSplusServer`, `SignalNode`, `FacilitySignal`, `DataAccess` and
`SignalRole`, and
[`schemas/facility_config.yaml`](../../../../imas_codex/schemas/facility_config.yaml).
Both schemas already enumerate `uda`, but `DataSystemsConfig` has **no UDA
configuration attribute**, and there is no registered UDA scanner. An enum value
alone does not make a source runnable. Add a typed CODAC configuration using
`DataSystemBase` for setup/interpreter settings, private server roles,
catalogue patterns, metadata-service settings, limits and a known-good check
selector. Avoid coercing a CODAC pulse into `get_facility_reference_shot()`:
that helper casts references to `int`.

Decode the variable naming convention during deterministic source discovery,
attached to the native `SignalNode` identity and carried as evidence into
`FacilitySignal` enrichment. Preserve the original selector byte for byte.
Store parsed plant-system/component code, control-function token, parser
provenance and an explicit parse outcome. Use a controlled mapping to a plant
breakdown `PlantComponent` entity, with parent hierarchy and a verified relation
from the signal/source node. A parsed name is a candidate join key, not proof of
physical identity. Unknown tokens remain unknown; they must not receive a
fabricated component or an LLM-invented identifier. `hardware_section` describes
static machine-description sections and should not be overloaded with this
plant hierarchy.

Treat pulse categories independently from plant grouping: an operating campaign
can contain many components and a variable can be read without any pulse.
Preserve absolute epoch nanoseconds alongside explicit relative-to-pulse seconds
at the API boundary. Separate the requested mode from sampling evidence:
`decimated=true` for a server envelope, `false` only when an independent cadence
has been checked, and `null` for an explicit raw request without that evidence.
The client flag is not sufficient. Irregular archived-on-change signals need a
sampling policy that does not pretend to have a fixed cadence.

## 5. Rename inventory and safe order

This is a tracked-tree inventory at base revision
`98b7ee1bb1abff52f64aa7289f0c91f96e63297d`, before the prototype. It covers all
8,099 tracked paths, excludes binary content from text matching, and excludes
private/generated files, installed environments and other repositories' working
trees. Searching `imas-codex|imas_codex|IMAS_CODEX|IMAS Codex|imas-cx` found
6,932 files, 166,585 matching lines and the following literal occurrences:

| Literal | Occurrences |
|---|---:|
| `imas-codex` | 39,426 |
| `imas_codex` | 138,394 |
| `IMAS_CODEX` | 821 |
| `IMAS Codex` | 85 |
| `imas-cx` | 13 |

These are lexical counts, not independent integrations. Historical evidence
contains many repetitions. There are 4,396 tracked paths containing one of the
spellings: 693 in the package directory and 3,703 elsewhere. Renaming all of
those blindly would alter evidence identities and old source-path citations.

| Surface group | Files containing matching text | Matching lines |
|---|---:|---:|
| Root packaging, config, container, README and lock files | 17 | 506 |
| Runtime package `imas_codex/` | 514 | 5,316 |
| Tests | 1,131 | 10,955 |
| Scripts | 48 | 148 |
| Docs, including historical evidence and plans | 5,144 | 149,217 |
| Crew configuration/records | 25 | 125 |
| GitHub automation/instructions | 17 | 181 |
| Shared agent docs | 5 | 55 |
| `.claude/` | 7 | 22 |
| `.cursor/` | 1 | 1 |
| `.vscode/` | 3 | 6 |
| Benchmarks | 9 | 20 |
| Research | 10 | 29 |
| SLURM scripts | 1 | 4 |

The complete per-file, per-line manifest, environment-token counts and matched
state paths are in this run's `rename-inventory.json`; `runtime-inventory.json`
records entry points and scheduler/discovery name tokens. Their absolute home is
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20261114T222330036277-codac-uda-loader-prototype/`.
The inventory script is retained there as `rename-inventory.py`. The scan
completed in 6.529 s and did not query the graph or touch installed services.

The actionable identity bindings are:

| Surface | Count and current binding | Rename implication |
|---|---|---|
| Distribution and import identity | 1 distribution `imas-codex`; 1 root package `imas_codex`, 693 tracked package files | Change build metadata, version lookup, package data, imports, typing/test configuration and generated-model/build-hook paths together |
| CLI entry points | 11 total in `pyproject.toml`; 1 branded entry `imas-codex = imas_codex.cli:main`, 10 capability-named build tools | Introduce `iter-codex`; update the branded target and package imports; capability-named tools can keep their names |
| Python module entry | Package `__main__.py` and root `__main__.py` | Verify `python -m iter_codex`, not only a console script |
| Settings namespace | 1 `[tool.imas-codex]` root with many subsections | Move to `[tool.iter-codex]`; update readers in `settings.py`, package initialization and build hooks atomically |
| Environment variables | 118 distinct lexical `IMAS_CODEX_*` tokens across tracked text; 90 in runtime/scripts/build surfaces | Includes generated prefixes and stale documentation, not 118 independently supported variables; enumerate actual readers before migration, set explicit new-name precedence and reject conflicting old/new values |
| MCP configuration and identity | 2 client configs use the `imas-cx` key; 2 runtime advertised names: `imas-codex`, `imas-codex-readonly` | Update `.mcp.json` and `.vscode/mcp.json` together; proposed short client key is `iter-cx`; decide runtime advertised names deliberately and reconnect clients |
| Container/image and registry naming | Application image family plus graph base, DD and facility-suffixed families; 3 Docker provenance labels `imas_codex.git_*` | Change Dockerfile, Compose service/image/dependencies, workflow tags, ACR name, OCI artifact resolution and graph export names together; retain immutable old digests/tags |
| GitHub repository identities | 2 configured remote repositories: fork and upstream; 4 project URL fields in `pyproject.toml` | Coordinate both repository renames, permissions, redirects, Actions conditions, release automation, registry ownership and badges; no repository was renamed here |
| State/cache/config roots | 3 XDG namespace families: `~/.local/share/imas-codex`, `~/.cache/imas-codex`, `~/.config/imas-codex`, plus named derivative caches | Paths include graph stores/active symlink, exports/backups, services, logs, credentials references, campaigns, checkpoints, staging, extraction results and remote venvs; do not move them under a running consumer |
| Service units and scheduler discovery | 8 concrete branded `.service`/`.timer` spellings in tracked text; 95 source/config lines match `codex-*` service/discovery tokens | Concrete unit references include graph push, Neo4j, LLM, embedding, database and tunnels; templated host-specific names add runtime instances; discovery currently relies on names such as `codex-neo4j`, `codex-embed`, `codex-llm` |
| Installed/remote execution | Shared checkout, editable install, script shebangs, remote venv location, SSH PATH prefixes, systemd `ExecStart`, SLURM scripts and queued run commands | Package migration does not rewrite already-launched interpreters or queued commands; retain their resolved paths until they drain |
| Graph and serialized records | Package import paths, access templates, stored filesystem paths and provenance; `GraphMeta` identifies facilities/IMAS content | Search and migrate actual stored values under an explicit graph migration; never global-replace scientific node IDs or source selectors |
| Documentation and developer tooling | Groups counted above, plus ASV, Makefile, release links, agent settings and Reckon mount/project links | Update current instructions and links; preserve historical evidence bytes and provide redirects/aliases for durable record paths |

Representative source bindings are
[`pyproject.toml`](../../../../pyproject.toml),
[`hatch_build_hooks.py`](../../../../hatch_build_hooks.py),
[`settings.py`](../../../../imas_codex/settings.py),
[`llm/server.py`](../../../../imas_codex/llm/server.py),
[`.mcp.json`](../../../../.mcp.json),
[`.vscode/mcp.json`](../../../../.vscode/mcp.json),
[`Dockerfile`](../../../../Dockerfile),
[`docker-compose.yml`](../../../../docker-compose.yml),
[`docker-build-push.yml`](../../../../.github/workflows/docker-build-push.yml),
[`release.yml`](../../../../.github/workflows/release.yml),
[`graph/registry.py`](../../../../imas_codex/cli/graph/registry.py),
[`graph/dirs.py`](../../../../imas_codex/graph/dirs.py),
[`graph/profiles.py`](../../../../imas_codex/graph/profiles.py),
[`graph/remote.py`](../../../../imas_codex/graph/remote.py),
[`remote/python.py`](../../../../imas_codex/remote/python.py),
[`remote/executor.py`](../../../../imas_codex/remote/executor.py),
and [`cli/logging.py`](../../../../imas_codex/cli/logging.py).
The remote identities are the
[fork](https://github.com/Simon-McIntosh/imas-codex) and
[upstream](https://github.com/iterorganization/imas-codex).

This inventory does **not** assert counts of live jobs, installed units, external
MCP clients, published images or database records. It records the code and
configuration that create or resolve them. Enumerating those installed consumers
and stored paths is required before scheduling a rename; reading this working
tree cannot prove there are no external consumers.

I would make the change in this order:

1. Freeze an inventory of installed consumers, graph templates, queued runs,
   editable installations and path bindings. Record active service job IDs,
   verified backups and registry digests. Lock in the public distribution,
   import, CLI and MCP names independently; a short MCP key need not dictate
   the Python package name.
2. Add the new user-facing name/entry point and an explicit migration resolver
   for settings, environment names and state roots. Keep one writable state
   root and one owner per service. Temporary aliases must resolve to that same
   root; two divergent caches or graph stores are not a migration strategy.
3. In one tested change, rename package/build paths and imports, rebuild models,
   regenerate the lock and entry scripts, and sync the **main checkout's**
   environment. Validate import provenance outside the working directory,
   package resources, CLI, build tools and MCP startup against the installed
   revision. Keep an old command/path compatibility route only until its named
   running/queued consumers drain; this is a migration boundary, not permanent
   backwards compatibility.
4. Publish new application and graph artifacts without deleting old tags.
   Update repository identities, release settings, permissions and public links
   in a coordinated cutover. Verify pulling by the new identity rather than
   merely observing that publication returned successfully.
5. Drain and replace services individually. Update discovery names and their
   producers together, restart MCP clients, and prove each consumer resolves
   the intended installed code. Move persistent state only with its owner
   stopped; keep old path resolution for readers that still hold it. Never
   rename a graph store while Neo4j has it open.
6. Update active docs, developer settings and Reckon registration, then remove
   temporary compatibility routes after an explicit zero-consumer check.
   Keep historical records and immutable artifact identities readable.

Retain `imas` wherever it names the actual science format or ecosystem:
`imas-python`, `imas-data-dictionary`, `imas-data-dictionaries`,
`imas-standard-names`, IMAS IDS paths, `IMASNode`, `IMASMapping`, IMAS schemas,
COCOS handling, the IMAS scanner, `data_systems.imas` and DD-only graph content.
The two Standard Names repositories are separate catalog/package products;
they are not implied rename targets. An image's project prefix may become
`iter-codex`, while its `-dd` or IMAS-content suffix continues to describe its
contents. Do not rename upstream science identifiers, source signal selectors
or standard-name identities to satisfy a product rebrand.

## 6. Disagreements and recommended boundaries

The rename should be a separately measured migration. It is unnecessary for
proving archive discovery and reading, and combining it with the first scanner
would make access defects hard to isolate. The prototype leaves all existing
source, schema and configuration untouched.

A raw request cannot certify its own fidelity merely by spelling
`decSamples=-1`, and a fixed-cadence check is unsuitable for archived-on-change
signals. Keep the requested mode and the sampling evidence separate. This
prototype deliberately refuses to page raw data without an independent cadence;
a production interface needs an explicit irregular-sampling policy as well.

The prototype limits catalogue size, samples, page attempts and subprocess
wall time, but its sample bound is checked **after** a native fetch and its
`subprocess.run()` captures stdout in memory. That is not a server-side memory
limit. Production must bound the requested interval/resolution before fetching,
cap serialized bytes, and use binary storage for larger arrays. Whole-pulse raw
reads should require a declared resource budget, not be the default tool action.

Do not advertise a variable as unavailable because one historical pulse is
missing, and do not infer full activity coverage from one envelope. Start with
catalogue and description search plus explicitly bounded loading. Activity
windows need a separate, resumable query-budgeted scan with positive controls
and a complete/incomplete outcome.

Envelope min/max and unweighted mean-of-bin-means are useful plot summaries,
not a reconstruction of raw sample statistics. The prototype labels them
`unweighted_bin_means`; archived-on-change sample means likewise are not
time-weighted physical averages. Structured records need a field selector and
strings should remain strings. Those distinctions belong in the production
result schema, not only in a plot caption.

The smallest next implementation is a typed CODAC config, the shared executor's
local module-setup support, a scanner returning shared `ScanResult`, and one
facility read service/tool using the same reader. Before calling that complete,
resolve the refused variable/archive binding, validate one current two-second
signal interval and inspect its live plot, then gate modified Cypher
against the graph on a compute partition. No live graph writes were performed
for this prototype.
