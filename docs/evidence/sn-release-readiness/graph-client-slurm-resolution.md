# Graph client address resolution inside a SLURM step

A `GraphClient()` constructed inside a SLURM step resolved a loopback tunnel
endpoint and failed to connect, while the same call on the login node reached
the graph. The address it should use is directly reachable from the compute
node. This records the reproduction, the code path responsible, the change, and
the measurement after it.

## The path that selects the loopback endpoint

| step | location | condition |
|---|---|---|
| locality decision | `imas_codex/graph/profiles.py:300` | `local = host is None or host == "local" or is_local_host(host)` |
| facility host patterns | `imas_codex/config/facilities/iter.yaml:32-36` | `login_nodes` lists `98dci4-srv-*` and `98dci4-gpu-*` only |
| branch taken | `imas_codex/graph/profiles.py:341-386` | `local` is False, so Mode 3 (remote) runs |
| address returned | `imas_codex/graph/profiles.py:386` | `return f"bolt://localhost:{tunnel_port_int}"` |

The condition choosing it: the hostname of a compute node
(`98dci4-clu-3141`) matches none of the facility's `login_nodes` patterns, so
`is_local_host("iter")` returns False. Resolution then takes the remote branch,
attempts to open an SSH tunnel out of the compute node, fails (`ssh: connect to
host iter port 22: Connection refused`), and returns the tunnel port anyway.

The graph itself runs on `98dci4-gpu-0002`, and that bolt port is directly
reachable from the compute step. On the login node `98dci4-srv-1006` the same
code returns `bolt://98dci4-gpu-0002:7687` with no tunnel in the path, because
`srv-1006` matches `98dci4-srv-*`.

## Reproduction, before the change

Log: `/home/ITER/mcintos/gcprobe/step_before.log`

```
srun --partition=all_debug --time=00:05:00 --cpus-per-task=2 --mem=8G bash /home/ITER/mcintos/gcprobe/run_step.sh /home/ITER/mcintos/gcprobe/wt /home/ITER/mcintos/gcprobe/step_before.log
```

Whole output, key lines:

```
host=98dci4-clu-3141.iter.org SLURM_JOB_ID=1273455
resolved profile uri: bolt://localhost:17687
GraphClient.uri: bolt://localhost:17687
neo4j.exceptions.ServiceUnavailable: Couldn't connect to localhost:17687
STEP_EXIT=1
```

## The change

`imas_codex/graph/client.py`. The default `uri` field no longer takes the
profile URI directly; it takes `_resolve_graph_uri()`, which returns the
profile URI unchanged outside a SLURM step and, inside a SLURM step, the
reachable address from `_slurm_service_uri()` when one can be resolved.

`_slurm_service_uri()` returns `None` — leaving the profile URI in place —
when the active location is not SLURM-scheduled or when no service node is
found, so no address is invented. When the service node is this node the
address is `bolt://localhost:<port>`; otherwise it is the node's own address.

## Measurement after the change

Log: `/home/ITER/mcintos/gcprobe/slurm_step.log` (command and full output)

```
srun --partition=all_debug --time=00:05:00 --cpus-per-task=2 --mem=8G bash /home/ITER/mcintos/gcprobe/run_probe_stdout.sh
```

```
host=98dci4-clu-3141.iter.org SLURM_JOB_ID=1273478
resolved profile uri: bolt://localhost:17687
GraphClient.uri: bolt://98dci4-gpu-0002:7687
count(StandardName): [{'c': 5130}]
CAPTURE_EXIT=0
```

The profile layer still reports its loopback URI, unmodified; the client
resolves the reachable address past it. The count is 5130.

## Login-node path, unchanged

Log: `/home/ITER/mcintos/gcprobe/login_node.log` (command and full output),
re-run with the changed code in the same evidence.

```
host=98dci4-srv-1006.iter.org SLURM_JOB_ID=unset
resolved profile uri: bolt://98dci4-gpu-0002:7687
GraphClient.uri: bolt://98dci4-gpu-0002:7687
count(StandardName): [{'c': 5130}]
CAPTURE_EXIT=0
```

Same address and same count as before the change: 5130 on both sides, and the
login-node reading taken through the same script that produced the pre-change
baseline.

## The four defects that left the repair unpinned

The first repair resolved the right address, but four things about it were
wrong or unwatched. Each was measured before it was closed. All readings are
in `/home/ITER/mcintos/gcprobe/s14pinned/`.

### An explicit `NEO4J_URI` lost to the address the repair discovered

`resolve_neo4j` applies `NEO4J_URI` last, documented in
`imas_codex/graph/profiles.py` as "Env var escape hatches (always win)". The
URI the client started from was therefore already the explicit one, and
`_resolve_graph_uri` replaced it with the address it had discovered — the
escape hatch was unreachable exactly inside the step that needed it.

Before, from `unit_before.log`, in-process with a SLURM step simulated:

```
tests/graph/test_client_address_resolution.py:101: in test_explicit_neo4j_uri_wins_inside_a_slurm_step
    assert _resolve_graph_uri() == EXPLICIT_URI
E   AssertionError: assert 'bolt://98dci4-gpu-0002:7687' == 'bolt://example.invalid:1'
```

After, inside a real SLURM step (`step_explicit_uri.log`):

```
host=98dci4-clu-3141.iter.org SLURM_JOB_ID=1273665
NEO4J_URI=bolt://example.invalid:1
profile uri: bolt://example.invalid:1
service-node discovery: bolt://98dci4-gpu-0002:7687
_resolve_graph_uri(): bolt://example.invalid:1
```

The discovery line is the control: the gpu address is still resolvable from
that step, so the explicit URI beats a live candidate rather than an empty
one. The precedence is now read from the environment before the profile or
the discovery runs, and pinned by
`test_explicit_neo4j_uri_wins_inside_a_slurm_step`.

### A failed location read restored the very address being repaired

The location read sat inside `_slurm_service_uri` behind a bare
`except Exception: return None`. `None` leaves the profile URI in place,
which inside a step is the loopback tunnel endpoint this module exists to
replace — so any fault in the read put the broken address back, silently.
The branch had no test at all, which is why nothing saw it.

Reproduction, `test_an_unreadable_location_is_not_swallowed`, before:

```
with pytest.raises(RuntimeError):
E   Failed: DID NOT RAISE <class 'RuntimeError'>
```

The branch is now uncaught and the same call raises. Letting it surface rather
than narrowing what is caught was the call because `resolve_location` has no
legitimate raise for a location it cannot find — an unknown or unconfigured
location answers a `scheduler="none"` `LocationInfo` — so an exception there is
a fault, and the caller's fallback address is the one known not to work.

### The dataclass default was unpinned

Reverting `field(default_factory=_resolve_graph_uri)` to `get_graph_uri` in a
scratch copy of the package (hard-linked, that one line reverted) fails one
test — `revert_run.log`:

```
1 failed, 8 passed

FAILED ...::test_the_client_default_uri_is_the_slurm_aware_resolver
E   assert <function get_graph_uri at 0x7f2227f4a340> is _resolve_graph_uri
```

The failure text names the reverted factory, which is the positive control
that the scratch module was the one loaded. Against the worktree the same 9
tests pass. With the profile URI as the factory the repair is inert for a
default `GraphClient()` — which is how the loopback endpoint reached the step
in the first place, so this is the line the suite exists to pin.

### The inert monkeypatch

The suite patched `client_module._resolve_compute_host`, an attribute the
function never reads: the name is bound inside the function body from
`imas_codex.remote.locations`. Removed as dead. The live target is pinned by
`test_the_fallback_is_read_from_the_module_it_is_imported_from`, which answers
from that module and sees the answer arrive.

## Re-measurement after both repairs

One SLURM step and one login-node reading, each through
`/home/ITER/mcintos/gcprobe/s14pinned/probe.py` with this worktree on
`PYTHONPATH`.

| reading | host | profile URI | client URI | count |
|---|---|---|---|---|
| compute node, default env | `98dci4-clu-3141` | `bolt://localhost:17687` | `bolt://98dci4-gpu-0002:7687` | 5130 |
| compute node, `NEO4J_URI` set | `98dci4-clu-3141` | `bolt://example.invalid:1` | `bolt://example.invalid:1` | not attempted |
| login node, default env | `98dci4-srv-1006` | `bolt://98dci4-gpu-0002:7687` | `bolt://98dci4-gpu-0002:7687` | 5130 |

The repair is intact: the compute step still resolves the gpu node and still
returns 5130, and the login-node reading is unchanged from the baseline above.
The profile layer still reports its loopback URI in the step, unmodified — the
client resolves past it.

## Tests

`tests/graph/test_client_address_resolution.py`, 9 tests, run under the
default marker selection (not `-m graph`), exit 0:

```
9 passed, 1 warning in 0.19s
```

They cover the peer-node branch, the on-node localhost branch, the explicit
`NEO4J_URI` precedence, the raised unreadable-location fault, the dataclass
default, and the branch that must not change: `None` rather than an invented
address when no service node is found or the location is not SLURM-scheduled.

## Scope

The profile layer's own resolution is untouched, so workstation users who
reach a remote graph through an explicit SSH tunnel are unaffected. Nothing
was changed in the facility host pattern lists; the client resolves the
service address directly rather than widening the definition of a local host.
An explicit `NEO4J_URI` continues to win over everything, as it did before.

## The suite no longer reads the ambient environment

The cases above pin the repair, but they asked the process environment two
questions first. `_resolve_graph_uri` consults `NEO4J_URI` and `SLURM_JOB_ID`
from `os.environ` before it reaches anything the tests patch, so a shell that
exported either decided what the cases observed. `NEO4J_URI` is exactly the
documented escape hatch this repair restores, so any machine or CI using the
hatch ran the pinning suite red on correct code.

Reproduced 2026-09-18 in the worktree at `d9634c141`, both readings from the
same command:

```
NEO4J_URI=bolt://example.invalid:1 \
UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-codex/.venv PYTHONPATH="$PWD" \
uv run --no-sync pytest -p no:cacheprovider \
  tests/graph/test_client_address_resolution.py
```

| `NEO4J_URI` in the environment | result | exit |
|---|---|---|
| `bolt://example.invalid:1` | **2 failed, 7 passed** | 1 |
| unset | 9 passed | 0 |

The two failures are the cases that do not set the variable themselves:

```
FAILED ...::test_direct_address_outside_a_slurm_step
E   AssertionError: assert 'bolt://example.invalid:1' == 'bolt://localhost:17687'

FAILED ...::test_the_client_default_uri_is_the_slurm_aware_resolver
E   AssertionError: assert 'bolt://example.invalid:1' == 'bolt://98dci4-gpu-0002:7687'
```

**The repair is in the tests, not in the assertions.** An autouse fixture
clears `NEO4J_URI` and `SLURM_JOB_ID` before every case; a case that exercises
one of them sets it. The explicit-URI precedence case still sets its own value
explicitly and still passes, and no case was deleted or relaxed. Landing commit
`8b76caacb`.

Measured after, same command, both environments:

| `NEO4J_URI` in the environment | result | exit |
|---|---|---|
| `bolt://example.invalid:1` | 9 passed | 0 |
| unset | 9 passed | 0 |

## The pins still fire

A green suite does not show that a pin fires, so each of the four reversions
in the scratch tree had to fail a named test. The scratch tree at `/dev/shm`
was a full copy of the worktree source with the pristine `client.py` restored
between runs; it passed 9 in its own right first, which is the load control
that shows the runs below were reading the reverted module rather than a stale
scratch copy.

| reversion | test that failed | result |
|---|---|---|
| default factory set back to `get_graph_uri` | `test_the_client_default_uri_is_the_slurm_aware_resolver` | 1 failed, 8 passed |
| explicit-`NEO4J_URI` early return removed | `test_explicit_neo4j_uri_wins_inside_a_slurm_step` (`assert 'bolt://98dci4-gpu-0002:7687' == 'bolt://example.invalid:1'`) | 1 failed, 8 passed |
| swallowed `resolve_location` exception restored | `test_an_unreadable_location_is_not_swallowed` | 1 failed, 8 passed |
| fallback import hoisted to module level | `test_the_fallback_is_read_from_the_module_it_is_imported_from` | 2 failed, 7 passed |

The second row shows the mechanism rather than a preference: with the early
return removed, the resolver returns the address it *discovered*, which is the
address the case exists to show losing.

## Logs

One log per reading, in the run directory
`~/.config/reckon/crew/runs/r-20260918T170835079529-n-the-address-resolution-tests-do-not-read-the-ambient-environment/logs/`:

| log | reading |
|---|---|
| `repro-B-default.log` | before, `NEO4J_URI` exported — 2 failed, 7 passed |
| `repro-C-unset.log` | before, unset — 9 passed |
| `after-A-exported.log` | after, exported — 9 passed |
| `after-B-unset.log` | after, unset — 9 passed |
| `rev-0-scratch-baseline.log` | scratch load control — 9 passed |
| `rev-1-factory.log` | reversion 1 — 1 failed, 8 passed |
| `rev-2-explicit.log` | reversion 2 — 1 failed, 8 passed |
| `rev-3-swallow.log` | reversion 3 — 1 failed, 8 passed |
| `rev-4-fallback-import.log` | reversion 4 — 2 failed, 7 passed |