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

Same address and same count as before the change: 5130 on both sides, with
the login-node reading taken through the same script that produced the
pre-change baseline.

## Tests

`tests/graph/test_client_address_resolution.py`, 5 tests, run under the
default marker selection (not `-m graph`), exit 0:

```
5 passed, 1 warning in 0.20s
```

They cover the peer-node branch, the on-node localhost branch, and the two
branches that must not change: no discovery outside a step, and `None` rather
than an invented address when no service node is found or the location is not
SLURM-scheduled.

## Scope

The profile layer's own resolution is untouched, so workstation users who
reach a remote graph through an explicit SSH tunnel are unaffected. Nothing
was changed in the facility host pattern lists; the client resolves the
service address directly rather than widening the definition of a local host.