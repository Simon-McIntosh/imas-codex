---
name: archive-signals
description: Find an ITER test or campaign's pulse categories and load named CODAC archive variables into NumPy arrays over bounded pulse windows from a shell. Use for array access; this skill does not plot.
---

# Load archive signals over test pulses

Use the imported graph catalogue to find categories, pulses and exact signal
identities. Use the existing `ArchiveSignalLoader` to load arrays. The graph
lookups below are metadata only; only `loader.load` reads sample data.

Keep real test descriptions, categories, pulse names, variable selectors and
endpoint details in private request files and receipts, never tracked code or
reports. Examples below are invented. Get real values from the user's request,
private configuration and graph results. Do not substitute a similar variable
when an exact named variable is absent or ambiguous.

## Access pattern

This block is the single source of the pattern: the server tool
`get_signal_load_recipe` returns the same text, and a test fails if the two
disagree. Change it in `iter_codex/remote/archive_access.py`, not here.

<!-- recipe:access-pattern:start -->
Access pattern, in order.

1. Find the pulse window. The server's search_pulses(designation=...) returns
   categories with id, name, archive, kind and pulse_count; pass a returned
   category id to the same tool to list its pulses with id, name, archive,
   status, start_ns, end_ns and description. From a shell, find_categories and
   list_pulses in iter_codex.remote.archive_access return the same rows. Times
   are absolute Unix epoch nanoseconds. A null end_ns is an open pulse and
   needs an explicit end when it is read.
2. Resolve the exact signal. The server's search_signals(facility, query=...)
   matches a graph name or accessor exactly, and find_signals returns the same
   rows from a shell; a returned signal id is the argument to load. The
   designation is a literal case-insensitive substring of a category name or
   description, never a wildcard or semantic match.
3. Load arrays with the package, not with the server:
   iter_codex.remote.codac_uda.ArchiveSignalLoader.load(signal_id, start=...,
   end=...) over an absolute window, or load(signal_id, pulse_id=...) over a
   closed pulse. An exact native selector with no graph identity is read through
   ArchiveReader.read(selector, ...); keep the requested selector instead of
   substituting a similar graph name.
4. Bound every read. Supply start and end as integer epoch nanoseconds, or a
   closed pulse_id, or a pulse with time_format="relative" and bounds in
   seconds. An open pulse needs explicit bounds. max_samples, max_bytes and
   max_window_ns are checked before any fetch, and an over-budget request is
   refused rather than truncated.

Returned values.

- signal.time and signal.values are NumPy arrays of equal length.
- metadata["unit"] is the archive's unit. An empty unit is unknown, not
  dimensionless.
- metadata["time_base"] is "epoch_ns" (integer Unix epoch nanoseconds) for an
  absolute window and "pulse_seconds" (floats relative to the pulse) for a
  pulse read with time_format="relative". Subtract an integer origin before
  converting epoch timestamps to float seconds.
- metadata also carries mode, decimated, sampling_verification, query and cache
  evidence. Raw mode is not proof of no decimation.
- An empty window is a refusal, never a measurement.

Limits.

- The search tools page at 100 rows and return a continuation for the next page.
- A read is refused when its window exceeds max_window_ns, its samples exceed
  max_samples, its serialized reply exceeds max_bytes, or its bounds are
  missing.
- Do not interpolate event values as continuous samples and do not resample
  across different time bases.
<!-- recipe:access-pattern:end -->

## Shell and environment

Run from the project checkout using its root environment, e.g. `uv run --no-sync
python /absolute/path/to/private_script.py`. In a worktree, point
`UV_PROJECT_ENVIRONMENT` at the main checkout's `.venv` and `PYTHONPATH` at your
worktree; reuse its `.env` symlink. No dependency install is needed. Put scripts,
receipts and arrays in the private output directory authorized for the session.
Set `ARCHIVE_REQUEST` to the private request JSON path and `ARCHIVE_OUTPUT` to
that output directory. Set `ARCHIVE_CONFIG` to a standalone private YAML when
needed; on this workstation it is normally
`~/.cache/iter-codex-codac-uda/catalogue-private-config.yaml`. Do not print its
contents. Keep any private script outside tracked paths.

Run live graph and archive work on a compute partition (`rigel` or `all_debug`).
From an existing allocation, a separate job is submitted with `env -u
SLURM_JOB_ID -u SLURM_JOBID sbatch --partition=rigel`; use a GPFS output log and
`export TMPDIR=/tmp` inside it. Do not create nested `srun` steps. A small bounded
read may run in an allocation already assigned for this work. Keep the whole
command's log and exit status; log no secrets to chat.

The reader resolves the data system's setup and interpreter itself through the
facility-aware executor. Its configured setup loads the archive client module
in the same shell as the native reader. Do not import the native archive client
or open another connection path. Endpoints come from private configuration;
never contact the excluded operator archive.

## Find categories, pulses and the named variable

Prepare a private JSON request with `test_article` (a designation or distinctive
substring of the test description) and `variable` (exact archive selector or
graph signal name). Example content: `{"test_article":"DEMO magnet",
"variable":"DEMO-PLANT:TEMP"}`. Pass private filenames through environment
variables so scripts and shell history need no real identities.

```python
import json
import os
from pathlib import Path

from iter_codex.graph.client import GraphClient
from iter_codex.remote.archive_access import (
    find_categories, find_signals, list_pulses,
)

request = json.loads(Path(os.environ["ARCHIVE_REQUEST"]).read_text())
output = Path(os.environ["ARCHIVE_OUTPUT"])
output.mkdir(parents=True, exist_ok=True)

with GraphClient() as graph:
    # Positive control: an empty search is credible only in a populated catalogue.
    control = graph.query(
        "MATCH (c:PulseCategory {facility_id: $facility}) "
        "RETURN count(c) AS total, count(c.name) AS named", facility="iter"
    )[0]
    if not control["total"] or control["named"] != control["total"]:
        raise RuntimeError("category catalogue unavailable or missing names")
    categories = find_categories(graph, request["test_article"])
    signals = find_signals(graph, request["variable"])
    listings = {}
    offset = 0
    while True:
        page = find_categories(graph, request["test_article"], offset=offset)
        for category in page["items"]:
            listing = list_pulses(graph, category["id"])
            listings[category["id"]] = listing
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]

receipt = {"categories": categories, "signals": signals, "pulses": listings}
(output / "discovery.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps({"categories": categories["total"], "signals": signals["total"],
                  "pulses_per_category": {k: v["total"] for k, v in listings.items()}}))
```

Inspect the private receipt before fetching. These helpers return `items`,
`total` and `next_offset`; pages hold at most 100 rows. For further pulse pages,
call `list_pulses(graph, category_id, offset=listing["next_offset"])` until the
continuation is `None`. Category matching is a literal, case-insensitive
substring in the name or description, not a wildcard or semantic search. If a
long prose request matches nothing, try its explicit article designation.
No match returns `{"items": [], "total": 0, "next_offset": null}`; report it
and do not fall back to listing everything. Signal matching is exact. More than
one signal match needs disambiguation. A named native variable can be absent
from the imported graph catalogue while the archive serves it: if there is no
graph match, retain the exact requested selector and use the direct reader path
below. Never select a similar graph name instead. Category and pulse IDs are
strings; preserve them exactly.

## Load bounded arrays

Select the intended categories from their names/descriptions, and three distinct
pulses from their listings. Retain the total number found separately from the
three selected. Store the selected pulse objects as `selected_pulses` and the
one signal object as `selected_signal` in the private script, or `None` when
the exact requested native variable has no graph match. Use absolute
integer epoch nanoseconds from the pulse windows. Choose an explicit interval
inside each pulse; two seconds at the start is an example, not a claim that the
variable has samples there. If needed, choose another bounded subwindow within
the same pulse, and record the selection. Never turn an empty read into a
successful measurement.

```python
import numpy as np
from iter_codex.remote.codac_uda import ArchiveReader, ArchiveSignalLoader

# The normal factory reads merged public/private ITER facility configuration.
# If the session instead supplies a private standalone YAML config, resolve it
# explicitly; keep cache files in the authorized private output directory.
import yaml
config_path = os.environ.get("ARCHIVE_CONFIG")
private_config = yaml.safe_load(Path(config_path).expanduser().read_text()) if config_path else None
if private_config is not None and "data_systems" not in private_config:
    private_config = {"data_systems": {"codac_uda": private_config}}

def reader_factory(facility, role):
    return ArchiveReader(facility, role, config=private_config,
                         cache_dir=output / "cache")

summaries = []
arrays = []
with GraphClient() as graph:
    loader = ArchiveSignalLoader(graph, reader_factory=reader_factory)
    for index, pulse in enumerate(selected_pulses):
        start = pulse["start_ns"]
        # An open pulse needs an explicitly chosen end; do not invent its end_ns.
        end = min(start + 2_000_000_000, pulse["end_ns"]) if pulse["end_ns"] is not None else start + 2_000_000_000
        parameters = dict(start=start, end=end, mode="raw",
                          max_samples=100_000, max_bytes=8_000_000)
        if selected_signal is not None:
            signal = loader.load(selected_signal["id"], **parameters)
        else:
            # Resolve the category's archive role; endpoints stay in private config.
            roles = graph.query(
                "MATCH (a:ArchiveEndpoint {id: $archive, facility_id: $facility}) "
                "RETURN a.role AS role, a.excluded AS excluded",
                archive=pulse["archive"], facility="iter",
            )
            if len(roles) != 1 or roles[0]["excluded"] is not False:
                raise RuntimeError("pulse archive is unavailable or excluded")
            reader = reader_factory("iter", roles[0]["role"])
            signal = reader.read(request["variable"], **parameters)
        if len(signal.time) == 0:
            raise RuntimeError("selected pulse window returned no samples")
        arrays.append(signal)  # NumPy signal.time and signal.values in this process.
        np.savez(output / f"pulse-{index}.npz", time=signal.time, values=signal.values)
        summaries.append({"pulse_id": pulse["id"], "start": start, "end": end,
                          "samples": len(signal.time), **signal.metadata})
(output / "samples.json").write_text(json.dumps(summaries, indent=2))
```

Use the configured sample/byte limits if they are smaller than the example;
`ArchiveReadError.category` and its message explain refusals. The configured
`max_window_ns` also bounds the interval. With neither explicit bounds nor a
closed `pulse_id` window, the loader refuses the load. A window beyond its bound
is refused before fetching. A full closed pulse can instead be loaded with
`loader.load(signal_id, pulse_id=pulse_id, ...)`, subject to the same limits;
an open pulse needs explicit bounds. `pulse_id` must belong to the signal's
archive. An explicit absolute interval uses the pulse listing as metadata and
does not require the metadata archive to be the signal's serving archive.

`signal.time` and `signal.values` are NumPy arrays; `metadata["unit"]` gives the
unit and `metadata["time_base"]` distinguishes `epoch_ns` (integer Unix epoch
nanoseconds) from `pulse_seconds` (floating seconds relative to the pulse). Subtract an integer
origin before converting epoch timestamps to float seconds. Keep `mode`,
`decimated` (true/false/unknown), `sampling_verification`, query and cache evidence
from the metadata. Raw mode is not proof of no decimation. Envelope/binned and
archived-on-change results must retain their sampling interpretation when
combined. Do not interpolate event values as though they were continuous samples.
An empty unit is unknown, not dimensionless. For several named variables, repeat
the exact lookup and bounded load for each; retain each array's own time base
and sampling evidence rather than automatically resampling them.

## Worked example

The server tool `get_signal_load_recipe` returns this same example; a test fails
if the two disagree. It runs in the project environment against a private
`ARCHIVE_CONFIG` file, with invented names.

<!-- recipe:example:start -->
```python
import os
from pathlib import Path

import numpy as np
import yaml

from iter_codex.remote.codac_uda import ArchiveReader

# The private configuration names the endpoints and the excluded operator
# archive; it is never tracked and never printed.
config = yaml.safe_load(Path(os.environ["ARCHIVE_CONFIG"]).expanduser().read_text())
if "data_systems" not in config:
    config = {"data_systems": {"codac_uda": config}}
reader = ArchiveReader("demo", "slow", config=config)

# One signal over an absolute window: integer Unix epoch nanoseconds.
window = reader.read(
    "DEMO-PLANT:TEMP",
    start=1_700_000_000_000_000_000,
    end=1_700_000_002_000_000_000,
    mode="raw",
    max_samples=100_000,
    max_bytes=8_000_000,
)

# The same signal over a closed pulse: seconds relative to the pulse start.
pulse = reader.read(
    "DEMO-PLANT:TEMP",
    pulse="DEMO:TRIAL/7",
    start=0.0,
    end=2.0,
    time_format="relative",
    mode="raw",
    max_samples=100_000,
    max_bytes=8_000_000,
)

print(np.shape(window.time), np.shape(window.values),
      window.metadata["unit"], window.metadata["time_base"])
print(np.shape(pulse.time), np.shape(pulse.values),
      pulse.metadata["unit"], pulse.metadata["time_base"])
```
<!-- recipe:example:end -->

Return the commands run, category and pulse counts, selected-window sample
counts, unit, time base, sampling evidence and private artifact paths. State
empty windows, refusals and any extra discovery needed beyond these instructions.
