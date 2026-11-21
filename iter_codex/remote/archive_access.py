"""Paged graph metadata and the load recipe for bounded archive signal reads.

The lookups never contact an archive or fetch samples. Sample arrays come
from ``ArchiveSignalLoader`` in ``iter_codex.remote.codac_uda``. The load
recipe below is the one source of the access pattern: the server tool returns
it and the repository's ``archive-signals`` skill quotes it verbatim.
"""

from __future__ import annotations

from typing import Any

LOAD_RECIPE_REQUIREMENTS = """\
Loading a signal needs three things this server cannot supply: the installed
iter-codex package, the archive client module (m-uda-client, loaded through the
facility's module environment), and the private host configuration that names
the archive endpoints and the excluded operator archive. This tool returns the
recipe only; it reads no samples and contacts no archive."""

LOAD_RECIPE_ACCESS = """\
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
  across different time bases."""

LOAD_RECIPE_EXAMPLE = """\
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
"""

SIGNAL_LOAD_RECIPE = "\n\n".join(
    (
        "Recipe for loading a plant signal from the CODAC archive.\n\n"
        + LOAD_RECIPE_REQUIREMENTS,
        LOAD_RECIPE_ACCESS,
        "Worked example, run in the project environment with invented names.\n"
        "ARCHIVE_CONFIG names a private YAML file; DEMO-PLANT:TEMP and "
        "DEMO:TRIAL/7 are invented.\n\n```python\n" + LOAD_RECIPE_EXAMPLE + "```",
    )
)


def load_recipe_text() -> str:
    """Return the load recipe shared with the repository's archive-signals skill."""
    return SIGNAL_LOAD_RECIPE


CATEGORY_MATCH = """
MATCH (c:PulseCategory {facility_id: $facility})
WHERE toLower(c.name) CONTAINS toLower($text)
   OR toLower(coalesce(c.description, '')) CONTAINS toLower($text)
"""
PULSE_MATCH = """
MATCH (p:Pulse {facility_id: $facility})-[:IN_PULSE_CATEGORY]->
      (c:PulseCategory {id: $category_id, facility_id: $facility})
"""
SIGNAL_MATCH = """
MATCH (s:FacilitySignal {facility_id: $facility})-[:DATA_ACCESS]->(a:DataAccess)
WHERE a.method_type = 'codac_uda' AND (s.name = $name OR s.accessor = $name)
"""


def _page(
    graph,
    match: str,
    symbol: str,
    projection: str,
    *,
    limit: int,
    offset: int,
    **parameters,
) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= 100:
        raise ValueError("page limit must be between 1 and 100")
    if type(offset) is not int or offset < 0:
        raise ValueError("page offset must be a nonnegative integer")
    (count,) = graph.query(
        match + f"RETURN count(DISTINCT {symbol}) AS total", **parameters
    )
    total = count["total"]
    items = graph.query(
        match
        + f"RETURN DISTINCT {projection} ORDER BY name, id SKIP $offset LIMIT $limit",
        **parameters,
        limit=limit,
        offset=offset,
    )
    return {
        "items": items,
        "total": total,
        "next_offset": offset + limit if offset + limit < total else None,
    }


def find_categories(
    graph, text: str, *, facility: str = "iter", limit: int = 100, offset: int = 0
) -> dict[str, Any]:
    """Match a literal case-insensitive substring of category name or description.

    An unmatched description returns ``items=[]``, ``total=0`` and no next page.
    Results describe the imported catalogue, not a fresh archive listing.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("category search requires a nonempty description")
    return _page(
        graph,
        CATEGORY_MATCH,
        "c",
        "c.id AS id, c.name AS name, c.archive AS archive, "
        "c.description AS description",
        facility=facility,
        text=text,
        limit=limit,
        offset=offset,
    )


def list_pulses(
    graph,
    category_id: str,
    *,
    facility: str = "iter",
    limit: int = 100,
    offset: int = 0,
) -> dict[str, Any]:
    """List exact category members with epoch-nanosecond bounds and descriptions.

    A null ``end_ns`` remains open-ended. Category and pulse IDs are opaque
    graph strings; retain the archive and every category-qualified path segment.
    """
    if not isinstance(category_id, str) or not category_id.strip():
        raise ValueError("pulse listing requires a category identity")
    return _page(
        graph,
        PULSE_MATCH,
        "p",
        "p.id AS id, p.name AS name, p.archive AS archive, "
        "p.start_ns AS start_ns, p.end_ns AS end_ns, p.description AS description",
        facility=facility,
        category_id=category_id,
        limit=limit,
        offset=offset,
    )


def find_signals(
    graph, name: str, *, facility: str = "iter", limit: int = 100, offset: int = 0
) -> dict[str, Any]:
    """Resolve an exact graph name or archive selector to loadable signal IDs."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("signal lookup requires an exact nonempty name")
    return _page(
        graph,
        SIGNAL_MATCH,
        "s",
        "s.id AS id, s.name AS name, s.accessor AS selector, s.unit AS unit, "
        "s.description AS description",
        facility=facility,
        name=name,
        limit=limit,
        offset=offset,
    )
