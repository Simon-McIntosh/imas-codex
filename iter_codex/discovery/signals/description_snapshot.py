"""Compare stored source descriptions against a pre-enrichment snapshot.

A generated description must never rewrite what the sources say, so a repair
that restores a feed label has to be checkable against the values the feed held
before enrichment ran. The comparison is a pure function over the snapshot and
the current rows, which lets the repair and its scratch-graph control read the
same verdict.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def compare_source_descriptions(
    rows: Iterable[Mapping[str, Any]],
    snapshot: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report every snapshot signal whose source description or marker drifted.

    ``snapshot`` entries carry ``id``, ``feed_description`` and ``feed_source``.
    ``rows`` are the current graph rows with ``id``, ``description`` and
    ``description_source``. A missing id, a changed description and a changed
    marker are reported separately so a reader can see which kind of drift it
    was, and ``all_equal`` is true only when all three lists are empty.
    """
    by_id = {row["id"]: row for row in rows}
    missing: list[str] = []
    description_mismatch: list[str] = []
    source_mismatch: list[str] = []
    size = 0
    for entry in snapshot:
        size += 1
        row = by_id.get(entry["id"])
        if row is None:
            missing.append(entry["id"])
            continue
        if (row.get("description") or "") != (entry.get("feed_description") or ""):
            description_mismatch.append(entry["id"])
        if (row.get("description_source") or "") != (entry.get("feed_source") or ""):
            source_mismatch.append(entry["id"])
    return {
        "snapshot_size": size,
        "missing": missing,
        "description_mismatch": description_mismatch,
        "source_mismatch": source_mismatch,
        "all_equal": not (missing or description_mismatch or source_mismatch),
    }
