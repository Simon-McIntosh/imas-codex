"""StandardName provenance-link guardrail for DD version-horizon updates.

A DD horizon roll-forward must never lose a StandardName provenance link.
The load-bearing guarantee is that ``IMASNode`` identity is a version-agnostic
DD path, written only via ``MERGE`` and never deleted — a path removed in a
newer DD version is *marked* (``lifecycle_status='removed'``), so the old node
persists and every ``StandardNameSource`` composed from it still resolves.

This module turns that guarantee into a measurable, enforced invariant:

- :func:`capture_sn_link_counts` snapshots the three provenance-edge counts
  and the dangling-link count.
- :func:`check_sn_links_safe` compares a before/after pair and returns the
  list of violations (empty == safe): every edge count must be
  non-decreasing and the dangling count must be zero afterwards.

The update command runs the capture before and after a build and fails on any
violation; the ``-m graph`` guardrail test asserts the same on the live graph.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Provenance edges from a StandardNameSource that must survive an update.
#: A source is composed from a DD path (``FROM_DD_PATH`` → IMASNode) and either
#: produces a name (``HAS_STANDARD_NAME``) or records a blocking gap
#: (``HAS_STANDARD_NAME_VOCAB_GAP``).
_EDGE_COUNT_QUERIES: dict[str, str] = {
    "from_dd_path": "MATCH ()-[r:FROM_DD_PATH]->() RETURN count(r) AS c",
    "has_standard_name": "MATCH ()-[r:HAS_STANDARD_NAME]->() RETURN count(r) AS c",
    "vocab_gap": "MATCH ()-[r:HAS_STANDARD_NAME_VOCAB_GAP]->() RETURN count(r) AS c",
}

#: A source that names a DD path but no longer resolves to an IMASNode — the
#: exact failure the never-delete invariant exists to prevent. Must always be 0.
_DANGLING_QUERY = """
MATCH (s:StandardNameSource)
WHERE s.dd_path IS NOT NULL
  AND NOT (s)-[:FROM_DD_PATH]->(:IMASNode)
RETURN count(s) AS c
"""


@dataclass(frozen=True)
class SNLinkCounts:
    """Snapshot of StandardName provenance-link health."""

    from_dd_path: int
    has_standard_name: int
    vocab_gap: int
    dangling: int

    @property
    def edge_counts(self) -> dict[str, int]:
        """The three provenance-edge counts that must be non-decreasing."""
        return {
            "from_dd_path": self.from_dd_path,
            "has_standard_name": self.has_standard_name,
            "vocab_gap": self.vocab_gap,
        }


def capture_sn_link_counts(gc=None) -> SNLinkCounts:
    """Snapshot the SN provenance-edge counts and dangling-link count.

    Args:
        gc: Optional open :class:`GraphClient`. When omitted, a client is
            opened and closed for the duration of the call.
    """
    from imas_codex.graph.client import GraphClient

    owns = gc is None
    gc = gc or GraphClient()
    try:
        counts = {
            key: (gc.query(query)[0]["c"] if gc.query(query) else 0)
            for key, query in _EDGE_COUNT_QUERIES.items()
        }
        dangling_rows = gc.query(_DANGLING_QUERY)
        dangling = dangling_rows[0]["c"] if dangling_rows else 0
        return SNLinkCounts(dangling=dangling, **counts)
    finally:
        if owns:
            gc.close()


def check_sn_links_safe(before: SNLinkCounts, after: SNLinkCounts) -> list[str]:
    """Return the list of invariant violations for a before/after pair.

    An empty list means the update preserved every StandardName provenance
    link. Violations are:

    - any provenance-edge count that decreased, and
    - a non-zero dangling-link count afterwards.
    """
    violations: list[str] = []
    for key, before_count in before.edge_counts.items():
        after_count = after.edge_counts[key]
        if after_count < before_count:
            violations.append(
                f"{key} decreased: {before_count} → {after_count} "
                f"({before_count - after_count} links lost)"
            )
    if after.dangling > 0:
        violations.append(
            f"{after.dangling} dangling SN source(s): a source names a DD path "
            "but no longer resolves to an IMASNode"
        )
    return violations
