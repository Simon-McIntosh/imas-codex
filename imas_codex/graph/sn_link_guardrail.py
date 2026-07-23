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


#: The provenance pairs the DD-side ``HAS_STANDARD_NAME`` edge must project.
#: Every ``(s)-[:PRODUCED_NAME]->(sn)`` on a non-terminal name whose source is
#: composed from an SN-eligible DD path (``FROM_DD_PATH``) is a pair the edge
#: is expected to materialize. Terminal stages (refined-away / given-up) are
#: excluded — they legitimately trail incomplete edges.
_TERMINAL_NAME_STAGES = ["superseded", "exhausted", "contested"]

_PROJECTION_PAIRS_QUERY = """
MATCH (s:StandardNameSource)-[:PRODUCED_NAME]->(sn:StandardName)
WHERE NOT (sn.name_stage IN $terminal)
MATCH (s)-[:FROM_DD_PATH]->(dd:IMASNode)
WHERE dd.node_category IN $categories
  AND NOT (dd)-[:HAS_STANDARD_NAME]->(sn)
OPTIONAL MATCH (dd)-[:HAS_UNIT]->(du:Unit)
RETURN DISTINCT dd.id AS dd_path, sn.name AS name, sn.unit AS sn_unit,
       du.id AS dd_unit
"""


def missing_projection_edges(gc=None) -> list[dict]:
    """Return provenance pairs that *should* carry an edge but don't.

    The completeness oracle for the DD-side ``HAS_STANDARD_NAME`` projection:
    an eligible, unit-agreeing provenance pair
    (``PRODUCED_NAME`` + ``FROM_DD_PATH`` on an SN-eligible DD node) with no
    materialized edge is drift the reconcile
    (:func:`imas_codex.standard_names.graph_ops.reconcile_standard_name_dd_edges`)
    is meant to erase. After a run this list must be empty.

    A pair whose DD path carries a *known* unit that ``units_agree`` rejects is
    excluded — those are intentionally dropped-and-triaged, not missing edges.
    A DD path with no unit edge is attached (nothing to disagree with), so it
    counts toward the projection here — symmetric with the reconcile gate.
    """
    from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES
    from imas_codex.graph.client import GraphClient
    from imas_codex.units.dd_unit_exceptions import units_agree

    owns = gc is None
    gc = gc or GraphClient()
    try:
        rows = gc.query(
            _PROJECTION_PAIRS_QUERY,
            terminal=_TERMINAL_NAME_STAGES,
            categories=list(SN_SOURCE_CATEGORIES),
        )
    finally:
        if owns:
            gc.close()
    return [
        r
        for r in rows
        if not (
            r["dd_unit"] and not units_agree(r["sn_unit"], r["dd_unit"], r["dd_path"])
        )
    ]


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
