"""Ledger invariants — read-only queries that assert provenance integrity.

The graph is the provenance ledger: every non-superseded/-exhausted
``StandardName`` MUST resolve to >=1 ``StandardNameSource`` via ``PRODUCED_NAME``
(``dd`` / ``derived`` / ``signal`` / ``manual`` — none privileged). These
queries surface the two ways that invariant breaks:

- **orphans** — a live name with no ``PRODUCED_NAME`` source at all.
- **edge/scalar desyncs** — a source whose ``produced_sn_id`` scalar names a
  live name but whose ``PRODUCED_NAME`` edge is missing (reattachable).

Deterministic error-siblings (``model='deterministic:dd_error_modifier'``) are
excluded from the orphan set by design: their source ``IMASNode`` is never
extracted as a ``StandardNameSource``; their provenance rides on
``StandardName.source_paths``.
"""

from __future__ import annotations

from typing import Any

from imas_codex.graph.client import GraphClient

#: Canonical "live" predicate used across the SN codebase — a name is live
#: unless it has been refined away (``superseded``) or hit the rotation cap
#: (``exhausted``). Both retain their producing source, so both are in scope.
LIVE_NAME = "NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted']"

#: Deterministic error-siblings carry no StandardNameSource by construction.
_NOT_ERROR_SIBLING = "coalesce(sn.model, '') <> 'deterministic:dd_error_modifier'"

_FIND_ORPHANS = f"""
    MATCH (sn:StandardName)
    WHERE {LIVE_NAME}
      AND {_NOT_ERROR_SIBLING}
      AND NOT (:StandardNameSource)-[:PRODUCED_NAME]->(sn)
    RETURN sn.id AS sn_id, sn.name_stage AS name_stage, sn.origin AS origin
    ORDER BY sn.id
"""

# A genuine desync is a source that HAS produced a name (status composed /
# attached) whose ``PRODUCED_NAME`` edge went missing. A source still pending
# (``extracted`` / ``drafted``) has not produced anything yet — its edge is
# absent by design, not by loss — so it is NOT a desync and must not be
# reattached (that would falsely mark the name sourced before the pipeline
# finishes composing it). Pending sources are handled by the compose pipeline
# (and the provenance-rebuild exclude-pending guard), never here.
_PRODUCED = "coalesce(sns.status, '') IN ['composed', 'attached']"

_FIND_DESYNCS = f"""
    MATCH (sns:StandardNameSource)
    WHERE sns.produced_sn_id IS NOT NULL AND {_PRODUCED}
    MATCH (sn:StandardName {{id: sns.produced_sn_id}})
    WHERE {LIVE_NAME}
      AND NOT (sns)-[:PRODUCED_NAME]->(sn)
    RETURN sns.id AS source_id, sns.produced_sn_id AS sn_id,
           sn.name_stage AS name_stage
    ORDER BY sns.id
"""

_REATTACH_DESYNCS = f"""
    MATCH (sns:StandardNameSource)
    WHERE sns.produced_sn_id IS NOT NULL AND {_PRODUCED}
    MATCH (sn:StandardName {{id: sns.produced_sn_id}})
    WHERE {LIVE_NAME}
      AND NOT (sns)-[:PRODUCED_NAME]->(sn)
    MERGE (sns)-[:PRODUCED_NAME]->(sn)
    RETURN count(*) AS reattached
"""


def find_provenance_orphans(*, gc: GraphClient | None = None) -> list[dict[str, Any]]:
    """Return live names with no ``PRODUCED_NAME`` source (excluding error-siblings).

    An empty list means the ledger invariant holds. Read-only.
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        return list(gc.query(_FIND_ORPHANS))
    finally:
        if owns:
            gc.close()


def find_edge_scalar_desyncs(*, gc: GraphClient | None = None) -> list[dict[str, Any]]:
    """Return sources whose ``produced_sn_id`` names a live name but whose
    ``PRODUCED_NAME`` edge is missing (reattachable desyncs). Read-only.
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        return list(gc.query(_FIND_DESYNCS))
    finally:
        if owns:
            gc.close()


def reattach_produced_name_edges(*, gc: GraphClient | None = None) -> int:
    """Heal edge/scalar desyncs by MERGEing the missing ``PRODUCED_NAME`` edge.

    For every source whose ``produced_sn_id`` scalar names a live name but
    whose ``PRODUCED_NAME`` edge is absent, create the edge. Idempotent (MERGE);
    returns the number of edges reattached.
    """
    owns = gc is None
    gc = gc or GraphClient()
    try:
        rows = gc.query(_REATTACH_DESYNCS)
        return int(rows[0]["reattached"]) if rows else 0
    finally:
        if owns:
            gc.close()
