"""Read-only provenance views over the graph ledger.

The shared source audit classifies all live names without a producing edge,
including exhausted names. The edge/scalar desync query separately finds
active source targets whose recorded scalar and edge disagree.
"""

from __future__ import annotations

from typing import Any

from imas_codex.graph.client import GraphClient

#: Current source targets for desync repair and other active-name maintenance.
#: Exhausted names still enter the source-loss audit through its own predicate.
LIVE_NAME = "NOT coalesce(sn.name_stage, '') IN ['superseded', 'exhausted']"

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
    """Return live source-loss names, including exhausted names. Read-only."""
    from imas_codex.standard_names.graph_store.graph_supersession import (
        classify_sourceless_standard_names,
    )

    owns = gc is None
    gc = gc or GraphClient()
    try:
        rows = classify_sourceless_standard_names(gc)
        return [
            {
                "sn_id": row["id"],
                "name_stage": row.get("name_stage"),
                "origin": row.get("origin"),
                "population": row["population"],
                "docs_stage": row.get("docs_stage"),
                "status": row.get("status"),
                "validation_status": row.get("validation_status"),
                "model": row.get("model"),
            }
            for row in rows
            if row["population"] in {"dropped_edge", "never_bound", "unchained_holder"}
        ]
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
