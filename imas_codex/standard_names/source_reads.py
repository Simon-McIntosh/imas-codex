"""Read a name's sources from its producer edges.

A name's sources are held twice in the graph: as ``PRODUCED_NAME`` edges from
the source nodes, which every writer maintains, and as a ``source_paths`` list
copied onto the name, which only some writers update. The edges are the
authority. Every reader that answers "what are this name's sources" asks this
module, so two readers cannot answer differently for one name, and a name whose
copied list is empty while its edges are not stops answering a search with no
source.

The copied list stays on the name as a second projection of the edges; nothing
here writes it and nothing here reads it.
"""

from __future__ import annotations

from typing import Any

#: The one live-name test a source count applies: a name is live when neither
#: its status nor its name stage reads superseded.
LIVE_NAME_TEST = (
    "coalesce(sn.status, '') <> 'superseded' "
    "AND coalesce(sn.name_stage, '') <> 'superseded'"
)

#: No source of any kind produces the name.
NO_PRODUCER_TEST = "NOT EXISTS { (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }"

#: Live children of the counted name, by the child test the writer invariant uses.
LIVE_CHILDREN_TEST = (
    "size([(child:StandardName)-[:HAS_PARENT]->(sn) "
    "WHERE NOT coalesce(child.name_stage, '') IN "
    "['superseded', 'exhausted', 'contested'] "
    "AND NOT coalesce(child.status, '') IN ['deprecated', 'superseded'] | child])"
)


def source_paths_projection(alias: str = "sn") -> str:
    """Cypher expression: the source ids the producer edges give for ``alias``.

    The ids carry the form the copied list stores, so a reader can substitute
    this expression for ``sn.source_paths`` and change nothing else. It is a
    pattern comprehension, which returns an empty list rather than null and
    needs no ``producer`` variable declared in the enclosing query, so it
    composes into a ``RETURN`` beside the other projections.

    The list is unordered; :func:`normalize_source_ids` gives the stable order a
    caller that exposes it as a value should use.
    """
    return f"[(producer:StandardNameSource)-[:PRODUCED_NAME]->({alias}) | producer.id]"


def normalize_source_ids(ids: Any) -> list[str]:
    """Stable, de-duplicated form of a source-id list."""
    return sorted({str(source_id) for source_id in (ids or []) if source_id})


def read_source_paths(gc: Any, name: str) -> list[str]:
    """The source ids the producer edges give for one name."""
    return read_source_paths_by_name(gc, [name]).get(name, [])


def read_source_paths_by_name(gc: Any, names: list[str]) -> dict[str, list[str]]:
    """The source ids the producer edges give for each name in ``names``.

    A name the graph does not hold maps to an empty list, so a caller that must
    tell "absent" from "sourceless" checks existence first.
    """
    wanted = [name for name in names if name]
    if not wanted:
        return {}
    rows = (
        gc.query(
            f"""
            UNWIND $names AS name_id
            MATCH (sn:StandardName {{id: name_id}})
            RETURN sn.id AS name,
                   {source_paths_projection()} AS source_paths
            """,
            names=wanted,
        )
        or []
    )
    return {
        str(row["name"]): normalize_source_ids(row.get("source_paths")) for row in rows
    }


def count_live_names_without_source(gc: Any) -> dict[str, Any]:
    """Count live names with no producer edge, applying one rule and no exemption.

    A name is counted when neither its status nor its name stage reads
    superseded and no producer edge points to it. Every name the rule finds is
    listed with its origin and its count of live children, because the
    exemptions that keep a name from being a loss — a derived parent, a name
    held under a release ruling — are the caller's to apply, not this count's
    to assume.

    A derived parent whose producer is a derived record has a producer edge, so
    it is not in this population. A name of derived origin that no edge produces
    at all is listed apart as an exempt derived parent and left out of
    ``count``: its producer is its children, by design, and a sourcelessness
    count that reported it would report every such parent as a loss.
    ``live_without_producer`` still counts every name the rule finds, so the
    return agrees with a direct read of the edges.
    """
    totals = (
        gc.query(
            f"""
            MATCH (sn:StandardName)
            RETURN sum(CASE WHEN {LIVE_NAME_TEST} THEN 1 ELSE 0 END) AS live_names,
                   sum(CASE WHEN {LIVE_NAME_TEST}
                             AND EXISTS {{ (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }}
                            THEN 1 ELSE 0 END) AS live_with_producer
            """
        )
        or [{"live_names": 0, "live_with_producer": 0}]
    )[0]

    rows = (
        gc.query(
            f"""
            MATCH (sn:StandardName)
            WHERE {LIVE_NAME_TEST} AND {NO_PRODUCER_TEST}
            RETURN sn.id AS id,
                   sn.origin AS origin,
                   sn.name_stage AS name_stage,
                   size(coalesce(sn.source_paths, [])) AS copied_count,
                   {LIVE_CHILDREN_TEST} AS live_children
            ORDER BY id
            """
        )
        or []
    )

    listed: list[dict[str, Any]] = []
    exempt: list[dict[str, Any]] = []
    for row in rows:
        entry = {
            "id": str(row["id"]),
            "origin": row.get("origin"),
            "name_stage": row.get("name_stage"),
            "live_children": int(row.get("live_children") or 0),
            "copied_count": int(row.get("copied_count") or 0),
        }
        if entry["origin"] == "derived" and entry["live_children"] > 0:
            exempt.append(entry)
        else:
            listed.append(entry)

    return {
        "live_names": int(totals.get("live_names") or 0),
        "live_with_producer": int(totals.get("live_with_producer") or 0),
        "live_without_producer": len(rows),
        "count": len(listed),
        "names": listed,
        "exempt_derived_parents": exempt,
    }
