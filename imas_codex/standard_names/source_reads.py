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

from imas_codex.standard_names.source_invariant import (
    derived_parent_cypher,
    live_name_cypher,
    producerless_cypher,
)

#: The contrast this module adds to the owner's rule: a name with no
#: ``PRODUCED_NAME`` edge of any kind. The rule the count applies is the owner's
#: (:func:`imas_codex.standard_names.source_invariant.producerless_cypher`),
#: which reads a stale producer as no producer; this one is reported beside it so
#: a reader can see where the two populations differ.
NO_EDGE_AT_ALL = "NOT EXISTS { (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }"


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
    """Count live names with no producer, under the owner's rule.

    The predicates are the ones :mod:`imas_codex.standard_names.source_invariant`
    owns, so this count and the export gate's ``find_sourceless_live_names``
    report the same population by construction. A name is live when neither its
    status nor its name stage reads superseded, and it is sourceless when no
    non-stale ``PRODUCED_NAME`` source produces it. A name of derived origin with
    a live child is listed apart rather than counted: its producer is its
    children by design.

    The count also reports the bare population — live names with no
    ``PRODUCED_NAME`` edge at all — and the names on which the two differ, which
    are exactly the names whose only producers are stale. A reader that wants to
    know whether a name is lost, and a reader that wants to know whether the
    graph still carries an edge, are asking different questions and this is where
    they separate.

    ``count`` and ``live_without_producer`` are the owner's rule;
    ``bare_count`` and ``bare_names`` are the edge-any-status contrast;
    ``differing_names`` is the difference between the counted populations.
    """

    def population(rule: str) -> tuple[list[dict[str, Any]], int]:
        rows = (
            gc.query(
                f"""
                MATCH (sn:StandardName)
                WHERE {live_name_cypher()} AND {rule}
                RETURN sn.id AS id,
                       sn.origin AS origin,
                       sn.name_stage AS name_stage,
                       size(coalesce(sn.source_paths, [])) AS copied_count,
                       {derived_parent_cypher()} AS derived_parent,
                       size([(child:StandardName)-[:HAS_PARENT]->(sn)
                             WHERE NOT coalesce(child.name_stage, '') IN
                               ['superseded', 'exhausted', 'contested']
                               AND NOT coalesce(child.status, '') IN
                                 ['deprecated', 'superseded'] | child]) AS live_children,
                       EXISTS {{ (:StandardNameSource)-[:PRODUCED_NAME]->(sn) }} AS has_edge
                ORDER BY id
                """
            )
            or []
        )
        found = []
        for row in rows:
            found.append(
                {
                    "id": str(row["id"]),
                    "origin": row.get("origin"),
                    "name_stage": row.get("name_stage"),
                    "live_children": int(row.get("live_children") or 0),
                    "copied_count": int(row.get("copied_count") or 0),
                    "has_edge": bool(row.get("has_edge")),
                }
            )
        return found, len(rows)

    counted, without_producer = population(producerless_cypher())
    bare, bare_without_producer = population(NO_EDGE_AT_ALL)

    def split(entries: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
        listed, exempt = [], []
        for entry in entries:
            if entry["origin"] == "derived" and entry["live_children"] > 0:
                exempt.append(entry)
            else:
                listed.append(entry)
        return listed, exempt

    listed, exempt = split(counted)
    bare_listed, _bare_exempt = split(bare)

    counted_ids = {entry["id"] for entry in listed}
    bare_ids = {entry["id"] for entry in bare_listed}
    totals = (
        gc.query(
            f"""
            MATCH (sn:StandardName)
            RETURN sum(CASE WHEN {live_name_cypher()} THEN 1 ELSE 0 END) AS live_names,
                   sum(CASE WHEN {live_name_cypher()}
                             AND NOT ({producerless_cypher()})
                            THEN 1 ELSE 0 END) AS live_with_producer
            """
        )
        or [{"live_names": 0, "live_with_producer": 0}]
    )[0]

    return {
        "live_names": int(totals.get("live_names") or 0),
        "live_with_producer": int(totals.get("live_with_producer") or 0),
        "live_without_producer": without_producer,
        "count": len(listed),
        "names": listed,
        "exempt_derived_parents": exempt,
        "bare_without_producer": bare_without_producer,
        "bare_count": len(bare_listed),
        "bare_names": [entry["id"] for entry in bare_listed],
        "differing_names": sorted(counted_ids - bare_ids),
    }
