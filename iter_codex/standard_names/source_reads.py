"""Read a name's sources from its producer edges.

A name's sources are held twice in the graph: as ``PRODUCED_NAME`` edges from
the source nodes, which every writer maintains, and as a ``source_paths`` list
copied onto the name, which only some writers update. The edges are the
authority. Every reader that answers "what are this name's sources" asks this
module, so two readers cannot answer differently for one name, and a name whose
copied list is empty while its edges are not stops answering a search with no
source.

The population of live names no source produces is enumerated by
:mod:`iter_codex.standard_names.source_invariant` — the module that owns the
rule — and this module presents it. Nothing here writes the copied list, and
nothing here reads it: a sourceless count taken from the copy answers a
different question, and answered it wrongly by hundreds.
"""

from __future__ import annotations

from typing import Any

from iter_codex.standard_names.source_invariant import (
    find_sourceless_live_name_rows,
)
from iter_codex.standard_names.source_paths import merge_source_paths


def source_paths_projection(alias: str = "sn") -> str:
    """Cypher expression: the source ids the producer edges give for ``alias``.

    The ids carry the form the copied list stores, so a reader can substitute
    this expression for ``sn.source_paths`` and change nothing else. It is a
    pattern comprehension, which returns an empty list rather than null and
    needs no ``producer`` variable declared in the enclosing query, so it
    composes into a ``RETURN`` beside the other projections.

    A stale producer is not a source: the writer's rule reads a stale edge as
    no edge, so a name whose only producer is stale answers here with nothing —
    the same answer the sourceless count gives for the same name.

    The list is unordered; :func:`normalize_source_ids` gives the stable order a
    caller that exposes it as a value should use.
    """
    return (
        f"[(producer:StandardNameSource)-[:PRODUCED_NAME]->({alias}) "
        "WHERE coalesce(producer.status, '') <> 'stale' | producer.id]"
    )


def normalize_source_ids(ids: Any) -> list[str]:
    """Stable, de-duplicated form of a source-id list.

    The union and its order are :func:`merge_source_paths`'s, so a list this
    module hands out and a list a writer merges read the same way.
    """
    return merge_source_paths([], [i for i in (ids or []) if i])


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
    """The live names no live source produces, under the owner's one rule.

    The enumeration is
    :func:`iter_codex.standard_names.source_invariant.find_sourceless_live_name_rows`;
    this module only names the shape a caller and the ``sn sourceless`` command
    read. The rule admits no exemption: a name is counted when neither its
    status nor its name stage reads superseded and no non-stale
    ``PRODUCED_NAME`` source produces it. A name of derived origin whose
    children are its producer is counted like any other and carries
    ``derived_parent`` beside its ``live_children``, so a reader that exempts a
    derived parent does so from a listed attribute rather than from a count
    that already removed it.

    ``count`` and ``names`` are the whole of the answer: one population, one
    number, and the same names the producer edges give.
    """
    names = find_sourceless_live_name_rows(gc, None, derived_parents_exempt=False)
    return {"count": len(names), "names": names}
