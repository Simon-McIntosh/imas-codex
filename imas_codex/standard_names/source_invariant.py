"""Refuse writes that strand a live standard name without a producer."""

from collections.abc import Iterable
from typing import Any


class MissingStandardNameSource(ValueError):
    """A live name would have no producing source after a write."""


def _live_child_cypher() -> str:
    """Return the evidence-bearing child test shared by writers and export."""
    return """EXISTS {
              MATCH (child:StandardName)-[:HAS_PARENT]->(sn)
              WHERE NOT coalesce(child.name_stage, '') IN
                ['superseded', 'exhausted', 'contested']
                AND NOT coalesce(child.status, '') IN
                  ['deprecated', 'superseded']
          }"""


def _derived_parent_cypher() -> str:
    """Return a two-valued derived-parent test for the graph predicate."""
    return f"coalesce(sn.origin, '') = 'derived' AND {_live_child_cypher()}"


def find_sourceless_live_names(
    query: Any,
    name_ids: Iterable[str],
    *,
    retired_is_exempt: bool = True,
    exhausted_exempt_ids: Iterable[str] = (),
) -> list[str]:
    """Find touched live names with no producer through the caller's graph handle.

    Superseded names are historical. A derived parent with a live child is
    structural and may legitimately have no direct producer.  A caller that has
    signed a narrow exemption names those ids in ``exhausted_exempt_ids``; an id
    is excused only while its own stage is ``exhausted``, so the exemption can
    never excuse a name the graph has since moved on from.
    """
    ids = sorted(set(name_ids))
    if not ids:
        return []
    exempt = sorted(set(exhausted_exempt_ids))
    rows = query.query(
        f"""
        // SOURCE_INVARIANT
        UNWIND $ids AS id
        MATCH (sn:StandardName {{id: id}})
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
        WHERE source.status <> 'stale'
        WITH sn, count(DISTINCT source) AS producers
        WHERE (NOT $retired_is_exempt OR (
            coalesce(sn.name_stage, '') <> 'superseded'
            AND coalesce(sn.status, '') <> 'superseded'))
          AND NOT ({_derived_parent_cypher()})
          AND producers = 0
          AND NOT (coalesce(sn.name_stage, '') = 'exhausted'
                   AND sn.id IN $exempt_ids)
        RETURN sn.id AS id ORDER BY id
        """,
        ids=ids,
        retired_is_exempt=retired_is_exempt,
        exempt_ids=exempt
    )
    return [str(row["id"]) for row in rows]


def require_producing_sources(
    query: Any,
    name_ids: Iterable[str],
    *,
    retired_is_exempt: bool = True,
    exhausted_exempt_ids: Iterable[str] = (),
) -> None:
    """Refuse a write that strands a live non-derived name without a source."""
    missing = find_sourceless_live_names(
        query,
        name_ids,
        retired_is_exempt=retired_is_exempt,
        exhausted_exempt_ids=exhausted_exempt_ids,
    )
    if missing:
        raise MissingStandardNameSource(
            "live standard name has no PRODUCED_NAME source: " + ", ".join(missing)
        )
