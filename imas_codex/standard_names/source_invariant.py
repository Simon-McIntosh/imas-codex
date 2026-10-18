"""Refuse writes that strand a live standard name without a producer."""

from collections.abc import Iterable
from typing import Any


class MissingStandardNameSource(ValueError):
    """A live name would have no producing source after a write."""


def _derived_parent_cypher() -> str:
    """Return a two-valued derived-parent test for the graph predicate."""
    return """coalesce(sn.origin, '') = 'derived' AND EXISTS {
              (child:StandardName)-[:HAS_PARENT]->(sn)
              WHERE coalesce(child.name_stage, '') <> 'superseded'
                AND coalesce(child.status, '') <> 'superseded'
          }"""


def find_sourceless_live_names(
    query: Any, name_ids: Iterable[str], *, retired_is_exempt: bool = True
) -> list[str]:
    """Find touched live names with no producer through the caller's graph handle.

    Superseded names are historical. A derived parent with a live child is
    structural and may legitimately have no direct producer.
    """
    ids = sorted(set(name_ids))
    if not ids:
        return []
    rows = query.query(
        f"""
        // SOURCE_INVARIANT
        UNWIND $ids AS id
        MATCH (sn:StandardName {{id: id}})
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
        WITH sn, count(DISTINCT source) AS producers
        WHERE (NOT $retired_is_exempt OR (
            coalesce(sn.name_stage, '') <> 'superseded'
            AND coalesce(sn.status, '') <> 'superseded'))
          AND NOT ({_derived_parent_cypher()})
          AND producers = 0
        RETURN sn.id AS id ORDER BY id
        """,
        ids=ids,
        retired_is_exempt=retired_is_exempt,
    )
    return [str(row["id"]) for row in rows]


def require_producing_sources(
    query: Any, name_ids: Iterable[str], *, retired_is_exempt: bool = True
) -> None:
    """Refuse a write that strands a live non-derived name without a source."""
    missing = find_sourceless_live_names(
        query, name_ids, retired_is_exempt=retired_is_exempt
    )
    if missing:
        raise MissingStandardNameSource(
            "live standard name has no PRODUCED_NAME source: " + ", ".join(missing)
        )
