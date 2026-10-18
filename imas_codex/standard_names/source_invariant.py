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


def _is_sourceless_name(
    origin: str | None, *, has_live_child: bool, producers: int
) -> bool:
    """Mirror the derived-parent and producer test for small fixtures."""
    return producers == 0 and not ((origin or "") == "derived" and has_live_child)


def require_producing_sources(
    query: Any, name_ids: Iterable[str], *, retired_is_exempt: bool = True
) -> None:
    """Check touched names through the writer's transaction before commit.

    Superseded names are historical. A derived parent with a live child is
    structural and may legitimately have no direct producer.
    """
    ids = sorted(set(name_ids))
    if not ids:
        return
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
    missing = [str(row["id"]) for row in rows]
    if missing:
        raise MissingStandardNameSource(
            "live standard name has no PRODUCED_NAME source: " + ", ".join(missing)
        )
