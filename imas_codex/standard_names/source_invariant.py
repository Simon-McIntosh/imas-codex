"""Refuse writes that strand a live standard name without a producer."""

from collections.abc import Iterable
from typing import Any


class MissingStandardNameSource(ValueError):
    """A live name would have no producing source after a write."""


def require_producing_sources(query: Any, name_ids: Iterable[str]) -> None:
    """Check touched names through the writer's transaction before commit.

    Superseded names are historical. A derived parent with a live child is
    structural and may legitimately have no direct producer.
    """
    ids = sorted(set(name_ids))
    if not ids:
        return
    rows = query.query(
        """
        // SOURCE_INVARIANT
        UNWIND $ids AS id
        MATCH (sn:StandardName {id: id})
        OPTIONAL MATCH (source:StandardNameSource)-[:PRODUCED_NAME]->(sn)
        WITH sn, count(DISTINCT source) AS producers
        WHERE coalesce(sn.name_stage, '') <> 'superseded'
          AND coalesce(sn.status, '') <> 'superseded'
          AND NOT (sn.origin = 'derived' AND EXISTS {
              (child:StandardName)-[:HAS_PARENT]->(sn)
              WHERE coalesce(child.name_stage, '') <> 'superseded'
                AND coalesce(child.status, '') <> 'superseded'
          })
          AND producers = 0
        RETURN sn.id AS id ORDER BY id
        """,
        ids=ids,
    )
    missing = [str(row["id"]) for row in rows]
    if missing:
        raise MissingStandardNameSource(
            "live standard name has no PRODUCED_NAME source: " + ", ".join(missing)
        )
