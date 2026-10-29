"""Refuse writes that strand a live standard name without a producer."""

from collections.abc import Iterable
from typing import Any


class MissingStandardNameSource(ValueError):
    """A live name would have no producing source after a write."""


def _live_child_cypher(parent: str = "sn") -> str:
    """Return the evidence-bearing child test shared by writers and export."""
    return f"""EXISTS {{
              MATCH (child:StandardName)-[:HAS_PARENT]->({parent})
              WHERE NOT coalesce(child.name_stage, '') IN
                ['superseded', 'exhausted', 'contested']
                AND NOT coalesce(child.status, '') IN
                  ['deprecated', 'superseded']
          }}"""


def _derived_parent_cypher(parent: str = "sn") -> str:
    """Return a two-valued derived-parent test for the graph predicate."""
    return f"coalesce({parent}.origin, '') = 'derived' AND {_live_child_cypher(parent)}"


def supersede_drained_name_cypher(
    *,
    name: str = "sn",
    successor: str = "new",
    operation: str = "last_source_supersede",
    reason: str = "$supersede_reason",
    origin: str = "$supersede_origin",
    run_id: str = "$supersede_run_id",
) -> str:
    """Return the clause that supersedes a drained name in the caller's write.

    A writer that must stay a single statement embeds this clause; a writer
    that can afford a statement of its own runs it as one.  Keeping the text in
    one builder means the successor edge and its receipt are written the same
    way wherever the supersession happens, and it commits or rolls back with
    the write that moved the source.
    """
    return f"""
        WHERE NOT coalesce({name}.name_stage, '') IN
                ['superseded', 'exhausted', 'contested']
          AND NOT coalesce({name}.status, '') IN ['deprecated', 'superseded']
          AND NOT ({_derived_parent_cypher(name)})
          AND NOT ({name})-[:REFINED_FROM*1..]->({successor})
          AND COUNT {{
                (source:StandardNameSource)-[:PRODUCED_NAME]->({name})
                WHERE coalesce(source.status, '') <> 'stale'
              }} = 0
        SET {name}.updated_at = datetime(),
            {name}.name_stage = 'superseded',
            {name}.status = 'superseded',
            {name}.claimed_at = null,
            {name}.claim_token = null
        MERGE ({successor})-[:REFINED_FROM]->({name})
        CREATE (change:StandardNameChange {{
          id: 'sn-change:' + randomUUID(),
          from_name: {name}.id, to_name: {successor}.id,
          operation: '{operation}',
          reason: {reason}, origin: {origin}, run_id: {run_id},
          changed_at: datetime(), internal: true
        }})
        CREATE ({successor})-[:HAS_INTERNAL_CHANGE]->(change)
    """


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
        WHERE coalesce(source.status, '') <> 'stale'
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
        exempt_ids=exempt,
    )
    return [str(row["id"]) for row in rows]


def supersede_names_losing_last_source(
    query: Any,
    moves: Iterable[tuple[str, str]],
    *,
    operation: str = "last_source_supersede",
    reason: str | None = None,
    origin: str | None = None,
    run_id: str | None = None,
) -> list[str]:
    """Supersede a name a write has just left with no producing source.

    ``moves`` pairs each name whose sources a write moved away with the name
    that now holds them.  A pair supersedes its predecessor only when the graph
    agrees it is live and has no remaining producer, so a caller may pass every
    name it touched without deciding which ones drained.  Derived parents are
    exempt for the same reason the invariant exempts them: a structural parent
    has children, not sources.

    Runs on the caller's own graph handle and transaction, so the supersession
    commits or rolls back with the write that caused it.  The successor edge is
    ``(new)-[:REFINED_FROM]->(old)`` and the receipt is the standard internal
    ``StandardNameChange`` event, so history stays on the usual chain.
    Returns the ids superseded, in output order.
    """
    pairs = sorted(
        {
            (str(old), str(new))
            for old, new in moves
            if old and new and str(old) != str(new)
        }
    )
    if not pairs:
        return []
    rows = query.query(
        f"""
        // LAST_SOURCE_SUPERSEDE
        UNWIND $pairs AS pair
        MATCH (sn:StandardName {{id: pair.old_name}}),
              (new:StandardName {{id: pair.new_name}})
        {supersede_drained_name_cypher(operation=operation)}
        RETURN sn.id AS old_id, new.id AS new_id
        ORDER BY old_id
        """,
        pairs=[{"old_name": old, "new_name": new} for old, new in pairs],
        supersede_reason=reason,
        supersede_origin=origin,
        supersede_run_id=run_id,
    )
    return [
        str(row["old_id"]) for row in rows or [] if "old_id" in row and "new_id" in row
    ]


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
