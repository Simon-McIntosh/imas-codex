"""Import archive-declared categories and exact pulse windows with graph receipts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from imas_codex.graph.models import ArchiveEndpoint, Pulse, PulseCategory

ARCHIVE_QUERY = """
MATCH (f:Facility {id: $facility})
MERGE (a:ArchiveEndpoint {id: $archive.id})
SET a += $archive
MERGE (a)-[:AT_FACILITY]->(f)
RETURN count(a) AS written
"""
CATEGORY_QUERY = """
MATCH (f:Facility {id: $facility})
MATCH (a:ArchiveEndpoint {id: $archive, facility_id: $facility, excluded: false})
UNWIND $items AS props
MERGE (c:PulseCategory {id: props.id})
SET c += props
MERGE (c)-[:AT_FACILITY]->(f)
MERGE (c)-[:IN_ARCHIVE]->(a)
RETURN count(c) AS written
"""
PULSE_QUERY = """
MATCH (f:Facility {id: $facility})
MATCH (a:ArchiveEndpoint {id: $archive, facility_id: $facility, excluded: false})
UNWIND $items AS props
MERGE (p:Pulse {id: props.id})
SET p += props
MERGE (p)-[:AT_FACILITY]->(f)
MERGE (p)-[:IN_ARCHIVE]->(a)
WITH p, props
UNWIND props.categories AS category_id
MATCH (c:PulseCategory {id: category_id, archive: $archive, facility_id: $facility})
MERGE (p)-[:IN_PULSE_CATEGORY]->(c)
RETURN count(DISTINCT p) AS written
"""


def pulse_distribution(
    categories: Sequence[PulseCategory], pulses: Sequence[Pulse]
) -> dict:
    """Summarize membership, retaining the zero-pulse categories explicitly."""
    members = Counter(
        category for pulse in pulses for category in pulse.categories or []
    )
    distribution = Counter(members[category.id] for category in categories)
    ended = sum(pulse.end_ns is not None for pulse in pulses)
    return {
        "categories": len(categories),
        "pulses": len(pulses),
        "pulses_per_category": dict(sorted(distribution.items())),
        "empty_categories": distribution[0],
        "with_end": ended,
        "without_end": len(pulses) - ended,
    }


def import_pulses(
    graph,
    archive: ArchiveEndpoint,
    categories: Sequence[PulseCategory],
    pulses: Sequence[Pulse],
    *,
    batch_size: int = 100,
) -> dict:
    """Upsert a validated snapshot and acknowledge each bounded transaction.

    Counts of created nodes come from transaction counters, not a count of
    processed rows. Null end times are included in SET so an open-ended source
    record cannot retain a stale graph end time. Missing parents roll back.
    """
    if batch_size < 1 or batch_size > 1000:
        raise ValueError("pulse import batch size must be between 1 and 1000")
    if archive.excluded:
        raise ValueError("excluded archive cannot own an imported pulse catalogue")
    category_ids = {category.id for category in categories}
    if len(category_ids) != len(categories) or len({p.id for p in pulses}) != len(
        pulses
    ):
        raise ValueError("pulse catalogue identities must be unique")
    for item in [*categories, *pulses]:
        if (
            item.archive != archive.id
            or item.facility_id != archive.facility_id
            or item.id != f"{archive.id}:{item.name}"
        ):
            raise ValueError("pulse catalogue identity disagrees with its archive")
    for pulse in pulses:
        if not pulse.categories or not set(pulse.categories) <= category_ids:
            raise ValueError("pulse memberships must reference listed categories")
        if pulse.end_ns is not None and pulse.end_ns < pulse.start_ns:
            raise ValueError("pulse end precedes its start")
    created = 0

    def write(tx, statement, expected, **parameters):
        result = tx.run(statement, **parameters)
        row = result.single()
        summary = result.consume()
        if row is None or row["written"] != expected:
            raise RuntimeError(
                "pulse import not acknowledged; check facility and category parents"
            )
        return summary.counters.nodes_created

    with graph.session() as session:
        created += session.execute_write(
            write,
            ARCHIVE_QUERY,
            1,
            facility=archive.facility_id,
            archive=archive.model_dump(mode="json"),
        )
        for statement, items in [(CATEGORY_QUERY, categories), (PULSE_QUERY, pulses)]:
            for offset in range(0, len(items), batch_size):
                batch = items[offset : offset + batch_size]
                created += session.execute_write(
                    write,
                    statement,
                    len(batch),
                    facility=archive.facility_id,
                    archive=archive.id,
                    items=[item.model_dump(mode="json") for item in batch],
                )
    return {**pulse_distribution(categories, pulses), "nodes_created": created}
