"""Facility-scoped graph projection of private archive component membership."""

from __future__ import annotations

from imas_codex.discovery.signals.archive_grouping import ArchiveGrouping
from imas_codex.discovery.signals.variable_names import PARSE_PROVENANCE
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import ArchiveGroupingState

PATTERN_QUERY = """
MATCH (f:Facility {id: $facility})
UNWIND $patterns AS props
MERGE (p:ArchiveFieldPattern {id: props.id})
SET p += props
MERGE (p)-[:AT_FACILITY]->(f)
RETURN count(p) AS written
"""

COMPONENT_QUERY = """
MATCH (f:Facility {id: $facility})
UNWIND $components AS props
MATCH (p:ArchiveFieldPattern {id: props.archive_field_pattern, facility_id: $facility})
MERGE (c:PlantComponent {id: props.id})
ON CREATE SET c.name = props.code, c.parse_provenance = $provenance,
              c.description = 'Candidate component from an unconfirmed selector grammar'
SET c += props
MERGE (c)-[:AT_FACILITY]->(f)
WITH c, p
OPTIONAL MATCH (c)-[old:HAS_ARCHIVE_FIELD_PATTERN]->()
DELETE old
MERGE (c)-[:HAS_ARCHIVE_FIELD_PATTERN]->(p)
RETURN count(DISTINCT c) AS written
"""

MEMBERSHIP_QUERY = """
UNWIND $items AS item
MATCH (s:FacilitySignal {id: item.id, facility_id: $facility})
WHERE s.accessor = item.selector AND s.metadata_archive IS NOT NULL
SET s.archive_grouping_state = item.state, s.archive_grouping_reason = item.reason,
    s.archive_component = item.component
WITH s, item
OPTIONAL MATCH (s)-[old:IN_ARCHIVE_COMPONENT]->()
DELETE old
WITH DISTINCT s, item
FOREACH (component_id IN CASE WHEN item.component IS NULL THEN [] ELSE [item.component] END |
    MERGE (c:PlantComponent {id: component_id})
    MERGE (s)-[:IN_ARCHIVE_COMPONENT]->(c)
)
RETURN count(DISTINCT s) AS written
"""


def snapshot_archive_signals(
    graph: GraphClient, facility: str, *, batch_size: int = 500
) -> list[dict]:
    """Read bounded pages and refuse missing selectors or changing populations."""
    if not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be between 1 and 1000")
    control_query = """
        MATCH (s:FacilitySignal {facility_id: $facility})
        WHERE s.metadata_archive IS NOT NULL
        RETURN count(s) AS total, count(s.accessor) AS selectors
    """
    control = graph.query(control_query, facility=facility)[0]
    if control["total"] != control["selectors"]:
        raise RuntimeError("archive graph contains missing selectors")
    snapshot = []
    after = ""
    while True:
        rows = graph.query(
            """
            MATCH (s:FacilitySignal {facility_id: $facility})
            WHERE s.metadata_archive IS NOT NULL AND s.id > $after
            RETURN s.id AS id, s.accessor AS selector
            ORDER BY s.id LIMIT $limit
            """,
            facility=facility,
            after=after,
            limit=batch_size,
        )
        if not rows:
            break
        snapshot.extend(rows)
        after = rows[-1]["id"]
    final = graph.query(control_query, facility=facility)[0]
    if len(snapshot) != control["total"] or final != control:
        raise RuntimeError("archive graph changed during snapshot; repeat grouping")
    return snapshot


def persist_archive_grouping(
    graph: GraphClient,
    grouping: ArchiveGrouping,
    facility: str,
    signals: list[dict],
    *,
    batch_size: int = 500,
) -> dict:
    """Project only existing signals; each acknowledged batch is transactional.

    Component counts and field sets include private-only members. The projection
    never creates a FacilitySignal or changes enrichment, claims or source groups.
    """
    if not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be between 1 and 1000")
    written = grouped = 0
    for offset in range(0, len(signals), batch_size):
        items, components, patterns = [], {}, {}
        for signal in signals[offset : offset + batch_size]:
            member = grouping.member(facility, signal["selector"])
            items.append({**signal, **member})
            if member["component"]:
                component = grouping.component(member["component"])
                components[component["id"]] = component
                pattern = grouping.pattern(component["archive_field_pattern"])
                patterns[pattern["id"]] = pattern

        def write_batch(tx, patterns=patterns, components=components, items=items):
            for query, key, values in (
                (PATTERN_QUERY, "patterns", list(patterns.values())),
                (COMPONENT_QUERY, "components", list(components.values())),
                (MEMBERSHIP_QUERY, "items", items),
            ):
                if not values:
                    continue
                rows = tx.run(
                    query,
                    facility=facility,
                    provenance=PARSE_PROVENANCE,
                    **{key: values},
                ).data()
                if len(rows) != 1 or rows[0]["written"] != len(values):
                    raise RuntimeError(
                        "archive grouping write not acknowledged; facility or selector changed"
                    )

        with graph.session() as session:
            session.execute_write(write_batch)
        written += len(items)
        grouped += sum(
            item["state"] == ArchiveGroupingState.grouped.value for item in items
        )
    return {"variables": written, "grouped": grouped, "ungrouped": written - grouped}
