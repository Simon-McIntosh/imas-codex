"""Project private component and control groups onto existing archive signals."""

from __future__ import annotations

from imas_codex.discovery.signals.archive_grouping import ArchiveGrouping
from imas_codex.discovery.signals.variable_name_graph import (
    PARSE_QUERY,
    name_parse_items,
    snapshot_archive_signals,
)
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import ArchiveGroupingState

__all__ = ["graph_group_counts", "persist_archive_grouping", "snapshot_archive_signals"]

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
MATCH (category:ArchiveFieldPattern {id: props.archive_category_pattern, facility_id: $facility})
MATCH (control:ControlNode {id: props.control_node, facility_id: $facility})
MERGE (c:PlantComponent {id: props.id})
SET c += props
MERGE (c)-[:AT_FACILITY]->(f)
MERGE (c)-[:AT_CONTROL_NODE]->(control)
WITH c, p, category
OPTIONAL MATCH (c)-[old:HAS_ARCHIVE_FIELD_PATTERN|HAS_ARCHIVE_CATEGORY_PATTERN]->()
DELETE old
MERGE (c)-[:HAS_ARCHIVE_FIELD_PATTERN]->(p)
MERGE (c)-[:HAS_ARCHIVE_CATEGORY_PATTERN]->(category)
RETURN count(DISTINCT c) AS written
"""

CONTROL_GROUP_QUERY = """
MATCH (f:Facility {id: $facility})
UNWIND $controls AS props
MATCH (c:ControlNode {id: props.id, facility_id: $facility})
SET c += props
MERGE (c)-[:AT_FACILITY]->(f)
RETURN count(c) AS written
"""

MEMBERSHIP_QUERY = """
UNWIND $items AS item
MATCH (s:FacilitySignal {id: item.id, facility_id: $facility})
WHERE s.accessor = item.selector AND s.metadata_archive IS NOT NULL
SET s.archive_grouping_state = item.state, s.archive_grouping_reason = item.reason,
    s.archive_component = item.component, s.archive_control_node = item.group_control_node
WITH s, item
OPTIONAL MATCH (s)-[old:IN_ARCHIVE_COMPONENT|IN_ARCHIVE_CONTROL_NODE]->()
DELETE old
WITH DISTINCT s, item
FOREACH (component_id IN CASE WHEN item.component IS NULL THEN [] ELSE [item.component] END |
    MERGE (c:PlantComponent {id: component_id})
    MERGE (s)-[:IN_ARCHIVE_COMPONENT]->(c)
)
FOREACH (control_id IN CASE WHEN item.group_control_node IS NULL THEN [] ELSE [item.group_control_node] END |
    MERGE (c:ControlNode {id: control_id})
    MERGE (s)-[:IN_ARCHIVE_CONTROL_NODE]->(c)
)
RETURN count(DISTINCT s) AS written
"""

RETIRE_UNUSED_GROUPS_QUERY = """
MATCH (c:PlantComponent {facility_id: $facility})
WHERE c.archive_field_pattern IS NOT NULL
  AND NOT EXISTS { MATCH (:FacilitySignal)-[:IN_ARCHIVE_COMPONENT]->(c) }
OPTIONAL MATCH (c)-[old:HAS_ARCHIVE_FIELD_PATTERN|HAS_ARCHIVE_CATEGORY_PATTERN]->()
DELETE old
REMOVE c.archive_field_pattern, c.archive_category_pattern, c.archive_member_count
RETURN count(DISTINCT c) AS retired
"""

RETIRE_UNUSED_CONTROL_GROUPS_QUERY = """
MATCH (c:ControlNode {facility_id: $facility})
WHERE c.archive_member_count IS NOT NULL
  AND NOT EXISTS { MATCH (:FacilitySignal)-[:IN_ARCHIVE_CONTROL_NODE]->(c) }
REMOVE c.archive_member_count
RETURN count(c) AS retired
"""


def graph_group_counts(graph: GraphClient, facility: str) -> dict:
    """Count active group and pattern identities after a projection or replay."""
    components = graph.query(
        """
        MATCH (c:PlantComponent {facility_id:$facility})
        WHERE c.archive_field_pattern IS NOT NULL
        RETURN count(c) AS total
    """,
        facility=facility,
    )[0]["total"]
    controls = graph.query(
        """
        MATCH (c:ControlNode {facility_id:$facility})
        WHERE c.archive_member_count IS NOT NULL
        RETURN count(c) AS total
    """,
        facility=facility,
    )[0]["total"]
    patterns = graph.query(
        """
        MATCH (p:ArchiveFieldPattern {facility_id:$facility})
        WHERE EXISTS { MATCH (:PlantComponent)-[:HAS_ARCHIVE_FIELD_PATTERN|HAS_ARCHIVE_CATEGORY_PATTERN]->(p) }
        RETURN p.key_kind AS kind,count(p) AS total
    """,
        facility=facility,
    )
    kinds = {row["kind"]: row["total"] for row in patterns}
    return {
        "component_groups": components,
        "control_node_groups": controls,
        "field_patterns": kinds.get("category_fields", 0),
        "category_patterns": kinds.get("category", 0),
    }


def persist_archive_grouping(
    graph: GraphClient,
    grouping: ArchiveGrouping,
    facility: str,
    signals: list[dict],
    *,
    batch_size: int = 500,
) -> dict:
    """Atomically refresh parse identities and grouping links for each bounded batch.

    These structural groups do not change SignalSource membership or enrichment.
    Both exact and category-only pattern nodes retain codes without meanings.
    """
    if not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be between 1 and 1000")
    counts = {state.value: 0 for state in ArchiveGroupingState}
    for offset in range(0, len(signals), batch_size):
        batch = signals[offset : offset + batch_size]
        items, components, patterns, controls = [], {}, {}, {}
        for signal in batch:
            member = grouping.member(facility, signal["selector"])
            group_control = (
                member["control_node"]
                if member["state"] == ArchiveGroupingState.control_node.value
                else None
            )
            items.append({**signal, **member, "group_control_node": group_control})
            if member["component"]:
                component = grouping.component(member["component"])
                components[component["id"]] = component
                for pattern_id in (
                    component["archive_field_pattern"],
                    component["archive_category_pattern"],
                ):
                    patterns[pattern_id] = grouping.pattern(pattern_id)
            elif group_control:
                controls[group_control] = grouping.control_group(group_control)
        parses = name_parse_items(
            [{**signal, "facility": facility} for signal in batch]
        )
        statements = (
            (PARSE_QUERY, "items", parses),
            (PATTERN_QUERY, "patterns", list(patterns.values())),
            (CONTROL_GROUP_QUERY, "controls", list(controls.values())),
            (COMPONENT_QUERY, "components", list(components.values())),
            (MEMBERSHIP_QUERY, "items", items),
        )

        def write_batch(tx, statements=statements):
            for query, key, values in statements:
                if not values:
                    continue
                rows = tx.run(query, facility=facility, **{key: values}).data()
                if len(rows) != 1 or rows[0]["written"] != len(values):
                    raise RuntimeError(
                        "archive grouping write not acknowledged; facility or selector changed"
                    )

        with graph.session() as session:
            session.execute_write(write_batch)
        for item in items:
            counts[item["state"]] += 1
    graph.query(RETIRE_UNUSED_GROUPS_QUERY, facility=facility)
    graph.query(RETIRE_UNUSED_CONTROL_GROUPS_QUERY, facility=facility)
    return {
        "variables": sum(counts.values()),
        "component_variables": counts[ArchiveGroupingState.component.value],
        "function_variables": counts[ArchiveGroupingState.control_node.value],
        "grouped": counts[ArchiveGroupingState.component.value]
        + counts[ArchiveGroupingState.control_node.value],
        "ungrouped": counts[ArchiveGroupingState.ungrouped.value],
    }
