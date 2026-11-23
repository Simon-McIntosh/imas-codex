"""Persist naming-convention parses and guard archive catalogue snapshots."""

from __future__ import annotations

from collections import Counter

from iter_codex.discovery.signals.variable_names import parse_variable_name
from iter_codex.graph.client import GraphClient
from iter_codex.graph.models import SignalNameParseOutcome

PARSE_QUERY = """
UNWIND $items AS item
MATCH (s:FacilitySignal {id: item.id}), (f:Facility {id: item.facility})
WHERE s.accessor = item.selector AND s.facility_id = item.facility
SET s += item.properties
REMOVE s.plant_system_code, s.control_function, s.variable_token,
       s.candidate_plant_component, s.archive_control_node
WITH s, f, item
OPTIONAL MATCH (s)-[old:CANDIDATE_PLANT_COMPONENT|AT_CONTROL_NODE]->()
DELETE old
WITH DISTINCT s, f, item
FOREACH (props IN item.control_nodes |
    MERGE (c:ControlNode {id: props.id})
    SET c += props
    MERGE (c)-[:AT_FACILITY]->(f)
)
FOREACH (props IN [p IN item.control_nodes WHERE p.parent_control_node IS NOT NULL] |
    MERGE (child:ControlNode {id: props.id})
    MERGE (parent:ControlNode {id: props.parent_control_node})
    MERGE (child)-[:PARENT_CONTROL_NODE]->(parent)
)
FOREACH (node_id IN CASE WHEN item.properties.control_node IS NULL
                        THEN [] ELSE [item.properties.control_node] END |
    MERGE (control:ControlNode {id: node_id})
    MERGE (s)-[:AT_CONTROL_NODE]->(control)
)
FOREACH (props IN item.components |
    MERGE (c:PlantComponent {id: props.id})
    SET c += props
    MERGE (c)-[:AT_FACILITY]->(f)
    MERGE (control:ControlNode {id: props.control_node})
    MERGE (c)-[:AT_CONTROL_NODE]->(control)
)
RETURN count(DISTINCT s) AS written
"""

CATALOGUE_QUERY = """
MATCH (s:FacilitySignal)
WHERE s.metadata_archive IS NOT NULL AND ($after IS NULL OR s.id > $after)
  AND ($facility IS NULL OR s.facility_id = $facility)
RETURN s.id AS id, s.facility_id AS facility, s.accessor AS selector
ORDER BY s.id LIMIT $limit
"""

COUNT_QUERY = """
MATCH (s:FacilitySignal)
WHERE s.metadata_archive IS NOT NULL
  AND ($facility IS NULL OR s.facility_id = $facility)
RETURN count(s) AS total, count(s.accessor) AS selectors
"""


def snapshot_archive_signals(
    graph: GraphClient,
    facility: str | None = None,
    *,
    batch_size: int = 500,
    require_nonempty: bool = False,
) -> list[dict]:
    """The shared selector-presence and population-stability guard for archive reads."""
    if not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be between 1 and 1000")
    control = graph.query(COUNT_QUERY, facility=facility)[0]
    expected = control["total"]
    if control["selectors"] != expected or require_nonempty and expected == 0:
        raise RuntimeError("archive catalogue is empty or contains missing selectors")
    snapshot = []
    after = None
    while True:
        rows = graph.query(
            CATALOGUE_QUERY, after=after, limit=batch_size, facility=facility
        )
        if not rows:
            break
        snapshot.extend(rows)
        after = rows[-1]["id"]
    if (
        len(snapshot) != expected
        or graph.query(COUNT_QUERY, facility=facility)[0] != control
    ):
        raise RuntimeError("archive catalogue changed during snapshot; repeat the pass")
    return snapshot


def name_parse_items(signals: list[dict]) -> list[dict]:
    """Build schema-owned parse properties and hierarchy rows for graph writers."""
    items = []
    for signal in signals:
        parsed = parse_variable_name(signal["selector"])
        items.append(
            {
                **signal,
                "properties": parsed.properties(signal["facility"]),
                "control_nodes": [
                    c.model_dump(mode="json", exclude_none=True)
                    for c in parsed.control_nodes(signal["facility"])
                ],
                "components": [
                    c.model_dump(mode="json", exclude_none=True)
                    for c in parsed.components(signal["facility"])
                ],
            }
        )
    return items


def persist_name_parses(graph: GraphClient, signals: list[dict]) -> Counter:
    """Derive every property and hierarchy link from each exact native selector."""
    items = name_parse_items(signals)
    if items:
        rows = graph.query(PARSE_QUERY, items=items)
        if not rows or rows[0]["written"] != len(items):
            raise RuntimeError(
                "name parse write not acknowledged; selector or facility changed"
            )
    return Counter(item["properties"]["name_parse_outcome"] for item in items)


def parse_archive_catalogue(
    graph: GraphClient, *, batch_size: int = 500, facility: str | None = None
) -> dict:
    """Apply bounded writes to the guarded snapshot and verify the population again."""
    snapshot = snapshot_archive_signals(
        graph, facility, batch_size=batch_size, require_nonempty=True
    )
    outcomes = Counter({outcome.value: 0 for outcome in SignalNameParseOutcome})
    for offset in range(0, len(snapshot), batch_size):
        outcomes.update(
            persist_name_parses(graph, snapshot[offset : offset + batch_size])
        )
    if graph.query(COUNT_QUERY, facility=facility)[0]["total"] != len(snapshot):
        raise RuntimeError("archive catalogue changed during parse; repeat the pass")
    return {
        "total": len(snapshot),
        "written": sum(outcomes.values()),
        "outcomes": dict(outcomes),
    }
