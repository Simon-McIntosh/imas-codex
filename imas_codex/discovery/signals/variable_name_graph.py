"""Persist candidate name parses and backfill the entire archive catalogue."""

from __future__ import annotations

from collections import Counter

from imas_codex.discovery.signals.variable_names import parse_variable_name
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import SignalNameParseOutcome

PARSE_QUERY = """
UNWIND $items AS item
MATCH (s:FacilitySignal {id: item.id}), (f:Facility {id: item.facility})
WHERE s.accessor = item.selector AND s.facility_id = item.facility
SET s += item.properties
WITH s, f, item
OPTIONAL MATCH (s)-[old:CANDIDATE_PLANT_COMPONENT]->()
DELETE old
WITH DISTINCT s, f, item
FOREACH (props IN item.components |
    MERGE (c:PlantComponent {id: props.id})
    SET c += props
    MERGE (c)-[:AT_FACILITY]->(f)
)
FOREACH (props IN [p IN item.components WHERE p.parent_component IS NOT NULL] |
    MERGE (child:PlantComponent {id: props.id})
    MERGE (parent:PlantComponent {id: props.parent_component})
    MERGE (child)-[:PARENT_COMPONENT]->(parent)
)
FOREACH (component_id IN CASE WHEN item.properties.candidate_plant_component IS NULL
                            THEN [] ELSE [item.properties.candidate_plant_component] END |
    MERGE (candidate:PlantComponent {id: component_id})
    MERGE (s)-[:CANDIDATE_PLANT_COMPONENT]->(candidate)
)
RETURN count(DISTINCT s) AS written
"""

CATALOGUE_QUERY = """
MATCH (s:FacilitySignal)
WHERE s.metadata_archive IS NOT NULL AND ($after IS NULL OR s.id > $after)
RETURN s.id AS id, s.facility_id AS facility, s.accessor AS selector
ORDER BY s.id LIMIT $limit
"""


def persist_name_parses(graph: GraphClient, signals: list[dict]) -> Counter:
    """Derive every property and link from each signal's exact native selector."""
    items = []
    outcomes = Counter()
    for signal in signals:
        parsed = parse_variable_name(signal["selector"])
        outcomes[parsed.outcome.value] += 1
        items.append(
            {
                **signal,
                "properties": parsed.properties(signal["facility"]),
                "components": [
                    c.model_dump(mode="json")
                    for c in parsed.components(signal["facility"])
                ],
            }
        )
    if items:
        rows = graph.query(PARSE_QUERY, items=items)
        if not rows or rows[0]["written"] != len(items):
            raise RuntimeError(
                "name parse write not acknowledged; selector or facility changed"
            )
    return outcomes


def parse_archive_catalogue(graph: GraphClient, *, batch_size: int = 500) -> dict:
    """Snapshot all catalogued archive selectors, then apply bounded writes.

    No cohort or facility predicate restricts this pass. The receipt includes
    all three outcomes, including zero counts. Missing selectors fail closed.
    Concurrent additions require another pass and cannot silently shrink coverage.
    """
    if not 1 <= batch_size <= 1000:
        raise ValueError("batch_size must be between 1 and 1000")
    control = graph.query("""
        MATCH (s:FacilitySignal) WHERE s.metadata_archive IS NOT NULL
        RETURN count(s) AS total, count(s.accessor) AS selectors
    """)
    expected = control[0]["total"]
    if expected == 0 or control[0]["selectors"] != expected:
        raise RuntimeError("archive catalogue is empty or contains missing selectors")
    snapshot = []
    after = None
    while True:
        rows = graph.query(CATALOGUE_QUERY, after=after, limit=batch_size)
        if not rows:
            break
        snapshot.extend(rows)
        after = rows[-1]["id"]
    if len(snapshot) != expected:
        raise RuntimeError("archive catalogue changed during snapshot; repeat the pass")
    outcomes = Counter({outcome.value: 0 for outcome in SignalNameParseOutcome})
    for offset in range(0, len(snapshot), batch_size):
        outcomes.update(
            persist_name_parses(graph, snapshot[offset : offset + batch_size])
        )
    final = graph.query("""
        MATCH (s:FacilitySignal) WHERE s.metadata_archive IS NOT NULL
        RETURN count(s) AS total
    """)
    if final[0]["total"] != expected:
        raise RuntimeError("archive catalogue changed during parse; repeat the pass")
    return {
        "total": expected,
        "written": sum(outcomes.values()),
        "outcomes": dict(outcomes),
    }
