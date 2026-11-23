"""Persist schema-owned plant evidence and verify facility-scoped recovery copies."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from iter_codex.discovery.signals.plant_context import PlantContextBundle
from iter_codex.graph.schema import GraphSchema

SIGNALS_QUERY = """
MATCH (s:FacilitySignal {facility_id: $facility})
WHERE s.metadata_archive IS NOT NULL
  AND (s.accessor STARTS WITH $control_prefix OR s.accessor STARTS WITH $function_prefix)
RETURN s.id AS id, s.accessor AS selector,
       s.archive_component AS component_id,
       s.plant_component_code AS component_code
ORDER BY s.id LIMIT $limit
"""

SIGNAL_STATE_QUERY = """
MATCH (f:Facility {id: $facility})
UNWIND $items AS item
MATCH (s:FacilitySignal {id: item.id, facility_id: $facility})
WHERE s.accessor = item.selector AND s.metadata_archive IS NOT NULL
SET s.plant_context_state = item.state, s.plant_context_scope = $scope,
    s.plant_context_links = item.links
WITH s, item
OPTIONAL MATCH (s)-[old:HAS_PLANT_CONTEXT]->(previous:SignalContextLink)
WHERE previous.context_scope = $scope
DELETE old
WITH DISTINCT s, item
CALL (s, item) {
    UNWIND item.links AS identity
    MATCH (link:SignalContextLink {id: identity, facility_id: $facility, context_scope: $scope})
    MERGE (s)-[r:HAS_PLANT_CONTEXT]->(link)
    SET r.context_scope = $scope
    RETURN count(r) AS linked
}
RETURN count(s) AS written, sum(linked) AS linked
"""

SCOPE_NODES_QUERY = """
CALL () {
    MATCH (n {context_scope: $scope}) RETURN n
    UNION
    MATCH (d:Document)-[:DOCUMENT_PLANT_STRUCTURE]->(p {context_scope: $scope})
    RETURN d AS n
}
WITH DISTINCT n
RETURN n.id AS id, n.facility_id AS facility,
       EXISTS { MATCH (n)-[:AT_FACILITY]->(:Facility {id: $facility}) } AS at_facility
ORDER BY id
LIMIT $limit
"""

SCOPE_LINKS_QUERY = """
MATCH (a)-[r {context_scope: $scope}]->(b)
RETURN a.id AS source, type(r) AS type, b.id AS target,
       (a:Facility AND a.id = $facility OR
        EXISTS { MATCH (a)-[:AT_FACILITY]->(:Facility {id: $facility}) }) AS source_scoped,
       (b:Facility AND b.id = $facility OR
        EXISTS { MATCH (b)-[:AT_FACILITY]->(:Facility {id: $facility}) }) AS target_scoped
ORDER BY source, type, target
LIMIT $limit
"""


def snapshot_pilot_signals(graph, facility: str, system: str, *, limit=10000):
    """Use a bounded system cohort and prove its selector instrument sees data."""
    if not 1 <= limit <= 100000:
        raise ValueError("Use a finite pilot catalogue bound")
    control = graph.query(
        """MATCH (s:FacilitySignal {facility_id:$facility})
        WHERE s.metadata_archive IS NOT NULL
        RETURN count(s) AS total, count(s.accessor) AS selectors""",
        facility=facility,
    )[0]
    if not control["total"] or control["total"] != control["selectors"]:
        raise RuntimeError(
            "Archive catalogue selector instrument is empty or incomplete"
        )
    parameters = {
        "facility": facility,
        "control_prefix": system + "-",
        "function_prefix": system + ":",
        "limit": limit + 1,
    }
    signals = graph.query(SIGNALS_QUERY, **parameters)
    if not signals or len(signals) > limit:
        raise RuntimeError("Pilot catalogue is empty or exceeds its declared bound")
    if graph.query(SIGNALS_QUERY, **parameters) != signals:
        raise RuntimeError("Pilot catalogue changed while taking its snapshot")
    return signals


def _acknowledge(tx, query, expected, **parameters):
    result = tx.run(query, **parameters).data()
    if len(result) != 1 or result[0]["written"] != expected:
        raise RuntimeError("Plant context write was not fully acknowledged")
    return result[0]


def context_statements(bundle: PlantContextBundle):
    """Derive node and relationship writes from the declared schema.

    Emit every node before any relationship so missing references fail the
    acknowledgement check rather than producing unscoped placeholder nodes.
    Existing catalogue signals and parsed components are never created here.
    """
    schema = GraphSchema()
    groups = bundle.node_groups()
    identities = set()
    nodes, relationships = [], []
    for label, models in groups.items():
        items = []
        for model in models:
            # A shared Document carries no context scope of its own; its signal
            # rows and its plant references carry the scope.
            scope = getattr(model, "context_scope", None)
            if model.facility_id != bundle.facility or (
                scope is not None and scope != bundle.scope
            ):
                raise ValueError(
                    "Imported node is outside its facility or context scope"
                )
            if model.id in identities or not model.id.startswith(bundle.facility + ":"):
                raise ValueError(
                    "Imported identities must be unique and facility-qualified"
                )
            identities.add(model.id)
            items.append(model.model_dump(mode="json"))
        if not items:
            continue
        query = f"""
        MATCH (f:Facility {{id: $facility}})
        UNWIND $items AS item
        MERGE (n:{label} {{id: item.id}})
        ON CREATE SET n.facility_id = $facility
        WITH n, item WHERE n.facility_id = $facility
        SET n = item
        RETURN count(n) AS written
        """
        nodes.append((query, items))
        for rel in schema.get_relationships_from(label):
            edges = []
            for item in items:
                value = item.get(rel.slot_name)
                targets = value if isinstance(value, list) else [value] if value else []
                edges.extend(
                    {"source": item["id"], "target": target} for target in targets
                )
            if not edges:
                continue
            query = f"""
            UNWIND $items AS item
            MATCH (n:{label} {{id: item.source, facility_id: $facility}})
            MATCH (t:{rel.to_class} {{id: item.target}})
            WHERE t.facility_id = $facility OR (t:Facility AND t.id = $facility)
            MERGE (n)-[r:{rel.cypher_type}]->(t)
            SET r.context_scope = $scope
            RETURN count(r) AS written
            """
            relationships.append((query, edges))
    return nodes + relationships


def write_context(tx, bundle: PlantContextBundle):
    """Apply one complete bundle within the caller's transaction."""
    parameters = {"facility": bundle.facility, "scope": bundle.scope}
    for query, items in context_statements(bundle):
        _acknowledge(tx, query, len(items), items=items, **parameters)
    if bundle.signals:
        result = _acknowledge(
            tx,
            SIGNAL_STATE_QUERY,
            len(bundle.signals),
            items=bundle.signals,
            **parameters,
        )
        if result["linked"] != len(bundle.links):
            raise RuntimeError(
                "Signal context associations were not fully acknowledged"
            )
    _write_component_bindings(tx, bundle)


def _write_component_bindings(tx, bundle):
    """Promote only unanimous, explicitly sourced component bindings."""
    rows = {row.id: row for row in bundle.rows}
    bindings = defaultdict(lambda: defaultdict(set))
    for link in bundle.links:
        if not link.component:
            continue
        row = rows[link.source_row]
        for name in ("plant_structure", "function_category"):
            if value := getattr(row, name):
                bindings[link.component][name].add(value)
    schema = GraphSchema()
    for rel in schema.get_relationships_from("PlantComponent"):
        if rel.slot_name not in {"plant_structure", "function_category"}:
            continue
        items = [
            {"source": source, "target": next(iter(values[rel.slot_name]))}
            for source, values in bindings.items()
            if len(values[rel.slot_name]) == 1
        ]
        if not items:
            continue
        query = f"""
        UNWIND $items AS item
        MATCH (c:PlantComponent {{id:item.source, facility_id:$facility}})
        MATCH (t:{rel.to_class} {{id:item.target, facility_id:$facility}})
        SET c.{rel.slot_name} = item.target
        MERGE (c)-[r:{rel.cypher_type}]->(t)
        SET r.context_scope = $scope
        RETURN count(r) AS written
        """
        _acknowledge(
            tx,
            query,
            len(items),
            items=items,
            facility=bundle.facility,
            scope=bundle.scope,
        )


def persist_plant_context(graph, bundle: PlantContextBundle):
    """An unacknowledged reference rolls back the whole import."""
    with graph.session() as session:
        session.execute_write(lambda tx: write_context(tx, bundle))
    return context_snapshot(graph, bundle.facility, bundle.scope)


def context_snapshot(
    graph, facility: str, scope: str, *, node_limit=10000, link_limit=50000
):
    """Count owned content and hash identities, checking both ends of every edge."""
    nodes = graph.query(
        SCOPE_NODES_QUERY, facility=facility, scope=scope, limit=node_limit + 1
    )
    links = graph.query(
        SCOPE_LINKS_QUERY, facility=facility, scope=scope, limit=link_limit + 1
    )
    if len(nodes) > node_limit or len(links) > link_limit:
        raise RuntimeError("Imported context exceeds its declared snapshot bound")
    if not nodes or not links:
        raise RuntimeError("Imported context snapshot is empty")
    if any(n["facility"] != facility or not n["at_facility"] for n in nodes):
        raise RuntimeError("Imported node does not reach its facility")
    if any(not r["source_scoped"] or not r["target_scoped"] for r in links):
        raise RuntimeError("Imported relationship does not reach its facility")
    if len({n["id"] for n in nodes}) != len(nodes):
        raise RuntimeError("Imported context has duplicate node identities")
    content = json.dumps({"nodes": nodes, "links": links}, sort_keys=True).encode()
    return {
        "nodes": len(nodes),
        "links": len(links),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def assert_context_archive_matches(graph_snapshot: dict, archive_snapshot: dict):
    """Read-back counts and identity digests must both match the live graph."""
    keys = ("nodes", "links", "sha256")
    if any(key not in graph_snapshot or key not in archive_snapshot for key in keys):
        raise ValueError("Both graph and archive need a complete context measurement")
    if any(graph_snapshot[key] != archive_snapshot[key] for key in keys):
        raise ValueError("Private archive does not match imported graph context")
