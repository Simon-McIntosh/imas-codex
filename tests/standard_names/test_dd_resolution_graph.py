"""Graph storage contract for Data Dictionary resolutions."""

from pathlib import Path

import yaml

from imas_codex.graph.schema import GraphSchema
from imas_codex.standard_names.dd_resolutions import _GRAPH_RESOLUTION_QUERY


def test_schema_declares_resolution_provenance_and_surviving_gates() -> None:
    schema = yaml.safe_load(Path("imas_codex/schemas/standard_name.yaml").read_text())
    assert "DDResolutionStateChange" not in schema["classes"]
    attributes = schema["classes"]["DDResolution"]["attributes"]

    assert {
        "published_value",
        "effective_value",
        "dd_version",
        "upstream_reference",
        "recorded_by",
        "recorded_at",
        "reason",
        "evidence",
    } <= attributes.keys()
    assert attributes["evidence"]["annotations"] == {
        "relationship_type": "EVIDENCED_BY",
        "target_label": "DDGap",
    }


def test_graph_schema_derives_required_relationship_directions() -> None:
    schema = GraphSchema("imas_codex/schemas/imas_dd.yaml")
    relationships = {
        relationship.cypher_type: (
            relationship.from_class,
            relationship.to_class,
        )
        for relationship in schema.relationships
        if (
            relationship.cypher_type == "BRIDGED_BY"
            and relationship.from_class == "IMASNode"
        )
        or (
            relationship.cypher_type == "EVIDENCED_BY"
            and relationship.from_class == "DDResolution"
        )
    }
    assert relationships["BRIDGED_BY"] == ("IMASNode", "DDResolution")
    assert relationships["EVIDENCED_BY"] == ("DDResolution", "DDGap")


def test_runtime_query_reads_both_gate_edges_and_bridge_direction() -> None:
    assert "(source:IMASNode)-[:BRIDGED_BY]->(resolution)" in _GRAPH_RESOLUTION_QUERY
    assert "(resolution)-[:EVIDENCED_BY]->(gap:DDGap)" in _GRAPH_RESOLUTION_QUERY
    assert (
        "(resolution)-[:FOR_DD_VERSION]->(version:DDVersion)" in _GRAPH_RESOLUTION_QUERY
    )
