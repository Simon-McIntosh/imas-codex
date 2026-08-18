"""Graph-port contract for active Data Dictionary resolutions."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from imas_codex.graph.schema import GraphSchema
from imas_codex.standard_names.dd_resolutions import (
    _GRAPH_PORT_CORRECT_QUERY,
    DDResolutionGraphPathAction,
    DDResolutionGraphPortConflict,
    DDResolutionGraphPortReceipt,
    _classify_graph_port_preflight,
    _graph_port_record_matches,
    _require_complete_graph_correction,
    port_active_dd_resolutions_to_graph,
)


class RecordingPort:
    def __init__(self) -> None:
        self.stored: dict[str, dict] = {}
        self.batches: list[tuple[dict, ...]] = []

    def apply(self, records: tuple[dict, ...]) -> dict[str, object]:
        self.batches.append(records)
        changed = {
            item["id"] for item in records if self.stored.get(item["id"]) != item
        }
        self.stored.update((item["id"], item) for item in records)
        return {
            "writes": len(changed),
            "nodes": len(self.stored),
            "bridged_edges": len(self.stored),
            "evidenced_edges": len(self.stored),
            "version_edges": len(self.stored),
            "path_receipts": [
                {
                    "resolution_id": item["id"],
                    "path": item["path"],
                    "published_value": item["published_value"],
                    "effective_value": item["effective_value"],
                    "action": "attached" if item["id"] in changed else "unchanged",
                }
                for item in records
            ],
        }


def test_graph_record_comparison_normalizes_neo4j_timestamp_text() -> None:
    expected = {
        "recorded_at": "2026-08-17T05:52:44.497040Z",
        "corrected_node": "wall/path",
        "evidence": "dd_gap:wall/path:unit_defect",
        "for_dd_version": "4.1.1",
    }
    current = {
        "properties": {**expected, "recorded_at": "2026-08-17T05:52:44.49704Z"},
        "corrected_nodes": [expected["corrected_node"]],
        "evidence": [expected["evidence"]],
        "dd_versions": [expected["for_dd_version"]],
    }

    assert _graph_port_record_matches(current, expected)


def test_schema_declares_resolution_provenance_and_relationships() -> None:
    schema_path = Path("imas_codex/schemas/standard_name.yaml")
    schema = yaml.safe_load(schema_path.read_text())
    attributes = schema["classes"]["DDResolution"]["attributes"]

    assert {
        "id",
        "path",
        "field",
        "published_value",
        "effective_value",
        "dd_version",
        "upstream_reference",
        "retiring_release",
        "recorded_by",
        "recorded_at",
        "reason",
        "corrected_node",
        "evidence",
        "for_dd_version",
    } <= attributes.keys()
    assert attributes["corrected_node"]["range"] == "string"
    assert "annotations" not in attributes["corrected_node"]
    assert attributes["evidence"]["range"] == "string"
    assert attributes["evidence"]["annotations"] == {
        "relationship_type": "EVIDENCED_BY",
        "target_label": "DDGap",
    }
    assert attributes["for_dd_version"]["annotations"]["relationship_type"] == (
        "FOR_DD_VERSION"
    )


def test_graph_schema_derives_bridge_from_imas_node() -> None:
    schema = GraphSchema("imas_codex/schemas/imas_dd.yaml")
    bridges = [
        relationship
        for relationship in schema.relationships
        if relationship.cypher_type == "BRIDGED_BY"
    ]
    evidence = [
        relationship
        for relationship in schema.relationships
        if relationship.cypher_type == "EVIDENCED_BY"
        and relationship.from_class == "DDResolution"
    ]

    assert [
        (relationship.from_class, relationship.to_class) for relationship in bridges
    ] == [("IMASNode", "DDResolution")]
    assert [
        (relationship.from_class, relationship.to_class) for relationship in evidence
    ] == [("DDResolution", "DDGap")]


def test_port_materializes_all_active_records_and_replays_without_writes() -> None:
    graph_port = RecordingPort()

    first = port_active_dd_resolutions_to_graph(graph_port=graph_port)
    replay = port_active_dd_resolutions_to_graph(graph_port=graph_port)

    assert isinstance(first, DDResolutionGraphPortReceipt)
    assert first.expected == 37
    assert first.writes == 37
    assert first.nodes == first.bridged_edges == first.evidenced_edges == 37
    assert first.version_edges == 37
    assert first.corrected == 0
    assert first.attached == 37
    assert first.unchanged == 0
    assert len(first.path_receipts) == 37
    assert {item.action for item in first.path_receipts} == {
        DDResolutionGraphPathAction.attached
    }
    assert first.replay is False
    assert replay.expected == 37
    assert replay.writes == 0
    assert replay.replay is True
    assert replay.nodes == replay.bridged_edges == replay.evidenced_edges == 37
    assert replay.version_edges == 37
    assert replay.corrected == 0
    assert replay.attached == 0
    assert replay.unchanged == 37
    assert len(replay.path_receipts) == 37
    assert replay.receipt_hash.startswith("sha256:")

    records = graph_port.batches[0]
    assert len(records) == 37
    assert len({item["id"] for item in records}) == 37
    assert all(item["path"] == item["corrected_node"] for item in records)
    assert all(item["evidence"].startswith("dd_gap:") for item in records)
    assert all(item["dd_version"] == item["for_dd_version"] for item in records)
    assert all(item["published_value"] != item["effective_value"] for item in records)
    assert all(item["upstream_reference"] for item in records)
    assert all(item["retiring_release"] for item in records)
    assert all(item["recorded_by"] and item["recorded_at"] for item in records)
    assert all(item["reason"] for item in records)


def test_graph_port_preflight_refuses_observed_value_cas_mismatch() -> None:
    record = {
        "id": "dd_resolution:" + "a" * 64,
        "path": "wall/description_2d(1)/mobile/unit(1)/outline/r",
        "published_value": '"m"',
        "effective_value": '"1"',
    }
    preflight = [
        {
            "id": record["id"],
            "node_count": 1,
            "gap_count": 1,
            "version_count": 1,
            "effective_unit_count": 1,
            "gap_paths": [record["path"]],
            "graph_value": "kg",
            "unit_ids": ["kg"],
        }
    ]

    with pytest.raises(DDResolutionGraphPortConflict) as exc_info:
        _classify_graph_port_preflight(preflight, {record["id"]: record})

    message = str(exc_info.value)
    assert record["path"] in message
    assert "published='m'" in message
    assert "observed='kg'" in message

    with pytest.raises(
        DDResolutionGraphPortConflict, match="compare-and-set"
    ) as correction_info:
        _require_complete_graph_correction([record], [])
    assert record["path"] in str(correction_info.value)
    assert "WHERE node.unit = b.published_graph_value" in _GRAPH_PORT_CORRECT_QUERY
    assert "unit_ids = [b.published_graph_value]" in _GRAPH_PORT_CORRECT_QUERY
