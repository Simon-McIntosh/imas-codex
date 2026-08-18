"""Graph storage contract for Data Dictionary resolutions."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from imas_codex.graph.models import (
    DDResolutionField,
    DDResolutionStatus,
    DDResolutionValueKind,
)
from imas_codex.graph.schema import GraphSchema
from imas_codex.standard_names.dd_resolutions import (
    _GRAPH_RESOLUTION_QUERY,
    IONISATION_POTENTIAL_RESOLUTION_PATHS,
    DDResolutionGraphPathAction,
    DDResolutionGraphPortConflict,
    DDResolutionManifest,
    DDResolutionRecord,
    DDResolutionValue,
    _classify_graph_port_preflight,
    ionisation_potential_resolution_records,
)


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


def _existing_parent(path: str) -> DDResolutionRecord:
    return DDResolutionRecord(
        id=f"dd_resolution:existing:{path}",
        gap_id=f"dd_gap:{path}:self_contradiction",
        path=path,
        dd_version="4.1.1",
        field=DDResolutionField.unit,
        observed=DDResolutionValue(kind=DDResolutionValueKind.string, value="e"),
        effective=DDResolutionValue(kind=DDResolutionValueKind.string, value="eV"),
        reason="Ionisation potential is an energy quantity.",
        recorded_by="test operator",
        recorded_at=datetime(2026, 8, 19, tzinfo=UTC),
        upstream_reference=(
            "https://github.com/iterorganization/IMAS-Data-Dictionary/pull/280"
        ),
        upstream_commit_reference=("commits:30a5ddd4b7037b9f93a8f00f7837809403349d99"),
        retiring_release="4.2.0",
        state=DDResolutionStatus.active,
    )


def test_ionisation_potential_cohort_is_exact_and_excludes_indices_and_charge() -> None:
    paths = set(IONISATION_POTENTIAL_RESOLUTION_PATHS)
    assert len(paths) == 14
    assert sum(path.endswith("ionisation_potential") for path in paths) == 2
    assert sum(path.endswith("error_lower") for path in paths) == 4
    assert sum(path.endswith("error_upper") for path in paths) == 4
    assert not any("index" in path for path in paths)
    assert not any(path.rsplit("/", 1)[-1].startswith("z_") for path in paths)


def test_cohort_builder_adds_only_twelve_uncovered_exact_paths() -> None:
    parent_paths = tuple(
        path
        for path in IONISATION_POTENTIAL_RESOLUTION_PATHS
        if path.endswith("ionisation_potential")
    )
    manifest = DDResolutionManifest(
        resolutions=tuple(_existing_parent(path) for path in parent_paths)
    )
    records = ionisation_potential_resolution_records(
        manifest=manifest,
        recorded_by="test operator",
        recorded_at=datetime(2026, 8, 19, tzinfo=UTC),
        reason="Official upstream change confirms the energy unit.",
    )
    assert len(records) == 12
    assert {record["path"] for record in records} == (
        set(IONISATION_POTENTIAL_RESOLUTION_PATHS) - set(parent_paths)
    )
    assert all(record["published_value"] == '"e"' for record in records)
    assert all(record["effective_value"] == '"eV"' for record in records)
    assert all(record["evidence"].endswith(":self_contradiction") for record in records)
    assert all(record["upstream_reference"].endswith("/pull/280") for record in records)


def test_graph_port_refuses_cas_mismatch_before_any_write() -> None:
    path = IONISATION_POTENTIAL_RESOLUTION_PATHS[1]
    record = ionisation_potential_resolution_records(
        manifest=DDResolutionManifest(resolutions=()),
        recorded_by="test operator",
        recorded_at=datetime(2026, 8, 19, tzinfo=UTC),
        reason="Official upstream change confirms the energy unit.",
    )[1]
    expected = {record["id"]: record}
    row = {
        "id": record["id"],
        "node_count": 1,
        "gap_count": 1,
        "version_count": 1,
        "effective_unit_count": 1,
        "gap_paths": [path],
        "gap_kinds": ["self_contradiction"],
        "gap_versions": ["4.1.1"],
        "gap_observed_values": ["e"],
        "gap_expected_values": ["eV"],
        "graph_value": "keV",
        "unit_ids": ["keV"],
        "claim_ids": [],
        "properties": None,
        "corrected_nodes": [],
        "evidence": [],
        "dd_versions": [],
    }
    with pytest.raises(DDResolutionGraphPortConflict, match="keV"):
        _classify_graph_port_preflight([row], expected)


def test_graph_port_classifies_effective_replay_as_unchanged() -> None:
    record = ionisation_potential_resolution_records(
        manifest=DDResolutionManifest(resolutions=()),
        recorded_by="test operator",
        recorded_at=datetime(2026, 8, 19, tzinfo=UTC),
        reason="Official upstream change confirms the energy unit.",
    )[0]
    row = {
        "id": record["id"],
        "node_count": 1,
        "gap_count": 1,
        "version_count": 1,
        "effective_unit_count": 1,
        "gap_paths": [record["path"]],
        "gap_kinds": ["self_contradiction"],
        "gap_versions": ["4.1.1"],
        "gap_observed_values": ["e"],
        "gap_expected_values": ["eV"],
        "graph_value": "eV",
        "unit_ids": ["eV"],
        "claim_ids": [record["id"]],
        "properties": record,
        "corrected_nodes": [record["path"]],
        "evidence": [record["evidence"]],
        "dd_versions": ["4.1.1"],
    }
    assert _classify_graph_port_preflight([row], {record["id"]: record}) == {
        record["id"]: DDResolutionGraphPathAction.unchanged
    }


@pytest.mark.graph
def test_live_ionisation_potential_cohort_has_exact_edges_and_units() -> None:
    from imas_codex.graph.client import GraphClient

    with GraphClient() as graph:
        rows = graph.query(
            """
            UNWIND $paths AS path
            MATCH (node:IMASNode {id: path})-[:BRIDGED_BY]->(resolution:DDResolution)
            MATCH (resolution)-[:EVIDENCED_BY]->(gap:DDGap)
            MATCH (resolution)-[:FOR_DD_VERSION]->(version:DDVersion)
            OPTIONAL MATCH (node)-[:HAS_UNIT]->(unit:Unit)
            RETURN path, node.unit AS unit, collect(DISTINCT unit.id) AS unit_ids,
                   count(DISTINCT resolution) AS resolutions,
                   count(DISTINCT gap) AS gaps,
                   count(DISTINCT version) AS versions,
                   collect(DISTINCT resolution.upstream_reference) AS upstream,
                   collect(DISTINCT resolution.upstream_commit_reference) AS commits
            ORDER BY path
            """,
            paths=list(IONISATION_POTENTIAL_RESOLUTION_PATHS),
        )
    assert len(rows) == 14
    for row in rows:
        assert row["unit"] == "eV"
        assert row["unit_ids"] == ["eV"]
        assert (row["resolutions"], row["gaps"], row["versions"]) == (1, 1, 1)
        assert row["upstream"] == [
            "https://github.com/iterorganization/IMAS-Data-Dictionary/pull/280"
        ]
        assert row["commits"] == ["commits:30a5ddd4b7037b9f93a8f00f7837809403349d99"]
