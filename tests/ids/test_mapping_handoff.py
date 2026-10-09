"""Contract tests for mapping hand-off export."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.map import map_cmd
from imas_codex.ids.handoff import build_mapping_handoff, check_handoff_document

FIXTURE = Path(__file__).parents[1] / "fixtures" / "mapping_handoff_example.json"
DOCUMENT_KEYS = {
    "format",
    "format_version",
    "facility",
    "dd_version",
    "exported_at",
    "ids",
}
IDS_KEYS = {"ids_name", "mapping_id", "status", "signals", "unexpanded"}
SIGNAL_KEYS = {
    "signal_id",
    "source_id",
    "data_source",
    "source_group",
    "source_array",
    "member_identifier",
    "source_property",
    "target_path",
    "transform_expression",
    "source_units",
    "target_units",
    "cocos_label",
    "cocos_label_source",
    "confidence",
    "evidence",
}
UNEXPANDED_KEYS = {"source_id", "target_path", "reason"}
ERROR_TARGET = "magnetics/b_field_pol_probe/field/data_error_upper"
SINGLETON = "jt-60sa:eddbreadTime('E101173', 'MDAC', 'magPbTC5', t1, t2)"
UNLABELLED_TARGET = "magnetics/saddle_coil/current/data"


def _graph() -> MagicMock:
    graph = MagicMock()
    target = "magnetics/b_field_pol_probe/field/data"

    def query(statement: str, **params):
        if "m.facility_id AS facility_id" in statement:
            return [
                {
                    "id": "jt-60sa:magnetics",
                    "facility_id": "jt-60sa",
                    "ids_name": "magnetics",
                    "dd_version": "4.1.1",
                    "status": "generated",
                    "provider": "imas-codex",
                }
            ]
        if "r.config AS config" in statement:
            return []
        if "r.source_property AS source_property" in statement:
            return [
                {
                    "source_id": "jt-60sa:magnetics:pickup_probe",
                    "target_id": target,
                    "transform_expression": None,
                    "source_units": "T",
                    "target_units": "T",
                    "source_property": "value",
                },
                {
                    "source_id": "jt-60sa:magnetics:flux_loop",
                    "target_id": "magnetics/flux_loop/flux/data",
                    "transform_expression": None,
                    "source_units": None,
                    "target_units": None,
                    "source_property": "value",
                },
                {
                    "source_id": "jt-60sa:magnetics:pickup_probe",
                    "target_id": ERROR_TARGET,
                    "transform_expression": "value",
                    "source_units": "T",
                    "target_units": "T",
                    "source_property": "value",
                },
                {
                    "source_id": SINGLETON,
                    "target_id": target,
                    "transform_expression": "value",
                    "source_units": "T",
                    "target_units": "T",
                    "source_property": "value",
                },
                {
                    "source_id": "jt-60sa:magnetics:unlabelled",
                    "target_id": UNLABELLED_TARGET,
                    "transform_expression": None,
                    "source_units": "A",
                    "target_units": "A",
                    "source_property": "value",
                },
            ]
        if "OPTIONAL MATCH (signal:FacilitySignal)-[:MEMBER_OF]->(source)" in statement:
            return (
                [
                    {
                        "source_id": "jt-60sa:magnetics:pickup_probe",
                        "target_id": target,
                        "signal_id": f"jt-60sa:general/mdac_magpbtc{number}",
                        "data_source": "edas",
                        "data_source_path": f"MDAC/magPbTC{number}",
                        "source_property": "value",
                        "mapping_type": "direct",
                        "derived_from": None,
                        "cocos_label": None,
                        "confidence": None,
                        "evidence": None,
                    }
                    for number in (10, 11)
                ]
                + [
                    {
                        "source_id": "jt-60sa:magnetics:pickup_probe",
                        "target_id": ERROR_TARGET,
                        "signal_id": f"jt-60sa:general/mdac_magpbtc{number}",
                        "data_source": "edas",
                        "data_source_path": f"MDAC/magPbTC{number}",
                        "source_property": "value",
                        "mapping_type": "error_derived",
                        "derived_from": target,
                        "cocos_label": None,
                        "confidence": None,
                        "evidence": None,
                    }
                    for number in (10, 11)
                ]
                + [
                    {
                        "source_id": SINGLETON,
                        "target_id": target,
                        "signal_id": "jt-60sa:general/mdac_magpbtc5",
                        "data_source": "edas",
                        "data_source_path": "MDAC/magPbTC5",
                        "source_property": "value",
                        "mapping_type": "direct",
                        "derived_from": None,
                        "cocos_label": None,
                        "confidence": None,
                        "evidence": None,
                    }
                ]
                + [
                    {
                        "source_id": "jt-60sa:magnetics:flux_loop",
                        "target_id": "magnetics/flux_loop/flux/data",
                        "signal_id": None,
                        "data_source": None,
                        "data_source_path": None,
                        "cocos_label": None,
                        "confidence": None,
                        "evidence": None,
                    },
                    {
                        "source_id": "jt-60sa:magnetics:unlabelled",
                        "target_id": UNLABELLED_TARGET,
                        "signal_id": "jt-60sa:general/eddbread_saddle1",
                        "data_source": "edas",
                        "data_source_path": "SAD/saddle1",
                        "source_property": "value",
                        "mapping_type": "direct",
                        "derived_from": None,
                        "confidence": None,
                        "evidence": None,
                    },
                ]
            )
        if "HAS_PARENT" in statement:
            return [
                {
                    "target_id": target,
                    "cocos_label": "one_like",
                    "cocos_label_source": "xml",
                }
            ]
        raise AssertionError(f"unexpected graph query: {statement}")

    graph.query.side_effect = query
    return graph


def test_builder_expands_members_and_carries_cocos_labels():
    graph = _graph()
    document = build_mapping_handoff("jt-60sa", ["magnetics"], gc=graph)

    check_handoff_document(document)
    assert set(document) == DOCUMENT_KEYS
    assert document["format"] == "imas-codex-mapping-handoff"
    assert document["format_version"] == 1
    assert document["facility"] == "jt-60sa"
    assert document["dd_version"] == "4.1.1"
    entry = document["ids"][0]
    assert set(entry) == IDS_KEYS
    assert entry["status"] == "generated"
    assert len(entry["signals"]) == 4
    assert [row["member_identifier"] for row in entry["signals"]] == [
        "10",
        "11",
        "5",
        "1",
    ]
    assert {row["source_property"] for row in entry["signals"]} == {"value"}
    for row in entry["signals"]:
        assert set(row) == SIGNAL_KEYS
        assert row["data_source"] == "edas"
        assert row["confidence"] is None
        assert row["evidence"] is None

    labelled = [
        row
        for row in entry["signals"]
        if row["target_path"] == "magnetics/b_field_pol_probe/field/data"
    ]
    unlabelled = [
        row for row in entry["signals"] if row["target_path"] == UNLABELLED_TARGET
    ]
    assert len(labelled) == 3
    assert len(unlabelled) == 1
    for row in labelled:
        assert row["source_group"] == "MDAC"
        assert row["cocos_label"] == "one_like"
        assert row["cocos_label_source"] == "xml"
    for row in unlabelled:
        assert row["source_group"] == "SAD"
        assert row["cocos_label"] == "none"
        assert row["cocos_label_source"] == "none"

    assert len(entry["unexpanded"]) == 2
    for row in entry["unexpanded"]:
        assert set(row) == UNEXPANDED_KEYS
    reasons = {row["target_path"]: row["reason"] for row in entry["unexpanded"]}
    assert "No FacilitySignal member" in reasons["magnetics/flux_loop/flux/data"]
    assert "no error signal exists" in reasons[ERROR_TARGET]
    assert all(row["target_path"] != ERROR_TARGET for row in entry["signals"])

    queries = [call.args[0] for call in graph.query.call_args_list]
    cocos_query = next(q for q in queries if "HAS_PARENT" in q)
    assert "cocos_transformation_type" in cocos_query
    assert "cocos_label_source" in cocos_query
    expansion_query = next(
        q for q in queries if "OPTIONAL MATCH (signal:FacilitySignal)" in q
    )
    for marker in (
        "FacilitySignal)-[:MEMBER_OF]->(source)",
        "signal.data_source_name",
        "signal.data_source_path",
        "binding.source_property",
        "binding.mapping_type",
        "binding.confidence",
        "binding.evidence",
    ):
        assert marker in expansion_query


def test_cocos_label_prefers_the_target_itself_over_its_parent():
    """A target labelled on its own node keeps its own label, not its parent's."""
    graph = MagicMock()
    target = "magnetics/self_probe/current/data"
    chain = {
        target: ("ip_like", "xml"),
        "magnetics/self_probe/current": ("one_like", "xml"),
    }
    chain_nodes = [
        target,
        "magnetics/self_probe/current",
        "magnetics/self_probe",
        "magnetics",
    ]

    def query(statement: str, **params):
        if "m.facility_id AS facility_id" in statement:
            return [
                {
                    "id": "jt-60sa:magnetics",
                    "facility_id": "jt-60sa",
                    "ids_name": "magnetics",
                    "dd_version": "4.1.1",
                    "status": "generated",
                    "provider": "imas-codex",
                }
            ]
        if "r.config AS config" in statement:
            return []
        if "r.source_property AS source_property" in statement:
            return [
                {
                    "source_id": "jt-60sa:magnetics:pickup_probe",
                    "target_id": target,
                    "transform_expression": None,
                    "source_units": "A",
                    "target_units": "A",
                    "source_property": "value",
                }
            ]
        if "OPTIONAL MATCH (signal:FacilitySignal)-[:MEMBER_OF]->(source)" in statement:
            return [
                {
                    "source_id": "jt-60sa:magnetics:pickup_probe",
                    "target_id": target,
                    "signal_id": "jt-60sa:general/mdac_selfpbtc1",
                    "data_source": "edas",
                    "data_source_path": "MDAC/selfPbTC1",
                    "source_property": "value",
                    "mapping_type": "direct",
                    "derived_from": None,
                    "cocos_label": None,
                    "confidence": None,
                    "evidence": None,
                }
            ]
        if "HAS_PARENT" in statement:
            start = 0 if "*0.." in statement else 1
            for node in chain_nodes[start:]:
                if node in chain:
                    label, source = chain[node]
                    return [
                        {
                            "target_id": target,
                            "cocos_label": label,
                            "cocos_label_source": source,
                        }
                    ]
            return []
        raise AssertionError(f"unexpected graph query: {statement}")

    graph.query.side_effect = query
    document = build_mapping_handoff("jt-60sa", ["magnetics"], gc=graph)
    row = document["ids"][0]["signals"][0]
    assert row["target_path"] == target
    assert row["cocos_label"] == "ip_like"
    assert row["cocos_label_source"] == "xml"


def test_fixture_has_the_exact_contract_and_realistic_rows():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    check_handoff_document(document)
    assert set(document) == DOCUMENT_KEYS
    assert [entry["ids_name"] for entry in document["ids"]] == [
        "magnetics",
        "pf_active",
    ]
    for entry in document["ids"]:
        assert set(entry) == IDS_KEYS
        assert entry["status"] == "generated"
        for row in entry["signals"]:
            assert set(row) == SIGNAL_KEYS
        for row in entry["unexpanded"]:
            assert set(row) == UNEXPANDED_KEYS
    magnetics, pf_active = document["ids"]
    assert len(magnetics["signals"]) == 2
    assert len(magnetics["unexpanded"]) == 1
    assert len(pf_active["signals"]) == 2
    assert {row["source_group"] for row in magnetics["signals"]} == {"MDAC"}
    assert {row["source_group"] for row in pf_active["signals"]} == {"MMSYS"}
    assert {row["cocos_label"] for row in magnetics["signals"]} == {"one_like"}
    assert {row["cocos_label_source"] for row in magnetics["signals"]} == {"xml"}
    assert {row["cocos_label"] for row in pf_active["signals"]} == {"ip_like"}
    assert {row["cocos_label_source"] for row in pf_active["signals"]} == {"xml"}


def test_checker_rejects_a_missing_key():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del document["ids"][0]["signals"][0]["cocos_label"]
    with pytest.raises(ValueError, match="signals\\[0\\]"):
        check_handoff_document(document)

    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del document["ids"][0]["signals"][0]["cocos_label_source"]
    with pytest.raises(ValueError, match="signals\\[0\\]"):
        check_handoff_document(document)


def test_export_command_writes_checked_json(tmp_path):
    graph = _graph()
    output = tmp_path / "handoff.json"
    with patch("imas_codex.graph.client.GraphClient", return_value=graph):
        result = CliRunner().invoke(
            map_cmd,
            ["export", "jt-60sa", "-i", "magnetics", "--out", str(output)],
        )
    assert result.exit_code == 0, result.output
    document = json.loads(output.read_text(encoding="utf-8"))
    check_handoff_document(document)
    assert document["ids"][0]["status"] == "generated"
