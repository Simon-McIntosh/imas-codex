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
    "target_path",
    "transform_expression",
    "source_units",
    "target_units",
    "cocos_label",
    "confidence",
    "evidence",
}
UNEXPANDED_KEYS = {"source_id", "target_path", "reason"}


def _graph() -> MagicMock:
    graph = MagicMock()
    target = "magnetics/b_field_pol_probe/field/data"

    def query(statement: str, **params):
        if "m.facility_id AS facility_id" in statement:
            return [
                {
                    "id": "jt-60sa:magnetics:4.1.1",
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
            ]
        if "OPTIONAL MATCH (signal:FacilitySignal)-[:MEMBER_OF]->(source)" in statement:
            return [
                {
                    "source_id": "jt-60sa:magnetics:pickup_probe",
                    "target_id": target,
                    "signal_id": f"jt-60sa:general/mdac_magpbtc{number}",
                    "data_source": "edas",
                    "data_source_path": f"MDAC/magPbTC{number}",
                    "cocos_label": None,
                    "confidence": None,
                    "evidence": None,
                }
                for number in (10, 11)
            ] + [
                {
                    "source_id": "jt-60sa:magnetics:flux_loop",
                    "target_id": "magnetics/flux_loop/flux/data",
                    "signal_id": None,
                    "data_source": None,
                    "data_source_path": None,
                    "cocos_label": None,
                    "confidence": None,
                    "evidence": None,
                }
            ]
        raise AssertionError(f"unexpected graph query: {statement}")

    graph.query.side_effect = query
    return graph


def test_builder_expands_members_and_keeps_missing_values_as_null():
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
    assert len(entry["signals"]) == 2
    assert [row["member_identifier"] for row in entry["signals"]] == ["10", "11"]
    for row in entry["signals"]:
        assert set(row) == SIGNAL_KEYS
        assert row["data_source"] == "edas"
        assert row["source_group"] == "MDAC"
        assert row["target_path"] == "magnetics/b_field_pol_probe/field/data"
        assert row["cocos_label"] is None
        assert row["confidence"] is None
        assert row["evidence"] is None
        assert row["transform_expression"] is None
    assert len(entry["unexpanded"]) == 1
    assert set(entry["unexpanded"][0]) == UNEXPANDED_KEYS
    assert "No FacilitySignal member" in entry["unexpanded"][0]["reason"]

    expansion_query = graph.query.call_args_list[-1].args[0]
    for marker in (
        "FacilitySignal)-[:MEMBER_OF]->(source)",
        "signal.data_source_name",
        "signal.data_source_path",
        "binding.cocos_label",
        "binding.confidence",
        "binding.evidence",
    ):
        assert marker in expansion_query


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
    assert all(
        row["cocos_label"] is None
        for entry in document["ids"]
        for row in entry["signals"]
    )


def test_checker_rejects_a_missing_key():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del document["ids"][0]["signals"][0]["cocos_label"]
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
