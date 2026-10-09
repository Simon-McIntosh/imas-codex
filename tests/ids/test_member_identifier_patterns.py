"""A singleton array's member identifier comes from facility configuration.

JT-60SA's MMSYS coil currents carry the member in the middle of the name,
followed by the acquisition chain (``curEF5LKAT``, ``curCS1HiTe``,
``curUFPLKAT``, ``cur1TFLKAT``), so the trailing number the exporter defaults to
is absent and every row exported null. The facility declares a per-source-group
pattern whose ``member`` named group captures the member; these tests show that
pattern driving the export, and that a group without one keeps the default.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from imas_codex.ids.handoff import (
    _member_identifier,
    _member_patterns,
    build_mapping_handoff,
)

MMSYS_TARGET = "pf_active/coil/current/data"
PSC_TARGET = "tf/coil/current/data"
UNMATCHED_TARGET = "pf_active/coil/resistance/data"


def test_mmsys_pattern_reads_the_member_from_the_facility_config():
    """The four coil current names give the member the plan calls for."""
    pattern = _member_patterns("jt-60sa")["MMSYS"]
    cases = {
        "curEF5LKAT": "EF5",
        "curCS1HiTe": "CS1",
        "curUFPLKAT": "UFP",
        "cur1TFLKAT": "1",
    }
    # Positive control: the pattern must match every name before its capture is
    # judged, so a pattern that silently stopped matching cannot pass as empty,
    # and the whole MMSYS family must resolve, not just the four named ones.
    for array, expected in cases.items():
        assert pattern.search(array) is not None, array
        assert _member_identifier(array, [array], pattern) == expected

    family = {
        "cur2TFLKAT": "2",
        "curEF1HiTe": "EF1",
        "curEF3LKAT": "EF3",
        "curCS2LKAT": "CS2",
        "curLFPHiTe": "LFP",
        "curUFPHiTe": "UFP",
    }
    for array, expected in family.items():
        assert _member_identifier(array, [array], pattern) == expected


def test_group_without_a_pattern_uses_the_trailing_number():
    """A source group with no declared pattern behaves as it did before."""
    assert "PSC" not in _member_patterns("jt-60sa")
    assert _member_identifier("curEFCC4", ["curEFCC4"], None) == "4"


def test_array_the_pattern_does_not_match_keeps_a_null_identifier():
    """An unmatched array is left null rather than guessed at."""
    pattern = _member_patterns("jt-60sa")["MMSYS"]
    assert pattern.search("nominalValue") is None
    assert _member_identifier("nominalValue", ["nominalValue"], pattern) is None
    # A group with no pattern and no trailing number is also null.
    assert _member_identifier("ppLFPPCVl", ["ppLFPPCVl"], None) is None


def _mmsys_graph() -> MagicMock:
    graph = MagicMock()

    def query(statement: str, **params):
        if "m.facility_id AS facility_id" in statement:
            return [
                {
                    "id": "jt-60sa:pf_active",
                    "facility_id": "jt-60sa",
                    "ids_name": "pf_active",
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
                    "source_id": "jt-60sa:pf_active:mmsys_ef5",
                    "target_id": MMSYS_TARGET,
                    "transform_expression": None,
                    "source_units": "A",
                    "target_units": "A",
                    "source_property": "value",
                },
                {
                    "source_id": "jt-60sa:pf_active:psc",
                    "target_id": PSC_TARGET,
                    "transform_expression": None,
                    "source_units": "A",
                    "target_units": "A",
                    "source_property": "value",
                },
                {
                    "source_id": "jt-60sa:pf_active:mmsys_nominal",
                    "target_id": UNMATCHED_TARGET,
                    "transform_expression": None,
                    "source_units": "ohm",
                    "target_units": "ohm",
                    "source_property": "value",
                },
            ]
        if "OPTIONAL MATCH (signal:FacilitySignal)-[:MEMBER_OF]->(source)" in statement:
            return [
                {
                    "source_id": "jt-60sa:pf_active:mmsys_ef5",
                    "target_id": MMSYS_TARGET,
                    "signal_id": "jt-60sa:general/mmsys_ef5",
                    "data_source": "edas",
                    "data_source_path": "MMSYS/curEF5LKAT",
                    "source_property": "value",
                    "mapping_type": "direct",
                    "derived_from": None,
                    "confidence": None,
                    "evidence": None,
                },
                {
                    "source_id": "jt-60sa:pf_active:psc",
                    "target_id": PSC_TARGET,
                    "signal_id": "jt-60sa:general/psc_efcc4",
                    "data_source": "edas",
                    "data_source_path": "PSC/curEFCC4",
                    "source_property": "value",
                    "mapping_type": "direct",
                    "derived_from": None,
                    "confidence": None,
                    "evidence": None,
                },
                {
                    "source_id": "jt-60sa:pf_active:mmsys_nominal",
                    "target_id": UNMATCHED_TARGET,
                    "signal_id": "jt-60sa:general/mmsys_nominal",
                    "data_source": "edas",
                    "data_source_path": "MMSYS/nominalValue",
                    "source_property": "value",
                    "mapping_type": "direct",
                    "derived_from": None,
                    "confidence": None,
                    "evidence": None,
                },
            ]
        if "HAS_PARENT" in statement:
            return []
        raise AssertionError(f"unexpected graph query: {statement}")

    graph.query.side_effect = query
    return graph


def test_export_applies_the_facility_pattern_per_source_group():
    """The exporter reads the row's group pattern before the default rule."""
    document = build_mapping_handoff("jt-60sa", ["pf_active"], gc=_mmsys_graph())
    signals = document["ids"][0]["signals"]
    by_target = {row["target_path"]: row for row in signals}

    assert by_target[MMSYS_TARGET]["source_group"] == "MMSYS"
    assert by_target[MMSYS_TARGET]["member_identifier"] == "EF5"
    # A group with no declared pattern keeps the trailing-number default.
    assert by_target[PSC_TARGET]["source_group"] == "PSC"
    assert by_target[PSC_TARGET]["member_identifier"] == "4"
    # An MMSYS array the pattern does not match stays null.
    assert by_target[UNMATCHED_TARGET]["source_group"] == "MMSYS"
    assert by_target[UNMATCHED_TARGET]["member_identifier"] is None
