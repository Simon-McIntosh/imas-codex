"""Literal EDAS call extraction and ingestion selection."""

from types import SimpleNamespace

import pytest

from imas_codex.discovery.base.facility import get_facility, list_facilities
from imas_codex.ingestion.extractors import (
    REFERENCE_HANDLERS,
    reference_handlers_for_systems,
)
from imas_codex.ingestion.extractors.edas import extract_edas_references
from imas_codex.ingestion.graph import link_chunks_to_edas_signals
from imas_codex.ingestion.pipeline import _split_and_extract


@pytest.mark.parametrize(
    ("source", "kind", "category", "data_name"),
    [
        (
            "db.eddbreadTime('E101173', 'CryoCP', 'm1TAF81', t1, t2)",
            "edas_eddb",
            "CryoCP",
            "m1TAF81",
        ),
        (
            "call eddbreadOne_f(shot, 'PSRC', 'calIp', step)",
            "edas_eddb",
            "PSRC",
            "calIp",
        ),
        ("call eddbreadPara_f(shot, 'TMS', 'neyag')", "edas_eddb", "TMS", "neyag"),
        (
            "db.eddbreadHeader('E101173', 'MMSYS', 'curCS1LKAT')",
            "edas_eddb",
            "MMSYS",
            "curCS1LKAT",
        ),
        (
            "db.get_time_slice_data(inp_shotnum=41698, inp_category='TMS', inp_dname='neyag')",
            "edas_eddb",
            "TMS",
            "neyag",
        ),
        (
            "db.get_time_nearest(41698, 3.0, 'teyag', 'TMS')",
            "edas_eddb",
            "TMS",
            "teyag",
        ),
        (
            "db.get_time_seriese_data(41698, 0, 1, 'teyag', 'TMS')",
            "edas_eddb",
            "TMS",
            "teyag",
        ),
        (
            "getseldata('E101173', 'PSRC', 'magFluxLp1')",
            "edas_eddb",
            "PSRC",
            "magFluxLp1",
        ),
        (
            "db.uddbreadConvert('E101173', '2111UA001', t1, t2)",
            "edas_uddb",
            None,
            "2111UA001",
        ),
        ("db.pmdbread('E101173', 'PF', 'coil')", "edas_pmdb", "PF", "coil"),
        ("db.lcdb_value('E101173', 'EQ', 'PSI')", "edas_lcdb", "EQ", "PSI"),
        ("db.mbdbread('E101173', 'MAG', 'Ip')", "edas_mbdb", "MAG", "Ip"),
    ],
)
def test_literal_calls(source, kind, category, data_name):
    assert [
        (r.ref_type, r.category, r.data_name) for r in extract_edas_references(source)
    ] == [(kind, category, data_name)]


@pytest.mark.parametrize(
    "source",
    [
        "db.eddbreadTime(shot, cat, dname, t1, t2)",
        "db.get_time_slice_data(inp_category=category, inp_dname=data_name)",
        "db.uddbreadConvert(shot, pid, t1, t2)",
        "db.eddbreadTime('E101173', 'TMS', 'ne' + 'yag', t1, t2)",
        "def eddbreadTime(self, shot, cat, dname): pass",
        "# db.eddbreadTime('E101173', 'TMS', 'neyag', t1, t2)",
    ],
)
def test_dynamic_or_non_call_is_not_a_data_reference(source):
    assert extract_edas_references(source) == []


def test_repeated_call_yields_one_identifier():
    source = "db.eddbreadTime('E101173', 'TMS', 'neyag'); db.eddbreadTime('E101174', 'TMS', 'neyag')"
    assert [r.raw_string for r in extract_edas_references(source)] == ["TMS/neyag"]


@pytest.mark.parametrize("facility", sorted(list_facilities()))
def test_configured_data_systems_select_reference_extractors(facility, monkeypatch):
    """Every public facility config controls its code extraction behavior."""
    config = get_facility(facility)
    systems = config.get("data_systems") or {}
    handlers = reference_handlers_for_systems(systems)
    assert {handler.extractor for handler in handlers} == {
        REFERENCE_HANDLERS[name].extractor
        for name in systems
        if name in REFERENCE_HANDLERS
    }
    assert len(handlers) == len({handler.extractor for handler in handlers})

    monkeypatch.setattr(
        "imas_codex.ingestion.pipeline.chunk_code",
        lambda content, **kwargs: [
            SimpleNamespace(text=content, start_line=1, end_line=1)
        ],
    )
    metadata = {"facility_id": facility}
    edas_source = "db.eddbreadTime('E101173', 'CryoCP', 'm1TAF81', t1, t2)"
    edas_chunks = _split_and_extract(edas_source, "python", metadata)
    assert sum(chunk.get("_edas_ref_count", 0) for chunk in edas_chunks) == (
        1 if "edas" in systems else 0
    )
    assert all("mdsplus_paths" not in chunk for chunk in edas_chunks)

    mdsplus_source = "conn.get('\\\\RESULTS::I_P')"
    mdsplus_chunks = _split_and_extract(mdsplus_source, "python", metadata)
    assert any(
        "\\RESULTS::I_P" in chunk.get("mdsplus_paths", []) for chunk in mdsplus_chunks
    ) == bool({"mdsplus", "tdi"} & systems.keys())
    assert all("_edas_ref_count" not in chunk for chunk in mdsplus_chunks)


def test_edas_backfill_requires_declared_data_system():
    with pytest.raises(ValueError, match="EDAS is not configured"):
        link_chunks_to_edas_signals(facility_id="tcv")
