"""Tests for the code-built case file record.

The verdict is the guard every consumer reads, so these tests make the guarded
thing happen: a record whose enclosing-structure text was withheld must refuse
use, a channel that failed must read as unavailable rather than empty, and a
channel with nothing to report must say so in prose. Assembly is also shown to
be a read: the reader refuses a write statement before it reaches the graph.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names import case_file
from imas_codex.standard_names.case_file import (
    CaseFile,
    assert_read_only,
    build_name_case_file,
    build_source_case_file,
    case_file_digest,
    completeness_verdict,
    render_case_file,
    separation,
)

_TARGET = {
    "name": {
        "id": "elongation_of_flux_surface",
        "kind": "scalar",
        "unit": "1",
        "description": "Elongation of a flux surface.",
        "documentation": "Elongation of flux surface.",
        "name_stage": "accepted",
        "source_paths": ["dd:equilibrium/time_slice/profiles_1d/elongation"],
    },
    "source_ids": ["dd:equilibrium/time_slice/profiles_1d/elongation"],
    "reason": {"review_axis": "name", "comments": "Source states a ratio."},
}

_SOURCE = {
    "id": "equilibrium/time_slice/profiles_1d/elongation",
    "exists": True,
    "unit": "1",
    "ndim": 1,
    "data_type": "FLT_1D",
    "documentation": "Elongation of the flux surface.",
    "description": "",
    "coordinates": ["1"],
}

_ANCESTORS = [
    {
        "path": "equilibrium/time_slice/profiles_1d",
        "depth": 1,
        "documentation": "Profiles of the equilibrium along the flux coordinate.",
        "description": "",
    },
    {
        "path": "equilibrium/time_slice",
        "depth": 2,
        "documentation": "Equilibrium at one time slice.",
        "description": "",
    },
    {
        "path": "equilibrium",
        "depth": 3,
        "documentation": "Equilibrium IDS.",
        "description": "",
    },
]


class _FakeGraph:
    """Answer case-file statements by their marker comment."""

    def __init__(self, **overrides: object) -> None:
        self.overrides = overrides
        self.calls: list[tuple[str, dict]] = []

    def query(self, statement: str, **params: object) -> list[dict]:
        self.calls.append((statement, params))
        for marker, rows in self.overrides.items():
            if marker in statement:
                return rows  # type: ignore[return-value]
        if "CASE_FILE_TARGET" in statement:
            return [_TARGET]
        if "CASE_FILE_SOURCE_TEXT" in statement:
            return [_SOURCE]
        if "CASE_FILE_SOURCE_CHAIN" in statement:
            return [{"id": _SOURCE["id"], "ancestors": _ANCESTORS}]
        if "CASE_FILE_SOURCE_ROOT" in statement:
            return [{"id": _SOURCE["id"], "roots": ["equilibrium"]}]
        if "CASE_FILE_FAMILY_SEED" in statement:
            return [
                {
                    "id": "elongation_of_flux_surface",
                    "physical_base": "elongation",
                    "name_stage": "accepted",
                }
            ]
        return []


def _family(name, *, gc=None):
    return {
        "parent": None,
        "anchor": None,
        "siblings": [
            {
                "name": "elongation_of_plasma_boundary",
                "description": "Elongation of the plasma boundary.",
                "operator_kind": "locus",
                "axis": "plasma_boundary",
            }
        ],
    }


def _neighbours(sn, *, gc=None):
    return {
        "vector_neighbours": [
            {
                "id": "inner_squareness_of_flux_surface",
                "description": "Squareness of a flux surface.",
                "unit": "1",
                "score": 0.9,
            }
        ],
        "same_base_neighbours": [],
        "same_path_neighbours": [],
        "sibling_family": _family(sn.get("id")),
        "channel_status": {
            "vector": "ok",
            "same_base": "skipped",
            "same_path": "skipped",
            "sibling_family": "ok",
        },
    }


def _locus(paths, *, sn_id="", gc=None):
    return [
        {
            "id": "volume_of_flux_surface",
            "description": "Volume enclosed by a flux surface.",
            "unit": "m^3",
            "shared_depth": 3,
            "shared_paths": ["equilibrium/time_slice/profiles_1d/volume"],
        }
    ]


_CONTEXT = {
    "vocabulary_sections": [
        {"segment": "physical_base", "tokens": ["elongation"]},
        {"segment": "position", "tokens": ["flux_surface"]},
    ]
}


def _build(**kwargs):
    return build_name_case_file(
        "elongation_of_flux_surface",
        gc=_FakeGraph(),
        compose_context=_CONTEXT,
        family_reader=_family,
        neighbour_reader=_neighbours,
        locus_reader=_locus,
        **kwargs,
    )


def test_complete_record_carries_content_digest_and_verdict():
    record = _build()

    assert isinstance(record, CaseFile)
    assert record.schema_id == case_file.CASE_FILE_SCHEMA
    assert record.completeness["verdict"] == "complete"
    assert record.completeness["usable"] is True
    assert record.digest == case_file_digest(record.content)
    assert record.tokens > 1

    source = record.content["sources"][0]
    assert source["id"] == "dd:equilibrium/time_slice/profiles_1d/elongation"
    assert source["unit"] == "1"
    assert source["coordinates"] == ["1"]
    assert [entry["path"] for entry in source["enclosing"]["statements"]] == [
        "equilibrium/time_slice/profiles_1d",
        "equilibrium/time_slice",
        "equilibrium",
    ]
    assert source["enclosing"]["chain_to_root"] is True
    assert source["enclosing"]["root"] == "equilibrium"
    assert record.content["incumbent"]["name"] == "elongation_of_flux_surface"
    assert record.content["adjudicator_reason"]["comments"] == (
        "Source states a ratio."
    )
    assert record.provenance["model_calls"] == 0
    assert record.provenance["graph_writes"] == 0
    assert record.provenance["graph_statements"] >= 1


def test_withheld_enclosing_text_is_refused_by_its_own_verdict():
    record = _build(withhold=["enclosing_structure_text"])

    assert record.completeness["verdict"] == "incomplete"
    assert record.completeness["usable"] is False
    assert record.completeness["missing"] == [
        "enclosing_structure_text:dd:equilibrium/time_slice/profiles_1d/elongation"
    ]
    source = record.content["sources"][0]
    assert source["enclosing"]["statements"] == []
    assert "was withheld" in source["enclosing"]["statement"]


def test_no_family_and_no_neighbour_are_stated_plainly_not_empty_blocks():
    record = build_name_case_file(
        "elongation_of_flux_surface",
        gc=_FakeGraph(),
        compose_context=_CONTEXT,
        family_reader=lambda name, *, gc=None: None,
        neighbour_reader=lambda sn, *, gc=None: {
            "vector_neighbours": [],
            "same_base_neighbours": [],
            "same_path_neighbours": [],
            "sibling_family": None,
            "channel_status": {
                "vector": "ok",
                "same_base": "skipped",
                "same_path": "skipped",
                "sibling_family": "ok",
            },
        },
        locus_reader=lambda paths, *, sn_id="", gc=None: [],
    )

    assert record.completeness["verdict"] == "complete"
    assert record.content["family"]["status"] == "none"
    assert record.content["family"]["members"] == []
    assert record.content["family"]["statement"].startswith("no family")
    for channel in ("by_subject", "by_locus"):
        block = record.content["neighbours"][channel]
        assert block["status"] == "none"
        assert block["members"] == []
        assert block["statement"]
    rendered = render_case_file(record)
    assert "no family" in rendered
    assert "no neighbour by subject" in rendered


def test_a_failed_lookup_is_not_read_as_an_absence():
    record = build_name_case_file(
        "elongation_of_flux_surface",
        gc=_FakeGraph(),
        compose_context=_CONTEXT,
        family_reader=_family,
        neighbour_reader=lambda sn, *, gc=None: {
            "vector_neighbours": [],
            "same_base_neighbours": [],
            "same_path_neighbours": [],
            "sibling_family": None,
            "channel_status": {
                "vector": "unavailable",
                "same_base": "skipped",
                "same_path": "skipped",
                "sibling_family": "unavailable",
            },
        },
        locus_reader=_locus,
    )

    assert record.content["neighbours"]["by_subject"]["status"] == "unavailable"
    assert record.completeness["usable"] is False
    assert "neighbours:by_subject" in record.completeness["missing"]


def test_a_vocabulary_that_cannot_be_read_refuses_the_record():
    class _BrokenContext(dict):
        def get(self, key, default=None):
            if key == "vocabulary_sections":
                raise RuntimeError("grammar context unavailable")
            return super().get(key, default)

    record = build_name_case_file(
        "elongation_of_flux_surface",
        gc=_FakeGraph(),
        compose_context=_BrokenContext({"broken": True}),
        family_reader=_family,
        neighbour_reader=_neighbours,
        locus_reader=_locus,
    )
    assert record.completeness["usable"] is False
    assert "grammar:vocabulary" in record.completeness["missing"]


def test_separation_names_each_differing_fact():
    text = separation(
        {
            "facts": {"base.token": "temperature", "locus.token": "magnetic_axis"},
            "unit": "eV",
        },
        {
            "facts": {"base.token": "temperature", "locus.token": "boundary"},
            "unit": "eV",
        },
    )
    assert "locus is boundary where this name's is magnetic_axis" in text
    assert "base" not in text

    units = separation(
        {"facts": {"base.token": "elongation"}, "unit": "1"},
        {"facts": {"base.token": "elongation"}, "unit": "m"},
    )
    assert "unit is m where this name's is 1" in units

    identical = separation(
        {"facts": {"base.token": "elongation"}, "unit": "1"},
        {"facts": {"base.token": "elongation"}, "unit": "1"},
    )
    assert identical.startswith("no grammar fact differs")


def test_digest_covers_content_alone():
    record = _build()
    rebuilt = _build()
    assert rebuilt.digest == record.digest

    altered = dict(record.content)
    altered["target"] = dict(record.content["target"], id="another_name")
    assert case_file_digest(altered) != record.digest


def test_assembly_refuses_a_write_statement_before_sending_it():
    assert_read_only(["\n// READ\nMATCH (n) RETURN n"])

    with pytest.raises(ValueError, match="may only read"):
        assert_read_only(["MATCH (n) SET n.x = true RETURN n"])

    graph_that_would_write = _FakeGraph()
    reader = case_file._Reader(graph_that_would_write, [])
    with pytest.raises(ValueError, match="may only read"):
        reader.query("MATCH (n) DETACH DELETE n")
    assert graph_that_would_write.calls == []


def test_assembly_issues_reads_only_against_a_live_shaped_graph():
    graph = _FakeGraph()
    build_name_case_file(
        "elongation_of_flux_surface",
        gc=graph,
        compose_context=_CONTEXT,
        family_reader=_family,
        neighbour_reader=_neighbours,
        locus_reader=_locus,
    )
    assert graph.calls
    assert_read_only([statement for statement, _ in graph.calls])


def test_source_target_states_what_it_has_not():
    record = build_source_case_file(
        ["dd:equilibrium/time_slice/profiles_1d/elongation"],
        gc=_FakeGraph(),  # no CASE_FILE_TARGET row: an unnamed source has no name
        locus_reader=_locus,
    )

    assert record.target["kind"] == "source"
    assert record.content["incumbent"] is None
    assert record.content["adjudicator_reason"] is None
    assert record.content["family"]["status"] == "none"
    assert record.content["neighbours"]["by_subject"]["status"] == "none"
    assert record.content["grammar"]["status"] == "not_a_name"
    assert record.content["neighbours"]["by_locus"]["members"][0]["name"] == (
        "volume_of_flux_surface"
    )
    assert record.completeness["verdict"] == "complete"


def test_a_derived_producer_is_not_read_as_a_missing_dd_source():
    """A composed name's provenance binds no DD text, so it is not a source."""
    target = dict(_TARGET, source_ids=["derived:elongation_of_flux_surface"])
    record = build_name_case_file(
        "elongation_of_flux_surface",
        gc=_FakeGraph(**{"CASE_FILE_TARGET": [target]}),
        compose_context=_CONTEXT,
        family_reader=_family,
        neighbour_reader=_neighbours,
        locus_reader=_locus,
    )

    assert record.content["sources"] == []
    assert record.content["incumbent"]["bound_paths"] == []
    assert record.completeness["missing"] == ["bound_source"]


def test_unknown_withheld_channel_is_refused():
    with pytest.raises(ValueError, match="unknown withheld channel"):
        _build(withhold=["everything"])


def test_rendering_shows_the_enclosing_text_and_the_verdict():
    record = _build()
    rendered = render_case_file(record)
    assert "Equilibrium IDS." in rendered
    assert "Completeness: complete; usable=true" in rendered

    withheld = render_case_file(_build(withhold=["enclosing_structure_text"]))
    assert "Completeness: incomplete; usable=false" in withheld
