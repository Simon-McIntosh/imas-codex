"""Synthetic source rows establish join boundaries and recovery checks."""

import io
import json

import pytest
from openpyxl import Workbook

from iter_codex.discovery.signals.plant_context import (
    PlantContextBundle,
    SpreadsheetLayout,
    extract_signal_rows,
    match_signal_row,
)
from iter_codex.discovery.signals.plant_context_graph import (
    assert_context_archive_matches,
    context_statements,
    snapshot_pilot_signals,
)
from iter_codex.graph.models import (
    Document,
    PlantContextMatch,
    PlantSignalRow,
)


@pytest.fixture
def document():
    return Document(
        id="synthetic:document:invented",
        facility_id="synthetic",
        source_uid="invented",
        version="draft",
        title="Invented spreadsheet",
        content_role="signal_list",
        sha256="a" * 64,
        status="discovered",
        context_system_code="DEMO-COIL",
        plant_structure=["synthetic:plant:demo"],
    )


def row(**changes):
    return PlantSignalRow(
        **{
            "id": "synthetic:row:invented",
            "facility_id": "synthetic",
            "context_scope": "synthetic:context:demo",
            "source_document": "synthetic:document:invented",
            "sheet": "Invented table",
            "row_number": 2,
            "description": "Invented sensor row",
            "cells_json": "{}",
            "control_path": "DEMO-COIL-CTRL",
            "component_category": "ZZ",
            "component_sequence": "42",
            **changes,
        }
    )


def bundle(rows):
    return PlantContextBundle(
        "synthetic", "synthetic:context:demo", "DEMO-COIL", [], [], [], rows
    )


def test_literal_selector_is_exact_and_component_only_match_is_derived():
    selector = "DEMO-COIL-CTRL:ZZ0042-TT"
    source = row(source_selectors=[selector])
    assert match_signal_row(selector, source, "DEMO-COIL") == PlantContextMatch.exact
    assert (
        match_signal_row("DEMO-COIL-CTRL:ZZ0042-SP", source, "DEMO-COIL")
        == PlantContextMatch.derived
    )


def test_component_absent_from_source_gets_no_link():
    assert (
        match_signal_row("DEMO-COIL-CTRL:ZZ0043-TT", row(), "DEMO-COIL")
        == PlantContextMatch.absent
    )
    b = bundle([row()])
    counts = b.link_signals(
        [{"id": "synthetic:signal:missing", "selector": "DEMO-COIL-CTRL:ZZ0043-TT"}]
    )
    assert counts == {"exact": 0, "derived": 0, "absent": 1}
    assert b.links == []
    assert b.signals[0]["links"] == []


def test_row_naming_different_system_cannot_link_even_with_embedded_selector():
    selector = "DEMO-COIL-CTRL:ZZ0042-TT"
    source = row(control_path="ELSE-COIL-CTRL", source_selectors=[selector])
    assert match_signal_row(selector, source, "DEMO-COIL") == PlantContextMatch.absent
    b = bundle([source])
    assert (
        b.link_signals([{"id": "synthetic:signal:wrong-system", "selector": selector}])[
            "absent"
        ]
        == 1
    )
    assert b.links == []


@pytest.mark.parametrize(
    "changes",
    [
        {"control_path": "DEMO-COIL-OTHER"},
        {"component_category": "YY"},
        {"component_sequence": None},
        {"control_path": "DEMO-COILISH-CTRL"},
        {"component_sequence": "42/43"},
    ],
)
def test_component_join_needs_full_unambiguous_identity(changes):
    assert (
        match_signal_row("DEMO-COIL-CTRL:ZZ0042-TT", row(**changes), "DEMO-COIL")
        == PlantContextMatch.absent
    )


def test_multiple_rows_keep_individual_kinds_and_exact_state_takes_precedence():
    selector = "DEMO-COIL-CTRL:ZZ0042-TT"
    b = bundle([row(), row(id="synthetic:row:explicit", source_selectors=[selector])])
    counts = b.link_signals([{"id": "synthetic:signal:present", "selector": selector}])
    assert counts == {"exact": 1, "derived": 0, "absent": 0}
    assert {link.match_kind for link in b.links} == {
        PlantContextMatch.exact,
        PlantContextMatch.derived,
    }
    assert len({link.id for link in b.links}) == 2


def test_function_variable_can_match_literal_but_never_bare_component():
    selector = "DEMO-COIL-CTRL:HEALTH"
    assert match_signal_row(selector, row(), "DEMO-COIL") == PlantContextMatch.absent
    assert (
        match_signal_row(selector, row(source_selectors=[selector]), "DEMO-COIL")
        == PlantContextMatch.exact
    )


def workbook_bytes(rows):
    wb = Workbook()
    ws = wb.active
    ws.title = "Invented table"
    for cells in rows:
        ws.append(cells)
    stream = io.BytesIO()
    wb.save(stream)
    wb.close()
    return stream.getvalue()


def test_spreadsheet_coordinates_and_text_survive_beyond_preview_limit(document):
    content = workbook_bytes(
        [["Control", "Category", "Number", "Meaning", "Selector"]]
        + [
            [
                "DEMO-COIL-CTRL",
                "ZZ",
                42,
                "Invented temperature",
                "DEMO-COIL-CTRL:ZZ0042-TT",
            ]
            for _ in range(510)
        ]
    )
    rows = extract_signal_rows(
        content,
        document,
        [
            SpreadsheetLayout(
                "Invented table", control_column=1, category_column=2, sequence_column=3
            )
        ],
        scope="synthetic:context:demo",
    )
    assert len(rows) == 510
    assert rows[-1].row_number == 511
    assert len({r.id for r in rows}) == 510
    assert rows[0].source_selectors == ["DEMO-COIL-CTRL:ZZ0042-TT"]
    assert json.loads(rows[0].cells_json)["3"] == 42
    assert "Meaning: Invented temperature" in rows[0].description


def test_spreadsheet_requires_every_sheet_and_valid_coordinates(document):
    content = workbook_bytes([["Name", "Value"], ["Invented", 1]])
    with pytest.raises(ValueError, match="Every worksheet"):
        extract_signal_rows(content, document, [], scope="synthetic:context:demo")
    with pytest.raises(ValueError, match="columns"):
        extract_signal_rows(
            content,
            document,
            [SpreadsheetLayout("Invented table", control_column=9)],
            scope="synthetic:context:demo",
        )


def test_cross_facility_row_is_refused_before_linking():
    with pytest.raises(ValueError, match="facility"):
        bundle([row(facility_id="elsewhere")]).link_signals([])


def test_schema_statements_do_not_create_existing_signals_or_components():
    b = bundle([row()])
    b.link_signals(
        [
            {
                "id": "synthetic:signal:present",
                "selector": "DEMO-COIL-CTRL:ZZ0042-TT",
                "component_id": "synthetic:component:present",
            }
        ]
    )
    queries = "\n".join(q for q, _ in context_statements(b))
    assert "MERGE (n:FacilitySignal" not in queries
    assert "MERGE (n:PlantComponent" not in queries
    assert "MATCH (t:FacilitySignal" in queries
    assert "MATCH (t:PlantComponent" in queries


def test_archive_with_one_imported_node_withheld_fails_count_comparison():
    live = {"nodes": 9, "links": 16, "sha256": "complete"}
    with pytest.raises(ValueError, match="does not match"):
        assert_context_archive_matches(live, {**live, "nodes": 8})
    assert_context_archive_matches(live, live)


def test_archive_equal_counts_do_not_hide_different_identities():
    live = {"nodes": 9, "links": 16, "sha256": "complete"}
    with pytest.raises(ValueError, match="does not match"):
        assert_context_archive_matches(live, {**live, "sha256": "different"})


def test_archive_incomplete_measurement_is_refused():
    with pytest.raises(ValueError, match="complete context measurement"):
        assert_context_archive_matches({"nodes": 9}, {"nodes": 9})


def test_pilot_snapshot_refuses_broken_selector_instrument():
    class EmptyGraph:
        def query(self, *args, **kwargs):
            return [{"total": 3, "selectors": 0}]

    with pytest.raises(RuntimeError, match="instrument"):
        snapshot_pilot_signals(EmptyGraph(), "synthetic", "DEMO-COIL")


def test_curated_source_alias_allows_only_unambiguous_component_join(document):
    document.source_system_codes = ["DEMO-TEST"]
    document.source_binding_note = (
        "Invented specification identifies both system spellings"
    )
    b = bundle([row(control_path="DEMO-TEST-SENS", plant_reference="invented-owner")])
    b.documents = [document]
    counts = b.link_signals(
        [{"id": "synthetic:signal:present", "selector": "DEMO-COIL-CTRL:ZZ0042-TT"}]
    )
    assert counts == {"exact": 0, "derived": 1, "absent": 0}
    assert b.rows[0].control_path == "DEMO-TEST-SENS"
    assert b.links[0].match_kind == PlantContextMatch.derived


def test_source_alias_without_evidence_is_refused(document):
    document.source_system_codes = ["DEMO-TEST"]
    b = bundle([row(control_path="DEMO-TEST-SENS")])
    b.documents = [document]
    with pytest.raises(ValueError, match="supporting evidence"):
        b.link_signals([])


def test_repeated_number_under_different_source_owners_stays_absent():
    b = bundle(
        [
            row(plant_reference="invented-owner-a"),
            row(id="synthetic:row:other", plant_reference="invented-owner-b"),
        ]
    )
    assert (
        b.link_signals(
            [{"id": "synthetic:signal:present", "selector": "DEMO-COIL-CTRL:ZZ0042-TT"}]
        )["absent"]
        == 1
    )
    assert b.links == []


def test_component_number_fallback_cannot_choose_between_archive_components():
    b = bundle([row(control_path="DEMO-COIL-SENS", plant_reference="invented-owner")])
    signals = [
        {"id": "synthetic:signal:a", "selector": "DEMO-COIL-LEFT:ZZ0042-TT"},
        {"id": "synthetic:signal:b", "selector": "DEMO-COIL-RGHT:ZZ0042-TT"},
    ]
    assert b.link_signals(signals)["absent"] == 2
    assert b.links == []


def test_nonconforming_pilot_name_is_counted_absent_without_inferred_component():
    b = bundle([row()])
    assert (
        b.link_signals(
            [
                {
                    "id": "synthetic:signal:unparsed",
                    "selector": "DEMO-COIL-LONGCTRL:ZZ0042-TT",
                }
            ]
        )["absent"]
        == 1
    )
    assert b.links == []


def test_literal_source_identity_does_not_need_a_convention_parse():
    selector = "DEMO-COIL-LONGCTRL:ZZ0042-TT"
    assert (
        match_signal_row(selector, row(source_selectors=[selector]), "DEMO-COIL")
        == PlantContextMatch.exact
    )
