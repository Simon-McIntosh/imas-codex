"""The stored assignment: source extraction, change detection and read-back shape.

The graph is replaced by a recording double, so these tests exercise the write
path's own decisions: which signal changed, what a repeated assignment does, and
what the receipt counts. Invented signals only.
"""

from __future__ import annotations

import json

import pytest

from iter_codex.discovery.signals.vocabulary_graph import (
    VOCABULARY_KEYS,
    archive_unit,
    assign_cohort,
    unit_agreement,
)

# One invented archive document in the shape the archive stores: a ``full`` list
# whose ``Units`` field is the archive's own unit.
ARCHIVE_DOCUMENT = json.dumps(
    {
        "full": [
            {"description": "", "name": "Units", "units": "", "value": "bar"},
            {"description": "", "name": "Format", "units": "", "value": "2"},
        ],
        "type": {"datatypes": []},
    }
)


def _signal(signal_id: str, **properties):
    signal = {
        "id": signal_id,
        "field_code": None,
        "unit": None,
        "description": None,
        "description_source": None,
        "archive_metadata": None,
    }
    signal.update(properties)
    return signal


class TestArchiveUnit:
    def test_the_units_field_supplies_the_unit(self):
        assert archive_unit(ARCHIVE_DOCUMENT) == "bar"

    def test_a_document_without_a_unit_supplies_none(self):
        assert archive_unit(json.dumps({"full": []})) is None
        assert archive_unit(None) is None

    def test_a_malformed_document_supplies_none_rather_than_raising(self):
        assert archive_unit("{not json") is None

    def test_an_empty_units_value_is_not_a_unit(self):
        document = json.dumps({"full": []})
        assert archive_unit(document) is None


class TestAssignCohort:
    def test_only_a_stored_value_that_differs_counts_as_a_change(self):
        signals = [
            _signal("inv:a", unit="bar", description_source="controlsmetadata:/feed"),
            _signal("inv:b", unit="K", description_source="controlsmetadata:/feed"),
        ]
        first = assign_cohort(signals, [])
        assert first["changed"] == 2

        # Write what the first assignment decided, then assign again.
        stored = []
        for record in first["ledger"]:
            row = {key: record[key] for key in VOCABULARY_KEYS}
            stored.append(
                _signal(
                    record["id"],
                    unit="bar" if record["id"] == "inv:a" else "K",
                    description_source="controlsmetadata:/feed",
                    **row,
                )
            )
        second = assign_cohort(stored, [])
        assert second["changed"] == 0
        assert second["pending"] == []

    def test_a_stale_member_left_by_an_earlier_assignment_is_cleared(self):
        signals = [
            _signal(
                "inv:c",
                vocabulary_quantity="flow",
                vocabulary_quantity_source="field_pattern",
                vocabulary_quantity_state="decided",
            )
        ]
        result = assign_cohort(signals, [])
        assert result["changed"] == 1
        assert "vocabulary_quantity" not in result["pending"][0]
        assert result["pending"][0]["vocabulary_quantity_state"] == "undecided"

    def test_a_controls_description_counts_only_when_the_feed_is_credited(self):
        """A description another writer wrote is not controls metadata."""
        signals = [
            _signal("inv:d", description="Water temperature", description_source="llm"),
        ]
        result = assign_cohort(signals, [])
        assert result["ledger"][0]["vocabulary_quantity_state"] == "undecided"

    def test_a_linked_row_supplies_the_row_text(self):
        signals = [_signal("inv:e")]
        rows = [
            {
                "id": "inv:e",
                "description": "Signal purpose: Absolute pressure",
                "field_code": "AA",
                "match_kind": "derived",
            }
        ]
        result = assign_cohort(signals, rows)
        record = result["ledger"][0]
        assert record["vocabulary_quantity"] == "pressure"
        assert record["vocabulary_quantity_source"] == "signal_list_row"

    def test_a_row_states_no_unit_because_the_layout_has_no_unit_column(self):
        """The formatted row carries every cell, and a unit-shaped cell is not a unit.

        The source layout declares a control, category, sequence, field and plant
        column and no unit column, so a row contributes text for the quantity and
        role axes only; a unit word inside a cell, such as a signal type, does not
        become the measured unit.
        """
        signals = [_signal("inv:u", field_code="AA")]
        rows = [
            {
                "id": "inv:u",
                "description": "Absolute pressure; signal type 2 wires 4-20 mA; supply 24 V",
                "field_code": "AA",
                "match_kind": "exact",
            }
        ]
        record = assign_cohort(signals, rows)["ledger"][0]
        assert record["vocabulary_quantity"] == "pressure"
        assert record["vocabulary_quantity_source"] == "signal_list_row"
        assert record["vocabulary_unit"] is None
        assert record["vocabulary_unit_state"] == "undecided"

    def test_a_sibling_channel_row_does_not_state_this_signal(self):
        """Several rows for one component describe their own channels only."""
        signals = [_signal("inv:sib", field_code="FL")]
        rows = [
            {
                "id": "inv:sib",
                "description": "Heater current measurement",
                "field_code": "IT",
                "match_kind": "derived",
            },
            {
                "id": "inv:sib",
                "description": "Heater fault state",
                "field_code": "FL",
                "match_kind": "derived",
            },
        ]
        result = assign_cohort(signals, rows)
        assert result["ledger"][0]["vocabulary_quantity_state"] == "undecided"

    def test_the_row_of_this_signals_own_field_code_states_it(self):
        signals = [_signal("inv:own", field_code="IT")]
        rows = [
            {
                "id": "inv:own",
                "description": "Heater current measurement",
                "field_code": "IT",
                "match_kind": "derived",
            },
            {
                "id": "inv:own",
                "description": "Heater fault state",
                "field_code": "FL",
                "match_kind": "derived",
            },
        ]
        result = assign_cohort(signals, rows)
        record = result["ledger"][0]
        assert record["vocabulary_quantity"] == "current"
        assert record["vocabulary_quantity_source"] == "signal_list_row"

    def test_an_exact_row_states_the_signal_whatever_its_field_code(self):
        signals = [_signal("inv:exact", field_code="XI")]
        rows = [
            {
                "id": "inv:exact",
                "description": "Signal purpose: Absolute pressure",
                "field_code": "PT",
                "match_kind": "exact",
            },
            {
                "id": "inv:exact",
                "description": "Sibling temperature channel",
                "field_code": "TT",
                "match_kind": "derived",
            },
        ]
        result = assign_cohort(signals, rows)
        record = result["ledger"][0]
        assert record["vocabulary_quantity"] == "pressure"
        assert record["vocabulary_quantity_state"] == "decided"


class TestUnitAgreement:
    def test_each_source_spelling_is_compared_with_the_stored_member(self):
        signals = [
            _signal(
                "inv:f",
                unit="KA",
                description_source="controlsmetadata:/feed",
                archive_metadata=json.dumps(
                    {"full": [{"name": "Units", "value": "kA"}]}
                ),
            )
        ]
        result = assign_cohort(signals, [])
        assert result["ledger"][0]["vocabulary_unit"] == "kiloampere"
        agreement = unit_agreement(signals, result["ledger"])
        assert agreement["signals_compared"] == 1
        assert agreement["source_units_compared"] == 2
        assert agreement["disagreements"] == 0

    def test_a_conflicted_unit_is_not_compared(self):
        """A conflict stores no member, so there is nothing to agree with."""
        signals = [
            _signal(
                "inv:g",
                unit="K",
                description_source="controlsmetadata:/feed",
                archive_metadata=json.dumps(
                    {"full": [{"name": "Units", "value": "bar"}]}
                ),
            )
        ]
        result = assign_cohort(signals, [])
        assert result["ledger"][0]["vocabulary_unit_state"] == "conflict"
        assert unit_agreement(signals, result["ledger"])["signals_compared"] == 0


def test_every_stored_key_is_a_declared_property():
    """The nine keys the write sets are the nine the schema generates."""
    from iter_codex.graph.models import FacilitySignal

    fields = set(FacilitySignal.model_fields)
    assert set(VOCABULARY_KEYS) <= fields
    assert len(VOCABULARY_KEYS) == 9


@pytest.mark.parametrize("bad", ["", "   "])
def test_an_archive_unit_of_whitespace_is_not_a_unit(bad):
    document = json.dumps({"full": [{"name": "Units", "value": bad}]})
    assert archive_unit(document) is None


class TestCatalogueAssignment:
    """Plant-wide assignment: the table's role and the pilot comparison."""

    def test_the_table_decides_a_newly_decided_pilot_role(self):
        from iter_codex.discovery.signals.field_codes import (
            FieldCodeEntry,
            FieldCodeTable,
        )

        table = FieldCodeTable(
            entries=(
                FieldCodeEntry(
                    code="QX",
                    component_kind=None,
                    role="measurement",
                    meaning="an invented reading",
                    source="controls_metadata",
                    support=1,
                ),
            )
        )
        signals = [_signal("inv:new", field_code="QX", plant_context_state="derived")]
        result = assign_cohort(signals, [], table=table)
        record = result["ledger"][0]
        assert record["vocabulary_role"] == "measurement"
        assert record["vocabulary_role_source"] == "field_pattern"

    def test_a_pilot_value_the_table_leaves_open_is_reported_as_changed(self):
        from iter_codex.discovery.signals.vocabulary_graph import pilot_comparison

        before = [
            _signal(
                "inv:lost",
                vocabulary_role="state",
                vocabulary_role_source="field_pattern",
                vocabulary_role_state="decided",
            )
        ]
        ledger = assign_cohort([_signal("inv:lost")], [], table=_empty_table())[
            "ledger"
        ]
        report = pilot_comparison(before, ledger)
        assert report["role"]["changed"] == 1
        assert report["role"]["changed_items"][0]["before"] == "state"
        assert report["role"]["changed_items"][0]["after"] is None

    def test_an_unchanged_pilot_value_is_counted_unchanged(self):
        from iter_codex.discovery.signals.vocabulary_graph import pilot_comparison

        before = [
            _signal(
                "inv:same",
                vocabulary_quantity="pressure",
                vocabulary_quantity_state="decided",
            )
        ]
        ledger = assign_cohort(
            [_signal("inv:same", description="invented pressure")],
            [{"id": "inv:same", "description": "invented pressure"}],
        )["ledger"]
        report = pilot_comparison(before, ledger)
        assert report["quantity"]["unchanged"] == 1
        assert report["quantity"]["changed"] == 0

    def test_a_hand_changed_pilot_value_is_caught_as_a_change(self):
        """A stored value that contradicts its own source cannot pass as unchanged."""
        from iter_codex.discovery.signals.vocabulary_graph import pilot_comparison

        before = [
            _signal(
                "inv:contradicted",
                field_code="QX",
                description="invented reading of the flow",
                vocabulary_role="alarm",
                vocabulary_role_source="controls_metadata",
                vocabulary_role_state="decided",
            )
        ]
        ledger = assign_cohort(
            [
                _signal(
                    "inv:contradicted",
                    field_code="QX",
                    description="invented reading of the flow",
                )
            ],
            [{"id": "inv:contradicted", "description": "invented reading of the flow"}],
            table=_empty_table(),
        )["ledger"]
        report = pilot_comparison(before, ledger)
        assert report["role"]["changed"] == 1
        assert report["role"]["changed_items"][0]["before"] == "alarm"
        assert report["role"]["changed_items"][0]["after"] == "measurement"

    def test_a_newly_decided_role_is_listed_with_its_source(self):
        from iter_codex.discovery.signals.field_codes import (
            FieldCodeEntry,
            FieldCodeTable,
        )
        from iter_codex.discovery.signals.vocabulary_graph import newly_decided

        table = FieldCodeTable(
            entries=(
                FieldCodeEntry(
                    code="QX",
                    component_kind="MF",
                    role="setpoint",
                    meaning="an invented threshold",
                    source="signal_list_row",
                    support=1,
                ),
            )
        )
        before = [_signal("inv:sp", field_code="QX", component_category="MF")]
        ledger = assign_cohort(before, [], table=table)["ledger"]
        items = newly_decided(before, ledger, "role")
        assert [(item["id"], item["value"], item["source"]) for item in items] == [
            ("inv:sp", "setpoint", "field_pattern")
        ]


def _empty_table():
    from iter_codex.discovery.signals.field_codes import FieldCodeTable

    return FieldCodeTable()


class TestCatalogueSnapshot:
    def test_a_snapshot_shorter_than_the_control_count_is_refused(self):
        from iter_codex.discovery.signals.vocabulary_graph import snapshot_catalogue

        class _ShortGraph:
            def query(self, statement, **parameters):
                if "count(s) AS total" in statement:
                    return [{"total": 3, "coded": 3}]
                if "LIMIT $limit" in statement:
                    return [{"id": "inv:one", "field_code": "QX"}]
                return []

        with pytest.raises(RuntimeError, match="every catalogued signal"):
            snapshot_catalogue(_ShortGraph(), "inv")

    def test_an_empty_catalogue_is_refused(self):
        from iter_codex.discovery.signals.vocabulary_graph import snapshot_catalogue

        class _EmptyGraph:
            def query(self, statement, **parameters):
                return [{"total": 0, "coded": 0}]

        with pytest.raises(RuntimeError, match="sees no signal"):
            snapshot_catalogue(_EmptyGraph(), "inv")
