"""The stored assignment: source extraction, change detection and read-back shape.

The graph is replaced by a recording double, so these tests exercise the write
path's own decisions: which signal changed, what a repeated assignment does, and
what the receipt counts. Invented signals only.
"""

from __future__ import annotations

import json

import pytest

from imas_codex.discovery.signals.vocabulary_graph import (
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
        rows = [{"id": "inv:e", "description": "Signal purpose: Absolute pressure"}]
        result = assign_cohort(signals, rows)
        record = result["ledger"][0]
        assert record["vocabulary_quantity"] == "pressure"
        assert record["vocabulary_quantity_source"] == "signal_list_row"


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
                archive_metadata=json.dumps({"full": [{"name": "Units", "value": "bar"}]}),
            )
        ]
        result = assign_cohort(signals, [])
        assert result["ledger"][0]["vocabulary_unit_state"] == "conflict"
        assert unit_agreement(signals, result["ledger"])["signals_compared"] == 0


def test_every_stored_key_is_a_declared_property():
    """The nine keys the write sets are the nine the schema generates."""
    from imas_codex.graph.models import FacilitySignal

    fields = set(FacilitySignal.model_fields)
    assert set(VOCABULARY_KEYS) <= fields
    assert len(VOCABULARY_KEYS) == 9


@pytest.mark.parametrize("bad", ["", "   "])
def test_an_archive_unit_of_whitespace_is_not_a_unit(bad):
    document = json.dumps({"full": [{"name": "Units", "value": bad}]})
    assert archive_unit(document) is None
