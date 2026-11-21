"""Closed-vocabulary assignment: precedence, conflict, ambiguity and patterns.

Every signal is invented. The rules under test are the declared rule set, and
each test names the behaviour a reader depends on: a value is stored only when
a source states it, credited to the first source in precedence; two sources that
disagree store nothing; a pattern states a member only when other signals with
that field code agree unanimously.
"""

from __future__ import annotations

import pytest

from iter_codex.discovery.signals.vocabulary import (
    SignalSources,
    VocabularyRules,
    assign_vocabulary,
    device_token,
    summarise,
    unit_member,
    values_from_text,
    vocabulary_properties,
)

# Invented selectors, field codes and wording: nothing here is a real signal.
ROW = "Signal purpose: Absolute Pressure at the cold box outlet"
ROW_MEASURED = "Signal purpose: Temperature of the shield support"


def _assign(*signals: SignalSources, rules: VocabularyRules | None = None):
    return {record["id"]: record for record in assign_vocabulary(list(signals), rules)}


def _rules(**overrides) -> VocabularyRules:
    return VocabularyRules(**overrides)


class TestPrecedence:
    def test_row_decision_is_credited_to_the_row_not_to_metadata(self):
        """Both sources agree; the row's precedence wins the credit."""
        record = _assign(
            SignalSources(
                "inv:one",
                row_texts=(ROW,),
                controls_text="Absolute pressure transmitter",
                controls_unit="bar",
            )
        )["inv:one"]
        assert record["vocabulary_quantity"] == "pressure"
        assert record["vocabulary_quantity_source"] == "signal_list_row"
        assert record["vocabulary_unit_source"] == "controls_metadata"

    def test_controls_metadata_decides_when_no_row_states_the_quantity(self):
        record = _assign(
            SignalSources("inv:two", controls_text="Raw water temperature H")
        )["inv:two"]
        assert record["vocabulary_quantity"] == "temperature"
        assert record["vocabulary_quantity_source"] == "controls_metadata"

    def test_archive_unit_decides_the_unit_when_nothing_else_does(self):
        record = _assign(SignalSources("inv:three", archive_unit="mbar"))["inv:three"]
        assert record["vocabulary_unit"] == "millibar"
        assert record["vocabulary_unit_source"] == "archive_unit"
        assert record["vocabulary_quantity_state"] == "undecided"

    def test_archive_unit_never_gives_a_quantity_or_a_role(self):
        """The archive source is declared for the unit axis only."""
        record = _assign(SignalSources("inv:four", archive_unit="K"))["inv:four"]
        assert record["vocabulary_quantity"] is None
        assert record["vocabulary_role"] is None

    def test_equivalent_spellings_agree_rather_than_conflict(self):
        """One unit written two ways is one unit, so the unit is decided."""
        record = _assign(
            SignalSources("inv:five", controls_unit="KA", archive_unit="kA")
        )["inv:five"]
        assert record["vocabulary_unit"] == "kiloampere"
        assert record["vocabulary_unit_state"] == "decided"


class TestConflict:
    def test_two_sources_with_different_units_store_no_unit(self):
        record = _assign(
            SignalSources("inv:six", controls_unit="bar", archive_unit="K")
        )["inv:six"]
        assert record["vocabulary_unit"] is None
        assert record["vocabulary_unit_source"] is None
        assert record["vocabulary_unit_state"] == "conflict"

    def test_two_sources_with_different_quantities_store_no_quantity(self):
        record = _assign(
            SignalSources(
                "inv:seven",
                row_texts=(ROW,),
                controls_text="Water temperature at the inlet",
            )
        )["inv:seven"]
        assert record["vocabulary_quantity"] is None
        assert record["vocabulary_quantity_state"] == "conflict"


class TestUndecided:
    def test_a_signal_with_no_source_is_undecided_on_every_axis(self):
        record = _assign(SignalSources("inv:eight", field_code="ZZ"))["inv:eight"]
        for axis in ("quantity", "unit", "role"):
            assert record[f"vocabulary_{axis}"] is None
            assert record[f"vocabulary_{axis}_source"] is None
            assert record[f"vocabulary_{axis}_state"] == "undecided"

    def test_wording_that_names_two_quantities_gives_none(self):
        assert (
            values_from_text(
                ["Pressure and temperature of the valve body"],
                _rules().quantity_matchers(),
            )
            is None
        )

    def test_an_unparseable_unit_gives_no_member(self):
        lookup = _rules().unit_lookup()
        assert unit_member("Second", lookup) is None
        assert unit_member("", lookup) is None
        assert unit_member(None, lookup) is None

    def test_an_identifier_containing_a_term_is_not_a_match(self):
        """``PressureCode`` is one token; the rule matches whole tokens only."""
        assert values_from_text(["PressureCode"], _rules().quantity_matchers()) is None


class TestRole:
    def test_a_stated_role_is_stored(self):
        record = _assign(
            SignalSources("inv:nine", controls_text="Cubicle door closed")
        )["inv:nine"]
        assert record["vocabulary_role"] == "state"

    def test_an_interface_word_does_not_state_a_role(self):
        """ "Analogue Input" names an interface, not a role."""
        record = _assign(
            SignalSources("inv:ten", controls_text="Analogue Input MP0000")
        )["inv:ten"]
        assert record["vocabulary_role"] is None


class TestFieldPattern:
    def _supporting(self, code: str, count: int) -> list[SignalSources]:
        return [
            SignalSources(
                f"inv:{code}:{index}", field_code=code, controls_text=ROW_MEASURED
            )
            for index in range(count)
        ]

    def test_unanimous_support_states_the_member_for_a_member_without_a_source(self):
        signals = [
            *self._supporting("TT", 3),
            SignalSources("inv:target", field_code="TT"),
        ]
        record = _assign(*signals)["inv:target"]
        assert record["vocabulary_quantity"] == "temperature"
        assert record["vocabulary_quantity_source"] == "field_pattern"

    def test_support_below_the_declared_minimum_states_nothing(self):
        signals = [
            *self._supporting("TT", 2),
            SignalSources("inv:target", field_code="TT"),
        ]
        record = _assign(*signals)["inv:target"]
        assert record["vocabulary_quantity_state"] == "undecided"

    def test_one_dissenting_member_suppresses_the_pattern(self):
        signals = [
            *self._supporting("TT", 3),
            SignalSources(
                "inv:dissenter", field_code="TT", controls_text="Current of the bus"
            ),
            SignalSources("inv:target", field_code="TT"),
        ]
        record = _assign(*signals)["inv:target"]
        assert record["vocabulary_quantity_state"] == "undecided"

    def test_a_pattern_does_not_cross_field_codes(self):
        signals = [
            *self._supporting("TT", 4),
            SignalSources("inv:target", field_code="PP"),
        ]
        record = _assign(*signals)["inv:target"]
        assert record["vocabulary_quantity_state"] == "undecided"

    def test_a_signal_never_counts_towards_its_own_pattern(self):
        """Three supporters are not enough when one of them is the signal itself."""
        signals = [
            *self._supporting("TT", 2),
            SignalSources("inv:target", field_code="TT"),
        ]
        record = _assign(*signals)["inv:target"]
        assert record["vocabulary_quantity_state"] == "undecided"

    def test_a_consensus_from_one_family_does_not_reach_another_family(self):
        """Four temperature transmitters do not state a pressure device's quantity."""
        signals = [
            *[
                SignalSources(
                    f"inv:{index}",
                    field_code="MST",
                    device_token="MT",
                    controls_text=ROW_MEASURED,
                )
                for index in range(4)
            ],
            SignalSources("inv:target", field_code="MST", device_token="MP"),
        ]
        record = _assign(*signals)["inv:target"]
        assert record["vocabulary_quantity_state"] == "undecided"

    def test_a_consensus_from_one_family_reaches_a_related_family(self):
        """MP and MPT are one instrument line, so the consensus transfers."""
        signals = [
            *[
                SignalSources(
                    f"inv:{index}",
                    field_code="MST",
                    device_token="MPT",
                    controls_text=ROW_MEASURED,
                )
                for index in range(4)
            ],
            SignalSources("inv:target", field_code="MST", device_token="MP"),
        ]
        record = _assign(*signals)["inv:target"]
        assert record["vocabulary_quantity"] == "temperature"

    def test_a_consensus_across_families_states_the_code_not_one_family(self):
        """Supporters of several device families state the code's own meaning."""
        signals = [
            SignalSources(
                "inv:a",
                field_code="YSTA",
                device_token="FAN",
                controls_text="Fan status",
            ),
            SignalSources(
                "inv:b",
                field_code="YSTA",
                device_token="BJ",
                controls_text="Door closed",
            ),
            SignalSources(
                "inv:c",
                field_code="YSTA",
                device_token="CU",
                controls_text="Power normal",
            ),
            SignalSources("inv:target", field_code="YSTA", device_token="CPC"),
        ]
        record = _assign(*signals)["inv:target"]
        assert record["vocabulary_role"] == "state"
        assert record["vocabulary_role_source"] == "field_pattern"

    def test_a_signal_without_a_device_token_takes_no_single_family_consensus(self):
        signals = [
            *[
                SignalSources(
                    f"inv:{index}",
                    field_code="MST",
                    device_token="MT",
                    controls_text=ROW_MEASURED,
                )
                for index in range(4)
            ],
            SignalSources("inv:target", field_code="MST"),
        ]
        record = _assign(*signals)["inv:target"]
        assert record["vocabulary_quantity_state"] == "undecided"

    def test_a_pattern_disagreement_with_an_own_source_is_a_conflict(self):
        signals = [
            *self._supporting("TT", 3),
            SignalSources(
                "inv:target",
                field_code="TT",
                controls_text="Absolute pressure of the line",
            ),
        ]
        record = _assign(*signals)["inv:target"]
        assert record["vocabulary_quantity_state"] == "conflict"
        assert record["vocabulary_quantity"] is None


class TestRecordShape:
    def test_summary_counts_sum_to_the_cohort_on_every_axis(self):
        records = assign_vocabulary(
            [
                SignalSources("inv:a", row_texts=(ROW,), controls_unit="bar"),
                SignalSources("inv:b"),
                SignalSources("inv:c", controls_unit="bar", archive_unit="K"),
            ]
        )
        summary = summarise(records)
        for axis in ("quantity", "unit", "role"):
            entry = summary[axis]
            assert entry["decided"] + entry["conflict"] + entry["undecided"] == 3
            assert sum(entry["decided_by_source"].values()) == entry["decided"]

    def test_stored_properties_omit_unset_members(self):
        records = _assign(SignalSources("inv:d", controls_unit="bar"))
        properties = vocabulary_properties(records["inv:d"])
        assert properties["vocabulary_unit"] == "bar"
        assert "vocabulary_quantity" not in properties
        assert properties["vocabulary_quantity_state"] == "undecided"

    def test_repeated_signal_identity_is_refused(self):
        with pytest.raises(ValueError, match="repeated"):
            assign_vocabulary([SignalSources("inv:dup"), SignalSources("inv:dup")])

    def test_a_rule_naming_a_member_outside_the_vocabulary_is_refused(self):
        with pytest.raises(ValueError, match="outside the vocabulary"):
            _rules(quantity_terms={"no_such_quantity": ("x",)})

    def test_the_assignment_is_deterministic(self):
        signals = [SignalSources("inv:e", controls_text=ROW)]
        assert assign_vocabulary(signals) == assign_vocabulary(signals)


class TestAccessionToken:
    def test_the_token_is_the_letters_before_the_number(self):
        assert device_token("inv:site:MT1100-TT-XI") == "MT"
        assert device_token("inv:site:MPT8101-PT-XI") == "MPT"
        assert device_token("inv:site:CPC4001-YSTA-XI") == "CPC"

    def test_an_accession_without_a_letter_prefix_has_no_token(self):
        assert device_token("inv:site:1234") is None
        assert device_token(None) is None
