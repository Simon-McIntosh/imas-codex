"""Field-code table: a role and meaning only where a source states one.

Every code, component kind and source text here is invented. The behaviours a
reader depends on: a code no source explains has no entry and is counted; a code
whose two sources disagree has no entry and a recorded conflict; a code's role
may differ by component kind, and a kind no entry covers resolves to nothing; an
entry needs both a stated role and a code-and-kind meaning; a rebuild changes
nothing.
"""

from __future__ import annotations

import pytest

from iter_codex.discovery.signals.field_codes import (
    TABLE_SOURCES,
    FieldCodeRules,
    FieldCodeStatement,
    build_field_code_table,
    declared_role_terms,
    role_stated,
    statements_from_texts,
)
from iter_codex.discovery.signals.field_codes_graph import meanings_from_config

SIGNAL_ROW = TABLE_SOURCES[0]
CONTROLS = TABLE_SOURCES[1]
ROW_TEXT = "Signal purpose: Absolute Pressure transducer at the cold box outlet"
CONTROLS_TEXT = "AHU supply fan status"


def _statement(
    code: str, kind: str | None, role: str, source: str = SIGNAL_ROW
) -> FieldCodeStatement:
    return FieldCodeStatement(
        code=code, component_kind=kind, role=role, source=source, evidence="invented"
    )


class TestARoleIsReadFromWording:
    def test_an_instrument_noun_states_a_measurement(self):
        assert role_stated("a pressure transducer on the outlet") == "measurement"

    def test_a_request_states_a_command(self):
        assert role_stated("fast discharge request") == "command"

    def test_a_threshold_states_a_setpoint(self):
        assert role_stated("maintain condition threshold") == "setpoint"

    def test_wording_naming_two_members_states_neither(self):
        assert role_stated("valve request status") is None

    def test_wording_naming_no_member_states_no_role(self):
        assert role_stated("panel number four") is None

    def test_the_rule_set_refuses_a_role_outside_the_vocabulary(self):
        with pytest.raises(ValueError, match="outside the vocabulary"):
            FieldCodeRules(role_terms={"estimate": ("estimated",)})

    def test_the_declared_terms_keep_the_assignment_terms_and_add_source_wording(self):
        terms = declared_role_terms()
        assert "transducer" in terms["measurement"]
        assert all(isinstance(item, tuple) for item in terms.values())


class TestAnEntryNeedsBothARoleAndAMeaning:
    def test_a_code_with_no_statement_has_no_entry_and_is_counted(self):
        table = build_field_code_table([], {}, codes_in_catalogue=["QX"])
        assert table.entries == ()
        assert [item["code"] for item in table.unestablished] == ["QX"]
        assert table.unestablished[0]["reason"] == "no source states a role"

    def test_a_stated_role_with_no_declared_meaning_has_no_entry(self):
        table = build_field_code_table(
            [_statement("QX", "MF", "measurement")], {}, codes_in_catalogue=["QX"]
        )
        assert table.entries == ()
        assert table.unestablished[0]["reason"] == "role stated but no meaning declared"

    def test_a_role_and_a_meaning_make_one_entry(self):
        table = build_field_code_table(
            [_statement("QX", "MF", "measurement")],
            {("QX", "MF"): "a pressure reading"},
        )
        assert [
            (e.code, e.component_kind, e.role, e.meaning) for e in table.entries
        ] == [("QX", "MF", "measurement", "a pressure reading")]


class TestRoleByComponentKind:
    def test_one_code_may_be_a_command_on_one_kind_and_a_state_on_another(self):
        table = build_field_code_table(
            [
                _statement("ZZ", "VC", "command"),
                _statement("ZZ", "CU", "state", CONTROLS),
            ],
            {("ZZ", "VC"): "the close command", ("ZZ", "CU"): "the closed condition"},
        )
        assert table.resolve("ZZ", "VC") == "command"
        assert table.resolve("ZZ", "CU") == "state"

    def test_a_signal_of_a_third_kind_resolves_to_no_role(self):
        table = build_field_code_table(
            [_statement("ZZ", "VC", "command")], {("ZZ", "VC"): "the close command"}
        )
        assert table.resolve("ZZ", "VC") == "command"
        assert table.resolve("ZZ", "MT") is None
        assert table.resolve("ZZ") is None

    def test_entries_are_counted_per_kind_and_per_code(self):
        table = build_field_code_table(
            [
                _statement("ZZ", "VC", "command"),
                _statement("ZZ", "CU", "state", CONTROLS),
            ],
            {("ZZ", "VC"): "the close command", ("ZZ", "CU"): "the closed condition"},
        )
        summary = table.summary()
        assert summary["entries"] == 2
        assert summary["codes_with_entry"] == 1
        assert summary["entries_for_one_kind"] == 2


class TestSourcesThatDisagreeLeaveNoEntry:
    def test_two_sources_giving_one_code_different_roles_conflict(self):
        table = build_field_code_table(
            [
                _statement("QQ", "MF", "measurement"),
                _statement("QQ", "MF", "state", CONTROLS),
            ],
            {("QQ", "MF"): "something"},
        )
        assert table.entries == ()
        assert len(table.conflicts) == 1
        assert table.conflicts[0].roles == ("measurement", "state")
        assert table.resolve("QQ", "MF") is None

    def test_one_source_stating_two_roles_for_a_key_conflicts(self):
        table = build_field_code_table(
            [
                _statement("QQ", "MF", "measurement"),
                _statement("QQ", "MF", "state"),
            ],
            {("QQ", "MF"): "something"},
        )
        assert table.entries == ()
        assert table.conflicts[0].sources == (SIGNAL_ROW,)

    def test_a_conflict_on_one_kind_leaves_another_kind_established(self):
        table = build_field_code_table(
            [
                _statement("QQ", "MF", "measurement"),
                _statement("QQ", "MF", "state", CONTROLS),
                _statement("QQ", "CU", "state", CONTROLS),
            ],
            {("QQ", "MF"): "something", ("QQ", "CU"): "a condition"},
        )
        assert [entry.component_kind for entry in table.entries] == ["CU"]
        assert table.resolve("QQ", "MF") is None
        assert table.resolve("QQ", "CU") == "state"


class TestPrecedenceAndFallback:
    def test_agreement_is_credited_to_the_first_source_in_precedence(self):
        table = build_field_code_table(
            [
                _statement("WZ", "MF", "measurement", CONTROLS),
                _statement("WZ", "MF", "measurement"),
            ],
            {("WZ", "MF"): "a reading"},
        )
        assert table.entries[0].source == SIGNAL_ROW
        assert table.entries[0].support == 2

    def test_a_meaning_declared_without_a_kind_serves_a_kind_the_source_omits(self):
        table = build_field_code_table(
            [_statement("WZ", None, "measurement")], {("WZ", None): "a reading"}
        )
        assert table.entries[0].component_kind is None
        assert table.resolve("WZ", "MP") == "measurement"

    def test_a_meaning_declared_without_a_kind_serves_a_named_kind(self):
        table = build_field_code_table(
            [_statement("WZ", "MF", "measurement")], {("WZ", None): "a reading"}
        )
        assert table.entries[0].component_kind == "MF"
        assert table.resolve("WZ", "MF") == "measurement"


class TestCountsAndStability:
    def test_entries_and_non_entries_sum_to_the_distinct_codes(self):
        signals_by_code = {"QX": 2, "ZZ": 3, "WZ": 1, "UN": 5}
        table = build_field_code_table(
            [_statement("QX", "MF", "measurement"), _statement("ZZ", "VC", "command")],
            {("QX", "MF"): "a reading", ("ZZ", "VC"): "a command"},
            codes_in_catalogue=signals_by_code,
        )
        summary = table.summary(signals_by_code)
        counted_without = {item["code"] for item in table.unestablished} | {
            item.code for item in table.conflicts
        }
        assert summary["entries"] == 2
        assert summary["codes_with_entry"] + len(counted_without) == len(
            signals_by_code
        )

    def test_the_summary_counts_entries_by_source_and_role(self):
        table = build_field_code_table(
            [
                _statement("QX", "MF", "measurement"),
                _statement("ZZ", "VC", "command", CONTROLS),
            ],
            {("QX", "MF"): "a reading", ("ZZ", "VC"): "a command"},
            codes_in_catalogue=["QX", "ZZ"],
        )
        summary = table.summary()
        assert summary["entries_by_source"][SIGNAL_ROW] == 1
        assert summary["entries_by_source"][CONTROLS] == 1
        assert summary["entries_by_role"]["measurement"] == 1
        assert summary["entries_by_role"]["command"] == 1

    def test_a_rebuild_changes_nothing(self):
        statements = [
            _statement("QX", "MF", "measurement"),
            _statement("ZZ", "VC", "command", CONTROLS),
        ]
        meanings = {("QX", "MF"): "a reading", ("ZZ", "VC"): "a command"}
        first = build_field_code_table(statements, meanings, ["QX", "ZZ", "UN"])
        second = build_field_code_table(statements, meanings, ["QX", "ZZ", "UN"])
        assert [e.as_dict() for e in first.entries] == [
            e.as_dict() for e in second.entries
        ]
        assert first.unestablished == second.unestablished


class TestStatementReading:
    def test_a_text_that_states_no_role_yields_no_statement(self):
        assert statements_from_texts("QX", "MF", SIGNAL_ROW, ["panel four"]) == []

    def test_a_code_and_kind_are_carried_with_the_evidence(self):
        (statement,) = statements_from_texts("QX", "MF", SIGNAL_ROW, [ROW_TEXT])
        assert (statement.code, statement.component_kind, statement.role) == (
            "QX",
            "MF",
            "measurement",
        )
        assert statement.evidence == ROW_TEXT

    def test_controls_wording_states_a_state(self):
        (statement,) = statements_from_texts("QX", "FAN", CONTROLS, [CONTROLS_TEXT])
        assert statement.role == "state"
        assert statement.source == CONTROLS


class TestDeclaredMeaningsFromConfiguration:
    def test_a_declared_meaning_is_keyed_by_code_and_kind(self):
        meanings = meanings_from_config(
            [{"code": "QX", "component_kind": "MF", "meaning": "a reading"}]
        )
        assert meanings == {("QX", "MF"): "a reading"}

    def test_a_meaning_without_a_code_is_refused(self):
        with pytest.raises(ValueError, match="no code"):
            meanings_from_config([{"code": " ", "meaning": "a reading"}])

    def test_an_empty_meaning_is_refused(self):
        with pytest.raises(ValueError, match="is empty"):
            meanings_from_config([{"code": "QX", "meaning": "  "}])

    def test_a_repeated_key_is_refused(self):
        with pytest.raises(ValueError, match="repeated"):
            meanings_from_config(
                [
                    {"code": "QX", "component_kind": "MF", "meaning": "one"},
                    {"code": "QX", "component_kind": "MF", "meaning": "two"},
                ]
            )
