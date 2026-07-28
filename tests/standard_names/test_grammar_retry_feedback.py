"""A failed candidate is told which token and which slot, not just that it failed.

ISN's ``UnknownBaseTokenError`` carries the offending token and its segment, so a
round-trip failure is diagnosable. Reporting only the composed name gives the
composer nothing to change and it re-proposes the same token on the next attempt;
the recorded gap population shows the same compounds offered repeatedly with an
empty reason.

These tests pin the diagnosis reaching both the retry directive and the stored
gap record.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.workers import (
    _build_grammar_retry_reason,
    _grammar_round_trip_failures,
)


def _isn_available() -> bool:
    try:
        import imas_standard_names.grammar  # noqa: F401
    except ImportError:
        return False
    return True


requires_isn = pytest.mark.skipif(
    not _isn_available(), reason="imas-standard-names not installed"
)


class _Candidate:
    """Minimal stand-in for a compose candidate."""

    def __init__(self, name: str, source_id: str = "a/b/c") -> None:
        self._name = name
        self.source_id = source_id

    def compose_name(self) -> str:
        if self._name is None:
            raise ValueError("cannot compose")
        return self._name


class TestRetryReasonWording:
    """The directive leads with advice whenever a failure was diagnosable."""

    def test_advice_is_used_when_present(self):
        reason = _build_grammar_retry_reason(
            ["some_name"],
            ["'square' is a registered OPERATOR, not a qualifier."],
        )
        assert "registered OPERATOR" in reason
        assert "produce a different name" not in reason

    def test_falls_back_only_when_nothing_was_diagnosable(self):
        reason = _build_grammar_retry_reason(["some_name"], [])
        assert "produce a different name" in reason
        assert "some_name" in reason

    def test_repeated_advice_is_stated_once(self):
        """Several candidates failing the same way should not repeat the line."""
        line = "'square' is a registered OPERATOR, not a qualifier."
        reason = _build_grammar_retry_reason(["a", "b", "c"], [line, line, line])
        assert reason.count("registered OPERATOR") == 1

    def test_advice_order_is_preserved(self):
        reason = _build_grammar_retry_reason(["a"], ["first advice", "second advice"])
        assert reason.index("first advice") < reason.index("second advice")


@requires_isn
class TestRoundTripDiagnosis:
    """The round-trip check reports what failed AND how to repair it."""

    def test_a_valid_name_produces_no_failure(self):
        names, advice = _grammar_round_trip_failures(
            [_Candidate("electron_temperature")]
        )
        assert names == []
        assert advice == []

    def test_an_unknown_base_is_reported_with_advice(self):
        names, advice = _grammar_round_trip_failures(
            [_Candidate("zzz_unregistered_quantity_xyzzy")]
        )
        assert names == ["zzz_unregistered_quantity_xyzzy"]
        assert advice, "an UnknownBaseTokenError must yield repair advice"
        assert "zzz_unregistered_quantity_xyzzy" in advice[0]

    def test_the_full_token_vocabulary_is_not_inlined(self):
        """The seat's prompt already carries it; repeating it crowds out the point."""
        _names, advice = _grammar_round_trip_failures(
            [_Candidate("zzz_unregistered_quantity_xyzzy")]
        )
        joined = " ".join(advice)
        # A vocabulary dump would be enormous and would name unrelated bases.
        assert len(joined) < 600
        assert "electron_temperature" not in joined

    def test_a_candidate_that_cannot_compose_is_still_reported(self):
        names, _advice = _grammar_round_trip_failures(
            [_Candidate(None, source_id="ids/path/x")]
        )
        assert names == ["ids/path/x"]

    def test_valid_and_failing_candidates_are_separated(self):
        names, advice = _grammar_round_trip_failures(
            [
                _Candidate("electron_temperature"),
                _Candidate("zzz_unregistered_quantity_xyzzy"),
            ]
        )
        assert names == ["zzz_unregistered_quantity_xyzzy"]
        assert len(advice) == 1


@requires_isn
class TestRescuedGapCarriesAReason:
    """A gap rescued from pydantic validation records why, not that."""

    def test_reason_names_the_slot_for_an_operator(self):
        from imas_codex.standard_names.models import _gap_reason

        reason = _gap_reason("qualifier", "square")
        assert "operator" in reason.lower()
        assert "qualifier" in reason

    def test_reason_is_not_a_restatement_of_the_fields(self):
        from imas_codex.standard_names.models import _gap_reason

        reason = _gap_reason("qualifier", "square")
        assert reason != "LLM proposed unregistered qualifier token"

    def test_absent_token_reason_states_the_deficiency(self):
        from imas_codex.standard_names.models import _gap_reason

        reason = _gap_reason("physical_base", "zzz_unknown_xyzzy")
        assert "zzz_unknown_xyzzy" in reason
        assert reason.strip()

    def test_rescued_gap_in_a_batch_has_a_nonempty_reason(self):
        from imas_codex.standard_names.models import StandardNameComposeBatch

        batch = StandardNameComposeBatch.model_validate(
            {
                "candidates": [
                    {
                        "source_id": "p/b",
                        "segments": {
                            "base_token": "torque",
                            "base_kind": "quantity",
                            "qualifiers": ["zzz_unregistered_xyzzy"],
                        },
                        "description": "Torque",
                        "reason": "test",
                    }
                ]
            }
        )
        assert len(batch.vocab_gaps) == 1
        gap_reason = batch.vocab_gaps[0].reason
        assert gap_reason.strip()
        assert "zzz_unregistered_xyzzy" in gap_reason
