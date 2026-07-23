"""Deterministic-failure classification for the refine-name pool.

A refine that proposes an ungrammatical name fails the SAME way every cycle
(the model keeps producing the same output for a given item). Such failures
must be classified as terminal so the item is marked ``exhausted`` rather than
reverted to ``reviewed`` and re-claimed — otherwise it re-charges a paid model
on every pool cycle in an infinite loop.

The classifier keys on the exception message. These tests pin the messages
that MUST be treated as terminal, including the ISN grammar-composition
``ValidationError`` raised when a proposed name carries an invalid operator
token — the failure mode that stalled a WEST batch drain in an infinite
refine loop.
"""

from __future__ import annotations

from imas_codex.standard_names.workers import _is_refine_grammar_failure


def test_isn_invalid_operator_token_is_terminal() -> None:
    """ISN's StandardName validation error for a bad operator token is terminal."""
    msg = (
        "1 validation error for StandardName\n"
        "transformation\n"
        "  Value error, Invalid operator token "
        "'derivative_with_respect_to_poloidal_magnetic_flux_coordinate': "
        "expected a registered operator token or a fused "
        "'<indexed_operator>_<coordinate>' form "
        "(e.g. 'derivative_with_respect_to_radial_coordinate'). "
        "[type=value_error, input_value='...', input_type=str]"
    )
    assert _is_refine_grammar_failure(ValueError(msg)) is True


def test_isn_standardname_validation_error_is_terminal() -> None:
    """Any ISN StandardName grammar/vocab validation failure is deterministic."""
    msg = (
        "1 validation error for StandardName\n"
        "physical_base\n"
        "  Value error, 'not_a_base' is not in the closed physical_base vocabulary"
    )
    assert _is_refine_grammar_failure(ValueError(msg)) is True


def test_existing_markers_still_terminal() -> None:
    """The pre-existing deterministic markers keep classifying as terminal."""
    assert _is_refine_grammar_failure(
        ValueError("Token 'foo' is not a registered qualifier")
    )
    assert _is_refine_grammar_failure(
        ValueError("1 validation error for RefinedName\nname\n  field required")
    )


def test_transient_error_is_not_terminal() -> None:
    """A genuinely unexpected/transient error must NOT be marked exhausted."""
    assert (
        _is_refine_grammar_failure(ConnectionError("connection reset by peer")) is False
    )
    assert _is_refine_grammar_failure(TimeoutError("request timed out")) is False
