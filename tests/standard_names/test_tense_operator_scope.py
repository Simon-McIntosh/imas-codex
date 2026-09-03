"""The tense-consistency guard must respect operator scope.

A temporal operator nested inside ONE operand of a binary construction
(``difference_of``, ``ratio_of``, ...) does not make the whole name a rate:
the derivative already carries the rate, and the outer combination does not
apply a second one. A difference of a heating power and a stored-energy time
derivative is dimensionally a power, matching the DD's own source path for
that difference. See ``test_attachment_guard.py`` for the lexical tense rule
covering everything except this operator-scope distinction.
"""

from imas_codex.standard_names.workers import _is_attachment_consistent


def test_binary_scoped_temporal_operand_accepts_matching_power_path() -> None:
    """A difference whose second operand carries a temporal operator accepts
    a base-quantity power path once the declared units agree."""
    ok, reason = _is_attachment_consistent(
        "core_profiles/global_quantities/heating_and_current_drive_power/total",
        "difference_of_heating_power_and_time_derivative_of_stored_energy",
        dd_unit="W",
        sn_unit="W",
    )
    assert ok, reason


def test_outermost_temporal_operator_still_requires_a_rate_path() -> None:
    """A name whose OUTERMOST operator is itself temporal is still a rate,
    even with matching units on offer — the operator scope is not binary."""
    ok, reason = _is_attachment_consistent(
        "core_profiles/global_quantities/heating_and_current_drive_power/total",
        "time_derivative_of_stored_energy",
        dd_unit="W",
        sn_unit="W",
    )
    assert not ok
    assert "tense mismatch" in reason
