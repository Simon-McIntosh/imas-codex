"""Audits judge a retired operator spelling on the semantics ISN publishes.

Retiring an operator to an advisory alias strips the semantics a live name
depends on: the retired spelling carries no ``semantic_effects``, so an audit
reading the name literally sees a quantity with no operator at all. The unit
audit then quarantines a per-time quantity for holding watts under an energy
base, and the attachment tense rule detaches the very data-dictionary
time-derivative path that justifies the name. Both must resolve the spelling
through the alias the grammar publishes before judging, and neither may admit a
prefix the grammar publishes no alias for.
"""

from __future__ import annotations

import pytest

from imas_codex.standard_names.attachment_audit import _attachment_consistency
from imas_codex.standard_names.audits import (
    _isn_operator_advisory_aliases,
    name_unit_consistency_check,
    resolve_retired_operator_spellings,
)

#: A data-dictionary path whose terminal segment differentiates with respect to
#: time, so the tense rule reads it as a change.
TIME_DERIVATIVE_DD_PATH = "core_profiles/global_quantities/energy_thermal/d_dt"

#: A prefix the grammar registers as neither an operator nor an alias.
UNKNOWN_PREFIX = "propensity"


@pytest.fixture(scope="module")
def temporal_alias() -> tuple[str, str]:
    """One retired spelling of an operator that divides a quantity by time."""
    from imas_standard_names import get_operator_semantics

    aliases = {
        alias: canonical
        for alias, canonical in _isn_operator_advisory_aliases().items()
        if "temporal_change" in get_operator_semantics(canonical)
    }
    if not aliases:
        pytest.skip("installed ISN grammar publishes no temporal-change alias")
    alias = sorted(aliases)[0]
    return alias, aliases[alias]


def test_retired_spelling_resolves_to_the_registered_operator(
    temporal_alias: tuple[str, str],
) -> None:
    """The resolver rewrites the alias as a whole token, never a substring."""
    alias, canonical = temporal_alias
    assert resolve_retired_operator_spellings(f"{alias}_of_plasma_current") == (
        f"{canonical}_of_plasma_current"
    )
    # A word that merely contains the alias is not an operator application.
    assert resolve_retired_operator_spellings(f"anti{alias}_of_plasma_current") == (
        f"anti{alias}_of_plasma_current"
    )


def test_unit_audit_accepts_watts_under_an_energy_base(
    temporal_alias: tuple[str, str],
) -> None:
    """An energy base differentiated in time is dimensionally a power."""
    alias, canonical = temporal_alias
    retired = f"{alias}_of_thermal_plasma_stored_energy"
    registered = f"{canonical}_of_thermal_plasma_stored_energy"

    # The registered spelling is the control: it already passes at watts.
    assert name_unit_consistency_check({"id": registered, "unit": "W"}) == []
    assert name_unit_consistency_check({"id": retired, "unit": "W"}) == []

    # Resolving the spelling does not disarm the audit: the undifferentiated
    # energy base still refuses watts.
    assert name_unit_consistency_check(
        {"id": "thermal_plasma_stored_energy", "unit": "W"}
    )


def test_unit_audit_still_refuses_an_unknown_prefix() -> None:
    """A prefix the grammar publishes no alias for carries no semantics."""
    unknown = f"{UNKNOWN_PREFIX}_of_thermal_plasma_stored_energy"
    assert resolve_retired_operator_spellings(unknown) == unknown
    issues = name_unit_consistency_check({"id": unknown, "unit": "W"})
    assert issues, "an unrecognised prefix must not license watts under energy"
    assert all("name_unit_consistency_check" in issue for issue in issues)


def test_tense_rule_accepts_a_time_derivative_path(
    temporal_alias: tuple[str, str],
) -> None:
    """The differentiating source justifies the name under either spelling."""
    alias, canonical = temporal_alias
    registered = f"{canonical}_of_thermal_plasma_stored_energy"
    retired = f"{alias}_of_thermal_plasma_stored_energy"

    assert _attachment_consistency(TIME_DERIVATIVE_DD_PATH, registered)[0] is True
    accepted, reason = _attachment_consistency(TIME_DERIVATIVE_DD_PATH, retired)
    assert accepted is True, reason


def test_tense_rule_still_refuses_a_base_quantity_path(
    temporal_alias: tuple[str, str],
) -> None:
    """Resolving the spelling keeps the tense rule symmetric, not permissive."""
    alias, _canonical = temporal_alias
    retired = f"{alias}_of_thermal_plasma_stored_energy"
    base_path = "core_profiles/global_quantities/energy_thermal"

    accepted, reason = _attachment_consistency(base_path, retired)
    assert accepted is False
    assert "tense mismatch" in reason


def test_tense_rule_still_refuses_an_unknown_prefix() -> None:
    """An unresolvable prefix leaves the name reading as a base quantity."""
    unknown = f"{UNKNOWN_PREFIX}_of_thermal_plasma_stored_energy"
    accepted, reason = _attachment_consistency(TIME_DERIVATIVE_DD_PATH, unknown)
    assert accepted is False
    assert "tense mismatch" in reason
