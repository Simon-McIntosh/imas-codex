"""Dimensional-sanity audit for standard name units.

Cross-checks the unit field against structural tokens in the standard name.
Returns tagged issue strings when a name's unit is physically implausible
given its semantic role (e.g. a ``_phase`` name must be ``rad`` or ``1``).

All rules are deterministic — no LLM inference.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Unit-equivalence helpers
# ---------------------------------------------------------------------------

# Tesla-equivalent unit strings (SI base or common derived forms).
_TESLA_EQUIVALENTS = frozenset(
    {
        "T",
        "kg.A^-1.s^-2",
        "kg.s^-2.A^-1",
        "kg/(A.s^2)",
        "V.s.m^-2",
        "Wb.m^-2",
    }
)

# Acceptable per-frequency / per-mode spectral denominators.
_SPECTRAL_FACTORS = re.compile(
    r"Hz\^-1|s(?:\^1)?(?![a-zA-Z])|rad\^-1|mode|"
    r"per.mode|per.Hz|per.frequency|per.toroidal|per.poloidal"
)


def _unit_contains_factor(unit: str, factor: str) -> bool:
    """Check if *unit* contains *factor* as a sub-expression."""
    return factor in unit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_unit_sanity(name: str, unit: str) -> list[str]:
    """Return issue tags for dimensional mismatches between *name* and *unit*.

    Each returned string has the format ``"unit_mismatch:<rule_id>"``.

    Rules implemented:

    1. ``_phase`` suffix → unit ∈ {``rad``, ``1``, ``dimensionless``}.
    2. ``wave_magnetic_field`` → unit must be T-equivalent.
    3. ``wave_vector`` → unit must contain ``m^-1``.
    4. ``_spectrum`` suffix → unit must contain a per-frequency or per-mode
       factor (``Hz^-1``, ``s``, ``rad^-1``, or mode-related token).
    5. ``_per_magnetic_field_strength`` or ``_over_magnetic_field_strength``
       → unit must include ``T^-1``.
    """
    issues: list[str] = []
    unit_stripped = (unit or "").strip()
    name_lower = name.lower()

    # Rule 1: phase → rad or 1
    if name_lower.endswith("_phase") or "_phase_" in name_lower:
        acceptable_phase = {"rad", "1", "dimensionless", ""}
        if unit_stripped not in acceptable_phase:
            issues.append("unit_mismatch:phase_must_be_rad")

    # Rule 2: wave magnetic field → T-equivalent
    if "wave_magnetic_field" in name_lower:
        if unit_stripped not in _TESLA_EQUIVALENTS:
            issues.append("unit_mismatch:wave_b_field_must_be_tesla")

    # Rule 3: wave vector → m^-1
    if "wave_vector" in name_lower:
        if not _unit_contains_factor(unit_stripped, "m^-1"):
            issues.append("unit_mismatch:wave_vector_must_be_per_metre")

    # Rule 4: spectrum → per-frequency or per-mode factor
    if name_lower.endswith("_spectrum"):
        # Allow genuinely dimensionless or empty (integer-mode style)
        if unit_stripped not in {"1", "dimensionless", ""}:
            if not _SPECTRAL_FACTORS.search(unit_stripped):
                issues.append("unit_mismatch:spectrum_missing_spectral_denominator")

    # Rule 5: per/over magnetic field strength → must include T^-1
    if (
        "_per_magnetic_field_strength" in name_lower
        or "_over_magnetic_field_strength" in name_lower
    ):
        if not _unit_contains_factor(unit_stripped, "T^-1"):
            issues.append("unit_mismatch:per_b_must_include_inverse_tesla")

    return issues


# ---------------------------------------------------------------------------
# Quantity spelling: which quantity a name states, and what dimension it has
# ---------------------------------------------------------------------------

# Quantities whose head is more than one token, longest first so that
# ``torque_density`` is read before ``density`` and ``torque``.
_COMPOUND_QUANTITY_HEADS: tuple[str, ...] = (
    "torque_density",
    "torque",
    "source_rate",
    "particle_flux",
    "energy_flux",
    "heat_flux",
    "current_density",
    "power_density",
    "energy_density",
    "magnetic_field",
    "temperature",
    "frequency",
    "pressure",
    "rate",
)

# Words that begin a qualifying clause: everything after the first of these
# describes where, when or why the quantity applies, not which quantity it is.
_QUALIFIER_MARKERS: tuple[str, ...] = (
    "_due_to_",
    "_relative_to_",
    "_with_respect_to_",
    "_at_",
    "_of_",
    "_in_",
    "_over_",
    "_for_",
    "_from_",
    "_per_",
)

# Unit spellings the dimension of a quantity admits. Only quantities that carry
# a dimensional commitment are listed; an unlisted quantity is not judged.
QUANTITY_EXPECTED_UNITS: dict[str, tuple[str, ...]] = {
    "torque": ("N.m", "J"),
    "torque_density": ("N.m^-2", "Pa.m^-1"),
    "source_rate": ("s^-1", "Hz"),
    "energy_flux": ("W.m^-2",),
    "particle_flux": ("s^-1", "m^-2.s^-1"),
}


def quantity_of_name(name: str) -> str:
    """Return the quantity a standard name states.

    The head clause is the part of the name before the first qualifying clause,
    and within it the longest registered compound head it ends with. When
    no compound head matches, the head clause itself stands as the quantity, so
    ``ion_torque``, ``toroidal_ion_torque`` and ``explicit_ion_torque`` are one
    quantity while ``electron_density`` and ``ion_density`` stay two: a species
    names a different quantity, and grouping them would report every unit
    difference between species to be a spelling dispute.
    """
    lowered = (name or "").lower()
    head = lowered
    for marker in _QUALIFIER_MARKERS:
        index = lowered.find(marker)
        while index > 0:
            candidate = lowered[:index]
            if candidate:
                head = candidate
                break
            index = lowered.find(marker, index + 1)
        if head is not lowered:
            break
    for compound in _COMPOUND_QUANTITY_HEADS:
        if head == compound or head.endswith("_" + compound):
            return compound
    return head


def unit_dimension(unit: str) -> str | None:
    """Return the Pint dimensionality of *unit*, or None when it does not parse."""
    from imas_codex.units import unit_registry

    stripped = (unit or "").strip()
    if stripped.lower() in {"1", "dimensionless", "-", "none"}:
        return ""
    try:
        return str(unit_registry.Unit(stripped).dimensionality)
    except Exception:
        return None


def units_are_dimensionally_equal(left: str, right: str) -> bool | None:
    """Return whether two unit spellings carry one dimension.

    None means the comparison could not be made because a spelling does not
    parse, so the caller reports a recorded unknown rather than a verdict.
    """
    from imas_codex.units.dd_unit_exceptions import canonical_or_none

    left_canonical = canonical_or_none(left)
    right_canonical = canonical_or_none(right)
    if left_canonical is None or right_canonical is None:
        return None
    left_dimension = unit_dimension(left_canonical)
    right_dimension = unit_dimension(right_canonical)
    if left_dimension is None or right_dimension is None:
        return None
    return left_dimension == right_dimension


def quantity_dimension_issues(name: str, unit: str) -> list[str]:
    """Return issue tags where *unit* does not fit the quantity *name* states."""
    quantity = quantity_of_name(name)
    expected = QUANTITY_EXPECTED_UNITS.get(quantity)
    if not expected or not unit:
        return []
    if any(units_are_dimensionally_equal(unit, candidate) for candidate in expected):
        return []
    return [
        f"unit_mismatch:quantity_dimension:{quantity}",
    ]
