"""Unit handling for IMAS Codex Server."""

import importlib.resources
import logging
from functools import lru_cache
from typing import Any

import pint

logger = logging.getLogger(__name__)


# register UDUNITS unit format with pint (guard against re-import)
def format_unit_simple(
    unit, registry: pint.UnitRegistry, **options: dict[str, Any]
) -> str:
    """Render a pint unit in IMAS dot-exponential notation (``W.m^-3``).

    Symbol ORDER is part of the convention, not cosmetic: a power density is
    written ``W.m^-3``, never ``m^-3.W``. ``unit.items()`` yields pint's internal
    order, which interleaves numerator and denominator factors, so factors are
    grouped here — positive exponents first, then negative — mirroring the
    numerator/denominator reading of the conventional form. The authority for
    units in the canonical vocabulary is
    ``imas_standard_names.canonical_unit``; this formatter serves the fallback
    path for units that vocabulary does not cover.
    """

    def _fmt_exp(p: float | int) -> str:
        # Coerce integer-valued floats (e.g. -1.0) to int to avoid 'ohm^-1.0'
        ip = int(p)
        return str(ip) if ip == p else str(p)

    # Stable sort: numerator factors (exponent > 0) precede denominators, and
    # each group keeps pint's relative order.
    ordered = sorted(unit.items(), key=lambda item: item[1] < 0)
    return ".".join(u if p == 1 else f"{u}^{_fmt_exp(p)}" for u, p in ordered)


if "U" not in pint.formatting.REGISTERED_FORMATTERS:
    pint.register_unit_format("U")(format_unit_simple)


# Initialize unit registry
unit_registry = pint.UnitRegistry()

# Load non-SI Data Dictionary unit aliases
with importlib.resources.as_file(
    importlib.resources.files("imas_codex.units").joinpath(
        "data_dictionary_unit_aliases.txt"
    )
) as resource_path:
    unit_registry.load_definitions(str(resource_path))


# Dimensionless markers. The IMAS DD writes "-" for a dimensionless quantity
# (imas-python returns it verbatim); "1" and "dimensionless" are equivalent
# spellings. These are a REAL unit — the canonical dimensionless unit "1" — not
# the absence of one, so they must normalize to "1" (the standard-name side and
# the Unit{id:'1'} node already use "1"). Mapping them to None dropped every
# dimensionless quantity's HAS_UNIT edge and desynced it from its standard name.
_DIMENSIONLESS_STRINGS = frozenset({"-", "1", "dimensionless"})

# DD "count" pseudo-units: the DD expresses "a number of things" with a plural
# noun (``electrons`` on summary/gas_injection_*, ``atoms`` on isotope element
# counts, ``events`` ratios). pint cannot parse these, so they were dropped and
# the quantity lost its unit — which also silently discarded the resolved
# ``as_parent`` inheritance for every gas-injection ``value`` field, since the
# unit those inherit from their grandparent IS ``electrons``. A count is the
# dimensionless unit 1.
_COUNT_PSEUDO_UNITS = frozenset({"electrons", "atoms", "events.neutron^-1"})

# Ambiguous DD unit spellings: the SAME string denotes different dimensionalities
# depending on the quantity, so a context-free normaliser must NOT guess.
# "Elementary Charge Unit" is written both on charge numbers (z_ion, z_min,
# z_max, z_n — a charge, correctly 'e') and on ionisation potentials, which are
# ENERGIES the DD carries as 'eV' elsewhere. Typing an energy as a charge is a
# dimensionality error, so this resolves to None here; the quantity-aware DD
# layer assigns it from the same quantity's unambiguous unit elsewhere.
_AMBIGUOUS_UNIT_STRINGS = frozenset({"Elementary Charge Unit"})

# Sentinel strings that are genuinely NOT a unit (no dimensionality to assign).
_NON_UNIT_STRINGS = frozenset(
    {
        "mixed",
        "as parent",
        "as_parent",
        "as_parent_level_2",
        "Toroidal angle",
        "",
    }
)


@lru_cache(maxsize=512)
def normalize_unit_symbol(raw: str) -> str | None:
    """Normalize a unit string to a canonical symbol via pint.

    Returns a dot-exponential notation for graph storage.  Uses the custom
    ``U`` pint formatter which joins base units with ``.`` and appends
    ``^exp`` for non-unity exponents.  Equivalent unit expressions
    (e.g., ``m.s^-1`` and ``m/s``) produce the same output.

    Examples:
        >>> normalize_unit_symbol("Ohm")
        'ohm'
        >>> normalize_unit_symbol("H.m^-1")
        'H.m^-1'
        >>> normalize_unit_symbol("m.s^-1")
        'm.s^-1'
        >>> normalize_unit_symbol("mixed")  # sentinel
        >>> normalize_unit_symbol("A/m^2")
        'A.m^-2'
        >>> normalize_unit_symbol("kg.m.s^-2")
        'kg.m.s^-2'

    Args:
        raw: Raw unit string from MDSplus or IMAS DD.

    Returns:
        Normalized symbol string, or None if unparseable/not a unit.
    """
    if not raw or raw in _NON_UNIT_STRINGS:
        return None
    if raw in _AMBIGUOUS_UNIT_STRINGS:
        return None
    if raw in _DIMENSIONLESS_STRINGS or raw in _COUNT_PSEUDO_UNITS:
        return "1"
    if raw.startswith("units given") or raw.startswith("as_parent"):
        return None

    # Canonicalise through the SAME authority the standard-name side uses, so a
    # unit has ONE spelling across the graph. Symbol order is meaningful — a
    # power density is conventionally 'W.m^-3', not 'm^-3.W' — and the pint
    # formatter orders factors by its own internal sequence, which disagreed
    # with the standard-name canonical form and produced two spellings of one
    # unit (30 Unit nodes were affected).
    try:
        from imas_standard_names import canonical_unit

        return canonical_unit(raw)
    except Exception:
        logger.debug("imas_standard_names could not canonicalise unit '%s'", raw)

    # Fallback: pint, for units the canonical vocabulary does not cover. Keeps
    # a DD-only unit resolvable rather than dropping it.
    try:
        parsed = unit_registry.parse_expression(raw)
        compact = f"{parsed.units:~U}"
        compact = compact.replace("Ω", "ohm")
        return compact
    except Exception:
        logger.debug("Could not normalize unit '%s'", raw)
        return None


def resolve_dd_unit(dd_path: str, raw: str | None) -> str | None:
    """Normalise a DD-declared unit, correcting curated dimensional defects.

    Wraps :func:`normalize_unit_symbol` with a path-aware step so a DD
    declaration that contradicts the quantity's own dimensionality is not
    propagated into the graph or into a standard name.

    The curation lives in ``dd_unit_exceptions.yaml`` — the single registry of
    DD-side unit bugs — so the mismatch axis and the build agree on one list.
    Most entries there are *suppressions*: the DD is wrong, the standard name is
    right, and the DD unit is left as declared so the axis can still report it.
    Only entries flagged ``correct_in_graph`` are rewritten here, reserved for
    the case where the DD contradicts **itself** on one quantity (so there is no
    single DD answer to mirror) and propagating the wrong dimensionality would
    corrupt a standard name.

    Args:
        dd_path: The DD path the unit was declared on (IDS-relative or full).
        raw: The raw unit string from the DD.

    Returns:
        The canonical unit symbol, or None when the string is not a unit.
    """
    if not raw:
        return None
    from imas_codex.units.dd_unit_exceptions import graph_unit_correction

    correction = graph_unit_correction(dd_path, raw)
    if correction is not None:
        logger.debug(
            "resolve_dd_unit: %s declares %r; using %r "
            "(DD declares this quantity inconsistently)",
            dd_path,
            raw,
            correction,
        )
        return normalize_unit_symbol(correction)
    return normalize_unit_symbol(raw)


def validate_unit(unit_str: str) -> str | None:
    """Validate unit string against pint and return canonical short form.

    Used as a post-enrichment validation step: if the LLM-extracted unit
    is invalid, returns None (clear rather than store garbage). If valid,
    returns the pint-canonical short form.

    Args:
        unit_str: Raw unit string from LLM enrichment.

    Returns:
        Canonical short-form unit string, or None if invalid.
    """
    if not unit_str or not unit_str.strip():
        return None
    return normalize_unit_symbol(unit_str.strip())
