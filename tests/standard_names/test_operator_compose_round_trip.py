"""Round-trip tests for operator joining in GrammarSegments.compose_name.

The compose model emits grammar SEGMENTS (operator_token + operator_kind).
``GrammarSegments.compose_name()`` delegates to the ISN model-layer composer,
which produces the CANONICAL standard name — the one the public ISN parser
accepts and round-trips. The grammar has three operator-join classes:

* bare-prefix transformations (volume_averaged, line_averaged, normalized,
  surface_integrated, per_toroidal_mode, ...) render BARE: ``op_<base>``. The
  public layer REJECTS the ``op_of_<base>`` form for these.
* ``_of_``-prefix transformations (time_derivative, gradient, real_part, ...)
  render with explicit scope: ``op_of_<base>``.
* postfix transformations (magnitude, *_postfix, ...) render as a suffix:
  ``<base>_op``.

The single binding contract is the ROUND-TRIP GATE: every name a composed
candidate produces must satisfy
``compose_standard_name(parse_standard_name(name)) == name``. The public ISN
parser is the oracle here — not any classification logic inside codex.
"""

from __future__ import annotations

import pytest

pytest.importorskip("imas_standard_names")

from imas_standard_names.grammar import (  # noqa: E402
    compose_standard_name,
    parse_standard_name,
)

from imas_codex.standard_names.models import GrammarSegments  # noqa: E402


def _public_round_trips(name: str) -> bool:
    """True iff ``name`` survives the public parse -> compose round-trip."""
    try:
        return compose_standard_name(parse_standard_name(name)) == name
    except Exception:
        return False


def _compose(
    base_token: str,
    operator_token: str,
    operator_kind: str,
    *,
    base_kind: str = "quantity",
    qualifiers: list[str] | None = None,
) -> str:
    seg = GrammarSegments(
        base_token=base_token,
        base_kind=base_kind,
        qualifiers=qualifiers or [],
        operator_token=operator_token,
        operator_kind=operator_kind,
    )
    return seg.compose_name()


# ---------------------------------------------------------------------------
# Class 1: averaging / integrating / per-mode prefixes -> BARE join
# ---------------------------------------------------------------------------

# (operator_token, base_token, qualifiers, expected canonical name)
_BARE_CASES = [
    (
        "volume_averaged",
        "temperature",
        ["electron"],
        "volume_averaged_electron_temperature",
    ),
    ("line_averaged", "density", ["electron"], "line_averaged_electron_density"),
    ("flux_surface_averaged", "pressure", [], "flux_surface_averaged_pressure"),
    ("normalized", "temperature", ["electron"], "normalized_electron_temperature"),
    ("surface_integrated", "pressure", [], "surface_integrated_pressure"),
    ("volume_integrated", "pressure", [], "volume_integrated_pressure"),
    (
        "per_toroidal_mode",
        "temperature",
        ["electron"],
        "per_toroidal_mode_electron_temperature",
    ),
]


@pytest.mark.parametrize("op,base,quals,expected", _BARE_CASES)
def test_bare_prefix_operators_compose_bare(op, base, quals, expected) -> None:
    """Averaging/integrating/per-mode prefixes render bare, never with _of_."""
    produced = _compose(base, op, "unary_prefix", qualifiers=quals)
    assert produced == expected, f"{op}: expected bare {expected!r}, got {produced!r}"
    assert "_of_" not in produced, f"{op} wrongly composed with _of_: {produced!r}"
    assert _public_round_trips(produced), (
        f"{op}: produced {produced!r} does not round-trip through the public parser"
    )


# ---------------------------------------------------------------------------
# Class 2: differential / etc. prefixes -> _of_ join
# ---------------------------------------------------------------------------

_OF_CASES = [
    (
        "time_derivative",
        "temperature",
        ["electron"],
        "time_derivative_of_electron_temperature",
    ),
    ("gradient", "pressure", ["electron"], "gradient_of_electron_pressure"),
    ("radial_derivative", "safety_factor", [], "radial_derivative_of_safety_factor"),
]


@pytest.mark.parametrize("op,base,quals,expected", _OF_CASES)
def test_of_prefix_operators_compose_with_scope(op, base, quals, expected) -> None:
    """Differential-class prefixes render with explicit _of_ scope."""
    produced = _compose(base, op, "unary_prefix", qualifiers=quals)
    assert produced == expected, f"{op}: expected {expected!r}, got {produced!r}"
    assert _public_round_trips(produced), (
        f"{op}: produced {produced!r} does not round-trip through the public parser"
    )


# ---------------------------------------------------------------------------
# Class 3: postfix transformations -> suffix join
# ---------------------------------------------------------------------------

_POSTFIX_CASES = [
    ("magnitude", "magnetic_field", "magnetic_field_magnitude"),
    # Scalar-extraction family — canonical POSTFIX (ISN ≥ rc41), consistent
    # with magnitude. The prefix `_of_` form is rejected; these also combine
    # with a projection (radial_electric_field_amplitude), unlike the old prefix.
    ("real_part", "electric_field", "electric_field_real_part"),
    ("imaginary_part", "electric_field", "electric_field_imaginary_part"),
    ("amplitude", "electric_field", "electric_field_amplitude"),
]


@pytest.mark.parametrize("op,base,expected", _POSTFIX_CASES)
def test_postfix_operators_compose_as_suffix(op, base, expected) -> None:
    produced = _compose(base, op, "unary_postfix")
    assert produced == expected, f"{op}: expected {expected!r}, got {produced!r}"
    assert _public_round_trips(produced), (
        f"{op}: produced {produced!r} does not round-trip through the public parser"
    )


# ---------------------------------------------------------------------------
# Class 4: bare-prefix transformation co-occurring with a projection axis
# fuses into a single compound axis token (normalized + radial ->
# normalized_radial), not the rejected `radial_normalized_` ordering.
# ---------------------------------------------------------------------------


def test_bare_prefix_with_projection_fuses_compound_axis() -> None:
    """`normalized` + projection axis `radial` -> compound axis `normalized_radial`."""
    seg = GrammarSegments(
        base_token="electric_field",
        base_kind="quantity",
        projection_axis="radial",
        projection_shape="component",
        operator_token="normalized",
        operator_kind="unary_prefix",
    )
    produced = seg.compose_name()
    assert produced == "normalized_radial_electric_field", (
        f"expected fused compound axis, got {produced!r}"
    )
    assert _public_round_trips(produced), (
        f"fused {produced!r} does not round-trip through the public parser"
    )


def test_mislabeled_operator_kind_does_not_misroute() -> None:
    """A registered op routes by its registry kind, not the LLM's operator_kind.

    Postfix `magnitude` mislabeled as unary_prefix must still render as a
    suffix, never `magnitude_of_...`.
    """
    seg = GrammarSegments(
        base_token="magnetic_field",
        base_kind="quantity",
        operator_token="magnitude",
        operator_kind="unary_prefix",  # deliberately wrong
    )
    produced = seg.compose_name()
    assert produced == "magnetic_field_magnitude", (
        f"registry kind should win; got {produced!r}"
    )
    assert _public_round_trips(produced)


# ---------------------------------------------------------------------------
# Comprehensive guard: EVERY registered prefix operator, routed through
# compose_name(), must produce a name that round-trips through the public
# parser. This is the round-trip gate applied across the whole operator
# vocabulary — the public parser, not codex's own routing, is the oracle.
# ---------------------------------------------------------------------------


def _registered_prefix_operators() -> list[str]:
    from imas_standard_names import get_grammar_context

    ops = get_grammar_context()["grammar"]["vocabularies"]["operators"]
    return sorted(
        tok for tok, meta in ops.items() if meta.get("kind") == "unary_prefix"
    )


@pytest.mark.parametrize("op", _registered_prefix_operators())
def test_every_prefix_operator_round_trips(op) -> None:
    """compose_name() for any registered unary_prefix op must round-trip."""
    produced = _compose("temperature", op, "unary_prefix", qualifiers=["electron"])
    assert _public_round_trips(produced), (
        f"{op}: compose_name produced {produced!r}, which is not canonical "
        f"(public parse->compose does not return it unchanged)"
    )


# ---------------------------------------------------------------------------
# Composed (nested) operator names — an outer differential/postfix operator
# over an inner bare-prefix transformation — are valid physics and must
# round-trip. codex composes one operator per candidate (GrammarSegments has a
# single operator_token), so these names arise when PARSING catalog/source
# names; the grammar (ISN) must round-trip them. These are the cases a prior
# session dropped instead of fixing the grammar.
# ---------------------------------------------------------------------------

_NESTED_NAMES = [
    "time_derivative_of_volume_averaged_electron_density",
    "gradient_of_normalized_electron_temperature",
    "volume_averaged_electron_density_magnitude",
    "time_derivative_of_volume_averaged_electron_density_at_magnetic_axis",
]


@pytest.mark.parametrize("name", _NESTED_NAMES)
def test_nested_operator_names_round_trip(name) -> None:
    """Composed operator-over-bare-transformation names round-trip via ISN."""
    assert _public_round_trips(name), (
        f"nested name {name!r} does not round-trip — ISN grammar regression"
    )
