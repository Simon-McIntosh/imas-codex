"""Round-trip tests for ordered operator expressions in GrammarSegments.

The compose model emits an outer-to-inner structured ``operators`` list.
``GrammarSegments.compose_name()`` delegates spelling and validation to public
ISN APIs. The grammar has three operator-join classes:

* bare-prefix transformations (volume_averaged, line_averaged, normalized,
  surface_integrated, per_toroidal_mode, ...) render BARE: ``op_<base>``. The
  public layer REJECTS the ``op_of_<base>`` form for these.
* ``_of_``-prefix transformations (time_derivative, gradient, real_part, ...)
  render with explicit scope: ``op_of_<base>``.
* postfix transformations (magnitude, *_postfix, ...) render as a suffix:
  ``<base>_op``.

The single binding contract is the ROUND-TRIP GATE: every name a composed
candidate produces must satisfy strict lossless parse followed by
``compose(result.ir) == name``. The public ISN parser is the oracle here — not
the lossy flat facade or any classification logic inside codex.
"""

from __future__ import annotations

import pytest

pytest.importorskip("imas_standard_names")

from imas_codex.standard_names.grammar_adapter import (  # noqa: E402
    is_canonical_name,
)
from imas_codex.standard_names.models import GrammarSegments  # noqa: E402


def _public_round_trips(name: str) -> bool:
    """True iff ``name`` survives the lossless public grammar round-trip."""
    return is_canonical_name(name)


def _compose(
    base_token: str,
    operator: str,
    *,
    base_kind: str = "quantity",
    qualifiers: list[str] | None = None,
    coordinate: str | None = None,
) -> str:
    seg = GrammarSegments(
        base_token=base_token,
        base_kind=base_kind,
        qualifiers=qualifiers or [],
        operators=[{"token": operator, "coordinate": coordinate}],
    )
    return seg.compose_name()


# ---------------------------------------------------------------------------
# Class 1: averaging / integrating / per-mode prefixes -> BARE join
# ---------------------------------------------------------------------------

# (operator, base_token, qualifiers, expected canonical name)
_BARE_CASES = [
    (
        "volume_averaged",
        "temperature",
        ["electron"],
        "volume_averaged_electron_temperature",
    ),
    ("line_averaged", "density", ["electron"], "line_averaged_electron_density"),
    (
        "flux_surface_averaged",
        "density",
        ["electron"],
        "flux_surface_averaged_electron_density",
    ),
    ("normalized", "temperature", ["electron"], "normalized_electron_temperature"),
    (
        "surface_integrated",
        "pressure",
        ["electron"],
        "surface_integrated_electron_pressure",
    ),
    (
        "volume_integrated",
        "pressure",
        ["electron"],
        "volume_integrated_electron_pressure",
    ),
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
    produced = _compose(base, op, qualifiers=quals)
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
    produced = _compose(base, op, qualifiers=quals)
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
    produced = _compose(base, op)
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
        operators=[{"token": "normalized"}],
    )
    produced = seg.compose_name()
    assert produced == "normalized_radial_electric_field", (
        f"expected fused compound axis, got {produced!r}"
    )
    assert _public_round_trips(produced), (
        f"fused {produced!r} does not round-trip through the public parser"
    )


def test_projection_and_transformation_follow_public_canonical_order() -> None:
    """A transformation follows a component when no fused component exists."""
    seg = GrammarSegments(
        base_token="flux",
        base_kind="quantity",
        projection_axis="perpendicular",
        qualifiers=["momentum"],
        process_token="e_cross_b_drift",
        operators=[{"token": "normalized"}],
    )

    produced = seg.compose_name()

    assert produced == ("perpendicular_normalized_momentum_flux_due_to_e_cross_b_drift")
    assert _public_round_trips(produced)


def test_operator_schema_does_not_ask_model_for_registry_kind() -> None:
    """The LLM selects a token; the live registry owns its attachment kind."""
    schema = GrammarSegments.model_json_schema()
    operator_schema = schema["$defs"]["GrammarOperator"]["properties"]
    assert "kind" not in operator_schema

    seg = GrammarSegments(
        base_token="magnetic_field",
        base_kind="quantity",
        operators=[{"token": "magnitude"}],
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


def _coord_indexed_prefix_operators() -> set[str]:
    from imas_standard_names import get_grammar_context

    ops = get_grammar_context()["grammar"]["vocabularies"]["operators"]
    return {
        tok
        for tok, meta in ops.items()
        if meta.get("indexed") and meta.get("kind") == "unary_prefix"
    }


@pytest.mark.parametrize("op", _registered_prefix_operators())
def test_every_prefix_operator_round_trips(op) -> None:
    """compose_name() for any registered unary_prefix op must round-trip.

    Coordinate-indexed prefix operators (``derivative_with_respect_to``) bind a
    coordinate via the operator item's ``coordinate``; the bare form is intentionally
    rejected (it would drop the index), so they are tested with a registered
    coordinate carrier.
    """
    coord = (
        "normalized_poloidal_flux_coordinate"
        if op in _coord_indexed_prefix_operators()
        else None
    )
    produced = _compose(
        "temperature",
        op,
        qualifiers=["electron"],
        coordinate=coord,
    )
    assert _public_round_trips(produced), (
        f"{op}: compose_name produced {produced!r}, which is not canonical "
        f"(public parse->compose does not return it unchanged)"
    )


# ---------------------------------------------------------------------------
# Composed operator names must both parse from source strings and be generatable
# from the structured outer-to-inner expression list.
# ---------------------------------------------------------------------------

_NESTED_NAMES = [
    "flux_surface_averaged_inverse_of_square_of_major_radius",
    "flux_surface_averaged_square_of_magnetic_field_magnitude",
    (
        "flux_surface_averaged_ratio_of"
        "_square_of_toroidal_flux_coordinate_gradient_magnitude"
        "_to_square_of_magnetic_field_magnitude"
    ),
]


@pytest.mark.parametrize("name", _NESTED_NAMES)
def test_nested_operator_names_round_trip(name) -> None:
    """Composed operator-over-bare-transformation names round-trip via ISN."""
    assert _public_round_trips(name), (
        f"nested name {name!r} does not round-trip — ISN grammar regression"
    )


def test_inverse_cannot_wrap_inner_flux_surface_average() -> None:
    seg = GrammarSegments(
        base_token="temperature",
        base_kind="quantity",
        qualifiers=["electron"],
        operators=[
            {"token": "inverse"},
            {"token": "flux_surface_averaged"},
        ],
    )
    with pytest.raises(
        ValueError,
        match=r"ISN rejected operator chain inverse -> flux_surface_averaged.*precedence",
    ):
        seg.compose_name()


def test_generates_flux_surface_average_over_binary_ratio() -> None:
    seg = GrammarSegments(
        base_token="radius",
        base_kind="quantity",
        qualifiers=["major"],
        operators=[
            {"token": "flux_surface_averaged"},
            {
                "token": "ratio",
                "secondary_operand": "square_of_minor_radius",
            },
            {"token": "square"},
        ],
    )
    produced = seg.compose_name()
    assert produced == (
        "flux_surface_averaged_ratio_of_square_of_major_radius"
        "_to_square_of_minor_radius"
    )
    assert _public_round_trips(produced)


def test_inverse_over_square_is_a_valid_authored_order() -> None:
    seg = GrammarSegments(
        base_token="temperature",
        base_kind="quantity",
        qualifiers=["electron"],
        operators=[
            {"token": "inverse"},
            {"token": "square"},
        ],
    )
    produced = seg.compose_name()

    assert produced == "inverse_of_square_of_electron_temperature"
    assert _public_round_trips(produced)


def test_no_operator_preserves_plain_composition() -> None:
    seg = GrammarSegments(
        base_token="temperature",
        base_kind="quantity",
        qualifiers=["electron"],
    )
    assert seg.compose_name() == "electron_temperature"
