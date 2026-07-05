"""Generation-time catch-and-promote for stray axis short-forms.

The LLM composer sometimes emits a DD-leaf short-form (``r``, ``phi``,
``tor``, ``pol``) instead of the canonical axis word. ``GrammarSegments``
promotes these to the canonical word before the registered-token check, so
composition succeeds with the one spelling the catalog actually stores
(cylindrical-axis-naming decision: words-canonical, one way). ``z`` is a
distinct, valid canonical Cartesian axis and must NOT be coerced.

This is compose-time only: the ISN parser itself stays strict and must keep
rejecting a stored name that uses the short form, proving the catcher does
not smuggle in a second accepted spelling.
"""

import pytest

from imas_codex.standard_names.models import GrammarSegments


@pytest.mark.parametrize(
    "short_form,canonical",
    [
        ("r", "radial"),
        ("phi", "toroidal"),
        ("tor", "toroidal"),
        ("pol", "poloidal"),
    ],
)
def test_short_form_promoted_to_canonical_word(short_form, canonical):
    seg = GrammarSegments(
        base_token="magnetic_field",
        base_kind="quantity",
        projection_axis=short_form,
    )
    assert seg.projection_axis == canonical


def test_z_axis_is_not_coerced():
    # 'z' is a valid canonical Cartesian axis (third member of an x, y, z
    # family); disambiguating it from cylindrical 'vertical' is a DD-ingest
    # concern, not a compose-time one, so the catcher must leave it alone.
    seg = GrammarSegments(
        base_token="direction_unit_vector",
        base_kind="geometry",
        projection_axis="z",
    )
    assert seg.projection_axis == "z"


def test_strict_parser_still_rejects_short_form_name():
    """The catcher promotes at generation time only — it must not create a
    second accepted spelling in the ISN parser itself."""
    from imas_standard_names.grammar import parse_standard_name

    # The canonical word form parses cleanly...
    parse_standard_name("radial_coordinate_of_flux_loop")

    # ...but the short-form spelling is still rejected outright.
    with pytest.raises(Exception):
        parse_standard_name("r_coordinate_of_flux_loop")
