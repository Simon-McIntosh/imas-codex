"""Projection shape is derived from base_kind, not trusted from the LLM."""

from imas_codex.standard_names.models import GrammarSegments


def test_geometry_base_coerces_component_to_coordinate():
    # The composer LLM frequently labels a unit-vector projection as a
    # 'component'; ISN's IR requires 'coordinate' for geometry carriers.
    seg = GrammarSegments(
        base_token="direction_unit_vector",
        base_kind="geometry",
        projection_axis="z",
        projection_shape="component",
    )
    assert seg.projection_shape == "coordinate"


def test_quantity_base_coerces_coordinate_to_component():
    seg = GrammarSegments(
        base_token="magnetic_field",
        base_kind="quantity",
        projection_axis="radial",
        projection_shape="coordinate",
    )
    assert seg.projection_shape == "component"


def test_missing_shape_derived_when_axis_present():
    seg = GrammarSegments(
        base_token="direction_unit_vector",
        base_kind="geometry",
        projection_axis="x",
    )
    assert seg.projection_shape == "coordinate"


def test_no_projection_untouched():
    seg = GrammarSegments(base_token="direction_unit_vector", base_kind="geometry")
    assert seg.projection_shape is None
