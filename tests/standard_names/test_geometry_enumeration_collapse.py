"""Enumerated geometry points collapse to one geometric-quantity name.

A standard name identifies a quantity-KIND by intrinsic physical identity.
Ordinal/enumerated geometry points are NOT separately named when the ordinal is
only array bookkeeping within one carrier and owner. The ordinal index lives in
the DD path/mapping (``dd_paths``), never in the name. Carrier, owner, axis, and
representation remain part of the physical identity: wall and plasma-boundary
outlines therefore remain distinct. A point earns a distinct name when it is a
distinct physical ENTITY (aperture vs wall), named by that entity.

A separate, orthogonal rule: DD local-coordinate axes ``x1``/``x2`` are
orthogonal directions of a local sensor frame, not ordinal samples. They map to
the registered semantic carriers ``first_local_tangential_coordinate`` and
``second_local_tangential_coordinate`` and stay distinct names.

These tests exercise the ISN composer directly via :class:`GrammarSegments` — no
LLM call. They pin the verified target forms and guard that ordinal-bearing base
tokens do not compose.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from imas_codex.standard_names.models import GrammarSegments
from imas_codex.standard_names.workers import normalize_spelling


def _name(seg: GrammarSegments) -> str:
    return normalize_spelling(seg.compose_name())


# ---------------------------------------------------------------------------
# Collapsed geometry carriers — ordinal vertices within one owner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "owner,axis,expected",
    [
        ("wall", "radial", "radial_outline_of_wall"),
        ("wall", "vertical", "vertical_outline_of_wall"),
        (
            "plasma_boundary",
            "radial",
            "radial_outline_of_plasma_boundary",
        ),
        (
            "plasma_boundary",
            "vertical",
            "vertical_outline_of_plasma_boundary",
        ),
    ],
)
def test_outline_ordinals_collapse_only_within_owner(owner, axis, expected):
    """An outline's vertex index is omitted while its owner remains."""
    seg = GrammarSegments(
        base_token="outline",
        base_kind="geometry",
        projection_axis=axis,
        projection_shape="coordinate",
        locus_token=owner,
        locus_relation="of",
        locus_type="position",
    )
    assert _name(seg) == expected


def test_outline_owners_stay_distinct():
    """Consistency within one vertex array cannot merge different objects."""
    wall = GrammarSegments(
        base_token="outline",
        base_kind="geometry",
        projection_axis="radial",
        projection_shape="coordinate",
        locus_token="wall",
        locus_relation="of",
        locus_type="position",
    )
    boundary = GrammarSegments(
        base_token="outline",
        base_kind="geometry",
        projection_axis="radial",
        projection_shape="coordinate",
        locus_token="plasma_boundary",
        locus_relation="of",
        locus_type="position",
    )

    assert _name(wall) == "radial_outline_of_wall"
    assert _name(boundary) == "radial_outline_of_plasma_boundary"
    assert _name(wall) != _name(boundary)


@pytest.mark.parametrize("axis", ["radial", "vertical", "toroidal"])
def test_line_of_sight_is_no_longer_a_geometry_carrier(axis):
    # line_of_sight migrated from geometry_carriers.yml to locus_registry.yml
    # (a path locus with relation 'along': toroidal_angle_along_line_of_sight).
    # Composing it as a carrier must fail; the endpoint collapse now happens
    # at the locus, not via a carrier base.
    with pytest.raises(ValidationError):
        GrammarSegments(
            base_token="line_of_sight",
            base_kind="geometry",
            projection_axis=axis,
            projection_shape="coordinate",
        )


# ---------------------------------------------------------------------------
# Local sensor-frame tangential axes — DISTINCT directions, registered
# descriptive carriers. The DD-shaped x1/x2 labels are removed from the
# grammar: the frame is expressed as first/second local tangential
# directions (e3 = plasma-facing normal, e1 = more-horizontal tangent in
# positive toroidal phi, e2 = e3 x e1).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "carrier",
    ["first_local_tangential_coordinate", "second_local_tangential_coordinate"],
)
def test_local_sensor_axes_use_registered_carriers(carrier):
    seg = GrammarSegments(base_token=carrier, base_kind="geometry")
    assert _name(seg) == carrier


@pytest.mark.parametrize("carrier", ["x1_coordinate", "x2_coordinate"])
def test_dd_shaped_local_axis_labels_are_rejected(carrier):
    """DD x1/x2 axis labels are not registered geometry carriers."""
    with pytest.raises(ValidationError):
        GrammarSegments(base_token=carrier, base_kind="geometry")


def test_local_sensor_axes_stay_distinct():
    """The two tangential axes differ — they must compose to different names."""
    first = GrammarSegments(
        base_token="first_local_tangential_coordinate", base_kind="geometry"
    )
    second = GrammarSegments(
        base_token="second_local_tangential_coordinate", base_kind="geometry"
    )
    assert _name(first) != _name(second)


# ---------------------------------------------------------------------------
# Entity-distinguished points — a point named by its physical ENTITY, never
# by its ordinal. (aperture is an object/entity locus; first_wall is a
# position locus — both render the entity-named coordinate form.)
# ---------------------------------------------------------------------------


def test_aperture_point_named_by_entity():
    seg = GrammarSegments(
        base_token="position",
        base_kind="geometry",
        projection_axis="radial",
        projection_shape="coordinate",
        locus_token="aperture",
        locus_relation="of",
        locus_type="entity",
    )
    assert _name(seg) == "radial_position_of_aperture"


def test_first_wall_point_named_by_entity():
    seg = GrammarSegments(
        base_token="position",
        base_kind="geometry",
        projection_axis="radial",
        projection_shape="coordinate",
        locus_token="first_wall",
        locus_relation="of",
        locus_type="position",
    )
    assert _name(seg) == "radial_position_of_first_wall"


# ---------------------------------------------------------------------------
# Negative guard — ordinal-bearing base tokens must NOT be registered carriers;
# they erase the source carrier and owner instead of representing the quantity.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_base",
    [
        "first_point",
        "second_point",
        "third_point",
        "outline_point",
        "first_coordinate",
        "second_coordinate",
    ],
)
def test_ordinal_base_tokens_do_not_compose(bad_base):
    """Ordinal-point base tokens are unregistered — the validator rejects them,
    so no ordinal-bearing name can be composed."""
    with pytest.raises(ValidationError):
        GrammarSegments(base_token=bad_base, base_kind="geometry")
