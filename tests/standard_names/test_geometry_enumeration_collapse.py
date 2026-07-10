"""Enumerated geometry points collapse to one geometric-quantity name.

A standard name identifies a quantity-KIND by intrinsic physical identity.
Ordinal/enumerated geometry points (line-of-sight endpoints, polygon outline
vertices, beam-path waypoints) are NOT separately named — they COLLAPSE to ONE
geometric-quantity standard name; the ordinal index lives in the DD path/mapping
(``dd_paths``), never in the name. A point earns a distinct name only when it is
a distinct physical ENTITY (aperture vs wall), named by that entity.

A separate, orthogonal rule: DD local-coordinate axes ``x1``/``x2``/``x3`` are
ORTHOGONAL directions of a local sensor frame (NOT ordinal samples). They use the
registered carriers ``x1_coordinate`` / ``x2_coordinate`` and stay DISTINCT names.

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
# Collapsed geometry carriers — line-of-sight endpoints + outline vertices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_token,axis,expected",
    [
        # outline vertex array -> ONE name per axis
        ("outline", "radial", "radial_outline"),
        ("outline", "vertical", "vertical_outline"),
    ],
)
def test_enumerated_geometry_collapses_to_one_carrier(base_token, axis, expected):
    seg = GrammarSegments(
        base_token=base_token,
        base_kind="geometry",
        projection_axis=axis,
        projection_shape="coordinate",
    )
    assert _name(seg) == expected


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
# Local sensor-frame axes x1/x2/x3 — DISTINCT directions, registered carriers
# (NOT ordinal points; they must NOT collapse, and must NOT use
# first_coordinate/second_coordinate which are unregistered)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "carrier",
    ["x1_coordinate", "x2_coordinate"],
)
def test_local_sensor_axes_use_registered_carriers(carrier):
    seg = GrammarSegments(base_token=carrier, base_kind="geometry")
    assert _name(seg) == carrier


def test_local_sensor_axes_stay_distinct():
    """x1 and x2 are different axes — they must compose to different names."""
    x1 = GrammarSegments(base_token="x1_coordinate", base_kind="geometry")
    x2 = GrammarSegments(base_token="x2_coordinate", base_kind="geometry")
    assert _name(x1) != _name(x2)


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
# Negative guard — ordinal-bearing base tokens must NOT be registered carriers
# (these are exactly the names the old prompt taught and that generated gaps).
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
