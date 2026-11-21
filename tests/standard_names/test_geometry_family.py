"""Shape-keyed grouping of fields that describe one geometric object."""

from __future__ import annotations


def _items(parent: str, *leaves: tuple[str, str]) -> list[dict[str, str]]:
    return [{"path": f"{parent}/{suffix}", "unit": unit} for suffix, unit in leaves]


def _suffixes(family: object) -> list[str]:
    return [member.suffix for member in family.members]


def test_rectangle_groups_coordinates_width_and_height() -> None:
    from iter_codex.standard_names.families import (
        GeometryFamily,
        detect_geometry_families,
    )

    parent = "pf_active/coil/element/geometry/rectangle"
    families = detect_geometry_families(
        _items(parent, ("r", "m"), ("z", "m"), ("width", "m"), ("height", "m"))
    )

    assert len(families) == 1
    assert isinstance(families[0], GeometryFamily)
    assert families[0].parent_path == parent
    assert families[0].primitive == "rectangle"
    assert _suffixes(families[0]) == ["r", "z", "width", "height"]


def test_oblique_groups_all_eight_shape_fields() -> None:
    from iter_codex.standard_names.families import detect_geometry_families

    parent = "pf_active/coil/element/geometry/oblique"
    families = detect_geometry_families(
        _items(
            parent,
            ("r", "m"),
            ("z", "m"),
            ("alpha", "rad"),
            ("beta", "rad"),
            ("length", "m"),
            ("length_alpha", "m"),
            ("length_beta", "m"),
            ("thickness", "m"),
        )
    )

    assert len(families) == 1
    assert families[0].primitive == "oblique"
    assert _suffixes(families[0]) == [
        "r",
        "z",
        "alpha",
        "beta",
        "length",
        "length_alpha",
        "length_beta",
        "thickness",
    ]


def test_annulus_groups_inner_and_outer_radii_with_coordinates() -> None:
    from iter_codex.standard_names.families import detect_geometry_families

    parent = "pf_active/coil/element/geometry/annulus"
    families = detect_geometry_families(
        _items(
            parent,
            ("r", "m"),
            ("z", "m"),
            ("radius_inner", "m"),
            ("radius_outer", "m"),
        )
    )

    assert len(families) == 1
    assert families[0].primitive == "annulus"
    assert _suffixes(families[0]) == ["r", "z", "radius_inner", "radius_outer"]


def test_arcs_of_circle_groups_curvature_radii_with_coordinates() -> None:
    from iter_codex.standard_names.families import detect_geometry_families

    parent = "pf_active/coil/element/geometry/arcs_of_circle"
    families = detect_geometry_families(
        _items(parent, ("r", "m"), ("z", "m"), ("curvature_radii", "m"))
    )

    assert len(families) == 1
    assert families[0].primitive == "arcs_of_circle"
    assert _suffixes(families[0]) == ["r", "z", "curvature_radii"]


def test_outline_groups_closed_state_with_coordinates() -> None:
    from iter_codex.standard_names.families import detect_geometry_families

    parent = "cryostat/description_2d/cryostat/unit/element/outline"
    families = detect_geometry_families(
        _items(parent, ("r", "m"), ("z", "m"), ("closed", ""))
    )

    assert len(families) == 1
    assert families[0].primitive == "outline"
    assert _suffixes(families[0]) == ["r", "z", "closed"]


def test_error_and_uncertainty_fields_are_excluded() -> None:
    from iter_codex.standard_names.families import detect_geometry_families

    parent = "pf_active/coil/element/geometry/rectangle"
    families = detect_geometry_families(
        _items(
            parent,
            ("r", "m"),
            ("z", "m"),
            ("width", "m"),
            ("height", "m"),
            ("r_error_lower", "m"),
            ("z_error_upper", "m"),
            ("width_error_index", ""),
        )
    )

    assert len(families) == 1
    assert _suffixes(families[0]) == ["r", "z", "width", "height"]


def test_axis_only_shape_does_not_duplicate_vector_family() -> None:
    from iter_codex.standard_names.families import detect_geometry_families

    parent = "wall/description_2d/mobile/unit/outline"
    families = detect_geometry_families(_items(parent, ("r", "m"), ("z", "m")))

    assert families == []


def test_shape_name_not_child_suffixes_controls_detection() -> None:
    from iter_codex.standard_names.families import detect_geometry_families

    parent = "pf_active/coil/element/geometry/not_a_primitive"
    families = detect_geometry_families(
        _items(parent, ("r", "m"), ("z", "m"), ("width", "m"), ("height", "m"))
    )

    assert families == []


def test_existing_detector_result_is_exactly_unchanged() -> None:
    from iter_codex.standard_names.families import (
        FamilyMember,
        VectorFamily,
        detect_families,
    )

    parent = "pf_active/coil/element/geometry/rectangle"
    items = _items(
        parent,
        ("r", "m"),
        ("z", "m"),
        ("width", "m"),
        ("height", "m"),
    )

    assert detect_families(items) == [
        VectorFamily(
            parent_path=parent,
            family_type="physical_vector",
            members=[
                FamilyMember(
                    dd_path=f"{parent}/r", axis="radial", unit="m", suffix="r"
                ),
                FamilyMember(
                    dd_path=f"{parent}/z", axis="vertical", unit="m", suffix="z"
                ),
            ],
            parent_name=None,
            unit_uniform=True,
            units={"m"},
        )
    ]
