"""Regression tests for the x/y/z coordinate & unit-vector triple completeness
gap found by the systematic review.

Root cause (confirmed against the live catalog, e.g.
``z_coordinate_of_ferritic_element_centroid`` in
``standard_names/magnetic_field_systems.yml``): x/y/z (and r/phi/z) sibling
triples are minted independently, one LLM candidate per DD leaf. When a
candidate for the third (commonly ``z``) leaf resolves as an **attach**
(merging its DD source onto an already-existing standard name) rather than a
fresh **compose**, no documentation is ever generated for it — the merge
target can sit with ``documentation=""`` indefinitely, since attach never
triggers doc generation. This produced 6/6 known casualties on the z member,
because z is conventionally the last-processed axis of the triple.

``vector_family_consistency_check`` (imas_codex/standard_names/audits.py)
already groups a DD vector node's minted siblings and checks structural
agreement (axis token, base carrier, locus, physics_domain, canonical axis
triple) but — before this fix — never checked documentation completeness,
so this exact defect shape sailed through every corpus audit undetected.
This test module locks in the added check (case 6 in the docstring).
"""

from __future__ import annotations

from imas_codex.standard_names.audits import vector_family_consistency_check


def _axis_member(
    axis_name: str,
    leaf: str,
    *,
    locus: str = "camera",
    domain: str = "magnetics",
    documentation: str = "some rich documentation text",
) -> dict:
    """A fabricated device-vector component sharing one DD vector node."""
    name = f"{axis_name}_direction_unit_vector"
    if locus:
        name = f"{name}_of_{locus}"
    return {
        "id": name,
        "physics_domain": domain,
        "documentation": documentation,
        "source_paths": [f"camera_ir/channel/camera/direction/{leaf}"],
    }


def test_fully_documented_triple_passes():
    """All three members documented — no completeness issue."""
    names = [
        _axis_member("x", "x"),
        _axis_member("y", "y"),
        _axis_member("z", "z"),
    ]
    assert vector_family_consistency_check(names) == []


def test_z_member_with_empty_documentation_is_flagged():
    """Reproduces the exact real-world defect: x and y are fully documented,
    z was merged via an attach-only edge and carries empty documentation."""
    names = [
        _axis_member("x", "x"),
        _axis_member("y", "y"),
        _axis_member("z", "z", documentation=""),
    ]
    issues = vector_family_consistency_check(names)
    matches = [i for i in issues if "empty documentation" in i]
    assert len(matches) == 1
    assert "z_direction_unit_vector_of_camera" in matches[0]
    assert "x_direction_unit_vector_of_camera" in matches[0]
    assert "y_direction_unit_vector_of_camera" in matches[0]


def test_cylindrical_z_member_with_whitespace_only_documentation_is_flagged():
    """A whitespace-only documentation string must not be mistaken for real
    content — it is exactly as incomplete as an empty string."""
    names = [
        _axis_member("radial", "r"),
        _axis_member("toroidal", "phi"),
        _axis_member("vertical", "z", documentation="   \n"),
    ]
    issues = vector_family_consistency_check(names)
    matches = [i for i in issues if "empty documentation" in i]
    assert len(matches) == 1
    assert "vertical_direction_unit_vector_of_camera" in matches[0]


def test_all_undocumented_triple_is_not_flagged_as_incomplete():
    """A triple still awaiting docs generation entirely (no sibling has
    documentation yet) is a pending-generation state, not a defect — only
    flag when siblings disagree on completeness."""
    names = [
        _axis_member("x", "x", documentation=""),
        _axis_member("y", "y", documentation=""),
        _axis_member("z", "z", documentation=""),
    ]
    issues = vector_family_consistency_check(names)
    assert not [i for i in issues if "empty documentation" in i]
