"""An x/y/z (or r/phi/z) sibling triple must be documented consistently.

Sibling triples are minted independently, one LLM candidate per DD leaf. When
a candidate resolves as an **attach** (merging its DD source onto an existing
standard name) rather than a fresh **compose**, no documentation is generated
for it — attach never triggers doc generation, so the merge target can sit
with ``documentation=""`` indefinitely. The casualty is systematically the
``z`` member, because z is conventionally the last-processed axis of the
triple, by which point a sibling has already created the name.

Structural agreement alone (axis token, base carrier, locus, physics_domain,
canonical axis triple) does not catch this shape: every structural field is
consistent and only the documentation is missing. These tests lock in the
documentation-completeness arm of ``vector_family_consistency_check``
(imas_codex/standard_names/audits.py) — case 6 of its docstring.
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
