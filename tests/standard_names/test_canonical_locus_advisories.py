"""Regression coverage for grammar-backed canonical-locus advisories."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.audits import canonical_locus_check


@pytest.mark.parametrize(
    "name",
    [
        "toroidal_coordinate_of_line_of_sight",
        "vertical_coordinate_of_detector_pixel",
        "poloidal_length_of_flux_surface",
        "vertical_coordinate_of_geometric_axis",
        "vertical_coordinate_of_magnetic_axis",
        "vertical_coordinate_of_measurement_position",
        "vertical_coordinate_of_primary_x_point",
        "vertical_coordinate_of_strike_point",
    ],
)
def test_isn_permitted_intrinsic_geometry_has_no_advisory(name: str) -> None:
    from imas_standard_names import get_grammar_context
    from imas_standard_names.grammar import parse_standard_name

    parsed = parse_standard_name(name)
    grammar = get_grammar_context()["grammar"]
    locus = parsed.geometry
    locus_token = str(getattr(locus, "value", locus))

    assert (
        "of"
        in grammar["vocabularies"]["locus_registry"][locus_token]["allowed_relations"]
    )
    assert canonical_locus_check({"id": name}) == []


def test_subject_quantity_at_position_advisory_is_derived_from_isn() -> None:
    """A grammar-derived subject quantity still requires field-evaluation form."""
    from imas_standard_names import get_grammar_context
    from imas_standard_names.grammar import parse_standard_name

    context = get_grammar_context()
    sections = {
        section["segment"]: section["tokens"]
        for section in context["vocabulary_sections"]
    }
    registry = context["grammar"]["vocabularies"]["locus_registry"]
    position_loci = [
        token
        for token, details in registry.items()
        if details["type"] == "position"
        and {"at", "of"}.issubset(details["allowed_relations"])
    ]

    positive: tuple[str, list[str]] | None = None
    for subject in sections["subject"]:
        for base in sections["physical_base"]:
            for locus in position_loci:
                name = f"{subject}_{base}_of_{locus}"
                try:
                    parse_standard_name(name)
                except Exception:
                    continue
                issues = canonical_locus_check({"id": name})
                if issues:
                    positive = name, issues
                    break
            if positive:
                break
        if positive:
            break

    assert positive is not None, "ISN exposes no parseable subject field at a position"
    name, issues = positive
    assert len(issues) == 1
    assert "field-evaluation structure" in issues[0]
    corrected = name.replace("_of_", "_at_", 1)
    assert f"Rewrite as '{corrected}'." in issues[0]
    assert corrected != name
