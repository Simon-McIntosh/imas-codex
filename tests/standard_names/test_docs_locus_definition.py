"""A locus is defined once, from the registry, for every name at it.

A name evaluated at a locus carries that locus's defining sentence. The
sentence comes from the locus registry's ``definition`` joined with the
conventions entry's equation and symbols, so it reads the same for two names
whose subjects differ. A name that states the locus in other words is a second
definition and is refined rather than accepted, and a forbidden variant is
refused through the same concept entry.
"""

from __future__ import annotations

import pytest
from imas_standard_names.grammar.terms import select_conventions

from imas_codex.standard_names import workers
from imas_codex.standard_names.context import conventions_context_for
from imas_codex.standard_names.docs_gates import (
    HouseStyleFindingClass,
    conformance_findings,
    locus_definition_sentence,
)

_SUBJECT_ELONGATION = "elongation_of_plasma_boundary"
_SUBJECT_MINOR_RADIUS = "minor_radius_of_plasma_boundary"

_BASE_DOCUMENT = "The model describes the plasma boundary for this quantity."
_REGISTRY_DEFINITION = (
    "The surface used to delimit the plasma domain for the quantity being described."
)


def _rendered(name: str, documentation: str) -> str:
    conventions = conventions_context_for(name)
    return workers._documentation_with_registry_form(name, documentation, conventions)


def _locus_sentence(concept: object) -> str:
    return locus_definition_sentence(concept)


def _plasma_boundary_concept(name: str):
    conventions = select_conventions(name)
    return next(c for c in conventions.concepts if c.token == "plasma_boundary")


def _last_paragraph(text: str) -> str:
    return text.rstrip().split("\n\n")[-1]


def test_two_subjects_render_the_identical_locus_sentence() -> None:
    elongation = _rendered(_SUBJECT_ELONGATION, _BASE_DOCUMENT)
    minor_radius = _rendered(_SUBJECT_MINOR_RADIUS, _BASE_DOCUMENT)

    expected = _locus_sentence(_plasma_boundary_concept(_SUBJECT_ELONGATION))

    assert _last_paragraph(elongation) == expected
    assert _last_paragraph(minor_radius) == expected
    assert _last_paragraph(elongation) == _last_paragraph(minor_radius)
    assert _REGISTRY_DEFINITION in expected


def test_locus_sentence_carries_the_registry_equation_and_symbols() -> None:
    sentence = _locus_sentence(_plasma_boundary_concept(_SUBJECT_ELONGATION))

    assert r"$\psi = \psi_{\mathrm{LCFS}}$" in sentence
    assert "poloidal magnetic flux" in sentence
    assert "last closed flux surface" in sentence


def test_departing_locus_sentence_is_refined_rather_than_accepted() -> None:
    documentation = (
        "The plasma boundary is the last closed magnetic flux surface "
        "satisfying the confinement requirement."
    )
    findings = conformance_findings(_SUBJECT_ELONGATION, documentation=documentation)
    classes = {finding.finding_class for finding in findings}
    assert HouseStyleFindingClass.LOCUS_DEFINITION in classes

    note = workers._documentation_conformance_note(
        {"id": _SUBJECT_ELONGATION, "description": "", "documentation": documentation}
    )
    assert note is not None
    assert "locus_definition" in note


@pytest.mark.parametrize("subject", [_SUBJECT_ELONGATION, _SUBJECT_MINOR_RADIUS])
def test_carried_registry_definition_is_accepted(subject: str) -> None:
    documentation = f"The model describes {subject.replace('_', ' ')}."
    conventions = conventions_context_for(subject)
    written = workers._documentation_with_registry_form(
        subject, documentation, conventions
    )

    classes = {
        finding.finding_class
        for finding in conformance_findings(subject, documentation=written)
    }
    assert HouseStyleFindingClass.LOCUS_DEFINITION not in classes


def test_forbidden_locus_variant_is_refined_rather_than_accepted() -> None:
    documentation = (
        "The plasma boundary is the separatrix in this configuration, "
        "and the quantity is evaluated there."
    )
    findings = conformance_findings(_SUBJECT_ELONGATION, documentation=documentation)
    classes = {finding.finding_class for finding in findings}
    assert HouseStyleFindingClass.FORBIDDEN_LOCUS_GLOSS in classes

    note = workers._documentation_conformance_note(
        {"id": _SUBJECT_ELONGATION, "description": "", "documentation": documentation}
    )
    assert note is not None
    assert "forbidden_locus_gloss" in note
