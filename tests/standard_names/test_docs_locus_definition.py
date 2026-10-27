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
from imas_codex.standard_names.worker_tasks import (
    worker_generate_docs as _owning_worker_generate_docs,
    worker_review_docs as _owning_worker_review_docs,
)

_SUBJECT_ELONGATION = "elongation_of_plasma_boundary"
_SUBJECT_MINOR_RADIUS = "minor_radius_of_plasma_boundary"

_BASE_DOCUMENT = "The model describes the plasma boundary for this quantity."


def _rendered(name: str, documentation: str) -> str:
    conventions = conventions_context_for(name)
    return _owning_worker_generate_docs._documentation_with_registry_form(
        name, documentation, conventions
    )


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

    concept = _plasma_boundary_concept(_SUBJECT_ELONGATION)
    expected = _locus_sentence(concept)

    assert _last_paragraph(elongation) == expected
    assert _last_paragraph(minor_radius) == expected
    assert _last_paragraph(elongation) == _last_paragraph(minor_radius)
    assert concept.definition in expected


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

    note = _owning_worker_review_docs._documentation_conformance_note(
        {"id": _SUBJECT_ELONGATION, "description": "", "documentation": documentation}
    )
    assert note is not None
    assert "locus_definition" in note


@pytest.mark.parametrize("subject", [_SUBJECT_ELONGATION, _SUBJECT_MINOR_RADIUS])
def test_carried_registry_definition_is_accepted(subject: str) -> None:
    documentation = f"The model describes {subject.replace('_', ' ')}."
    conventions = conventions_context_for(subject)
    written = _owning_worker_generate_docs._documentation_with_registry_form(
        subject, documentation, conventions
    )

    classes = {
        finding.finding_class
        for finding in conformance_findings(subject, documentation=written)
    }
    assert HouseStyleFindingClass.LOCUS_DEFINITION not in classes


_ANCHOR = "flux_surface_averaged_xenon_density_at_plasma_boundary"

# The opening a boundary-averaged density name actually writes: the subject is
# the quantity, and the locus appears inside the prepositional phrase that says
# where the quantity is evaluated.
_ANCHOR_OPENING = (
    "The flux-surface-averaged xenon density at the plasma boundary is the "
    "poloidal flux-surface average of the local xenon ion number density "
    "evaluated on the last closed flux surface."
)


def _locus_definition_classes(
    name: str, documentation: str
) -> set[HouseStyleFindingClass]:
    return {
        finding.finding_class
        for finding in conformance_findings(name, documentation=documentation)
    }


def test_opening_that_evaluates_at_the_locus_is_not_a_second_definition() -> None:
    classes = _locus_definition_classes(_ANCHOR, _ANCHOR_OPENING)

    assert HouseStyleFindingClass.LOCUS_DEFINITION not in classes


def test_locus_subject_definition_is_refined_rather_than_accepted() -> None:
    documentation = (
        "The plasma boundary is the flux surface satisfying the confinement "
        "requirement."
    )
    classes = _locus_definition_classes(_SUBJECT_ELONGATION, documentation)

    assert HouseStyleFindingClass.LOCUS_DEFINITION in classes


def test_locus_subject_noun_phrase_definition_is_still_a_finding() -> None:
    documentation = (
        "The plasma-boundary surface is the flux surface satisfying "
        r"$\psi=\psi_{\mathrm{LCFS}}$."
    )
    classes = _locus_definition_classes(_ANCHOR, documentation)

    assert HouseStyleFindingClass.LOCUS_DEFINITION in classes


def test_locus_subject_with_a_long_modifier_is_still_a_definition() -> None:
    documentation = (
        "The plasma boundary of this configuration is the last closed flux surface."
    )
    classes = _locus_definition_classes(_ANCHOR, documentation)

    assert HouseStyleFindingClass.LOCUS_DEFINITION in classes


def test_prepositional_locus_mention_is_not_a_definition() -> None:
    documentation = (
        "The xenon density at the plasma boundary is the local ion number "
        "density on the last closed flux surface."
    )
    classes = _locus_definition_classes(_ANCHOR, documentation)

    assert HouseStyleFindingClass.LOCUS_DEFINITION not in classes


def test_registry_locus_sentence_is_not_a_second_definition() -> None:
    sentence = _locus_sentence(_plasma_boundary_concept(_SUBJECT_ELONGATION))
    documentation = f"{_BASE_DOCUMENT}\n\n{sentence}"
    classes = _locus_definition_classes(_ANCHOR, documentation)

    assert HouseStyleFindingClass.LOCUS_DEFINITION not in classes


def test_forbidden_locus_variant_is_refined_rather_than_accepted() -> None:
    documentation = (
        "The plasma boundary is the separatrix in this configuration, "
        "and the quantity is evaluated there."
    )
    findings = conformance_findings(_SUBJECT_ELONGATION, documentation=documentation)
    classes = {finding.finding_class for finding in findings}
    assert HouseStyleFindingClass.FORBIDDEN_LOCUS_GLOSS in classes

    note = _owning_worker_review_docs._documentation_conformance_note(
        {"id": _SUBJECT_ELONGATION, "description": "", "documentation": documentation}
    )
    assert note is not None
    assert "forbidden_locus_gloss" in note
