"""The phrase check flags a narrowing phrase, and a convention suppresses its flag.

Both tests call the check with the phrase records and the two graded judgments
supplied, so the decision code is exercised without a model call. Each test sits
beside the control it is measured against: the same inputs with the closed
phrase question removed, and with the recorded conventions removed.
"""

from __future__ import annotations

from imas_codex.standard_names.review.name_definition_phrases import (
    DISPOSITION_BY_KIND,
    phrase_check,
    phrases_hold,
    reverse_check,
)

RESTRICTED = {
    "id": "temperature",
    "unit": "eV",
    "description": (
        "Temperature is the mean kinetic energy per particle of the plasma, "
        "evaluated on the divertor target plate only."
    ),
}
RESTRICTED_PHRASES = [
    {
        "kind": "restricts",
        "phrase": "evaluated on the divertor target plate only",
        "adds": "limits the quantity to one device part",
        "arm": "chat",
    }
]
RESTRICTED_JUDGMENTS = {
    "temperature#0": {
        "relation": "narrower_different_or_additional",
        "p_flag": 0.92,
        "answered": True,
    }
}


def test_a_device_part_restriction_is_flagged_under_an_unrestricted_name():
    findings = phrase_check(RESTRICTED, RESTRICTED_PHRASES, RESTRICTED_JUDGMENTS)

    assert [finding["pattern"] for finding in findings] == ["restricts"]
    assert findings[0]["phrase"] == "evaluated on the divertor target plate only"
    assert findings[0]["finding_type"] == "name_definition_phrase_disagreement"
    assert findings[0]["disposition"] == DISPOSITION_BY_KIND["restricts"]
    assert phrases_hold(findings) is not None


def test_the_restriction_flag_needs_the_phrase_question():
    """Control: with the closed question removed, the same inputs cannot flag."""

    findings = phrase_check(
        RESTRICTED, RESTRICTED_PHRASES, RESTRICTED_JUDGMENTS, apply_judgment=False
    )

    assert findings == []


SPECIES = {
    "id": "argon_density",
    "unit": "m^-3",
    "description": (
        "Argon density is the number density of argon ions in the plasma, "
        "summed over all charge states of the species, with neutral atoms "
        "excluded."
    ),
}
SPECIES_PHRASES = [
    {
        "kind": "narrows",
        "phrase": "summed over all charge states of the species",
        "adds": "fixes which charge states are counted",
        "arm": "chat",
    }
]
SPECIES_JUDGMENTS = {
    "argon_density#0": {
        "relation": "narrower_different_or_additional",
        "p_flag": 0.88,
        "answered": True,
    }
}


def test_a_recorded_species_convention_suppresses_the_flag():
    findings = phrase_check(SPECIES, SPECIES_PHRASES, SPECIES_JUDGMENTS)

    assert findings == []


def test_the_suppressed_flag_returns_when_the_convention_is_removed():
    """Control: the convention is what suppresses it, not the phrase or the grade."""

    findings = phrase_check(
        SPECIES, SPECIES_PHRASES, SPECIES_JUDGMENTS, apply_conventions=False
    )

    assert [finding["pattern"] for finding in findings] == ["narrows"]
    assert (
        phrase_check(
            SPECIES,
            SPECIES_PHRASES,
            SPECIES_JUDGMENTS,
            apply_conventions=False,
            apply_judgment=False,
        )
        == []
    )


def test_a_phrase_below_the_threshold_never_flags():
    judgments = {
        "argon_density#0": {
            "relation": "stated_by_name",
            "p_flag": 0.1,
            "answered": True,
        }
    }

    assert phrase_check(SPECIES, SPECIES_PHRASES, judgments) == []


def test_the_reverse_direction_flags_an_unsupported_name_qualifier():
    item = {
        "id": "effective_ion_energy_diffusivity",
        "description": (
            "Ion energy diffusivity is the coefficient linking the ion heat flux "
            "to the ion temperature gradient."
        ),
    }
    judgments = {
        "effective_ion_energy_diffusivity@effective": {
            "relation": "unsupported_by_definition",
            "p_flag": 0.81,
            "answered": True,
        }
    }

    findings = reverse_check(item, judgments)

    assert [finding["qualifier"] for finding in findings] == ["effective"]
    assert reverse_check(item, judgments, apply_judgment=False) == []
