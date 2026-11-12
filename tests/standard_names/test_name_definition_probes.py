"""The slot probes find what the name states and what the definition adds."""

from __future__ import annotations

import pytest

from imas_codex.standard_names.review.name_definition_probes import (
    SLOT_BY_KEY,
    compare_slots,
    name_closure,
    name_slots,
    probe_check,
)


def answers(**options: tuple[str, float]) -> dict[str, dict]:
    """Definition answers, one per slot, as the probe call returns them."""

    return {
        slot: {
            "ref": "fixture",
            "option": option,
            "phrase": "the definition wording",
            "probability": probability,
            "answered": True,
        }
        for slot, (option, probability) in options.items()
    }


def test_definition_states_a_normalisation_the_name_lacks():
    """A quantity defined per unit volume under a name that states no scale."""

    item = {"id": "particle_source", "description": "Particles per unit volume."}
    found = probe_check(item, answers(normalisation=("per unit volume", 0.9)))
    assert found, "an unnormalised name whose definition divides by a volume must flag"
    assert found[0]["slot"] == "normalisation"
    assert found[0]["direction"] == "definition states more"
    assert found[0]["finding_type"] == "name_definition_probe_disagreement"


def test_a_convention_suppresses_its_flag():
    """The catalog's one sign and reference convention is not an extra qualifier."""

    item = {"id": "wave_phase", "description": "Phase of the wave."}
    definition = answers(normalisation=("relative to a reference value", 0.9))
    assert probe_check(item, definition) == [], (
        "the recorded reference convention must suppress the normalisation candidate"
    )
    unfiltered = probe_check(item, definition, apply_conventions=False)
    assert [finding["slot"] for finding in unfiltered] == ["normalisation"], (
        "removing the convention must leave the candidate, so the suppression "
        "is what the passing test rests on"
    )


def test_a_default_aggregation_is_not_an_extra_qualifier():
    """A local value is what a name that states no aggregation already means."""

    item = {"id": "electron_temperature", "description": "Local temperature."}
    found = compare_slots(
        record_id(item),
        name_slots(item["id"]),
        answers(aggregation=("local value at a point", 0.9)),
    )
    assert found == []


def test_name_states_more_than_the_definition_says():
    """The reverse direction is a candidate too."""

    item = {"id": "flux_surface_averaged_density", "description": "Density."}
    found = compare_slots(
        item["id"],
        name_slots(item["id"]),
        answers(aggregation=("none stated", 0.9)),
    )
    assert [finding["direction"] for finding in found] == ["name states more"]


def test_every_slot_question_offers_the_answer_code_reads():
    """The closed vocabulary is closed: 'none stated' is always an option."""

    for slot in SLOT_BY_KEY.values():
        assert "none stated" in slot.options
        assert slot.options[-1] == "none stated"


@pytest.mark.parametrize(
    ("name", "slot", "value"),
    [
        ("volume_averaged_ion_density", "aggregation", "volume average"),
        ("accumulated_tritium_count", "aggregation", "time-accumulated total"),
        ("electron_density", "normalisation", "per unit volume"),
        ("electron_temperature", None, None),
    ],
)
def test_name_token_table_reads_the_slots(name, slot, value):
    stated = name_slots(name)
    if slot is None:
        assert all(entry is None for entry in stated.values())
        return
    assert stated[slot] == value


def test_an_object_name_does_not_imply_a_part_of_it():
    """Naming the injector is not naming its ion source, so restriction is free."""

    closure = name_closure("neutral_beam_injector_power")
    assert all(implication.slot != "restriction" for implication in closure), (
        "an object token must not suppress a restriction onto one part of it"
    )


def record_id(item: dict) -> str:
    return str(item.get("id") or item.get("name") or "")
