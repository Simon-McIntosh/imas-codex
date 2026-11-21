"""One stored sentence per species family, rendered into every member.

A species family is the set of names that differ only in the species they name.
Its members must carry the family's stored sentence with the species
substituted, so a family states one wording rather than one template per
member. These tests hold the stored sentences to the ruled form and hold a
member's stored definition to the rendering; a member written in other words is
refused before it is persisted.
"""

from __future__ import annotations

import re

import pytest

from iter_codex.standard_names.context import conventions_context_for
from iter_codex.standard_names.docs_gates import (
    SPECIES_FAMILY_SENTENCES,
    SPECIES_POPULATION_SENTENCE,
    HouseStyleFindingClass,
    SharedPartAltered,
    conformance_findings,
    description_with_population,
    refuse_altered_shared_parts,
    shared_part_sentences,
    species_family,
    species_family_description,
    species_family_sentence,
)
from iter_codex.standard_names.worker_tasks import (
    worker_generate_docs as owning_worker,
)

#: The families whose members count a species number density. The ruled
#: population statement is the one every such family states.
DENSITY_FAMILIES = (
    "<X>_density",
    "<X>_density_at_divertor_target",
    "<X>_density_at_internal_transport_barrier",
    "<X>_density_at_limiter",
    "<X>_density_at_magnetic_axis",
    "<X>_density_at_pedestal_top",
    "<X>_density_at_plasma_boundary",
    "flux_surface_averaged_<X>_density_at_plasma_boundary",
    "line_averaged_<X>_density",
    "volume_averaged_<X>_density",
    "ratio_of_<X>_density_to_electron_density",
    "ratio_of_volume_averaged_<X>_density_to_volume_averaged_electron_density",
)

#: One member name per family: the family key with its species substituted.
FAMILY_MEMBERS = {key: key.replace("<X>", "argon") for key in SPECIES_FAMILY_SENTENCES}


def _render(name: str) -> str:
    return owning_worker._documentation_with_registry_form(
        name, "A note specific to this name alone.", conventions_context_for(name)
    )


def test_every_stored_family_sentence_varies_only_in_the_species() -> None:
    """Every family has a sentence whose only variable is the species."""

    assert len(SPECIES_FAMILY_SENTENCES) == 23
    for key, sentence in SPECIES_FAMILY_SENTENCES.items():
        assert sentence.count("<X>") == 1, key
        assert "COCOS" not in sentence, key


def test_the_density_families_state_the_ruled_population_statement() -> None:
    """The ruled statement appears as ruled in every density family's sentence."""

    assert len(DENSITY_FAMILIES) == 12
    for key in DENSITY_FAMILIES:
        assert SPECIES_POPULATION_SENTENCE in SPECIES_FAMILY_SENTENCES[key], key


def test_no_family_sentence_states_a_population_including_neutrals() -> None:
    """No stored sentence contradicts the ruled population."""

    for key, sentence in SPECIES_FAMILY_SENTENCES.items():
        lowered = sentence.lower()
        assert "including neutral" not in lowered, key
        assert "with neutral atoms included" not in lowered, key
        assert "including all particles" not in lowered, key


def test_a_member_written_away_from_its_family_sentence_is_refused() -> None:
    """A stored definition that is not the rendering is refused, family by family.

    The guarded thing is made to happen: every family's member is handed a
    definition that does not carry the family's sentence, and the writer's own
    refusal rejects it before anything is stored.
    """

    for name in FAMILY_MEMBERS.values():
        with pytest.raises(SharedPartAltered) as raised:
            refuse_altered_shared_parts(
                name,
                "The species population is described in this name's own words.",
                None,
            )
        assert "species_family:" in str(raised.value), name


def test_a_rendered_member_carries_its_family_sentence() -> None:
    """The writer installs the family sentence, with the species substituted."""

    families = (
        "<X>_density",
        "line_averaged_<X>_density",
        "toroidal_<X>_velocity_at_plasma_boundary",
    )
    for key in families:
        name = key.replace("<X>", "argon")
        sentence = species_family_sentence(name)
        assert sentence is not None
        rendered = _render(name)
        assert sentence in rendered, name


def test_the_members_of_a_family_carry_one_description() -> None:
    """Every member's description is the family's one wording, species apart.

    Each member is handed a written description of its own, differing in a way
    the species substitution cannot collapse, and the stored description is
    asserted directly. Deleting the family branch of
    ``description_with_population`` leaves three distinct templates and three
    written descriptions here, so this test goes red on that deletion.
    """

    for key in SPECIES_FAMILY_SENTENCES:
        members = [key.replace("<X>", token) for token in ("argon", "neon", "iron")]
        word_re = re.compile(
            "|".join(sorted(("argon", "neon", "iron"), key=len, reverse=True)),
            re.IGNORECASE,
        )
        templates = {
            re.sub(
                r"\s+",
                " ",
                word_re.sub(
                    "<X>",
                    description_with_population(
                        member,
                        f"Written in the model's own words, variant {members.index(member)}.",
                    ),
                ),
            )
            for member in members
        }
        assert len(templates) == 1, (key, templates)
        for member in members:
            assert description_with_population(
                member, "Any other wording the model offered."
            ) == species_family_description(member), key


def test_a_family_sentence_is_carried_beside_the_operator_wording() -> None:
    """The family sentence is the family's part; operators are rendered too."""

    name = "flux_surface_averaged_argon_density_at_plasma_boundary"
    anchors = {anchor for anchor, _ in shared_part_sentences(name, None)}
    assert anchors == {
        "species_family:flux_surface_averaged_<X>_density_at_plasma_boundary"
    }
    rendered = owning_worker._documentation_with_registry_form(
        name,
        "A note specific to this name alone.",
        conventions_context_for(name),
    )
    assert "Flux-surface-averaged number density of argon ions" in rendered
    assert SPECIES_POPULATION_SENTENCE in rendered
