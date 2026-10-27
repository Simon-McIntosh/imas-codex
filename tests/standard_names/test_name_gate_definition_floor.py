"""The name-review semantic gate scores a bare base against its definition.

The gate is a cosine between a name and its description at a floor calibrated
on multi-word identifiers.  A name that is a single vocabulary token cannot
reach that floor at any description — measured against the live embedding
server, ``beta`` scores 0.4877 against its own description while its
multi-word family members reach 0.770-0.849 — so the coupling rejects the
closed vocabulary's own bare tokens by construction rather than on merit.

Substituting the vocabulary's definition of the token as the name-side text
moves the whole live bare-base population clear: 25 identities measured, four
below the 0.55 floor before and none after, with ``beta`` at 0.8295 and a
population minimum of 0.7577.  The
floor itself is unchanged; only the text it scores moved.

These tests pin the substitution and its boundaries without reaching the
embedding server or the graph.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_codex.standard_names.audits import semantic_similarity_check
from imas_codex.standard_names.defaults import SEMANTIC_SIM_CRITICAL
from imas_codex.standard_names.worker_tasks import worker_review_names


def semantic_gate_name_text(name: str) -> str:
    """Resolve the gate's name-side text builder, or fail naming what is gone.

    Looked up on the module rather than imported so that removing the
    substitution fails these tests individually instead of collapsing the
    file into one collection error.
    """
    builder = getattr(worker_review_names, "semantic_gate_name_text", None)
    assert builder is not None, (
        "the review gate no longer builds its name-side text through a "
        "vocabulary definition lookup"
    )
    return builder(name)


def closed_vocabulary_definitions() -> dict[str, str]:
    """Resolve the token-to-definition mapping, or fail naming what is gone."""
    loader = getattr(worker_review_names, "_closed_vocabulary_definitions", None)
    assert loader is not None, (
        "the closed vocabulary's token definitions are no longer read"
    )
    return loader()


# Bare bases whose live description scored below or on the 0.55 floor under
# the identifier coupling, with the score measured against the live embedding
# server before and after the substitution.
MEASURED_BARE_BASES = {
    "mach_number": (0.4106, 0.9619),
    "safety_factor": (0.4391, 0.9675),
    "beta": (0.4877, 0.8295),
    "momentum": (0.5425, 0.8707),
    "magnetic_field": (0.5921, 0.9532),
}


def test_bare_base_gate_text_is_the_vocabulary_definition() -> None:
    """A single defined token is scored against its definition, not itself."""
    definitions = closed_vocabulary_definitions()
    text = semantic_gate_name_text("beta")

    assert text != "beta"
    assert text == definitions["beta"]
    assert "pressure" in text


@pytest.mark.parametrize("name", sorted(MEASURED_BARE_BASES))
def test_every_measured_bare_base_resolves_to_a_definition(name: str) -> None:
    """Each bare base that failed or hugged the floor now carries prose."""
    text = semantic_gate_name_text(name)

    assert text != name
    assert len(text.split()) > 5


def test_geometry_carriers_are_defined_too() -> None:
    """Carrier tokens live in a separate vocabulary file and are reachable."""
    text = semantic_gate_name_text("radial_coordinate")

    assert text != "radial_coordinate"
    assert "radius" in text.lower()


def test_multi_token_name_keeps_the_identifier_text() -> None:
    """Multi-word names keep today's coupling; the floor was calibrated there."""
    for name in ("plasma_beta", "toroidal_beta", "normalized_toroidal_plasma_beta"):
        assert semantic_gate_name_text(name) == name


def test_undefined_token_falls_through_and_fails_closed() -> None:
    """A vocabulary token with no definition gets no admission path."""
    definitions = closed_vocabulary_definitions()
    undefined = "capacitance"

    assert undefined not in definitions
    assert semantic_gate_name_text(undefined) == undefined


def test_unparseable_name_falls_through_to_the_identifier() -> None:
    """A name the strict grammar rejects is scored exactly as before."""
    for name in ("not_a_vocabulary_token_at_all", "", "beta beta"):
        assert semantic_gate_name_text(name) == name


def test_the_critical_floor_is_unchanged() -> None:
    """One floor serves both texts: the substitution moved the text, not 0.55."""
    assert SEMANTIC_SIM_CRITICAL == 0.55


def test_the_gate_call_site_builds_its_text_through_the_lookup() -> None:
    """The review batch reaches the gate through the definition substitution.

    The processor's own code object is read, not the entrypoint forwarder's:
    the forwarder delegates and names neither the builder nor the gate, so a
    read of it would pass whatever the implementation did.
    """
    co_names = (
        worker_review_names.process_review_name_batch.__wrapped__.__code__.co_names
    )

    assert "semantic_gate_name_text" in co_names


def test_gate_embeds_the_definition_and_clears_the_floor(monkeypatch) -> None:
    """End to end at the audit boundary, with a stand-in for the embedder.

    The stand-in returns a unit vector per text and makes the description
    agree with the definition rather than with the token, which is the
    geometry the live measurement found.
    """
    embedded: list[str] = []
    # semantic_similarity_check renders its name-side text with underscores
    # replaced by spaces, which is a no-op on prose apart from symbol names.
    definition = semantic_gate_name_text("beta").replace("_", " ")
    description = "Ratio of plasma pressure to magnetic pressure."

    vectors = {
        "beta": [1.0, 0.0, 0.0],
        definition: [0.0, 1.0, 0.0],
        description: [0.0, 1.0, 0.0],
    }

    def fake_embed(items, text_field="_text"):
        for item in items:
            text = item[text_field]
            embedded.append(text)
            item["embedding"] = np.asarray(vectors[text], dtype=np.float32)

    monkeypatch.setattr(
        "imas_codex.embeddings.description.embed_descriptions_batch", fake_embed
    )

    identifier_sim, identifier_issues = semantic_similarity_check("beta", description)
    definition_sim, definition_issues = semantic_similarity_check(
        semantic_gate_name_text("beta"), description
    )

    assert definition in embedded
    assert identifier_sim == pytest.approx(0.0, abs=1e-6)
    assert identifier_sim < SEMANTIC_SIM_CRITICAL
    assert identifier_issues
    assert definition_sim == pytest.approx(1.0, abs=1e-6)
    assert definition_sim >= SEMANTIC_SIM_CRITICAL
    assert not definition_issues


# One-word tokens the substitution must not admit: a strict-parse reject and a
# token the vocabulary leaves undefined each keep the identifier text and are
# scored against the unchanged floor at the audit boundary.
UNRESOLVED_ONE_WORD_TOKENS = ("quux", "capacitance")


@pytest.mark.parametrize("token", UNRESOLVED_ONE_WORD_TOKENS)
def test_unresolved_one_word_token_fails_the_gate(token: str, monkeypatch) -> None:
    """A rejected or undefined one-word token never clears the floor.

    Drives the token through the gate's own text builder into the audit, with
    the same stand-in embedder as the clearing test. The identifier text does
    not agree with the description, so the score stays below the unchanged
    floor and the gate returns its issue.
    """
    description = "A quantity that has no vocabulary definition."
    name_text = semantic_gate_name_text(token)

    # The fall-through is what keeps the token scored as itself; a gate that
    # synthesised definition prose for every token would not reach this state.
    assert name_text == token

    embedded: list[str] = []
    vectors = {
        token: [1.0, 0.0, 0.0],
        description: [0.0, 1.0, 0.0],
    }

    def fake_embed(items, text_field="_text"):
        for item in items:
            text = item[text_field]
            embedded.append(text)
            item["embedding"] = np.asarray(
                vectors.get(text, [1.0, 0.0, 0.0]), dtype=np.float32
            )

    monkeypatch.setattr(
        "imas_codex.embeddings.description.embed_descriptions_batch", fake_embed
    )

    sim, issues = semantic_similarity_check(name_text, description)

    assert name_text in embedded
    assert sim is not None
    assert sim < SEMANTIC_SIM_CRITICAL
    assert issues
