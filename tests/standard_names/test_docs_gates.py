"""Tests for deterministic standard-name documentation content gates."""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

from imas_codex.standard_names import docs_gates
from imas_codex.standard_names.docs_gates import (
    DOCUMENTATION_GATE_NAMES,
    MIN_DOCUMENTATION_WORDS,
    NORMATIVE_GATE_NAMES,
    score_documentation,
)

HOLDOUT_PATH = Path("tests/standard_names/eval_sets/docs_holdout.json")
SYSTEM_PROMPT_PATH = Path("imas_codex/llm/prompts/sn/generate_docs_system.md")
SHARED_FORMAT_PATH = Path("imas_codex/llm/prompts/shared/sn/_docs_format.md")


def _catalog_documentation(name: str) -> str:
    rows = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    return next(
        row["catalog_documentation"] for row in rows if row["catalog_name"] == name
    )


def test_catalog_documentation_passes_every_content_gate() -> None:
    score = score_documentation(_catalog_documentation("poloidal_magnetic_flux"))

    assert score.gate_vector == dict.fromkeys(DOCUMENTATION_GATE_NAMES, True)
    assert score.passed_count == score.total_count == 9
    assert score.word_count >= MIN_DOCUMENTATION_WORDS


def test_stub_fails_required_content() -> None:
    score = score_documentation("A plasma quantity.")

    for gate in (
        "physical_meaning",
        "defining_equation",
        "scope",
        "exclusions_or_distinctions",
        "relationship_link_or_phrase_witness",
        "minimum_word_count",
    ):
        assert score.gate_vector[gate] is False
    assert score.passed_count < score.total_count


def test_gate_vocabulary_matches_normative_policy() -> None:
    assert len(NORMATIVE_GATE_NAMES) == 7
    assert set(NORMATIVE_GATE_NAMES) < set(DOCUMENTATION_GATE_NAMES)
    assert not {
        "typical_values",
        "measurement_methods",
        "general_measurement",
    } & set(DOCUMENTATION_GATE_NAMES)


def test_documentation_gate_list_has_no_maximum_word_count() -> None:
    assert "maximum_word_count" not in DOCUMENTATION_GATE_NAMES


def test_relationship_gate_records_only_a_link_or_phrase_witness() -> None:
    relationship_gate = next(
        gate for gate in NORMATIVE_GATE_NAMES if "relationship" in gate
    )
    detector_doc = inspect.getdoc(docs_gates._has_relationship_link_or_phrase_witness)

    assert relationship_gate == "relationship_link_or_phrase_witness"
    assert detector_doc is not None
    assert "witness" in detector_doc
    assert "Markdown link" in detector_doc
    assert "allow-listed relationship phrase" in detector_doc
    assert "does not assess" in detector_doc


def test_relationship_prose_without_a_lexical_witness_records_false() -> None:
    documentation = (
        "The safety factor is the number of toroidal circuits made by a magnetic "
        "field line for each poloidal circuit."
    )

    score = score_documentation(documentation)

    assert "[" not in documentation
    assert score.gate_vector["relationship_link_or_phrase_witness"] is False


def test_malformed_name_link_fails_hygiene() -> None:
    text = _catalog_documentation("poloidal_magnetic_flux").replace(
        "[poloidal_magnetic_field](#poloidal_magnetic_field)",
        "[poloidal_magnetic_field](poloidal_magnetic_field)",
    )

    assert score_documentation(text).gate_vector["link_hygiene"] is False


def test_bare_name_brackets_fail_hygiene() -> None:
    text = _catalog_documentation("poloidal_magnetic_flux").replace(
        "[flux_loop_voltage](#flux_loop_voltage)", "[flux_loop_voltage]"
    )

    assert score_documentation(text).gate_vector["link_hygiene"] is False


def test_mathematical_brackets_do_not_fail_link_hygiene() -> None:
    text = _catalog_documentation("poloidal_magnetic_flux").replace(
        "The poloidal magnetic flux $\\psi$",
        "The poloidal magnetic flux $\\psi = C[f_q]$",
    )

    assert score_documentation(text).gate_vector["link_hygiene"] is True


def test_sign_convention_must_be_the_final_plain_paragraph() -> None:
    text = _catalog_documentation("poloidal_magnetic_flux").replace(
        "Sign convention: Positive", "**Sign convention:** Positive"
    )

    assert score_documentation(text).gate_vector["sign_convention"] is False


def test_absent_sign_convention_is_conditionally_valid() -> None:
    text = _catalog_documentation("electron_temperature")

    assert "Sign convention:" not in text
    assert score_documentation(text).gate_vector["sign_convention"] is True


def test_minimum_word_floor_is_enforced() -> None:
    below = score_documentation("plasma " * (MIN_DOCUMENTATION_WORDS - 1))
    at_floor = score_documentation("plasma " * MIN_DOCUMENTATION_WORDS)

    assert below.gate_vector["minimum_word_count"] is False
    assert at_floor.gate_vector["minimum_word_count"] is True


def test_long_content_dense_documentation_passes_every_remaining_gate() -> None:
    base = _catalog_documentation("poloidal_magnetic_flux")
    additional_context = """The value identifies an entire family of nested magnetic surfaces within a magnetic equilibrium. Equality of the value at two spatial points means that they lie on the same flux surface when the equilibrium admits nested surfaces; it does not mean that their local magnetic-field components are equal. The magnetic-axis value provides the inner reference, while the plasma-boundary value provides the outer reference for normalized radial coordinates. These references distinguish the absolute flux from normalized poloidal flux, which rescales the interval between the axis and boundary.

Because the definition is surface-based, the chosen surface and its orientation are part of the quantity's scope. The boundary of the integration surface follows the toroidal contour associated with the evaluation location. Reversing the surface orientation reverses the signed flux without changing the underlying equilibrium. The quantity is distinct from toroidal magnetic flux, which threads a poloidal cross-section, and from flux-loop voltage, which depends on its time derivative rather than its instantaneous value.

Within an axisymmetric equilibrium, contours of the quantity organize the nested surfaces used to express profiles and averages. Its gradient relates to the poloidal magnetic field, while differences between two contours represent the flux contained between their surfaces. These relationships make the quantity a coordinate label and an integrated field measure; neither role substitutes for the defining surface integral."""
    documentation = base.replace(
        "\n\nSign convention:",
        f"\n\n{additional_context}\n\nSign convention:",
    )

    score = score_documentation(documentation)

    assert score.word_count > 250
    assert score.gate_vector == dict.fromkeys(DOCUMENTATION_GATE_NAMES, True)
    assert score.passed_count == score.total_count == 9


def test_documentation_prompts_have_no_upper_word_bound() -> None:
    upper_word_bound = re.compile(
        r"`documentation`[^\n]*(?:\d+\s*[–-]\s*\d+\s+words|max(?:imum)?\s+\d+\s+words)",
        re.IGNORECASE,
    )

    for prompt_path in (SYSTEM_PROMPT_PATH, SHARED_FORMAT_PATH):
        prompt = prompt_path.read_text(encoding="utf-8")
        assert upper_word_bound.search(prompt) is None, prompt_path


def test_description_sentence_and_character_limits_remain_separate() -> None:
    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

    assert "`description` | 15–30 words, 1 sentence" in prompt
    assert "max 250 chars" in prompt
    assert "`description` — **1 concise sentence" in prompt
    assert "≤250 characters" in prompt
