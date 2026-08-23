"""Tests for deterministic standard-name documentation content gates."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from imas_codex.standard_names import docs_gates
from imas_codex.standard_names.docs_gates import (
    DOCUMENTATION_GATE_NAMES,
    MAX_DOCUMENTATION_WORDS,
    MIN_DOCUMENTATION_WORDS,
    NORMATIVE_GATE_NAMES,
    score_documentation,
)

HOLDOUT_PATH = Path("tests/standard_names/eval_sets/docs_holdout.json")


def _catalog_documentation(name: str) -> str:
    rows = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    return next(
        row["catalog_documentation"] for row in rows if row["catalog_name"] == name
    )


def test_catalog_documentation_passes_every_content_gate() -> None:
    score = score_documentation(_catalog_documentation("poloidal_magnetic_flux"))

    assert score.gate_vector == dict.fromkeys(DOCUMENTATION_GATE_NAMES, True)
    assert score.passed_count == score.total_count == 10
    assert MIN_DOCUMENTATION_WORDS <= score.word_count <= MAX_DOCUMENTATION_WORDS


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


def test_sign_convention_must_be_the_final_plain_paragraph() -> None:
    text = _catalog_documentation("poloidal_magnetic_flux").replace(
        "Sign convention: Positive", "**Sign convention:** Positive"
    )

    assert score_documentation(text).gate_vector["sign_convention"] is False


def test_absent_sign_convention_is_conditionally_valid() -> None:
    text = _catalog_documentation("electron_temperature")

    assert "Sign convention:" not in text
    assert score_documentation(text).gate_vector["sign_convention"] is True


def test_word_bounds_are_independent() -> None:
    below = score_documentation("plasma " * (MIN_DOCUMENTATION_WORDS - 1))
    above = score_documentation("plasma " * (MAX_DOCUMENTATION_WORDS + 1))

    assert below.gate_vector["minimum_word_count"] is False
    assert below.gate_vector["maximum_word_count"] is True
    assert above.gate_vector["minimum_word_count"] is True
    assert above.gate_vector["maximum_word_count"] is False
