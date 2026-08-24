"""Tests for deterministic standard-name documentation content gates."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from imas_codex.standard_names.docs_gates import (
    DOCUMENTATION_GATE_NAMES,
    MIN_DOCUMENTATION_WORDS,
    NORMATIVE_GATE_NAMES,
    DocumentationGateOutcome,
    DocumentationGateResult,
    DocumentationGateScore,
    DocumentationPhysicsContext,
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


def _outcomes(score: DocumentationGateScore) -> dict[str, DocumentationGateOutcome]:
    return {gate: result.outcome for gate, result in score.gate_vector.items()}


def _result(outcome: DocumentationGateOutcome, reason: str) -> DocumentationGateResult:
    return DocumentationGateResult(outcome=outcome, reason=reason)


def _electron_pressure_relation(*, include_boltzmann_constant: bool) -> str:
    product = "n_e k_B T_e" if include_boltzmann_constant else "n_e T_e"
    constant_definition = (
        ", $k_B$ is the Boltzmann constant in J/K,"
        if include_boltzmann_constant
        else ""
    )
    return f"""Electron pressure $p_e$ is related to particle density and temperature by the ideal-gas relation.

$$p_e = {product}$$

where $n_e$ is the electron density in m$^{{-3}}${constant_definition} and $T_e$ is the thermodynamic temperature in K. The relation distinguishes pressure from density and temperature individually. See [electron_density](name:electron_density) for the related particle quantity."""


def test_catalog_documentation_passes_every_content_gate() -> None:
    score = score_documentation(
        _catalog_documentation("poloidal_magnetic_flux"),
        physics_context=DocumentationPhysicsContext(
            dd_path="core_sources/source/profiles_1d/grid/psi",
            declared_unit="Wb",
            cocos_transformation_type="psi_like",
        ),
    )

    assert _outcomes(score) == dict.fromkeys(
        DOCUMENTATION_GATE_NAMES, DocumentationGateOutcome.PASS
    )
    assert score.passed_count == score.total_count == 6
    assert score.failed_count == score.not_evaluable_count == 0
    assert score.evaluable_count == 6
    assert score.word_count >= MIN_DOCUMENTATION_WORDS


def test_defining_relation_fails_a_genuine_unit_mismatch() -> None:
    score = score_documentation(
        _electron_pressure_relation(include_boltzmann_constant=False),
        physics_context=DocumentationPhysicsContext(
            dd_path="equilibrium/time_slice/profiles_1d/electrons/pressure",
            declared_unit="Pa",
        ),
    )

    result = score.gate_vector["defining_equation"]
    assert result.outcome is DocumentationGateOutcome.FAIL
    assert (
        result.reason == "defining relation dimensions contradict the DD-declared unit"
    )


def test_defining_relation_passes_when_dimensions_match_declared_unit() -> None:
    score = score_documentation(
        _electron_pressure_relation(include_boltzmann_constant=True),
        physics_context=DocumentationPhysicsContext(
            dd_path="equilibrium/time_slice/profiles_1d/electrons/pressure",
            declared_unit="Pa",
        ),
    )

    result = score.gate_vector["defining_equation"]
    assert result.outcome is DocumentationGateOutcome.PASS
    assert result.reason == "defining relation reproduces the DD-declared unit"


def test_defining_relation_is_not_evaluable_without_a_declared_unit() -> None:
    score = score_documentation(
        _electron_pressure_relation(include_boltzmann_constant=True),
        physics_context=DocumentationPhysicsContext(
            dd_path="equilibrium/time_slice/profiles_1d/electrons/pressure",
            declared_unit=None,
        ),
    )

    result = score.gate_vector["defining_equation"]
    assert result.outcome is DocumentationGateOutcome.NOT_EVALUABLE
    assert result.reason == "DD-declared unit is unavailable"


def test_defining_relation_is_not_evaluable_with_an_unbound_symbol() -> None:
    documentation = _electron_pressure_relation(
        include_boltzmann_constant=True
    ).replace(
        ", $k_B$ is the Boltzmann constant in J/K,",
        ", $k_B$ is the Boltzmann constant,",
    )

    score = score_documentation(
        documentation,
        physics_context=DocumentationPhysicsContext(
            dd_path="equilibrium/time_slice/profiles_1d/electrons/pressure",
            declared_unit="Pa",
        ),
    )

    result = score.gate_vector["defining_equation"]
    assert result.outcome is DocumentationGateOutcome.NOT_EVALUABLE
    assert result.reason == "relation cannot bind every symbol to a stated unit"


def test_stub_fails_required_content() -> None:
    score = score_documentation(
        "A plasma quantity.",
        physics_context=DocumentationPhysicsContext(
            dd_path="equilibrium/time_slice/profiles_1d/q",
            declared_unit="1",
        ),
    )

    for gate in (
        "defining_equation",
        "relationship_link",
        "minimum_word_count",
    ):
        assert score.gate_vector[gate].outcome is DocumentationGateOutcome.FAIL
    assert score.passed_count < score.total_count


def test_not_evaluable_is_distinct_from_a_failed_gate() -> None:
    outcomes = {
        gate: _result(DocumentationGateOutcome.PASS, "authoritative check passed")
        for gate in DOCUMENTATION_GATE_NAMES
    }
    outcomes["defining_equation"] = _result(
        DocumentationGateOutcome.NOT_EVALUABLE,
        "declared unit is unavailable",
    )

    score = DocumentationGateScore(gate_vector=outcomes, word_count=48)

    assert score.gate_vector["defining_equation"].outcome == "not_evaluable"
    assert score.gate_vector["defining_equation"].reason == (
        "declared unit is unavailable"
    )
    assert score.passed_count == 5
    assert score.failed_count == 0
    assert score.not_evaluable_count == 1
    assert score.evaluable_count == 5
    assert score.total_count == 6


def test_physics_context_is_retained_by_the_scorer() -> None:
    context = DocumentationPhysicsContext(
        dd_path="equilibrium/time_slice/profiles_1d/q",
        declared_unit="1",
        cocos_transformation_type=None,
    )

    score = score_documentation(
        _catalog_documentation("safety_factor"),
        physics_context=context,
    )

    assert score.physics_context is context


def test_documentation_gate_names_are_exact() -> None:
    assert DOCUMENTATION_GATE_NAMES == (
        "defining_equation",
        "symbol_definitions",
        "relationship_link",
        "sign_convention",
        "link_hygiene",
        "minimum_word_count",
    )
    assert NORMATIVE_GATE_NAMES == DOCUMENTATION_GATE_NAMES[:4]


def test_keyword_presence_gates_are_absent() -> None:
    assert not {
        "physical_meaning",
        "scope",
        "exclusions_or_distinctions",
    } & set(DOCUMENTATION_GATE_NAMES)
    assert not {
        "typical_values",
        "measurement_methods",
        "general_measurement",
    } & set(DOCUMENTATION_GATE_NAMES)


def test_documentation_gate_list_has_no_maximum_word_count() -> None:
    assert "maximum_word_count" not in DOCUMENTATION_GATE_NAMES


def test_relationship_gate_requires_a_resolving_name_link() -> None:
    phrase_only = "This quantity depends on and is proportional to the magnetic field."
    name_link = (
        "This quantity depends on the "
        "[poloidal magnetic field](name:poloidal_magnetic_field)."
    )

    assert (
        score_documentation(phrase_only).gate_vector["relationship_link"].outcome
        is DocumentationGateOutcome.FAIL
    )
    assert (
        score_documentation(name_link).gate_vector["relationship_link"].outcome
        is DocumentationGateOutcome.PASS
    )


def test_malformed_name_link_fails_hygiene() -> None:
    text = _catalog_documentation("poloidal_magnetic_flux").replace(
        "[poloidal_magnetic_field](#poloidal_magnetic_field)",
        "[poloidal_magnetic_field](poloidal_magnetic_field)",
    )

    assert (
        score_documentation(text).gate_vector["link_hygiene"].outcome
        is DocumentationGateOutcome.FAIL
    )


def test_bare_name_brackets_fail_hygiene() -> None:
    text = _catalog_documentation("poloidal_magnetic_flux").replace(
        "[flux_loop_voltage](#flux_loop_voltage)", "[flux_loop_voltage]"
    )

    assert (
        score_documentation(text).gate_vector["link_hygiene"].outcome
        is DocumentationGateOutcome.FAIL
    )


def test_mathematical_brackets_do_not_fail_link_hygiene() -> None:
    text = _catalog_documentation("poloidal_magnetic_flux").replace(
        "The poloidal magnetic flux $\\psi$",
        "The poloidal magnetic flux $\\psi = C[f_q]$",
    )

    assert (
        score_documentation(text).gate_vector["link_hygiene"].outcome
        is DocumentationGateOutcome.PASS
    )


def test_sign_convention_must_be_the_final_plain_paragraph() -> None:
    text = _catalog_documentation("poloidal_magnetic_flux").replace(
        "Sign convention: Positive", "**Sign convention:** Positive"
    )

    assert (
        score_documentation(
            text,
            physics_context=DocumentationPhysicsContext(
                cocos_transformation_type="psi_like"
            ),
        )
        .gate_vector["sign_convention"]
        .outcome
        is DocumentationGateOutcome.FAIL
    )


def test_sensitive_quantity_requires_a_sign_convention() -> None:
    text = _catalog_documentation("electron_temperature")

    assert "Sign convention:" not in text
    result = score_documentation(
        text,
        physics_context=DocumentationPhysicsContext(
            cocos_transformation_type="psi_like"
        ),
    ).gate_vector["sign_convention"]

    assert result.outcome is DocumentationGateOutcome.FAIL
    assert result.reason == "COCOS-sensitive quantity omits a sign convention"


def test_invariant_quantity_forbids_a_sign_convention() -> None:
    result = score_documentation(
        "Sign convention: Positive when the measured value is above zero.",
        physics_context=DocumentationPhysicsContext(
            cocos_transformation_type="one_like"
        ),
    ).gate_vector["sign_convention"]

    assert result.outcome is DocumentationGateOutcome.FAIL
    assert result.reason == "COCOS-invariant quantity states a sign convention"


def test_sign_convention_is_not_evaluable_without_a_transformation_class() -> None:
    result = score_documentation(
        _catalog_documentation("electron_temperature"),
        physics_context=DocumentationPhysicsContext(cocos_transformation_type=None),
    ).gate_vector["sign_convention"]

    assert result.outcome is DocumentationGateOutcome.NOT_EVALUABLE
    assert result.reason == "COCOS transformation class is unavailable"


def test_catalog_text_must_not_expose_cocos_metadata() -> None:
    for metadata in ("COCOS 17", "psi_like"):
        result = score_documentation(
            f"This quantity follows {metadata}.",
            physics_context=DocumentationPhysicsContext(
                cocos_transformation_type="psi_like"
            ),
        ).gate_vector["sign_convention"]

        assert result.outcome is DocumentationGateOutcome.FAIL
        assert result.reason == "documentation exposes catalog-internal COCOS metadata"


def test_holdout_sign_gate_outcome_distribution_is_complete() -> None:
    rows = json.loads(HOLDOUT_PATH.read_text(encoding="utf-8"))
    outcomes = Counter(
        score_documentation(
            row["catalog_documentation"],
            physics_context=DocumentationPhysicsContext(
                dd_path=row["dd_path"],
                declared_unit=row["declared_unit"],
                cocos_transformation_type=row["cocos_transformation_type"],
            ),
        )
        .gate_vector["sign_convention"]
        .outcome
        for row in rows
    )
    one_like_count = sum(row["cocos_transformation_type"] == "one_like" for row in rows)

    assert len(rows) == 85
    assert one_like_count == 2
    assert outcomes == {
        DocumentationGateOutcome.PASS: 22,
        DocumentationGateOutcome.FAIL: 2,
        DocumentationGateOutcome.NOT_EVALUABLE: 61,
    }
    assert sum(outcomes.values()) == len(rows)


def test_minimum_word_floor_is_enforced() -> None:
    below = score_documentation("plasma " * (MIN_DOCUMENTATION_WORDS - 1))
    at_floor = score_documentation("plasma " * MIN_DOCUMENTATION_WORDS)

    assert (
        below.gate_vector["minimum_word_count"].outcome is DocumentationGateOutcome.FAIL
    )
    assert (
        at_floor.gate_vector["minimum_word_count"].outcome
        is DocumentationGateOutcome.PASS
    )


def test_long_content_dense_documentation_passes_every_remaining_gate() -> None:
    base = _catalog_documentation("poloidal_magnetic_flux")
    additional_context = """The value identifies an entire family of nested magnetic surfaces within a magnetic equilibrium. Equality of the value at two spatial points means that they lie on the same flux surface when the equilibrium admits nested surfaces; it does not mean that their local magnetic-field components are equal. The magnetic-axis value provides the inner reference, while the plasma-boundary value provides the outer reference for normalized radial coordinates. These references distinguish the absolute flux from normalized poloidal flux, which rescales the interval between the axis and boundary.

Because the definition is surface-based, the chosen surface and its orientation are part of the quantity's scope. The boundary of the integration surface follows the toroidal contour associated with the evaluation location. Reversing the surface orientation reverses the signed flux without changing the underlying equilibrium. The quantity is distinct from toroidal magnetic flux, which threads a poloidal cross-section, and from flux-loop voltage, which depends on its time derivative rather than its instantaneous value.

Within an axisymmetric equilibrium, contours of the quantity organize the nested surfaces used to express profiles and averages. Its gradient relates to the poloidal magnetic field, while differences between two contours represent the flux contained between their surfaces. These relationships make the quantity a coordinate label and an integrated field measure; neither role substitutes for the defining surface integral."""
    documentation = base.replace(
        "\n\nSign convention:",
        f"\n\n{additional_context}\n\nSign convention:",
    )

    score = score_documentation(
        documentation,
        physics_context=DocumentationPhysicsContext(
            dd_path="core_sources/source/profiles_1d/grid/psi",
            declared_unit="Wb",
            cocos_transformation_type="psi_like",
        ),
    )

    assert score.word_count > 250
    assert _outcomes(score) == dict.fromkeys(
        DOCUMENTATION_GATE_NAMES, DocumentationGateOutcome.PASS
    )
    assert score.passed_count == score.total_count == 6


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
