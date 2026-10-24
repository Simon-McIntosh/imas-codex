"""One owner decides coordinate-ness, and both documentation paths ask it."""

from __future__ import annotations

import pytest

from imas_codex.standard_names import derivation, docs_gates
from imas_codex.standard_names.docs_gates import (
    DocumentationGateOutcome,
    DocumentationPhysicsContext,
    score_documentation,
)
from imas_codex.standard_names.docs_holdout_eval import evaluate_docs_holdout
from imas_codex.standard_names.review.audits import run_documentation_gate_checks

COORDINATE_DOCUMENT = (
    "The radial position of the magnetic axis in the right-handed cylindrical "
    "(R, phi, Z) frame, where R is the major radius, phi the toroidal angle and "
    "Z the vertical direction."
)

QUANTITY_DOCUMENT = (
    "The electron temperature in the right-handed cylindrical (R, phi, Z) frame."
)

DD_PATH = "equilibrium/time_slice/profiles_1d/position"


def _holdout_row(name: str, documentation: str, unit: str) -> dict:
    return {
        "split_key": f"coordinate-decision/{name}",
        "dd_path": DD_PATH,
        "declared_unit": unit,
        "cocos_transformation_type": None,
        "catalog_name": name,
        "catalog_description": documentation,
        "catalog_documentation": documentation,
        "catalog_source": "catalog",
        "catalog_commit": "0000000",
    }


def _physics_context(unit: str) -> DocumentationPhysicsContext:
    return DocumentationPhysicsContext(dd_path=DD_PATH, declared_unit=unit)


def _lint_verdict(name: str, documentation: str, unit: str) -> str:
    """The production lint verdict for one gate, as ``LintFinding`` or ``pass``."""
    findings = [
        finding
        for finding in run_documentation_gate_checks(
            [{"id": name, "documentation": documentation, "unit": unit}]
        )
        if finding.finding_type == "defining_equation"
    ]
    return findings[0].finding_type if findings else "pass"


def test_the_gate_keeps_no_second_owner_of_the_decision() -> None:
    assert not hasattr(docs_gates, "_names_a_position")
    assert callable(derivation.is_coordinate_name)


def test_model_level_parser_alone_recognises_a_coordinate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _reject(name: str) -> None:
        raise ValueError(f"structural parser cannot parse {name!r}")

    derivation.is_coordinate_name.cache_clear()
    monkeypatch.setattr(derivation, "_parse_structural_ir", _reject)

    assert derivation.is_coordinate_name("vertical_coordinate") is True
    score = score_documentation(
        COORDINATE_DOCUMENT,
        physics_context=_physics_context("m"),
        documenting_name="vertical_coordinate",
    )
    assert (
        score.gate_vector["defining_equation"].outcome is DocumentationGateOutcome.PASS
    )


async def test_holdout_and_lint_agree_on_a_coordinate_document() -> None:
    name = "radial_position"
    row = _holdout_row(name, COORDINATE_DOCUMENT, "m")

    report = await evaluate_docs_holdout("catalog", rows=[row])
    holdout_outcome = report.row_scores[0].catalog_gates["defining_equation"].outcome

    assert holdout_outcome is DocumentationGateOutcome.PASS
    assert _lint_verdict(name, COORDINATE_DOCUMENT, "m") == "pass"


async def test_non_coordinate_quantity_still_needs_a_defining_relation() -> None:
    name = "electron_temperature"
    row = _holdout_row(name, QUANTITY_DOCUMENT, "eV")

    assert derivation.is_coordinate_name(name) is False
    report = await evaluate_docs_holdout("catalog", rows=[row])
    holdout_outcome = report.row_scores[0].catalog_gates["defining_equation"].outcome

    assert holdout_outcome is DocumentationGateOutcome.FAIL
    assert _lint_verdict(name, QUANTITY_DOCUMENT, "eV") == "defining_equation"
