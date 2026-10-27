"""Reviewed documentation is accepted only if it conforms to the registry.

The review score judges prose and cannot see an opening that follows none of
its selected templates, a forbidden equation variant, or an equation for a
registered concept written other than in that concept's canonical form. The
canonical form is the converged statement and is accepted. Each test drives the
documentation-review worker with the model and graph calls stubbed and a
review score of 1.0, so the only thing that can change the outcome is the
conformance check before acceptance.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.llm.prompt_loader import render_prompt
from imas_codex.standard_names import workers
from imas_codex.standard_names.context import conventions_context_for
from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.docs_gates import (
    HouseStyleFindingClass,
    conformance_findings,
)
from imas_codex.standard_names.graph_store import (
    graph_catalog as _owning_graph_catalog,
    graph_review_persistence as _owning_graph_review_persistence,
)
from imas_codex.standard_names.worker_tasks import (
    worker_common as _owning_worker_common,
    worker_review_docs as _owning_worker_review_docs,
    worker_review_names as _owning_worker_review_names,
)

_AVERAGED = "flux_surface_averaged_argon_density_at_plasma_boundary"
_UNTEMPLATED = "magnetic_field"
_UNPARSEABLE = "vacuum_toroidal_field_strength"

_CONFORMING_OPENING = "Number density of argon ions on the last closed flux surface."
_PROSE = (
    "The quantity is the argon number density taken as a flux-surface average, "
    "see [flux_surface_averaged_electron_density_at_plasma_boundary]"
    "(name:flux_surface_averaged_electron_density_at_plasma_boundary) for the "
    "electron counterpart."
)
_CANONICAL_EQUATION = (
    r"$$\langle n\rangle = \oint n\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|$$"
)
_LINE_FORM_EQUATION = r"$$\langle n\rangle = \oint n\,dl/B \,/\, \oint dl/B$$"
_SUBSCRIPTED_EQUATION = (
    r"$$\langle n\rangle_S = \oint n\,dS/|\nabla\psi| \,/\, \oint dS/|\nabla\psi|$$"
)
_FRACTION_EQUATION = (
    r"$$\left\langle n_{\mathrm{Ar}} \right\rangle_S = "
    r"\frac{\oint_S n_{\mathrm{Ar}}\,dS/\lvert\nabla\psi\rvert}"
    r"{\oint_S dS/\lvert\nabla\psi\rvert}$$"
)


def _documentation(*, opening: str = _CONFORMING_OPENING, equation: str = "") -> str:
    return "\n\n".join(part for part in (opening, _PROSE, equation) if part)


def _quorum(score: float = 1.0) -> dict[str, Any]:
    return {
        "records": [],
        "winning_score": score,
        "winning_scores": {"documentation_quality": 20},
        "winning_comments": "Clear and complete.",
        "winning_comments_per_dim": {"documentation_quality": "Clear."},
        "canonical_model": "test/model",
        "total_cost": 0.0,
        "total_tokens_in": 0,
        "total_tokens_out": 0,
        "resolution_method": "quorum_consensus",
        "dd_gaps": [],
    }


def _review(name: str, documentation: str, *, score: float = 1.0) -> dict[str, Any]:
    """Run the review worker over one drafted name and return what it persisted."""
    import imas_codex.standard_names.graph_ops as graph_ops

    request = {
        "response_model": MagicMock(),
        "user_prompt": "review",
        "system_prompt": "review",
        "models": ["test/model"],
        "disagreement_threshold": 0.2,
        "rubric_dims": ["documentation_quality"],
        "reasoning_effort": None,
        "escalation_reasoning_effort": None,
    }
    item = {
        "id": name,
        "claim_token": "token",
        "claim_seq": 1,
        "description": "Argon number density.",
        "documentation": documentation,
    }
    persist = MagicMock(return_value="accepted")
    manager = MagicMock()
    manager.run_id = "run"

    async def prepare(_item: dict[str, Any]) -> dict[str, Any]:
        return request

    async def quorum(**_kwargs: Any) -> dict[str, Any]:
        return _quorum(score)

    with (
        patch.object(
            _owning_worker_review_docs, "prepare_docs_review_request", prepare
        ),
        patch.object(_owning_worker_review_names, "_run_rd_quorum_cycles", quorum),
        patch.object(
            _owning_worker_common, "_review_dd_source_bindings", return_value={}
        ),
        patch.object(_owning_worker_common, "_persist_dd_gap_evidence"),
        patch.object(_owning_graph_catalog, "write_reviews"),
        patch.object(_owning_graph_catalog, "update_review_aggregates"),
        patch.object(
            _owning_graph_review_persistence, "persist_reviewed_docs", persist
        ),
    ):
        processed = asyncio.run(
            _owning_worker_review_docs.process_review_docs_batch(
                [item], manager, asyncio.Event()
            )
        )

    assert processed == 1
    persist.assert_called_once()
    return persist.call_args.kwargs


def _routed_to_refinement(persisted: dict[str, Any]) -> bool:
    return persisted["score"] < persisted["min_score"]


def test_conforming_documentation_is_accepted_on_score() -> None:
    documentation = _documentation()
    assert conformance_findings(_AVERAGED, documentation=documentation) == []

    persisted = _review(_AVERAGED, documentation)

    assert persisted["score"] == 1.0
    assert persisted["min_score"] == DEFAULT_MIN_SCORE
    assert not _routed_to_refinement(persisted)


def test_canonical_statement_of_a_registered_concept_is_accepted() -> None:
    documentation = _documentation(equation=_CANONICAL_EQUATION)
    assert conformance_findings(_AVERAGED, documentation=documentation) == []

    persisted = _review(_AVERAGED, documentation)

    assert persisted["score"] == 1.0
    assert not _routed_to_refinement(persisted)


def test_off_template_opening_is_refined_at_a_perfect_score() -> None:
    documentation = _documentation(opening="The argon density at the boundary.")
    findings = conformance_findings(_AVERAGED, documentation=documentation)
    assert {f.finding_class for f in findings} == {
        HouseStyleFindingClass.TEMPLATE_OPENING
    }

    persisted = _review(_AVERAGED, documentation)

    assert _routed_to_refinement(persisted)
    assert "template_opening" in persisted["comments"]
    assert "template_opening" in persisted["comments_per_dim"]["registry_conformance"]


def test_forbidden_variant_equation_is_refined_at_a_perfect_score() -> None:
    documentation = _documentation(equation=_LINE_FORM_EQUATION)
    classes = {
        f.finding_class
        for f in conformance_findings(_AVERAGED, documentation=documentation)
    }
    assert classes == {HouseStyleFindingClass.FORBIDDEN_EQUATION_VARIANT}

    persisted = _review(_AVERAGED, documentation)

    assert _routed_to_refinement(persisted)
    assert "forbidden_equation_variant" in persisted["comments"]


@pytest.mark.parametrize("equation", [_FRACTION_EQUATION, _SUBSCRIPTED_EQUATION])
def test_non_canonical_statement_of_a_registered_concept_is_refined(
    equation: str,
) -> None:
    documentation = _documentation(equation=equation)
    classes = {
        f.finding_class
        for f in conformance_findings(_AVERAGED, documentation=documentation)
    }
    assert classes == {HouseStyleFindingClass.NON_CANONICAL_EQUATION}

    persisted = _review(_AVERAGED, documentation)

    assert _routed_to_refinement(persisted)
    assert "non_canonical_equation" in persisted["comments"]
    assert "flux-surface average" in persisted["comments"]


def test_name_without_selected_template_is_accepted_on_score() -> None:
    documentation = (
        "The magnetic field, a free-form opening no template governs.\n\n"
        r"$$\mathbf{B} = \nabla \times \mathbf{A}$$"
    )
    assert conformance_findings(_UNTEMPLATED, documentation=documentation) == []

    persisted = _review(_UNTEMPLATED, documentation)

    assert persisted["score"] == 1.0
    assert not _routed_to_refinement(persisted)


def test_name_the_grammar_cannot_parse_is_accepted_on_score() -> None:
    documentation = _documentation(opening="An opening no template can govern.")

    persisted = _review(_UNPARSEABLE, documentation)

    assert persisted["score"] == 1.0
    assert not _routed_to_refinement(persisted)


def test_canonical_equation_of_a_required_restatement_is_accepted() -> None:
    documentation = (
        "Elongation of the plasma boundary.\n\n"
        r"$$\kappa = (Z_{\max} - Z_{\min})/(R_{\max} - R_{\min})$$"
    )
    classes = {
        f.finding_class
        for f in conformance_findings(
            "elongation_of_plasma_boundary", documentation=documentation
        )
    }

    assert HouseStyleFindingClass.NON_CANONICAL_EQUATION not in classes


def test_below_threshold_score_is_not_raised_by_the_conformance_check() -> None:
    persisted = _review(_AVERAGED, _documentation(), score=0.5)

    assert persisted["score"] == 0.5
    assert _routed_to_refinement(persisted)


def test_finding_names_the_concept_entry_and_its_canonical_form() -> None:
    findings = conformance_findings(
        _AVERAGED, documentation=_documentation(equation=_FRACTION_EQUATION)
    )

    [finding] = findings
    assert finding.registry_entry == "concept:flux_surface_averaged"
    assert r"\langle f\rangle = \oint f\,dS/|\nabla\psi|" in finding.detail


def test_review_note_is_serialisable_as_per_dimension_comment() -> None:
    persisted = _review(_AVERAGED, _documentation(equation=_FRACTION_EQUATION))

    assert json.loads(json.dumps(persisted["comments_per_dim"]))["registry_conformance"]


def _equation_placement(rendered: str) -> list[str]:
    return [line for line in rendered.splitlines() if "Equation placement" in line]


def test_conventions_block_allows_only_the_canonical_form_of_a_named_concept() -> None:
    context = conventions_context_for(_AVERAGED)
    rendered = render_prompt("sn/_conventions", {"conventions": context})

    placements = _equation_placement(rendered)
    assert len(placements) == 2
    for line in placements:
        assert "unless needed" not in line
        assert "name the concept and link to it" in line
        assert "canonical equation above, written exactly that way" in line
    assert "unless needed" not in rendered
    assert context["concepts"][0]["equation"] in rendered


def test_conventions_block_still_asks_for_a_required_equation() -> None:
    context = conventions_context_for("elongation_of_plasma_boundary")
    assert context is not None
    rendered = render_prompt("sn/_conventions", {"conventions": context})

    by_concept = dict(
        zip(
            [c["token"] for c in context["concepts"]],
            _equation_placement(rendered),
            strict=True,
        )
    )
    assert "write this equation in the documentation" in by_concept["elongation"]
    assert "unless needed" not in by_concept["plasma_boundary"]
