"""The contradiction veto reads a claim against every binding a name carries.

A base name such as ``electrostatic_potential`` spans seven ``phi_potential``
bindings, and its documentation states the scope common to them. Reading one
sentence against one narrow binding finds a conflict that the whole binding set
does not have, so the clause is judged binding by binding and the readings are
folded: a claim contradicts only when no binding supports it.
"""

from __future__ import annotations

from typing import Any

import pytest

from imas_codex.standard_names.review import jev_review
from imas_codex.standard_names.review.audits import AuditReport
from imas_codex.standard_names.review.checks import load_review_checks

ELECTROSTATIC_SCOPE = (
    "This quantity is the electrostatic potential in the plasma or at a "
    "plasma-facing surface, relative to a chosen electrical reference. "
    "The general quantity is not restricted to the core, edge, scrape-off "
    "layer, or a particular surface. "
    "It encompasses local values and explicitly identified "
    "magnetic-flux-surface averages; averaging is not implicit in the "
    "unqualified quantity."
)

PHI_BINDINGS = [
    (
        "plasma_profiles/profiles_1d/phi_potential",
        "Electrostatic potential, averaged on the magnetic flux surface",
    )
]
PHI_BINDINGS += [
    (
        f"plasma_profiles/profiles_1d/phi_potential_{index}",
        "Electrostatic potential at a local position.",
    )
    for index in range(2, 8)
]

S0010_NAME = (
    "flux_surface_averaged_time_averaged_normalized_perturbed_parallel_"
    "magnetic_field_magnitude"
)
S0010_SOURCE = (
    "gyrokinetics_local/non_linear/fields_intensity_1d/b_field_parallel_perturbed_norm"
)
S0010_ELECTROMAGNETIC_SOURCE = (
    "gyrokinetics_local/non_linear/fields_intensity_1d/b_field_parallel_perturbed"
)
S0010_DOCUMENTATION = (
    "The quantity applies to the parallel magnetic perturbation only; it "
    "excludes perpendicular perturbations and the unperturbed equilibrium field."
)


def _item(
    documentation: str, bindings: list[tuple[str, str]], name: str
) -> dict[str, Any]:
    return {
        "id": name,
        "documentation": documentation,
        "description": documentation.split(".")[0],
        "dd_source_docs": [
            {"id": path, "documentation": text} for path, text in bindings
        ],
    }


def _answers(bindings: list[tuple[str, str]], claim_index: int, **by_path: str):
    """One reading per binding, keyed by the binding it was made against."""
    answers: dict[str, dict[str, Any]] = {}
    for position, (path, _text) in enumerate(bindings):
        choice = by_path.get(path, "says nothing")
        answers[jev_review.claim_binding_key(claim_index, position)] = {
            "choice": choice,
            "probabilities": {choice: 1.0},
        }
    return answers


def _verdict(item: dict[str, Any], answers: dict[str, Any], claims: int = 0):
    checks = [check for check in load_review_checks()[0] if check.axis == "docs"]
    state = jev_review.build_docs_state(item, checks=checks)
    return state, jev_review.aggregate_docs_review(
        state, checks, answers, audit_report=AuditReport()
    )


def test_a_broad_base_name_claim_is_not_vetoed_by_one_narrow_binding():
    item = _item(ELECTROSTATIC_SCOPE, PHI_BINDINGS, "electrostatic_potential")
    answers = _answers(
        PHI_BINDINGS,
        0,
        **{PHI_BINDINGS[0][0]: "contradicts"},
        **{path: "supports" for path, _text in PHI_BINDINGS[1:]},
    )
    state, verdict = _verdict(item, answers)
    claim = state["documentation_claims"][0]
    assert verdict["contradiction_veto"] is False, verdict["contradiction_reason"]
    answer = verdict["claim_answers"][0]
    assert answer is not None
    assert answer["choice"] == "supports"
    assert claim == state["documentation_claims"][0]


def test_a_claim_false_for_every_binding_is_vetoed():
    item = _item(ELECTROSTATIC_SCOPE, PHI_BINDINGS, "electrostatic_potential")
    answers = _answers(
        PHI_BINDINGS,
        0,
        **{path: "contradicts" for path, _text in PHI_BINDINGS},
    )
    state, verdict = _verdict(item, answers)
    assert verdict["contradiction_veto"] is True
    assert state["documentation_claims"][0] in verdict["contradiction_reason"]
    assert verdict["claim_answers"][0]["choice"] == "contradicts"


def test_the_electrostatic_potential_scope_sentence_is_judged_once_per_binding():
    item = _item(ELECTROSTATIC_SCOPE, PHI_BINDINGS, "electrostatic_potential")
    state = jev_review.build_docs_state(item)
    assert state["binding_ids"] == [path for path, _text in PHI_BINDINGS]
    table = jev_review.docs_question_table(
        [check for check in load_review_checks()[0] if check.axis == "docs"],
        state["documentation_claims"],
        state["binding_ids"],
    )
    keys = [key for key in table if "claim_contradicts_source" in key]
    assert len(keys) == len(PHI_BINDINGS) * len(state["documentation_claims"])
    assert jev_review.claim_binding_key(0, 6) in table


def test_the_s0010_generic_claim_survives_a_sibling_binding_that_narrows_it():
    bindings = [
        (S0010_SOURCE, "Normalised perturbed parallel magnetic field"),
        (
            S0010_ELECTROMAGNETIC_SOURCE,
            "Perturbed magnetic field, summed over every component.",
        ),
    ]
    item = _item(S0010_DOCUMENTATION, bindings, S0010_NAME)
    answer = _answers(
        bindings,
        0,
        **{S0010_SOURCE: "supports"},
        **{S0010_ELECTROMAGNETIC_SOURCE: "contradicts"},
    )
    _state, verdict = _verdict(item, answer)
    assert verdict["contradiction_veto"] is False
    assert verdict["claim_answers"][0]["choice"] == "supports"


class _Lease:
    def charge_event(self, *_args, **_kwargs) -> None:
        return None

    def release_unused(self) -> None:
        return None


class _Budget:
    def reserve(self, *_args, **_kwargs) -> _Lease:
        return _Lease()


@pytest.mark.asyncio
async def test_the_review_docs_note_reads_every_binding_and_holds_on_none(
    monkeypatch,
):
    from imas_codex.standard_names.worker_tasks import worker_review_docs as module

    item = _item(ELECTROSTATIC_SCOPE, PHI_BINDINGS, "electrostatic_potential")

    async def decide(states, questions, **_kwargs):
        results = []
        for position, _state in enumerate(states):
            answers: dict[str, Any] = {}
            for key in questions:
                if "claim_contradicts_source" not in key:
                    continue
                if ":binding:" in key:
                    target = int(key.rsplit(":", 1)[1])
                    choice = "contradicts" if target == 0 else "supports"
                else:
                    choice = "contradicts" if position == 0 else "supports"
                answers[key] = {"choice": choice, "probabilities": {choice: 1.0}}
            results.append((answers, 0.0))
        return results, 0.0

    monkeypatch.setattr(jev_review, "decide_batch", decide)
    from imas_codex import settings as settings_module

    monkeypatch.setattr(settings_module, "get_model", lambda _seat: "judge-model")
    dependency = lambda name: {  # noqa: E731
        "_bound_docs_review_dd_paths": lambda _item: [],
        "_has_dd_source_reference": lambda _item: False,
        "strip_dd_prefix": lambda path: path,
    }[name]
    note = await module._documentation_contradiction_note(
        dependency, item=item, mgr=_Budget()
    )
    assert note is None


@pytest.mark.asyncio
async def test_the_review_docs_note_still_holds_a_claim_no_binding_supports(
    monkeypatch,
):
    from imas_codex.standard_names.worker_tasks import worker_review_docs as module

    item = _item(ELECTROSTATIC_SCOPE, PHI_BINDINGS, "electrostatic_potential")

    async def decide(states, questions, **_kwargs):
        results = []
        for _state in states:
            answers: dict[str, Any] = {
                key: {"choice": "contradicts", "probabilities": {"contradicts": 1.0}}
                for key in questions
                if "claim_contradicts_source" in key
            }
            results.append((answers, 0.0))
        return results, 0.0

    monkeypatch.setattr(jev_review, "decide_batch", decide)
    from imas_codex import settings as settings_module

    monkeypatch.setattr(settings_module, "get_model", lambda _seat: "judge-model")
    dependency = lambda name: {  # noqa: E731
        "_bound_docs_review_dd_paths": lambda _item: [],
        "_has_dd_source_reference": lambda _item: False,
        "strip_dd_prefix": lambda path: path,
    }[name]
    note = await module._documentation_contradiction_note(
        dependency, item=item, mgr=_Budget()
    )
    assert note is not None
    assert note.startswith("contradicts source:")
