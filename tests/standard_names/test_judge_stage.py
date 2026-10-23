"""The report-only judge stage asks a Choice per judged entry and never caps.

The judge stage sits beside the audit stage in the enrich worker: the entries a
pattern cannot decide — a selected concept that carries forbidden variants —
ask the judge one categorising Choice each through the shared decisions owner.
Its results join the same per-name findings the reviewer receives, marked
report-only, and are never booked as error-lint findings, so they never cap a
score.

The Choice offers the forms as words, never as the registry's patterns, so a
name written correctly is not read as forbidden by a pattern for a different
form. The stage runs whatever the audit does: skipping the audit skips the
deterministic checks, not the judge.

The decisions call is stubbed at the judgment layer's call seam
(``acall_decisions``), so the batch core runs for real and no test reaches the
live endpoint.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from imas_codex.discovery.base import judgment
from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import PipelinePhase
from imas_codex.standard_names.context import conventions_context_for
from imas_codex.standard_names.review import pipeline as pl
from imas_codex.standard_names.review.judge_stage import (
    _CONCEPT_FORMS,
    FORBIDDEN,
    QUESTION_NAME,
    _verdict,
)

# A carbon boundary entry writes the forbidden line form of the flux-surface
# average and calls the boundary the separatrix; a beryllium entry writes the
# canonical surface form and names the boundary as the last closed flux surface.
CARBON_DOC = (
    "Flux-surface average of the carbon density, "
    "$$\\langle n_C\\rangle = \\oint n_C\\,dl/B \\,/\\, \\oint dl/B$$, "
    "evaluated on the separatrix, the outermost closed magnetic flux surface."
)
BERYLLIUM_DOC = (
    "Flux-surface average of the beryllium density, "
    "$$\\langle n_{Be}\\rangle = \\oint n_{Be}\\,dS/|\\nabla\\psi| \\,/\\, "
    "\\oint dS/|\\nabla\\psi|$$, evaluated on the last closed flux surface."
)

CARBON_ID = "flux_surface_averaged_carbon_density_at_plasma_boundary"
BERYLLIUM_ID = "flux_surface_averaged_beryllium_density_at_plasma_boundary"


def _judge_answer(state: dict) -> dict:
    """Classify one judged entry from the documentation the stage carries.

    Returns an answer whose ``choice`` names one of the offered forms, the shape
    the decisions endpoint validates against the question's criteria.
    """
    concept = state["concept"]["token"]
    doc = state["entry"]["documentation"]
    if concept == "flux_surface_averaged":
        if "dl/B_p" in doc or "dl_p/B_p" in doc:
            choice = "line_poloidal"
        elif "dl/B" in doc:
            choice = "line_total_field"
        else:
            choice = "surface"
    elif concept == "plasma_boundary":
        choice = (
            "separatrix_equated" if "separatrix" in doc else "last_closed_flux_surface"
        )
    else:  # pragma: no cover - the registry holds no other forbidden concept yet
        choice = "none"
    return {
        QUESTION_NAME: {
            "type": "choice",
            "choice": choice,
            "probabilities": {choice: 1.0},
            "confidence": 1.0,
        }
    }


def _stub_decisions(monkeypatch):
    """Install a recording decisions stub and return the recorded states."""
    seen: list[dict] = []

    async def fake(model, state, questions, *, service, **_kwargs):
        seen.append(state)
        return _judge_answer(state), 0.0

    monkeypatch.setattr(judgment, "acall_decisions", fake)
    return seen


class _NoGraph:
    """A graph double that answers every query with an empty result.

    The score cap reconciles the name's documentation lint holds on each pass,
    which writes to the graph. A default-tier test injects this instead of
    reaching a database: the cap's own behaviour is under test here, not the
    hold store's.
    """

    def query(self, _cypher: str, **_params) -> list[dict]:
        return []


def _batch(names) -> dict:
    return {"names": names, "audit_findings": [], "error_lint_findings": {}}


def _state(batch: dict, *, skip_audit: bool = False):
    return SimpleNamespace(
        review_batches=[batch],
        all_names=[],
        neighborhood_k=10,
        audit_report=None,
        skip_audit=skip_audit,
        enrich_stats=WorkerStats(),
        enrich_phase=PipelinePhase("enrich"),
        should_stop=lambda: False,
    )


async def _run_enrich(monkeypatch, names, *, skip_audit: bool = False) -> dict:
    """Drive the enrich worker, neutralising the context lookups it also does."""
    from imas_codex.standard_names.review import enrichment

    monkeypatch.setattr(
        enrichment, "build_neighborhood_context", lambda *a, **k: [], raising=True
    )
    monkeypatch.setattr(pl, "_fetch_review_dd_context", lambda *a, **k: None)
    monkeypatch.setattr(pl, "_fetch_review_derived_children", lambda *a, **k: None)

    batch = _batch(names)
    await pl.enrich_review_worker(_state(batch, skip_audit=skip_audit))
    return batch


@pytest.mark.asyncio
async def test_one_choice_per_judged_entry_reaches_the_reviewer_findings(
    monkeypatch,
) -> None:
    """Each judged entry is asked one Choice; report-only findings reach the batch.

    Driving the enrich worker — not the stage directly — is what makes this fail
    when the stage is unwired from the pipeline.
    """
    seen = _stub_decisions(monkeypatch)

    batch = await _run_enrich(
        monkeypatch,
        [
            {"id": CARBON_ID, "description": "carbon", "documentation": CARBON_DOC},
            {
                "id": BERYLLIUM_ID,
                "description": "beryllium",
                "documentation": BERYLLIUM_DOC,
            },
        ],
    )

    # One state per judged entry: two boundary names x two forbidden-variant
    # concepts. Entries are grouped by concept to send each concept its own
    # options, so the request order is concept-major rather than name-major.
    assert len(seen) == 4
    assert sorted((s["entry"]["name"], s["concept"]["token"]) for s in seen) == sorted(
        [
            (CARBON_ID, "flux_surface_averaged"),
            (CARBON_ID, "plasma_boundary"),
            (BERYLLIUM_ID, "flux_surface_averaged"),
            (BERYLLIUM_ID, "plasma_boundary"),
        ]
    )
    # The state never carries the registry's forbidden-variant patterns.
    assert all("forbidden" not in s["concept"] for s in seen)

    findings = batch["audit_findings"]
    assert findings, "the judge stage's findings must reach the reviewer's list"
    assert all("[report-only:judge:concept_form" in line for line in findings)
    # Carbon writes both forbidden forms; beryllium writes neither.
    assert sum(CARBON_ID in line for line in findings) == 2
    assert not any(BERYLLIUM_ID in line for line in findings)


@pytest.mark.asyncio
async def test_report_only_result_leaves_the_score_uncapped(monkeypatch) -> None:
    """A judge finding is report-only: it never enters the error-lint cap input.

    The score cap reads only error-severity lint findings, so a report-only
    judge finding cannot cap; the entry's reviewer score survives the cap pass.
    """
    _stub_decisions(monkeypatch)
    batch = await _run_enrich(
        monkeypatch,
        [{"id": CARBON_ID, "description": "carbon", "documentation": CARBON_DOC}],
    )

    assert any("[report-only:judge" in line for line in batch["audit_findings"])
    # No audit ran for this batch, so the cap's input map is absent rather than
    # empty. The report-only findings are absent from it either way.
    assert batch["error_lint_findings"] is None

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    scored = [{"id": CARBON_ID, "reviewer_score": 0.9}]
    pl._cap_docs_scores_for_lint_errors(scored, {}, wlog, graph=_NoGraph())

    assert scored[0]["reviewer_score"] == 0.9
    assert "audit_cap_applied" not in scored[0]
    assert "audit_cap_finding" not in scored[0]


@pytest.mark.asyncio
async def test_a_name_with_no_judged_entry_makes_no_call(monkeypatch) -> None:
    """A name whose selected entries carry no forbidden variant is not judged."""
    seen = _stub_decisions(monkeypatch)

    batch = await _run_enrich(
        monkeypatch,
        [
            {
                "id": "plasma_current",
                "description": "plasma current",
                "documentation": "Total toroidal plasma shell current.",
            }
        ],
    )

    assert seen == []
    assert batch["audit_findings"] == []


@pytest.mark.asyncio
async def test_skip_audit_leaves_the_judge_stage_running(monkeypatch) -> None:
    """Skipping the deterministic audit does not skip the judge stage.

    With ``state.skip_audit`` true the audit report is absent, so the stage must
    still be reached: it makes its calls and its report-only findings reach the
    batch's findings list.
    """
    seen = _stub_decisions(monkeypatch)

    batch = await _run_enrich(
        monkeypatch,
        [{"id": CARBON_ID, "description": "carbon", "documentation": CARBON_DOC}],
        skip_audit=True,
    )

    assert len(seen) == 2, "the judge stage must still run when the audit is skipped"
    assert any("[report-only:judge" in line for line in batch["audit_findings"])


@pytest.mark.asyncio
async def test_no_option_text_carries_a_registry_forbidden_pattern(
    monkeypatch,
) -> None:
    """No registry forbidden pattern reaches the model through options or state.

    The registry states a forbidden variant as a regular expression. The stage
    must offer the forms as words: neither the question's options nor the state
    may contain a registry pattern, or the model is asked to reason about the
    pattern rather than the physics.
    """
    questions_sent: list[str] = []
    concepts_sent: list[str] = []

    async def fake(model, state, questions, *, service, **_kwargs):
        questions_sent.append(json.dumps(questions))
        concepts_sent.append(json.dumps(state.get("concept", {})))
        return _judge_answer(state), 0.0

    monkeypatch.setattr(judgment, "acall_decisions", fake)

    await _run_enrich(
        monkeypatch,
        [
            {"id": CARBON_ID, "description": "carbon", "documentation": CARBON_DOC},
            {
                "id": BERYLLIUM_ID,
                "description": "beryllium",
                "documentation": BERYLLIUM_DOC,
            },
        ],
    )

    patterns: list[str] = []
    for name_id in (CARBON_ID, BERYLLIUM_ID):
        conventions = conventions_context_for(name_id) or {"concepts": []}
        for concept in conventions["concepts"]:
            patterns.extend(concept.get("forbidden_variants", []))

    # Positive control: the registry really does state patterns for these names,
    # so an empty scan would be the instrument failing, not a clean payload.
    assert any("dl_p/B_p" in pattern for pattern in patterns), (
        "the registry's equation pattern was not loaded, so the scan proves nothing"
    )
    assert any(pattern == "the separatrix" for pattern in patterns), (
        "the registry's boundary pattern was not loaded, so the scan proves nothing"
    )

    # The option text the stage offers — the question's instructions and
    # criteria — carries no registry pattern. The entry's own written text is
    # the subject being judged, not an option the stage authors, so it is
    # checked only for the concept payload the stage builds.
    assert questions_sent, "the stage sent no question to inspect"
    for payload in [*questions_sent, *concepts_sent]:
        for pattern in patterns:
            assert pattern not in payload, (
                f"registry forbidden pattern {pattern!r} reached the model"
            )


def test_canonical_boundary_criterion_admits_no_separatrix_synonym() -> None:
    """The canonical boundary criterion admits no separatrix synonym.

    The registry lists the diverted-configuration name as a forbidden variant
    of the plasma boundary, so the canonical form must describe the boundary as
    the last closed flux surface without offering that name as an allowed
    alternative. The option for an entry that equates the boundary with the
    diverted name must map to the forbidden verdict. Restoring the clause that
    admitted the synonym as canonical fails this test.
    """
    spec = _CONCEPT_FORMS["plasma_boundary"]
    canonical = spec["criteria"]["last_closed_flux_surface"].lower()
    forbidden_form = spec["criteria"]["separatrix_equated"].lower()
    # The canonical form must not admit the diverted name as a synonym: the
    # clause "may also give ... as a synonym" is the defect this guards.
    assert "synonym" not in canonical
    assert "may also give" not in canonical
    # It still names the boundary as the last closed flux surface.
    assert "last closed flux surface" in canonical
    # The option that equates the boundary with the separated name is
    # forbidden, so an equated entry is judged non-conforming, not canonical.
    assert "synonym" in forbidden_form
    assert spec["verdicts"]["separatrix_equated"] == FORBIDDEN
    assert _verdict("plasma_boundary", "separatrix_equated") == FORBIDDEN
