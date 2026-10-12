"""The report-only judge stage asks a Choice per judged entry and never caps.

The judge stage sits beside the audit stage in the enrich worker: the entries a
pattern cannot decide — a selected concept that carries forbidden variants —
ask the judge one categorising Choice each through the shared decisions owner.
Its results join the same per-name findings the reviewer receives, marked
report-only, and are never booked as error-lint findings, so they never cap a
score.

The decisions call is stubbed at the judgment layer's call seam
(``acall_decisions``), so the batch core runs for real and no test reaches the
live endpoint.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from imas_codex.discovery.base import judgment
from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.base.supervision import PipelinePhase
from imas_codex.standard_names.review import pipeline as pl
from imas_codex.standard_names.review.judge_stage import QUESTION_NAME

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
    """Classify one judged entry from the documentation the stage carries."""
    concept = state["concept"]["token"]
    doc = state["entry"]["documentation"]
    if concept == "flux_surface_averaged":
        choice = "forbidden" if ("dl/B" in doc or "dl_p/B_p" in doc) else "canonical"
    elif concept == "plasma_boundary":
        choice = "forbidden" if "separatrix" in doc else "canonical"
    else:  # pragma: no cover - the registry holds no other forbidden concept yet
        choice = "absent"
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


def _batch(names) -> dict:
    return {"names": names, "audit_findings": [], "error_lint_findings": {}}


def _state(batch: dict):
    return SimpleNamespace(
        review_batches=[batch],
        all_names=[],
        neighborhood_k=10,
        audit_report=None,
        skip_audit=False,
        enrich_stats=WorkerStats(),
        enrich_phase=PipelinePhase("enrich"),
        should_stop=lambda: False,
    )


async def _run_enrich(monkeypatch, names) -> dict:
    """Drive the enrich worker, neutralising the context lookups it also does."""
    from imas_codex.standard_names.review import enrichment

    monkeypatch.setattr(
        enrichment, "build_neighborhood_context", lambda *a, **k: [], raising=True
    )
    monkeypatch.setattr(pl, "_fetch_review_dd_context", lambda *a, **k: None)
    monkeypatch.setattr(pl, "_fetch_review_derived_children", lambda *a, **k: None)

    batch = _batch(names)
    await pl.enrich_review_worker(_state(batch))
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
    # concepts.
    assert len(seen) == 4
    assert [s["concept"]["token"] for s in seen] == [
        "flux_surface_averaged",
        "plasma_boundary",
        "flux_surface_averaged",
        "plasma_boundary",
    ]

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
    # The report-only findings are absent from the cap's input mapping.
    assert batch["error_lint_findings"] == {}

    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})
    scored = [{"id": CARBON_ID, "reviewer_score": 0.9}]
    pl._cap_docs_scores_for_lint_errors(scored, batch["error_lint_findings"], wlog)

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
