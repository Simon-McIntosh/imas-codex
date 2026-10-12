"""Audit findings reach the reviewer, and an error caps the documentation score.

Two behaviours are guarded here. The review prompt must render the batch's
audit findings as deterministic input — a Jinja template that never reads
``audit_findings`` discards them silently, so the test renders the real
template and asserts the finding text is a substring of the result. And an
unresolved error-severity lint finding must hold the documentation score below
the acceptance threshold, so a document the reviewer would otherwise accept is
refused until the finding is gone.
"""

from __future__ import annotations

import logging

import pytest

from imas_codex.standard_names.defaults import DEFAULT_MIN_SCORE
from imas_codex.standard_names.models import (
    StandardNameQualityReviewDocs,
    StandardNameQualityScoreDocs,
)

FINDING_MARKER = "MARKER_AUDIT_FINDING_ROUND_TRIP"
ERROR_FINDING = (
    "[error:lint:round_trip_failure] plasma_current: "
    "name does not survive parser round-trip"
)


# ---------------------------------------------------------------------------
# Rendering — the finding text reaches the reviewer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prompt_name", ["sn/review_docs", "sn/review_names"])
def test_audit_finding_text_reaches_rendered_prompt(prompt_name: str) -> None:
    """A rendered review prompt carries the batch's audit finding text."""
    from imas_codex.llm.prompt_loader import render_prompt

    rendered = render_prompt(
        prompt_name,
        {
            "items": [],
            "audit_findings": [FINDING_MARKER],
        },
    )

    assert FINDING_MARKER in rendered


def test_audit_findings_block_is_absent_when_no_findings() -> None:
    """The audit-findings section does not appear when the batch carries none."""
    from imas_codex.llm.prompt_loader import render_prompt

    rendered = render_prompt(
        "sn/review_docs",
        {
            "items": [],
            "audit_findings": [],
        },
    )

    assert FINDING_MARKER not in rendered


# ---------------------------------------------------------------------------
# Score cap — an error finding refuses the document
# ---------------------------------------------------------------------------


def _docs_review(
    source_id: str, score_total: int = 80
) -> StandardNameQualityReviewDocs:
    """A stubbed docs review that scores a document as acceptable."""
    quarter = score_total // 4
    return StandardNameQualityReviewDocs(
        source_id=source_id,
        standard_name=source_id,
        scores=StandardNameQualityScoreDocs(
            description_quality=quarter,
            documentation_quality=quarter,
            completeness=quarter,
            physics_accuracy=quarter,
        ),
        reasoning="stub reviewer",
    )


async def _run_docs_review(
    monkeypatch,
    *,
    error_lint_findings: dict[str, str],
) -> list[dict]:
    """Drive ``_review_single_batch`` for a single docs candidate via a stub LLM."""
    from imas_codex.standard_names.review import pipeline as pl

    def fake_render_prompt(prompt_name: str, context: dict) -> str:
        return "stub"

    async def fake_call_llm_structured(**kwargs):
        response_model = kwargs["response_model"]
        reviews = [
            _docs_review("plasma_current"),
            _docs_review("electron_temperature"),
        ]
        return response_model(reviews=reviews), 0.0, 0

    import imas_codex.discovery.base.llm as dbllm
    from imas_codex.llm import prompt_loader

    monkeypatch.setattr(prompt_loader, "render_prompt", fake_render_prompt)
    monkeypatch.setattr(dbllm, "acall_llm_structured", fake_call_llm_structured)

    names = [
        {"id": "plasma_current", "source_id": "plasma_current"},
        {"id": "electron_temperature", "source_id": "electron_temperature"},
    ]
    wlog = logging.LoggerAdapter(logging.getLogger("test"), {})

    result = await pl._review_single_batch(
        names=names,
        model="fake/model",
        grammar_enums={},
        compose_ctx={},
        batch_context="",
        neighborhood=[],
        audit_findings=[ERROR_FINDING],
        error_lint_findings=error_lint_findings,
        wlog=wlog,
        target="docs",
    )
    return result["_items"]


@pytest.mark.asyncio
async def test_error_lint_finding_caps_docs_score_below_acceptance(monkeypatch) -> None:
    """A document carrying an unresolved error lint error cannot be accepted."""
    items = await _run_docs_review(
        monkeypatch,
        error_lint_findings={"plasma_current": ERROR_FINDING},
    )
    by_id = {item["id"]: item for item in items}

    capped = by_id["plasma_current"]
    assert capped["reviewer_score"] < DEFAULT_MIN_SCORE, (
        "an unresolved lint error must hold the score below acceptance"
    )
    assert capped["audit_cap_applied"] is True
    assert capped["audit_cap_finding"] == ERROR_FINDING

    # The twin with no error finding keeps the reviewer's own score.
    untouched = by_id["electron_temperature"]
    assert untouched["reviewer_score"] >= DEFAULT_MIN_SCORE
    assert "audit_cap_applied" not in untouched


@pytest.mark.asyncio
async def test_docs_document_without_error_finding_is_accepted(monkeypatch) -> None:
    """The same document is accepted once the error finding is absent."""
    items = await _run_docs_review(monkeypatch, error_lint_findings={})
    by_id = {item["id"]: item for item in items}

    assert by_id["plasma_current"]["reviewer_score"] >= DEFAULT_MIN_SCORE
    assert "audit_cap_applied" not in by_id["plasma_current"]
