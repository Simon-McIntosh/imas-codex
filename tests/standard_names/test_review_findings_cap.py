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
# A distinctive substring of the audit-findings section heading. The heading
# is enclosed by the template's ``{% if audit_findings %}`` guard, so it is
# absent from a render with no findings and appears the moment the guard is
# dropped — which is what the absence test asserts.
AUDIT_SECTION_HEADING = "Layer 1 Audit Findings"


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
    """The audit-findings section does not appear when the batch carries none.

    The assertion names the section heading, which the template's
    ``{% if audit_findings %}`` guard encloses: dropping the guard renders the
    heading even with no findings, so the test reddens on exactly the change it
    guards. A marker only the test could supply would pass either way.
    """
    from imas_codex.llm.prompt_loader import render_prompt

    rendered = render_prompt(
        "sn/review_docs",
        {
            "items": [],
            "audit_findings": [],
        },
    )

    assert AUDIT_SECTION_HEADING not in rendered


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


class _NullGraph:
    """A stand-in for the graph client: it opens, closes and queries nothing.

    The cap takes one client for the whole batch, so a pass that reaches it
    opens one whether or not the stubbed writers use it. No test in this module
    may reach a database, so this stands in for the client.
    """

    def __enter__(self) -> _NullGraph:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def query(self, *_args, **_kwargs) -> list[dict]:
        return []


async def _run_docs_review(
    monkeypatch,
    *,
    error_lint_findings: dict[str, str],
    score_totals: dict[str, int] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Drive ``_review_single_batch`` for a single docs candidate via a stub LLM.

    ``score_totals`` maps a source id to the reviewer's raw total out of 80,
    defaulting to a full-marks review for both candidates. Returns the scored
    items and the review holds the pass raised.
    """
    from imas_codex.standard_names.review import holds, pipeline as pl

    totals = score_totals or {}
    raised: list[dict] = []

    def record_hold(
        name_id: str,
        axis: str,
        kind: str,
        finding: str,
        evidence: str,
        raised_by: str,
        *,
        graph=None,
    ) -> dict:
        raised.append(
            {
                "name_id": name_id,
                "axis": axis,
                "kind": kind,
                "finding": finding,
            }
        )
        return {"id": f"{name_id}:{axis}:{kind}", "finding_key": kind}

    monkeypatch.setattr(holds, "raise_hold", record_hold)
    # The cap reconciles as well as raises: a pass that finds no error clears a
    # hold an earlier pass raised. Stub the writer so the pass runs without a
    # graph and the clearing call is not an unrecorded graph write.
    monkeypatch.setattr(
        holds,
        "clear_holds_not_raised",
        lambda name_id, axis, kinds, raised_keys, *, graph=None: 0,
    )
    import imas_codex.graph.client as graph_client

    monkeypatch.setattr(graph_client, "GraphClient", _NullGraph)

    def fake_render_prompt(prompt_name: str, context: dict) -> str:
        return "stub"

    async def fake_call_llm_structured(**kwargs):
        response_model = kwargs["response_model"]
        reviews = [
            _docs_review("plasma_current", totals.get("plasma_current", 80)),
            _docs_review(
                "electron_temperature", totals.get("electron_temperature", 80)
            ),
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
    return result["_items"], raised


@pytest.mark.asyncio
async def test_error_lint_finding_raises_a_hold_and_keeps_the_judged_score(
    monkeypatch,
) -> None:
    """A document carrying an unresolved error lint error cannot be accepted.

    The finding is raised as a review hold and the reviewer's judged score is
    left exactly as written, so the refusal is recorded as its own dimension
    rather than as a lowered number.
    """
    items, raised = await _run_docs_review(
        monkeypatch,
        error_lint_findings={"plasma_current": ERROR_FINDING},
    )
    by_id = {item["id"]: item for item in items}

    held = by_id["plasma_current"]
    assert held["reviewer_score"] >= DEFAULT_MIN_SCORE, (
        "the judged score must survive an unresolved lint error"
    )
    assert held["audit_cap_finding"] == ERROR_FINDING
    assert [(hold["name_id"], hold["kind"], hold["finding"]) for hold in raised] == [
        ("plasma_current", "docs_lint_error", ERROR_FINDING)
    ]

    # The twin with no error finding keeps the reviewer's own score, unheld.
    untouched = by_id["electron_temperature"]
    assert untouched["reviewer_score"] >= DEFAULT_MIN_SCORE
    assert "audit_cap_finding" not in untouched


@pytest.mark.asyncio
async def test_docs_document_without_error_finding_is_accepted(monkeypatch) -> None:
    """The same document is accepted once the error finding is absent."""
    items, raised = await _run_docs_review(monkeypatch, error_lint_findings={})
    by_id = {item["id"]: item for item in items}

    assert by_id["plasma_current"]["reviewer_score"] >= DEFAULT_MIN_SCORE
    assert "audit_cap_finding" not in by_id["plasma_current"]
    assert raised == []


@pytest.mark.asyncio
async def test_error_finding_is_recorded_on_already_below_acceptance_entry(
    monkeypatch,
) -> None:
    """An entry already below acceptance still records its error finding.

    The reviewer scored this document below the threshold itself, so the hold
    changes no score — but the finding stands against the name and the record
    must name it.
    """
    items, raised = await _run_docs_review(
        monkeypatch,
        error_lint_findings={"plasma_current": ERROR_FINDING},
        score_totals={"plasma_current": 40, "electron_temperature": 40},
    )
    by_id = {item["id"]: item for item in items}

    below = by_id["plasma_current"]
    assert below["reviewer_score"] == 40 / 80
    assert below["audit_cap_finding"] == ERROR_FINDING
    assert [hold["kind"] for hold in raised] == ["docs_lint_error"]

    # A twin below acceptance with no error finding records nothing.
    twin = by_id["electron_temperature"]
    assert "audit_cap_finding" not in twin
