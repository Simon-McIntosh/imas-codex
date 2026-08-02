"""Automated DD-defect evidence remains exact, additive, and non-authoritative."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.models import DDGapEvidence

PATH = "equilibrium/time_slice/profiles_1d/electrons/temperature"
REFERENCE_PATH = "equilibrium/time_slice/profiles_2d/electrons/temperature"


def _evidence(path: str = PATH, *, reference_path: str | None = None) -> DDGapEvidence:
    payload = {
        "path": path,
        "kind": "unit_defect",
        "reason": "The declared unit contradicts the independent quantity definition.",
        "observed_value": "1",
        "expected_value": "Pa",
        "evidence_rule": "unit_equals_expected",
    }
    if reference_path is not None:
        payload["reference_path"] = reference_path
        payload["reference_value"] = "Pa"
    return DDGapEvidence.model_validate(payload)


def test_compose_sanitizer_keeps_only_exact_claimed_dd_evidence() -> None:
    from imas_codex.standard_names.workers import _sanitize_compose_result_sources

    result = SimpleNamespace(
        candidates=[],
        attachments=[],
        skipped=[],
        vocab_gaps=[],
        dd_gaps=[_evidence(), _evidence("unclaimed/path")],
    )

    _sanitize_compose_result_sources(result, {PATH}, phase="generate_name")

    assert [report.path for report in result.dd_gaps] == [PATH]


def test_reference_evidence_must_also_be_in_claimed_batch() -> None:
    from imas_codex.standard_names.workers import _sanitize_dd_gap_evidence

    evidence = [_evidence(reference_path=REFERENCE_PATH)]

    assert _sanitize_dd_gap_evidence(evidence, {PATH}, phase="review_name") == []
    assert (
        _sanitize_dd_gap_evidence(
            evidence,
            {PATH, REFERENCE_PATH},
            phase="review_name",
        )
        == evidence
    )


def test_persisted_evidence_injects_writer_and_dd_version() -> None:
    from imas_codex.standard_names.workers import _persist_dd_gap_evidence

    with patch(
        "imas_codex.standard_names.dd_gaps.write_dd_gaps",
        return_value={"reported": 1},
    ) as write:
        reported = _persist_dd_gap_evidence(
            [_evidence()],
            {PATH},
            phase="generate_name",
            reporter="compose",
            observed_dd_version="4.1.1",
        )

    assert reported == 1
    report = write.call_args.args[0][0]
    assert report["path"] == PATH
    assert report["reporter"] == "compose"
    assert report["observed_dd_version"] == "4.1.1"
    assert "status" not in report


def test_evidence_writer_failure_isolated_from_pipeline_result() -> None:
    from imas_codex.standard_names.workers import _persist_dd_gap_evidence

    with patch(
        "imas_codex.standard_names.dd_gaps.write_dd_gaps",
        side_effect=RuntimeError("graph unavailable"),
    ):
        assert (
            _persist_dd_gap_evidence(
                [_evidence()],
                {PATH},
                phase="review_docs",
                reporter="review-docs",
            )
            == 0
        )


def test_no_evidence_performs_no_graph_write() -> None:
    from imas_codex.standard_names.workers import _persist_dd_gap_evidence

    with patch("imas_codex.standard_names.dd_gaps.write_dd_gaps") as write:
        assert (
            _persist_dd_gap_evidence(
                [],
                {PATH},
                phase="review_name",
                reporter="review-name",
            )
            == 0
        )
    write.assert_not_called()


def test_unit_injection_reports_only_physical_scalar_edge_conflicts() -> None:
    from imas_codex.standard_names.sources.dd import (
        _unit_declaration_conflict_reports,
    )

    rows = [
        {"path": PATH, "unit": "1", "unit_from_rel": "Pa"},
        {
            "path": "equilibrium/same",
            "unit": "W.m^-3",
            "unit_from_rel": "m^-3.W",
        },
        {"path": "equilibrium/missing", "unit": None, "unit_from_rel": "Pa"},
    ]

    reports = _unit_declaration_conflict_reports(rows, "4.1.1")

    assert reports == [
        {
            "path": PATH,
            "kind": "self_contradiction",
            "reason": (
                "The DD node unit property contradicts its authoritative "
                "HAS_UNIT relationship."
            ),
            "observed_dd_version": "4.1.1",
            "observed_value": "1",
            "expected_value": "Pa",
            "evidence_rule": "unit_equals_expected",
            "reporter": "dd-unit-injection",
        }
    ]


def test_unit_injection_writer_failure_does_not_block_extraction() -> None:
    from imas_codex.standard_names.sources.dd import (
        _persist_unit_declaration_conflicts,
    )

    with patch(
        "imas_codex.standard_names.dd_gaps.write_dd_gaps",
        side_effect=RuntimeError("graph unavailable"),
    ):
        assert (
            _persist_unit_declaration_conflicts(
                [{"path": PATH, "unit": "1", "unit_from_rel": "Pa"}],
                "4.1.1",
            )
            == 0
        )


def test_attachment_evidence_uses_fetched_unit_property_and_relationship() -> None:
    from imas_codex.standard_names.attachment_audit import (
        _attachment_dd_gap_evidence,
    )

    reports = _attachment_dd_gap_evidence(
        [
            {
                "dd_path": PATH,
                "dd_declared_unit": "1",
                "dd_relationship_unit": "Pa",
                "dd_version": "4.1.1",
            }
        ]
    )

    assert len(reports) == 1
    assert reports[0]["path"] == PATH
    assert reports[0]["observed_value"] == "1"
    assert reports[0]["expected_value"] == "Pa"
    assert reports[0]["reporter"] == "attachment-audit"
    assert reports[0]["observed_dd_version"] == "4.1.1"


@pytest.mark.asyncio
async def test_review_quorum_collects_evidence_without_changing_score() -> None:
    from imas_codex.standard_names.workers import _run_rd_quorum_cycles

    scores = {"grammar": 18, "semantic": 18, "convention": 18, "completeness": 18}

    async def call_llm(**_kwargs):
        result = SimpleNamespace(
            scores=SimpleNamespace(score=0.9, model_dump=lambda: scores),
            comments=None,
            reasoning="The name satisfies the rubric independently of DD evidence.",
            dd_gaps=[_evidence()],
        )
        return result, 0.01, 100

    result = await _run_rd_quorum_cycles(
        sn_id="electron_temperature",
        review_axis="names",
        response_model=object,
        user_prompt="review",
        system_prompt="system",
        models=["reviewer"],
        disagreement_threshold=0.2,
        rubric_dims=("grammar", "semantic", "convention", "completeness"),
        lease=None,
        phase="review_name",
        acall_llm_structured=call_llm,
    )

    assert result is not None
    assert result["winning_score"] == pytest.approx(0.9)
    assert result["resolution_method"] == "single_review"
    assert result["dd_gaps"] == [_evidence()]


def test_attachment_reconcile_dry_run_never_writes_evidence() -> None:
    from imas_codex.standard_names.attachment_audit import (
        reconcile_attachment_consistency,
    )

    gc = MagicMock()
    gc.query.return_value = [
        {
            "source_node_id": f"dd:{PATH}",
            "dd_path": PATH,
            "sn_id": "electron_temperature",
            "name_stage": "drafted",
            "origin": "pipeline",
            "dd_unit": "Pa",
            "dd_declared_unit": "1",
            "dd_relationship_unit": "Pa",
            "sn_unit": "Pa",
            "other_live_names": 0,
        }
    ]

    with patch("imas_codex.standard_names.dd_gaps.write_dd_gaps") as write:
        result = reconcile_attachment_consistency(gc, dry_run=True)

    assert result.dd_gap_evidence
    write.assert_not_called()
