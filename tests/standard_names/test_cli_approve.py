"""Tests for the ``sn approve`` CLI verb (fold a reviewed catalog PR into the graph)."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.promote import ApprovalReport, UndoApprovalReport

MOCK_TARGET = "imas_codex.standard_names.promote.run_approval"


def test_sn_approve_forwards_flags_and_reports():
    report = ApprovalReport(
        threshold=0.85,
        dry_run=True,
        changes_seen=5,
        accepted=["a", "b"],
        staged_for_review=["docs_pending"],
        quarantined=[{"sn_id": "c", "target_id": "c", "score": 0.4}],
        blocked=[],
        unmatched=["d"],
    )
    with patch(MOCK_TARGET, return_value=report) as m:
        result = CliRunner().invoke(
            sn, ["approve", "--isnc", "/tmp/isnc", "--base", "main", "--dry-run"]
        )

    assert result.exit_code == 0, result.output
    assert m.called
    kw = m.call_args.kwargs
    assert kw["base_ref"] == "main"
    assert kw["dry_run"] is True
    assert "2" in result.output  # accepted count surfaced
    assert "staged for review" in result.output.lower()
    assert "complete quorum review" in result.output.lower()


def test_sn_approve_nonzero_exit_on_blocked():
    """Entries that could not be attached (blocked) are an error → nonzero exit."""
    report = ApprovalReport(
        changes_seen=3,
        accepted=["a"],
        blocked=[{"sn_id": "b", "reason": "collision"}],
    )
    with patch(MOCK_TARGET, return_value=report):
        result = CliRunner().invoke(
            sn, ["approve", "--isnc", "/tmp/isnc", "--base", "main"]
        )

    assert result.exit_code != 0, result.output
    assert "blocked" in result.output.lower()


def test_sn_approve_names_refused_promotions_and_exits_nonzero():
    report = ApprovalReport(
        promotion_refused=[
            {
                "sn_id": "electron_temperature",
                "target_id": "electron_temperature",
                "reason": "catalog lifecycle promotion preconditions were not met",
            }
        ]
    )
    with patch(MOCK_TARGET, return_value=report):
        result = CliRunner().invoke(
            sn, ["approve", "--isnc", "/tmp/isnc", "--base", "main"]
        )

    assert result.exit_code != 0, result.output
    assert "promotion" in result.output.lower()
    assert "electron_temperature" in result.output
    assert "preconditions were not met" in result.output


def test_sn_approve_undo_reports_both_lifecycle_reversals():
    target = "imas_codex.standard_names.promote.undo_approval"
    report = UndoApprovalReport(pr_number=12, demoted=["a"])
    with patch(target, return_value=report):
        result = CliRunner().invoke(
            sn,
            ["approve", "--isnc", "/tmp/isnc", "--pr-number", "12", "--undo"],
        )

    assert result.exit_code == 0, result.output
    assert "approved/active" in result.output
    assert "accepted/draft" in result.output


def test_sn_merge_alias_is_absent():
    result = CliRunner().invoke(sn, ["merge", "--help"])

    assert result.exit_code == 2
    assert "no such command" in result.output.lower()


def test_sn_resolve_keeps_contested_override_semantics():
    target = "imas_codex.standard_names.promote.resolve_contested_override"
    with patch(target, return_value=True) as resolve:
        result = CliRunner().invoke(
            sn,
            [
                "resolve",
                "plasma_current",
                "--override",
                "--reason",
                "Expert adjudication.",
            ],
        )

    assert result.exit_code == 0, result.output
    resolve.assert_called_once_with("plasma_current", reason="Expert adjudication.")
