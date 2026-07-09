"""Tests for the ``sn merge`` CLI verb (fold a reviewed catalog PR into the graph)."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.merge import MergeReport

MOCK_TARGET = "imas_codex.standard_names.merge.run_merge"


def test_sn_merge_forwards_flags_and_reports():
    report = MergeReport(
        threshold=0.85,
        dry_run=True,
        changes_seen=5,
        accepted=["a", "b"],
        quarantined=[{"sn_id": "c", "target_id": "c", "score": 0.4}],
        blocked=[],
        unmatched=["d"],
    )
    with patch(MOCK_TARGET, return_value=report) as m:
        result = CliRunner().invoke(
            sn, ["merge", "--isnc", "/tmp/isnc", "--base", "main", "--dry-run"]
        )

    assert result.exit_code == 0, result.output
    assert m.called
    kw = m.call_args.kwargs
    assert kw["base_ref"] == "main"
    assert kw["dry_run"] is True
    assert "2" in result.output  # accepted count surfaced


def test_sn_merge_nonzero_exit_on_blocked():
    """Entries that could not be attached (blocked) are an error → nonzero exit."""
    report = MergeReport(
        changes_seen=3,
        accepted=["a"],
        blocked=[{"sn_id": "b", "reason": "collision"}],
    )
    with patch(MOCK_TARGET, return_value=report):
        result = CliRunner().invoke(sn, ["merge", "--isnc", "/tmp/isnc", "--base", "main"])

    assert result.exit_code != 0, result.output
    assert "blocked" in result.output.lower()
