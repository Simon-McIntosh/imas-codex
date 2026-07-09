"""Tests for the ``sn rebuild-provenance`` CLI verb.

Mocks the rebuild to avoid graph access, verifying the CLI forwards flags
(dry-run, ref, isnc) and reports the summary.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn

MOCK_TARGET = "imas_codex.standard_names.provenance_rebuild.rebuild_provenance"


def test_rebuild_provenance_dry_run_forwards_flag():
    summary = {
        "orphans_before": 1818,
        "bound_from_map": 1600,
        "bound_derived": 200,
        "bound_manual": 18,
        "dry_run": True,
    }
    with patch(MOCK_TARGET, return_value=summary) as mock_rebuild:
        result = CliRunner().invoke(
            sn, ["rebuild-provenance", "--dry-run", "--ref", "a2f8831"]
        )

    assert result.exit_code == 0, result.output
    assert mock_rebuild.called
    kwargs = mock_rebuild.call_args.kwargs
    assert kwargs["dry_run"] is True
    assert kwargs["ref"] == "a2f8831"
    # the summary counts are surfaced to the operator
    assert "1818" in result.output


def test_rebuild_provenance_reports_residual_orphans_nonzero_exit():
    """If orphans remain after a real (non-dry) rebuild, exit non-zero so the
    operator/CI notices the ledger is still broken.
    """
    summary = {
        "orphans_before": 1818,
        "bound_from_map": 1600,
        "bound_derived": 200,
        "bound_manual": 18,
        "orphans_after": 5,
        "dry_run": False,
    }
    with patch(MOCK_TARGET, return_value=summary):
        result = CliRunner().invoke(sn, ["rebuild-provenance"])

    assert result.exit_code != 0, result.output
    assert "5" in result.output
