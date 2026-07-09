"""Tests for the ``sn reconcile-catalog`` CLI verb (graph restore from catalog)."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.catalog_reconcile import ReconcileReport

MOCK_TARGET = "imas_codex.standard_names.catalog_reconcile.reconcile_catalog"


def test_reconcile_catalog_dry_run_forwards_flag():
    report = ReconcileReport(matched=2100, updated=3, sources_bound=1941, dry_run=True)
    with patch(MOCK_TARGET, return_value=report) as mock_rc:
        result = CliRunner().invoke(
            sn, ["reconcile-catalog", "--isnc", "/tmp/isnc", "--dry-run"]
        )

    assert result.exit_code == 0, result.output
    assert mock_rc.called
    assert mock_rc.call_args.kwargs["dry_run"] is True
    assert "2100" in result.output


def test_reconcile_catalog_reports_missing_names_nonzero_exit():
    """Catalog entries with no graph node are a restore signal → nonzero exit."""
    report = ReconcileReport(matched=2000, missing=["a", "b"], dry_run=False)
    with patch(MOCK_TARGET, return_value=report):
        result = CliRunner().invoke(sn, ["reconcile-catalog", "--isnc", "/tmp/isnc"])

    assert result.exit_code != 0, result.output
    assert "2" in result.output  # missing count surfaced
