"""CLI safety contracts for exact source-authority reconciliation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


def test_source_authority_reconciliation_is_dry_run_by_default(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "authority.json"
    manifest.write_text("{}")
    receipt = {
        "schema": "imas-codex.source-authority-reconciliation-receipt",
        "mode": "dry_run",
        "operation": "repair_identity_scalar",
        "counts": {"planned": 1, "applied": 0},
        "receipt_hash": "abc",
    }
    with patch(
        "imas_codex.standard_names.source_authority_reconciliation.reconcile_source_authority",
        return_value=receipt,
    ) as reconcile:
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-source-authority",
                "--manifest",
                str(manifest),
                "--reason",
                "repair exact DD authority",
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == receipt
    assert reconcile.call_args.kwargs["apply"] is False
    assert reconcile.call_args.kwargs["expected_manifest_hash"] is None


def test_source_authority_reconciliation_apply_is_hash_bound(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "authority.json"
    manifest.write_text("{}")
    receipt = {"mode": "applied"}
    with patch(
        "imas_codex.standard_names.source_authority_reconciliation.reconcile_source_authority",
        return_value=receipt,
    ) as reconcile:
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-source-authority",
                "--manifest",
                str(manifest),
                "--reason",
                "repair exact DD authority",
                "--apply",
                "--manifest-sha256",
                "a" * 64,
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == receipt
    assert reconcile.call_args.kwargs["apply"] is True
    assert reconcile.call_args.kwargs["expected_manifest_hash"] == "a" * 64


def test_source_authority_reconciliation_apply_requires_manifest_hash(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "authority.json"
    manifest.write_text("{}")
    with patch(
        "imas_codex.standard_names.source_authority_reconciliation.reconcile_source_authority"
    ) as reconcile:
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-source-authority",
                "--manifest",
                str(manifest),
                "--reason",
                "repair exact DD authority",
                "--apply",
            ],
        )

    assert result.exit_code != 0
    assert "--apply requires --manifest-sha256" in result.output
    reconcile.assert_not_called()
