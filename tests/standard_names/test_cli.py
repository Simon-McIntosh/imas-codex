"""CLI safety contract for exact terminal attachment recovery."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


def test_terminal_attachment_recovery_is_dry_run_by_default(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "terminal-recovery.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = {
        "schema": "imas-codex.terminal-attachment-recovery-receipt",
        "mode": "dry_run",
        "counts": {"planned": 9, "applied": 0},
        "receipt_hash": "abc",
    }
    with patch(
        "imas_codex.standard_names.attachment_audit.recover_terminal_attachments",
        return_value=receipt,
    ) as recover:
        result = CliRunner().invoke(
            sn,
            [
                "recover-terminal-attachments",
                "--manifest",
                str(manifest),
                "--reason",
                "recover exact terminal source bindings",
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == receipt
    assert recover.call_args.kwargs["apply"] is False
    assert recover.call_args.kwargs["expected_manifest_hash"] is None


def test_terminal_attachment_recovery_apply_is_hash_bound(tmp_path: Path) -> None:
    manifest = tmp_path / "terminal-recovery.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = {"mode": "applied"}
    with patch(
        "imas_codex.standard_names.attachment_audit.recover_terminal_attachments",
        return_value=receipt,
    ) as recover:
        result = CliRunner().invoke(
            sn,
            [
                "recover-terminal-attachments",
                "--manifest",
                str(manifest),
                "--reason",
                "recover exact terminal source bindings",
                "--apply",
                "--manifest-sha256",
                "a" * 64,
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == receipt
    assert recover.call_args.kwargs["apply"] is True
    assert recover.call_args.kwargs["expected_manifest_hash"] == "a" * 64


def test_terminal_attachment_recovery_apply_requires_manifest_hash(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "terminal-recovery.json"
    manifest.write_text("{}", encoding="utf-8")
    with patch(
        "imas_codex.standard_names.attachment_audit.recover_terminal_attachments"
    ) as recover:
        result = CliRunner().invoke(
            sn,
            [
                "recover-terminal-attachments",
                "--manifest",
                str(manifest),
                "--reason",
                "recover exact terminal source bindings",
                "--apply",
            ],
        )

    assert result.exit_code != 0
    assert "--apply requires --manifest-sha256" in result.output
    recover.assert_not_called()


def test_structural_closure_reconciliation_dry_run(tmp_path: Path) -> None:
    manifest = tmp_path / "structural-closure.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = {"mode": "dry_run", "counts": {"planned": 1, "refused": 0}}
    with patch(
        "imas_codex.standard_names.structural_closure.reconcile_structural_closure",
        return_value=receipt,
    ) as reconcile:
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-structural-closure",
                "--manifest",
                str(manifest),
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == receipt
    assert reconcile.call_args.kwargs == {
        "dry_run": True,
        "include_accepted": False,
        "expected_manifest_hash": None,
    }


def test_structural_closure_apply_is_hash_bound(tmp_path: Path) -> None:
    manifest = tmp_path / "structural-closure.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = {"mode": "applied"}
    with patch(
        "imas_codex.standard_names.structural_closure.reconcile_structural_closure",
        return_value=receipt,
    ) as reconcile:
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-structural-closure",
                "--manifest",
                str(manifest),
                "--include-accepted",
                "--manifest-sha256",
                "a" * 64,
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == receipt
    assert reconcile.call_args.kwargs == {
        "dry_run": False,
        "include_accepted": True,
        "expected_manifest_hash": "a" * 64,
    }


def test_structural_closure_apply_requires_manifest_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "structural-closure.json"
    manifest.write_text("{}", encoding="utf-8")
    with patch(
        "imas_codex.standard_names.structural_closure.reconcile_structural_closure"
    ) as reconcile:
        result = CliRunner().invoke(
            sn,
            ["reconcile-structural-closure", "--manifest", str(manifest)],
        )

    assert result.exit_code != 0
    assert "apply requires --manifest-sha256 or use --dry-run" in result.output
    reconcile.assert_not_called()
