"""CLI safety contracts for exact protected-structure reconciliation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from imas_codex.cli.sn import sn


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    authority = tmp_path / "authority.json"
    authority.write_bytes(b'{"signed":"authority"}\n')
    authority_hash = hashlib.sha256(authority.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(
        json.dumps(
            {
                "catalog_contract": {
                    "authority_evidence_path": str(authority.resolve()),
                    "authority_evidence_sha256": authority_hash,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    return manifest, authority, hashlib.sha256(manifest.read_bytes()).hexdigest()


def _receipt(*, mode: str = "dry_run") -> dict[str, object]:
    return {
        "mode": mode,
        "manifest_hash": "f" * 64,
        "counts": {
            "allowlisted": 2,
            "planned": 2 if mode == "dry_run" else 0,
            "refused": 0,
            "applied": 2 if mode == "applied" else 0,
        },
        "receipt_hash": "a" * 64,
    }


def _assert_receipt_collision_refused(
    manifest: Path,
    authority: Path,
    receipt: Path,
) -> None:
    manifest_bytes = manifest.read_bytes()
    authority_bytes = authority.read_bytes()
    with (
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "reconcile_protected_structure"
        ) as reconcile,
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation.GraphClient"
        ) as graph_client,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-protected-structure",
                "--manifest",
                str(manifest),
                "--authority-artifact",
                str(authority),
                "--receipt",
                str(receipt),
            ],
        )

    assert result.exit_code != 0
    assert "--receipt must not overwrite an input artifact" in result.output
    reconcile.assert_not_called()
    graph_client.assert_not_called()
    assert manifest.read_bytes() == manifest_bytes
    assert authority.read_bytes() == authority_bytes


def test_protected_structure_is_zero_write_dry_run_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    manifest, authority, manifest_hash = _write_inputs(tmp_path)
    operator_receipt = _receipt()
    with (
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "reconcile_protected_structure",
            return_value=operator_receipt,
        ) as reconcile,
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "census_protected_structural_release"
        ) as census,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-protected-structure",
                "--manifest",
                str(manifest),
                "--authority-artifact",
                str(authority),
            ],
        )

    assert result.exit_code == 0, result.output
    reconcile.assert_called_once_with(
        manifest.resolve(),
        apply=False,
        expected_manifest_hash=manifest_hash,
    )
    census.assert_not_called()
    assert "mode=dry_run" in result.output
    assert f"manifest_sha256={manifest_hash}" in result.output
    assert "counts=allowlisted=2,applied=0,planned=2,refused=0" in result.output
    assert "release_census=not_run" in result.output
    receipt_line = next(
        line for line in result.output.splitlines() if line.startswith("receipt=")
    )
    output = Path(receipt_line.removeprefix("receipt="))
    assert output.parent.name == "protected-structural-reconciliation"
    assert json.loads(output.read_bytes()) == {
        "manifest_hash": manifest_hash,
        "receipt": operator_receipt,
        "release_census": None,
    }


def test_protected_structure_apply_requires_manifest_hash_before_operator(
    tmp_path: Path,
) -> None:
    manifest, authority, _ = _write_inputs(tmp_path)
    with (
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "reconcile_protected_structure"
        ) as reconcile,
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation.GraphClient"
        ) as graph_client,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-protected-structure",
                "--manifest",
                str(manifest),
                "--authority-artifact",
                str(authority),
                "--apply",
            ],
        )

    assert result.exit_code != 0
    assert "--apply requires --manifest-sha256" in result.output
    reconcile.assert_not_called()
    graph_client.assert_not_called()


def test_protected_structure_apply_rejects_hash_mismatch_before_operator(
    tmp_path: Path,
) -> None:
    manifest, authority, _ = _write_inputs(tmp_path)
    with (
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "reconcile_protected_structure"
        ) as reconcile,
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation.GraphClient"
        ) as graph_client,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-protected-structure",
                "--manifest",
                str(manifest),
                "--authority-artifact",
                str(authority),
                "--apply",
                "--manifest-sha256",
                "0" * 64,
            ],
        )

    assert result.exit_code != 0
    assert "does not match the exact manifest bytes" in result.output
    reconcile.assert_not_called()
    graph_client.assert_not_called()


def test_protected_structure_rejects_unbound_authority_before_operator(
    tmp_path: Path,
) -> None:
    manifest, authority, _ = _write_inputs(tmp_path)
    authority.write_bytes(b'{"signed":"different"}\n')
    with patch(
        "imas_codex.standard_names.protected_structural_reconciliation."
        "reconcile_protected_structure"
    ) as reconcile:
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-protected-structure",
                "--manifest",
                str(manifest),
                "--authority-artifact",
                str(authority),
            ],
        )

    assert result.exit_code != 0
    assert "authority artifact SHA-256 does not match" in result.output
    reconcile.assert_not_called()


def test_protected_structure_refuses_direct_input_receipt_path(tmp_path: Path) -> None:
    manifest, authority, _ = _write_inputs(tmp_path)

    _assert_receipt_collision_refused(manifest, authority, manifest)


def test_protected_structure_refuses_symlinked_input_receipt_path(
    tmp_path: Path,
) -> None:
    manifest, authority, _ = _write_inputs(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.symlink_to(authority)

    _assert_receipt_collision_refused(manifest, authority, receipt)


def test_protected_structure_refuses_hardlinked_input_receipt_path(
    tmp_path: Path,
) -> None:
    manifest, authority, _ = _write_inputs(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.hardlink_to(manifest)

    _assert_receipt_collision_refused(manifest, authority, receipt)


def test_protected_structure_authorized_apply_forwards_and_certifies(
    tmp_path: Path,
) -> None:
    manifest, authority, manifest_hash = _write_inputs(tmp_path)
    output = tmp_path / "receipt.json"
    operator_receipt = _receipt(mode="applied")
    release_census = {
        "release_ready": True,
        "census_hash": "b" * 64,
    }
    with (
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "reconcile_protected_structure",
            return_value=operator_receipt,
        ) as reconcile,
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "census_protected_structural_release",
            return_value=release_census,
        ) as census,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-protected-structure",
                "--manifest",
                str(manifest),
                "--authority-artifact",
                str(authority),
                "--apply",
                "--manifest-sha256",
                manifest_hash,
                "--receipt",
                str(output),
            ],
        )

    assert result.exit_code == 0, result.output
    reconcile.assert_called_once_with(
        manifest.resolve(),
        apply=True,
        expected_manifest_hash=manifest_hash,
    )
    census.assert_called_once_with(
        manifest.resolve(),
        operator_receipt,
        expected_receipt_hash="a" * 64,
    )
    assert json.loads(output.read_bytes()) == {
        "manifest_hash": manifest_hash,
        "receipt": operator_receipt,
        "release_census": release_census,
    }
    assert "mode=applied" in result.output
    assert "receipt_hash=" + "a" * 64 in result.output
    assert "release_census=release_ready=true,census_hash=" + "b" * 64 in result.output
    assert f"receipt={output.resolve()}" in result.output


def test_protected_structure_receipt_refuses_mismatched_existing_file(
    tmp_path: Path,
) -> None:
    manifest, authority, _ = _write_inputs(tmp_path)
    manifest_bytes = manifest.read_bytes()
    authority_bytes = authority.read_bytes()
    output = tmp_path / "receipt.json"
    prior_receipt = b"complete prior receipt\n"
    output.write_bytes(prior_receipt)
    operator_receipt = _receipt()

    with (
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "reconcile_protected_structure",
            return_value=operator_receipt,
        ),
    ):
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-protected-structure",
                "--manifest",
                str(manifest),
                "--authority-artifact",
                str(authority),
                "--receipt",
                str(output),
            ],
        )

    assert result.exit_code != 0
    assert "different content" in result.output
    assert output.read_bytes() == prior_receipt
    assert manifest.read_bytes() == manifest_bytes
    assert authority.read_bytes() == authority_bytes
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_protected_structure_receipt_write_failure_cleans_temporary_file(
    tmp_path: Path,
) -> None:
    manifest, authority, _ = _write_inputs(tmp_path)
    manifest_bytes = manifest.read_bytes()
    authority_bytes = authority.read_bytes()
    output = tmp_path / "receipt.json"

    with (
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "reconcile_protected_structure",
            return_value=_receipt(),
        ),
        patch(
            "imas_codex.standard_names.receipt_store.os.link",
            side_effect=OSError("install refused"),
        ),
    ):
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-protected-structure",
                "--manifest",
                str(manifest),
                "--authority-artifact",
                str(authority),
                "--receipt",
                str(output),
            ],
        )

    assert result.exit_code != 0
    assert "cannot write receipt: install refused" in result.output
    assert not output.exists()
    assert manifest.read_bytes() == manifest_bytes
    assert authority.read_bytes() == authority_bytes
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_protected_structure_apply_refuses_success_when_receipt_cannot_persist(
    tmp_path: Path,
) -> None:
    manifest, authority, manifest_hash = _write_inputs(tmp_path)
    operator_receipt = _receipt(mode="applied")
    release_census = {"release_ready": True, "census_hash": "b" * 64}

    with (
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "reconcile_protected_structure",
            return_value=operator_receipt,
        ) as reconcile,
        patch(
            "imas_codex.standard_names.protected_structural_reconciliation."
            "census_protected_structural_release",
            return_value=release_census,
        ) as census,
        patch(
            "imas_codex.standard_names.receipt_store.persist_receipt",
            side_effect=OSError("durable storage unavailable"),
        ),
    ):
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-protected-structure",
                "--manifest",
                str(manifest),
                "--authority-artifact",
                str(authority),
                "--apply",
                "--manifest-sha256",
                manifest_hash,
            ],
        )

    assert result.exit_code != 0
    assert "durable storage unavailable" in result.output
    assert "mode=applied" not in result.output
    reconcile.assert_called_once()
    census.assert_called_once()


def test_protected_structure_operator_failure_is_a_cli_error(tmp_path: Path) -> None:
    manifest, authority, _ = _write_inputs(tmp_path)
    with patch(
        "imas_codex.standard_names.protected_structural_reconciliation."
        "reconcile_protected_structure",
        side_effect=ValueError("authority evidence refused"),
    ):
        result = CliRunner().invoke(
            sn,
            [
                "reconcile-protected-structure",
                "--manifest",
                str(manifest),
                "--authority-artifact",
                str(authority),
            ],
        )

    assert result.exit_code != 0
    assert "Error: authority evidence refused" in result.output
    assert "Traceback" not in result.output


def test_protected_structure_help_exposes_only_fail_closed_controls() -> None:
    result = CliRunner().invoke(sn, ["reconcile-protected-structure", "--help"])

    assert result.exit_code == 0, result.output
    help_text = " ".join(result.output.split())
    assert "--manifest" in result.output
    assert "--authority-artifact" in result.output
    assert "--apply" in result.output
    assert "default is a zero-write dry run" in help_text
    assert "--manifest-sha256" in result.output
    assert "required with --apply" in result.output
    assert "--receipt" in result.output
    assert "--force" not in result.output
    assert "--bypass" not in result.output
