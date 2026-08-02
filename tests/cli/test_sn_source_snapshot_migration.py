"""CLI safety coverage for bounded source maintenance."""

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


def test_reviewer_help_names_only_configured_profiles() -> None:
    runner = CliRunner()

    run_help = runner.invoke(sn, ["run", "--help"])
    review_help = runner.invoke(sn, ["review", "--help"])

    assert run_help.exit_code == 0, run_help.output
    assert review_help.exit_code == 0, review_help.output
    assert "quality-cost-balanced" in run_help.output
    assert "quality-cost-balanced" in review_help.output
    assert "--reviewer-profile pilot" not in run_help.output
    assert "--reviewer-profile pilot" not in review_help.output


def test_bounded_drain_refuses_explicit_single_reviewer_profile() -> None:
    with (
        patch(
            "imas_codex.settings.get_sn_review_profile_models",
            return_value=["openrouter/anthropic/one-reviewer"],
        ),
        patch(
            "imas_codex.standard_names.sources_manifest.resolve_batch_token"
        ) as resolve,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "run",
                "--drain-batch",
                "bounded.json",
                "--reviewer-profile",
                "opus-only",
                "--dry-run",
            ],
        )

    assert result.exit_code != 0
    assert "at least two reviewer models" in result.output
    resolve.assert_not_called()


def test_bounded_drain_refuses_environment_single_reviewer_profile() -> None:
    with (
        patch(
            "imas_codex.settings.get_sn_review_profile_models",
            return_value=["openrouter/anthropic/one-reviewer"],
        ),
        patch(
            "imas_codex.standard_names.sources_manifest.resolve_batch_token"
        ) as resolve,
    ):
        result = CliRunner().invoke(
            sn,
            ["run", "--drain-batch", "bounded.json", "--dry-run"],
            env={"IMAS_CODEX_SN_REVIEW_PROFILE": "opus-only"},
        )

    assert result.exit_code != 0
    assert "at least two reviewer models" in result.output
    resolve.assert_not_called()


def test_source_snapshot_migration_is_dry_run_by_default(tmp_path: Path) -> None:
    manifest = tmp_path / "bounded.json"
    manifest.write_text("{}")
    receipt = {
        "schema": "imas-codex.source-snapshot-migration-receipt",
        "mode": "dry_run",
        "counts": {"planned": 1, "applied": 0},
        "receipt_hash": "abc",
    }
    with patch(
        "imas_codex.standard_names.source_snapshot_migration.migrate_source_snapshots",
        return_value=receipt,
    ) as migrate:
        result = CliRunner().invoke(
            sn,
            [
                "migrate-source-snapshots",
                "--manifest",
                str(manifest),
                "--from-version",
                "old-dd",
                "--reason",
                "refresh immutable authority",
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == receipt
    assert migrate.call_args.kwargs["apply"] is False
    assert migrate.call_args.kwargs["expected_manifest_hash"] is None


def test_source_snapshot_migration_requires_explicit_apply_flag(tmp_path: Path) -> None:
    manifest = tmp_path / "bounded.json"
    manifest.write_text("{}")
    receipt = {
        "schema": "imas-codex.source-snapshot-migration-receipt",
        "mode": "applied",
        "counts": {"planned": 1, "applied": 1},
        "receipt_hash": "def",
    }
    with patch(
        "imas_codex.standard_names.source_snapshot_migration.migrate_source_snapshots",
        return_value=receipt,
    ) as migrate:
        result = CliRunner().invoke(
            sn,
            [
                "migrate-source-snapshots",
                "--manifest",
                str(manifest),
                "--from-version",
                "old-dd",
                "--reason",
                "refresh immutable authority",
                "--apply",
                "--manifest-sha256",
                "a" * 64,
            ],
        )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == receipt
    assert migrate.call_args.kwargs["apply"] is True
    assert migrate.call_args.kwargs["expected_manifest_hash"] == "a" * 64


def test_source_snapshot_apply_requires_manifest_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "bounded.json"
    manifest.write_text("{}")
    with patch(
        "imas_codex.standard_names.source_snapshot_migration.migrate_source_snapshots"
    ) as migrate:
        result = CliRunner().invoke(
            sn,
            [
                "migrate-source-snapshots",
                "--manifest",
                str(manifest),
                "--from-version",
                "old-dd",
                "--reason",
                "refresh immutable authority",
                "--apply",
            ],
        )

    assert result.exit_code != 0
    assert "--apply requires --manifest-sha256" in result.output
    migrate.assert_not_called()


def test_source_snapshot_dry_run_may_verify_manifest_hash(tmp_path: Path) -> None:
    manifest = tmp_path / "bounded.json"
    manifest.write_text("{}")
    receipt = {"mode": "dry_run"}
    with patch(
        "imas_codex.standard_names.source_snapshot_migration.migrate_source_snapshots",
        return_value=receipt,
    ) as migrate:
        result = CliRunner().invoke(
            sn,
            [
                "migrate-source-snapshots",
                "--manifest",
                str(manifest),
                "--from-version",
                "old-dd",
                "--reason",
                "refresh immutable authority",
                "--manifest-sha256",
                "b" * 64,
            ],
        )

    assert result.exit_code == 0, result.output
    assert migrate.call_args.kwargs["apply"] is False
    assert migrate.call_args.kwargs["expected_manifest_hash"] == "b" * 64
