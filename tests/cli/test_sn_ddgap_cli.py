"""CLI coverage for Data Dictionary defect evidence."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn


def test_human_flag_requires_path_kind_and_reason() -> None:
    result = CliRunner().invoke(sn, ["ddgap", "equilibrium/path"])
    assert result.exit_code == 2
    assert "PATH, --kind" in result.output


def test_human_flag_dry_run_routes_through_instrument() -> None:
    with patch(
        "imas_codex.standard_names.dd_gaps.write_dd_gaps",
        return_value={
            "reported": 1,
            "relationships": 1,
            "ids": ["dd_gap:equilibrium/path:unit_defect"],
            "dry_run": True,
        },
    ) as write:
        result = CliRunner().invoke(
            sn,
            [
                "ddgap",
                "equilibrium/path",
                "--kind",
                "unit_defect",
                "--reason",
                "measured twin declares pressure",
                "--dry-run",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "would flag 1 DD-gap fact" in result.output
    write.assert_called_once_with(
        [
            {
                "path": "equilibrium/path",
                "kind": "unit_defect",
                "reason": "measured twin declares pressure",
                "reporter": "human",
            }
        ],
        dry_run=True,
    )


def test_registry_sync_cannot_mix_human_arguments() -> None:
    result = CliRunner().invoke(sn, ["ddgap", "path", "--sync-registry"])
    assert result.exit_code == 2
    assert "cannot be combined" in result.output


def test_registry_sync_reports_provenance_counts() -> None:
    with patch(
        "imas_codex.standard_names.dd_gaps.sync_dd_unit_exception_gaps",
        return_value={
            "registry_entries": 34,
            "reported": 35,
            "relationships": 450,
            "matched_paths": 370,
            "dry_run": True,
        },
    ):
        result = CliRunner().invoke(sn, ["ddgap", "--sync-registry", "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "would sync 34 registry entries into 35 DD-gap facts" in result.output
    assert "450 path evidence link(s)" in result.output
