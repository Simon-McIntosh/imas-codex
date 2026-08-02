"""CLI coverage for Data Dictionary defect evidence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names.dd_gaps import DDGapTransitionConflict


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
    ) as sync:
        result = CliRunner().invoke(sn, ["ddgap", "--sync-registry"])

    assert result.exit_code == 0, result.output
    assert "would sync 34 registry entries into 35 DD-gap facts" in result.output
    assert "450 path evidence link(s)" in result.output
    sync.assert_called_once_with(dry_run=True)


def test_registry_sync_mutation_requires_explicit_apply() -> None:
    with patch(
        "imas_codex.standard_names.dd_gaps.sync_dd_unit_exception_gaps",
        return_value={
            "registry_entries": 34,
            "reported": 35,
            "relationships": 450,
            "matched_paths": 370,
            "dry_run": False,
        },
    ) as sync:
        result = CliRunner().invoke(sn, ["ddgap", "--sync-registry", "--apply"])

    assert result.exit_code == 0, result.output
    assert "synced 34 registry entries into 35 DD-gap facts" in result.output
    sync.assert_called_once_with(dry_run=False)


def test_list_uses_exact_read_filters_and_does_not_call_writers() -> None:
    rows = [
        {
            "id": "dd_gap:equilibrium/path:unit_defect",
            "path": "equilibrium/path",
            "kind": "unit_defect",
            "status": "flagged",
            "affected_path_count": 1,
        }
    ]
    with (
        patch(
            "imas_codex.standard_names.dd_gaps.list_dd_gaps", return_value=rows
        ) as listing,
        patch("imas_codex.standard_names.dd_gaps.write_dd_gaps") as write,
        patch("imas_codex.standard_names.dd_gaps.transition_dd_gap") as transition,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "ddgap",
                "equilibrium/path",
                "--list",
                "--kind",
                "unit_defect",
                "--status",
                "flagged",
                "--name",
                "plasma_pressure",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "dd_gap:equilibrium/path:unit_defect\tflagged" in result.output
    listing.assert_called_once_with(
        statuses=["flagged"],
        kinds=["unit_defect"],
        path_ids=["equilibrium/path"],
        name_ids=("plasma_pressure",),
    )
    write.assert_not_called()
    transition.assert_not_called()


def test_show_reports_exact_paths_and_is_read_only() -> None:
    fact = {
        "id": "dd_gap:equilibrium/path:unit_defect",
        "path": "equilibrium/path",
        "kind": "unit_defect",
        "status": "triaged",
        "affected_path_count": 1,
        "observed_dd_version": "4.1.0",
        "observed_value": "1",
        "expected_value": "Pa",
        "evidence_rule": "unit_equals_expected",
        "source_paths": ["equilibrium/path"],
        "affected_name_ids": ["plasma_pressure"],
        "observations": [
            {
                "id": "observation:1",
                "source_path": "equilibrium/path",
                "reporter": "unit-audit",
                "reason": "measured twin declares pressure",
                "observed_dd_version": "4.1.0",
                "observed_value": "1",
                "expected_value": "Pa",
                "evidence_rule": "unit_equals_expected",
                "first_observed_at": "2026-08-01T10:00:00Z",
                "last_observed_at": "2026-08-02T10:00:00Z",
            }
        ],
        "state_changes": [
            {
                "id": "change:1",
                "from_status": "flagged",
                "to_status": "triaged",
                "actor": "operator@example.org",
                "reason": "evidence checked against the declaration",
                "changed_at": "2026-08-02T11:00:00Z",
            }
        ],
    }
    with (
        patch("imas_codex.standard_names.dd_gaps.get_dd_gap", return_value=fact),
        patch("imas_codex.standard_names.dd_gaps.write_dd_gaps") as write,
        patch("imas_codex.standard_names.dd_gaps.transition_dd_gap") as transition,
        patch("imas_codex.standard_names.dd_gaps.reconcile_dd_gaps") as reconcile,
    ):
        result = CliRunner().invoke(
            sn,
            ["ddgap", "--show", "dd_gap:equilibrium/path:unit_defect"],
        )

    assert result.exit_code == 0, result.output
    assert "source_paths:\n  - equilibrium/path" in result.output
    assert "observed_value: 1" in result.output
    assert "expected_value: Pa" in result.output
    assert "evidence_rule: unit_equals_expected" in result.output
    assert "observations (1):" in result.output
    assert "reporter: unit-audit" in result.output
    assert "reason: measured twin declares pressure" in result.output
    assert "state_changes (1):" in result.output
    assert "from_status: flagged" in result.output
    assert "to_status: triaged" in result.output
    assert "actor: operator@example.org" in result.output
    write.assert_not_called()
    transition.assert_not_called()
    reconcile.assert_not_called()


def test_show_surfaces_invalid_or_missing_exact_id() -> None:
    with patch(
        "imas_codex.standard_names.dd_gaps.get_dd_gap",
        side_effect=ValueError("DD-gap id must use the exact form"),
    ):
        invalid = CliRunner().invoke(sn, ["ddgap", "--show", "not-an-id"])
    assert invalid.exit_code == 1
    assert "exact form" in invalid.output

    with patch("imas_codex.standard_names.dd_gaps.get_dd_gap", return_value=None):
        missing = CliRunner().invoke(
            sn, ["ddgap", "--show", "dd_gap:missing/path:unit_defect"]
        )
    assert missing.exit_code == 1
    assert "was not found" in missing.output


def test_human_transition_requires_explicit_apply() -> None:
    result = CliRunner().invoke(
        sn,
        [
            "ddgap",
            "--triage",
            "dd_gap:equilibrium/path:unit_defect",
            "--expected-status",
            "flagged",
            "--to-status",
            "triaged",
            "--actor",
            "operator@example.org",
            "--reason",
            "evidence checked",
        ],
    )
    assert result.exit_code == 2
    assert "requires --apply" in result.output


def test_human_transition_forwards_only_governed_disposition_fields() -> None:
    with (
        patch(
            "imas_codex.standard_names.dd_gaps.transition_dd_gap",
            return_value={
                "id": "dd_gap:equilibrium/path:unit_defect",
                "from_status": "triaged",
                "status": "upstream_issue",
            },
        ) as transition,
        patch("imas_codex.standard_names.dd_gaps.write_dd_gaps") as write,
    ):
        result = CliRunner().invoke(
            sn,
            [
                "ddgap",
                "--triage",
                "dd_gap:equilibrium/path:unit_defect",
                "--expected-status",
                "triaged",
                "--to-status",
                "upstream_issue",
                "--actor",
                "operator@example.org",
                "--reason",
                "maintainers reproduced the declaration defect",
                "--upstream-url",
                "https://example.invalid/issue/1",
                "--apply",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "triaged -> upstream_issue" in result.output
    transition.assert_called_once_with(
        "dd_gap:equilibrium/path:unit_defect",
        expected_status="triaged",
        new_status="upstream_issue",
        actor="operator@example.org",
        reason="maintainers reproduced the declaration defect",
        upstream_url="https://example.invalid/issue/1",
        resolved_dd_version=None,
        registry_backend=None,
        validation_evidence=None,
    )
    write.assert_not_called()


def test_stale_expected_state_is_a_clear_operator_failure() -> None:
    with patch(
        "imas_codex.standard_names.dd_gaps.transition_dd_gap",
        side_effect=DDGapTransitionConflict("expected 'flagged' but it changed"),
    ):
        result = CliRunner().invoke(
            sn,
            [
                "ddgap",
                "--triage",
                "dd_gap:equilibrium/path:unit_defect",
                "--expected-status",
                "flagged",
                "--to-status",
                "rejected",
                "--actor",
                "operator@example.org",
                "--reason",
                "different physical quantity",
                "--apply",
            ],
        )

    assert result.exit_code == 1
    assert "stale DD-gap state" in result.output
    assert "expected 'flagged'" in result.output


def test_release_reconcile_is_read_only_by_default(tmp_path: Path) -> None:
    facts_path = tmp_path / "release-facts.json"
    facts_path.write_text(json.dumps({"equilibrium/path": {"unit": "Pa"}}))
    summary = {
        "dd_version": "4.1.1",
        "evaluated": 1,
        "resolved": 0,
        "would_resolve": ["dd_gap:equilibrium/path:unit_defect"],
        "manual_required": [],
        "unchanged": [],
        "conflicts": [],
        "stale_registry_entries": [],
        "dry_run": True,
    }
    with patch(
        "imas_codex.standard_names.dd_gaps.reconcile_dd_gaps",
        return_value=summary,
    ) as reconcile:
        result = CliRunner().invoke(
            sn,
            [
                "ddgap",
                "--reconcile",
                "4.1.1",
                "--release-facts",
                str(facts_path),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "dry-run: DD=4.1.1" in result.output
    assert "would resolve: dd_gap:equilibrium/path:unit_defect" in result.output
    assert "stale registry entries: none" in result.output
    reconcile.assert_called_once_with(
        "4.1.1",
        {"equilibrium/path": {"unit": "Pa"}},
        require_current=True,
        dry_run=True,
    )


def test_release_reconcile_apply_and_conflict_reporting(tmp_path: Path) -> None:
    facts_path = tmp_path / "release-facts.json"
    facts_path.write_text(json.dumps([{"path": "equilibrium/path", "unit": "Pa"}]))
    summary = {
        "dd_version": "4.1.0",
        "evaluated": 1,
        "resolved": 0,
        "would_resolve": [],
        "manual_required": [],
        "unchanged": [],
        "conflicts": ["dd_gap:equilibrium/path:unit_defect"],
        "stale_registry_entries": ["dd_gap:equilibrium/path:unit_defect"],
        "dry_run": False,
    }
    with patch(
        "imas_codex.standard_names.dd_gaps.reconcile_dd_gaps",
        return_value=summary,
    ) as reconcile:
        result = CliRunner().invoke(
            sn,
            [
                "ddgap",
                "--reconcile",
                "4.1.0",
                "--release-facts",
                str(facts_path),
                "--allow-noncurrent",
                "--apply",
            ],
        )

    assert result.exit_code == 0, result.output
    assert "applied: DD=4.1.0" in result.output
    assert "conflict: dd_gap:equilibrium/path:unit_defect" in result.output
    assert "stale registry entries requiring governed YAML cleanup:" in result.output
    assert "  - dd_gap:equilibrium/path:unit_defect" in result.output
    reconcile.assert_called_once_with(
        "4.1.0",
        {"equilibrium/path": {"unit": "Pa"}},
        require_current=False,
        dry_run=False,
    )


def test_release_facts_reject_patterns_before_reconcile(tmp_path: Path) -> None:
    facts_path = tmp_path / "release-facts.json"
    facts_path.write_text(json.dumps({"*/pressure": {"unit": "Pa"}}))
    with patch("imas_codex.standard_names.dd_gaps.reconcile_dd_gaps") as reconcile:
        result = CliRunner().invoke(
            sn,
            [
                "ddgap",
                "--reconcile",
                "4.1.1",
                "--release-facts",
                str(facts_path),
            ],
        )
    assert result.exit_code == 1
    assert "exact non-pattern paths" in result.output
    reconcile.assert_not_called()
