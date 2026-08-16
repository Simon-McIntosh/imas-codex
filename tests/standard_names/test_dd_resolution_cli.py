from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.graph.models import DDResolutionField, DDResolutionStatus
from imas_codex.standard_names import dd_resolutions as dd_resolution_module
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionValue,
    load_dd_resolution_manifest,
    resolve_dd_field,
)

_ACTIVE_RESOURCE = (
    Path(__file__).parents[2]
    / "imas_codex"
    / "standard_names"
    / "config"
    / "dd_resolutions.yaml"
)
_APPROVABLE_ROW = "U25"
_APPROVABLE_PATH = "equilibrium/time_slice/constraints/pressure/reconstructed"
_OBSERVATION_ID = f"dd_gap_observation:{'a' * 64}"
_EVIDENCE_TOKEN = f"dd-gap-evidence:{'b' * 64}"


def _temporary_manifest(tmp_path: Path, monkeypatch) -> Path:
    manifest_path = tmp_path / "dd_resolutions.yaml"
    manifest_path.write_bytes(_ACTIVE_RESOURCE.read_bytes())
    monkeypatch.setattr(
        dd_resolution_module,
        "dd_resolution_manifest_path",
        lambda: manifest_path,
    )
    return manifest_path


def _approval_args(row: str, *, path: str = _APPROVABLE_PATH) -> list[str]:
    return [
        "ddres",
        "approve",
        row,
        "--path",
        path,
        "--gap-kind",
        "unit_defect",
        "--observation-id",
        _OBSERVATION_ID,
        "--evidence-token",
        _EVIDENCE_TOKEN,
        "--actor",
        "catalog-review-board",
        "--reason",
        "Reviewed raw release fact and exact DDGap evidence support the correction.",
        "--revision",
        "1",
    ]


def test_list_reports_all_candidates_as_not_approved() -> None:
    result = CliRunner().invoke(sn, ["ddres", "list"])

    assert result.exit_code == 0, result.output
    rows = [line for line in result.output.splitlines() if line.startswith(("U", "O"))]
    assert len(rows) == 21
    assert all(line.endswith("\tno") for line in rows)


def test_show_reports_candidate_and_upstream_solution() -> None:
    result = CliRunner().invoke(sn, ["ddres", "show", _APPROVABLE_ROW])

    assert result.exit_code == 0, result.output
    assert f"candidate: {_APPROVABLE_ROW}" in result.output
    assert f"path: {_APPROVABLE_PATH}" in result.output
    assert (
        "upstream_url: https://github.com/iterorganization/IMAS-Data-Dictionary/pull/281"
        in result.output
    )
    assert "approved: no" in result.output


def test_approve_refuses_broad_scope_hold_without_writing() -> None:
    before = _ACTIVE_RESOURCE.read_bytes()

    result = CliRunner().invoke(
        sn,
        _approval_args(
            "O20", path="equilibrium/time_slice/constraints/pressure/reconstructed"
        ),
    )

    assert result.exit_code != 0
    assert "broad-scope hold" in result.output
    assert _ACTIVE_RESOURCE.read_bytes() == before


def test_approve_refuses_candidate_with_unresolved_release_conflict() -> None:
    before = _ACTIVE_RESOURCE.read_bytes()

    result = CliRunner().invoke(
        sn,
        _approval_args(
            "U19",
            path="edge_profiles/ggd/ion/state/ionisation_potential",
        ),
    )

    assert result.exit_code != 0
    assert "unresolved graph conflict" in result.output
    assert _ACTIVE_RESOURCE.read_bytes() == before


def test_approve_requires_positive_revision() -> None:
    args = _approval_args(_APPROVABLE_ROW)
    args[args.index("1")] = "0"

    result = CliRunner().invoke(sn, args)

    assert result.exit_code != 0
    assert "0 is not in the range" in result.output


def test_approve_refuses_malformed_ddgap_evidence() -> None:
    args = _approval_args(_APPROVABLE_ROW)
    args[args.index(_OBSERVATION_ID)] = "not-an-observation"

    result = CliRunner().invoke(sn, args)

    assert result.exit_code != 0
    assert "content-addressed DDGap observation" in result.output


def test_approve_promotes_one_exact_path_with_strict_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _temporary_manifest(tmp_path, monkeypatch)

    result = CliRunner().invoke(sn, _approval_args(_APPROVABLE_ROW))

    assert result.exit_code == 0, result.output
    assert "approved U25" in result.output
    document = yaml.safe_load(manifest_path.read_text())
    assert len(document["resolutions"]) == 1
    record = document["resolutions"][0]
    assert record["path"] == _APPROVABLE_PATH
    assert record["approved_by"] == "catalog-review-board"
    assert record["reason"].startswith("Reviewed raw release fact")
    assert record["resolution_revision"] == 1
    assert record["approval_receipt"].startswith("dd-resolution-approval:sha256:")
    assert record["state"] == "active"

    manifest = load_dd_resolution_manifest()
    resolved = resolve_dd_field(
        path=_APPROVABLE_PATH,
        dd_version="4.1.1",
        field=DDResolutionField.unit,
        raw_value=DDResolutionValue(kind="string", value="1"),
        manifest=manifest,
    )
    assert resolved.effective.value == "Pa"
    assert resolved.applied is True


def test_revoke_appends_receipt_and_preserves_resolution_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _temporary_manifest(tmp_path, monkeypatch)
    runner = CliRunner()
    approved = runner.invoke(sn, _approval_args(_APPROVABLE_ROW))
    assert approved.exit_code == 0, approved.output
    approved_document = yaml.safe_load(manifest_path.read_text())
    active_id = approved_document["resolutions"][0]["id"]

    revoked = runner.invoke(
        sn,
        [
            "ddres",
            "revoke",
            active_id,
            "--actor",
            "catalog-review-board",
            "--reason",
            "New contradictory evidence requires withdrawal pending review.",
        ],
    )

    assert revoked.exit_code == 0, revoked.output
    document = yaml.safe_load(manifest_path.read_text())
    assert len(document["resolutions"]) == 2
    assert {record["state"] for record in document["resolutions"]} == {
        "active",
        "withdrawn",
    }
    assert len(document["state_changes"]) == 1
    receipt = document["state_changes"][0]
    assert receipt["from_resolution_id"] == active_id
    assert receipt["actor"] == "catalog-review-board"
    assert receipt["from_status"] == "active"
    assert receipt["to_status"] == "withdrawn"
    assert receipt["id"].startswith("dd-resolution-state-change:sha256:")

    manifest = load_dd_resolution_manifest()
    resolved = resolve_dd_field(
        path=_APPROVABLE_PATH,
        dd_version="4.1.1",
        field=DDResolutionField.unit,
        raw_value=DDResolutionValue(kind="string", value="1"),
        manifest=manifest,
    )
    assert resolved.effective.value == "1"
    assert resolved.applied is False
    assert any(
        record.state == DDResolutionStatus.withdrawn for record in manifest.resolutions
    )
