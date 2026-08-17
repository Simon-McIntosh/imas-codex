from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.graph.models import DDResolutionField, DDResolutionStatus
from imas_codex.standard_names import dd_resolutions as dd_resolution_module
from imas_codex.standard_names.dd_gaps import _evidence_token
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionValue,
    load_dd_resolution_manifest,
    resolve_dd_field,
)

_EMPTY_AUTHORITY = b"schema_version: 1\nresolutions: []\n"
_APPROVABLE_ROW = "U25"
_APPROVABLE_PATH = "equilibrium/time_slice/constraints/pressure/reconstructed"
_OBSERVATION_ID = f"dd_gap_observation:{'a' * 64}"


def _snapshot(**overrides: object) -> dict[str, Any]:
    fact: dict[str, Any] = {
        "id": f"dd_gap:{_APPROVABLE_PATH}:unit_defect",
        "path": _APPROVABLE_PATH,
        "kind": "unit_defect",
        "status": "upstream_issue",
        "example_count": 1,
        "first_seen_at": "2026-08-16T08:00:00Z",
        "last_seen_at": "2026-08-16T08:00:00Z",
        "observed_dd_version": "4.1.1",
        "observed_value": "1",
        "expected_value": "Pa",
        "evidence_rule": "unit_equals_expected",
        "reference_path": None,
        "reference_value": None,
        "registry_backend": None,
        "source_paths": [_APPROVABLE_PATH],
        "observations": [{"id": _OBSERVATION_ID}],
    }
    fact.update(overrides)
    fact["evidence_token"] = _evidence_token(fact)
    return fact


class _FakeGapReader:
    def __init__(self, *snapshots: dict[str, Any] | None) -> None:
        self.snapshots = list(snapshots)
        self.calls = 0

    def get_gap(self, gap_id: str) -> dict[str, Any] | None:
        index = min(self.calls, len(self.snapshots) - 1)
        self.calls += 1
        snapshot = self.snapshots[index]
        return dict(snapshot) if snapshot is not None else None


def _install_reader(monkeypatch, *snapshots: dict[str, Any] | None) -> _FakeGapReader:
    reader = _FakeGapReader(*snapshots)
    monkeypatch.setattr(
        dd_resolution_module,
        "dd_resolution_graph_reader",
        lambda: reader,
    )
    return reader


def _temporary_manifest(tmp_path: Path, monkeypatch) -> Path:
    manifest_path = tmp_path / "dd_resolutions.yaml"
    manifest_path.write_bytes(_EMPTY_AUTHORITY)
    monkeypatch.setattr(
        dd_resolution_module,
        "dd_resolution_manifest_path",
        lambda: manifest_path,
    )
    return manifest_path


def _approval_args(
    row: str,
    *,
    path: str = _APPROVABLE_PATH,
    snapshot: dict[str, Any] | None = None,
    expected_digest: str | None = None,
    revision: int = 1,
) -> list[str]:
    evidence = snapshot or _snapshot()
    return [
        "ddres",
        "approve",
        row,
        "--path",
        path,
        "--gap-kind",
        "unit_defect",
        "--observation-id",
        evidence["observations"][0]["id"],
        "--evidence-token",
        evidence["evidence_token"],
        "--expected-manifest-digest",
        expected_digest or load_dd_resolution_manifest().digest,
        "--actor",
        "catalog-review-board",
        "--reason",
        "Reviewed raw release fact and exact DDGap evidence support the correction.",
        "--revision",
        str(revision),
    ]


def test_list_reports_governed_candidate_authority() -> None:
    result = CliRunner().invoke(sn, ["ddres", "list"])

    assert result.exit_code == 0, result.output
    rows = [line for line in result.output.splitlines() if line.startswith(("U", "O"))]
    assert len(rows) == 21
    statuses = {line.split("\t")[0]: line.split("\t")[-1] for line in rows}
    assert statuses == {
        "U11": "yes",
        "U12": "yes",
        "U13": "yes",
        "U14": "yes",
        "U15": "yes",
        "U16": "yes",
        "U19": "no",
        "U21": "yes",
        "U22": "yes",
        "U25": "yes",
        "U26": "yes",
        "U27": "yes",
        "U28": "yes",
        "U29": "yes",
        "U32": "yes",
        "O17": "yes",
        "O20": "no",
        "O21": "no",
        "O22": "no",
        "O23": "no",
        "O24": "no",
    }


def test_show_reports_approved_candidate_and_upstream_solution() -> None:
    result = CliRunner().invoke(sn, ["ddres", "show", _APPROVABLE_ROW])

    assert result.exit_code == 0, result.output
    assert f"candidate: {_APPROVABLE_ROW}" in result.output
    assert f"path: {_APPROVABLE_PATH}" in result.output
    assert (
        "upstream_url: https://github.com/iterorganization/IMAS-Data-Dictionary/pull/281"
        in result.output
    )
    assert "approved: yes" in result.output


def test_approve_refuses_broad_scope_hold_without_writing(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _temporary_manifest(tmp_path, monkeypatch)

    result = CliRunner().invoke(
        sn,
        _approval_args(
            "O20", path="equilibrium/time_slice/constraints/pressure/reconstructed"
        ),
    )

    assert result.exit_code != 0
    assert "broad-scope hold" in result.output
    assert manifest_path.read_bytes() == _EMPTY_AUTHORITY


def test_approve_refuses_candidate_with_unresolved_release_conflict(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = _temporary_manifest(tmp_path, monkeypatch)

    result = CliRunner().invoke(
        sn,
        _approval_args(
            "U19",
            path="edge_profiles/ggd/ion/state/ionisation_potential",
        ),
    )

    assert result.exit_code != 0
    assert "unresolved graph conflict" in result.output
    assert manifest_path.read_bytes() == _EMPTY_AUTHORITY


def test_approve_requires_positive_revision(tmp_path: Path, monkeypatch) -> None:
    _temporary_manifest(tmp_path, monkeypatch)
    args = _approval_args(_APPROVABLE_ROW)
    args[-1] = "0"

    result = CliRunner().invoke(sn, args)

    assert result.exit_code != 0
    assert "0 is not in the range" in result.output


def test_approve_refuses_malformed_ddgap_evidence(tmp_path: Path, monkeypatch) -> None:
    _temporary_manifest(tmp_path, monkeypatch)
    _install_reader(monkeypatch, _snapshot())
    args = _approval_args(_APPROVABLE_ROW)
    args[args.index(_OBSERVATION_ID)] = "not-an-observation"

    result = CliRunner().invoke(sn, args)

    assert result.exit_code != 0
    assert "observation set differs from the reviewed set" in result.output


def test_approve_promotes_one_exact_path_with_strict_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _temporary_manifest(tmp_path, monkeypatch)
    snapshot = _snapshot()
    reader = _install_reader(monkeypatch, snapshot, snapshot)

    result = CliRunner().invoke(sn, _approval_args(_APPROVABLE_ROW, snapshot=snapshot))

    assert result.exit_code == 0, result.output
    assert "approved U25" in result.output
    document = yaml.safe_load(manifest_path.read_text())
    assert len(document["resolutions"]) == 1
    record = document["resolutions"][0]
    assert record["path"] == _APPROVABLE_PATH
    assert record["approved_by"] == "catalog-review-board"
    assert record["reason"].startswith("Reviewed raw release fact")
    assert record["resolution_revision"] == 1
    assert record["approval_receipt"].startswith("dd-resolution-approval:U25:sha256:")
    assert record["state"] == "active"
    assert reader.calls == 2

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
    snapshot = _snapshot()
    _install_reader(monkeypatch, snapshot, snapshot)
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
            "--expected-manifest-digest",
            load_dd_resolution_manifest().digest,
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
    original, successor = manifest.resolutions
    original_payload = original.model_dump(mode="json", exclude={"id", "state"})
    successor_payload = successor.model_dump(mode="json", exclude={"id", "state"})
    assert successor_payload == original_payload


def test_fabricated_digest_shaped_evidence_is_refused(tmp_path, monkeypatch) -> None:
    manifest_path = _temporary_manifest(tmp_path, monkeypatch)
    snapshot = _snapshot()
    _install_reader(monkeypatch, snapshot)
    args = _approval_args(_APPROVABLE_ROW, snapshot=snapshot)
    args[args.index(snapshot["evidence_token"])] = f"dd-gap-evidence:{'b' * 64}"

    result = CliRunner().invoke(sn, args)

    assert result.exit_code != 0
    assert "canonical graph evidence token" in result.output
    assert manifest_path.read_bytes() == _EMPTY_AUTHORITY


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"path": "equilibrium/other"}, "path"),
        ({"kind": "self_contradiction"}, "kind"),
        ({"observed_dd_version": "4.0.0"}, "DD version"),
        ({"observed_value": "Pa"}, "observed value"),
    ],
)
def test_graph_snapshot_must_match_candidate_exactly(
    tmp_path, monkeypatch, override, message
) -> None:
    _temporary_manifest(tmp_path, monkeypatch)
    snapshot = _snapshot(**override)
    _install_reader(monkeypatch, snapshot)

    result = CliRunner().invoke(sn, _approval_args(_APPROVABLE_ROW, snapshot=snapshot))

    assert result.exit_code != 0
    assert message in result.output


def test_graph_change_after_guard_refuses_mutation(tmp_path, monkeypatch) -> None:
    manifest_path = _temporary_manifest(tmp_path, monkeypatch)
    reviewed = _snapshot()
    changed = _snapshot(
        observations=[
            {"id": _OBSERVATION_ID},
            {"id": f"dd_gap_observation:{'c' * 64}"},
        ],
        example_count=2,
        last_seen_at="2026-08-16T09:00:00Z",
    )
    _install_reader(monkeypatch, reviewed, changed)

    result = CliRunner().invoke(sn, _approval_args(_APPROVABLE_ROW, snapshot=reviewed))

    assert result.exit_code != 0
    assert "changed during approval" in result.output
    assert manifest_path.read_bytes() == _EMPTY_AUTHORITY


def test_stale_manifest_digest_refuses_lost_update(tmp_path, monkeypatch) -> None:
    _temporary_manifest(tmp_path, monkeypatch)
    snapshot = _snapshot()
    _install_reader(monkeypatch, snapshot, snapshot)
    stale_digest = load_dd_resolution_manifest().digest
    first = CliRunner().invoke(
        sn,
        _approval_args(
            _APPROVABLE_ROW,
            snapshot=snapshot,
            expected_digest=stale_digest,
        ),
    )
    assert first.exit_code == 0, first.output

    second = CliRunner().invoke(
        sn,
        _approval_args(
            _APPROVABLE_ROW,
            snapshot=snapshot,
            expected_digest=stale_digest,
            revision=2,
        ),
    )

    assert second.exit_code != 0
    assert "manifest changed" in second.output
    assert len(load_dd_resolution_manifest().resolutions) == 1


def test_stale_revoke_digest_cannot_overwrite_approval(tmp_path, monkeypatch) -> None:
    _temporary_manifest(tmp_path, monkeypatch)
    snapshot = _snapshot()
    _install_reader(monkeypatch, snapshot, snapshot)
    stale_digest = load_dd_resolution_manifest().digest
    approved = CliRunner().invoke(
        sn,
        _approval_args(
            _APPROVABLE_ROW,
            snapshot=snapshot,
            expected_digest=stale_digest,
        ),
    )
    assert approved.exit_code == 0, approved.output
    resolution_id = load_dd_resolution_manifest().resolutions[0].id

    revoked = CliRunner().invoke(
        sn,
        [
            "ddres",
            "revoke",
            resolution_id,
            "--actor",
            "catalog-review-board",
            "--reason",
            "Contradictory evidence requires withdrawal.",
            "--expected-manifest-digest",
            stale_digest,
        ],
    )

    assert revoked.exit_code != 0
    assert "manifest changed" in revoked.output
    assert len(load_dd_resolution_manifest().resolutions) == 1


def test_manifest_lock_serializes_contending_revocations(tmp_path, monkeypatch) -> None:
    _temporary_manifest(tmp_path, monkeypatch)
    snapshot = _snapshot()
    _install_reader(monkeypatch, snapshot, snapshot)
    approved = CliRunner().invoke(
        sn, _approval_args(_APPROVABLE_ROW, snapshot=snapshot)
    )
    assert approved.exit_code == 0, approved.output
    authority = load_dd_resolution_manifest()
    predecessor_digest = authority.digest
    resolution_id = authority.resolutions[0].id

    first_at_write = Event()
    second_at_write = Event()
    release_first = Event()
    write_count_lock = Lock()
    outcome_lock = Lock()
    outcomes: dict[str, object] = {}
    write_count = 0
    original_write = dd_resolution_module._write_manifest

    def pause_first_write(*args, **kwargs) -> None:
        nonlocal write_count
        with write_count_lock:
            write_count += 1
            call_number = write_count
        if call_number == 1:
            first_at_write.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("manifest lock contention release timed out")
        else:
            second_at_write.set()
        original_write(*args, **kwargs)

    def revoke(name: str, changed_at: datetime) -> None:
        try:
            outcome: object = dd_resolution_module.revoke_dd_resolution(
                resolution_id,
                actor=name,
                reason="Concurrent evidence withdrawal test.",
                expected_manifest_digest=predecessor_digest,
                changed_at=changed_at,
            )
        except Exception as exc:  # noqa: BLE001 - the exact loser is asserted below
            outcome = exc
        with outcome_lock:
            outcomes[name] = outcome

    monkeypatch.setattr(dd_resolution_module, "_write_manifest", pause_first_write)
    first = Thread(
        target=revoke,
        args=("first-contender", datetime(2026, 8, 16, 12, tzinfo=UTC)),
    )
    second = Thread(
        target=revoke,
        args=("second-contender", datetime(2026, 8, 16, 13, tzinfo=UTC)),
    )
    first.start()
    assert first_at_write.wait(timeout=5)
    second.start()
    second_reached_write_while_locked = second_at_write.wait(timeout=0.25)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert second_reached_write_while_locked is False
    assert write_count == 1
    assert not isinstance(outcomes["first-contender"], Exception)
    assert isinstance(
        outcomes["second-contender"],
        dd_resolution_module.DDResolutionManifestConflict,
    )
    final = load_dd_resolution_manifest()
    assert len(final.resolutions) == 2
    assert len(final.state_changes) == 1
    assert final.state_changes[0].from_resolution_id == resolution_id
    assert final.state_changes[0].actor == "first-contender"


def test_active_key_collision_is_refused(tmp_path, monkeypatch) -> None:
    _temporary_manifest(tmp_path, monkeypatch)
    snapshot = _snapshot()
    _install_reader(monkeypatch, snapshot, snapshot)
    first = CliRunner().invoke(sn, _approval_args(_APPROVABLE_ROW, snapshot=snapshot))
    assert first.exit_code == 0, first.output

    fresh = _snapshot(
        observations=[{"id": f"dd_gap_observation:{'d' * 64}"}],
        last_seen_at="2026-08-16T10:00:00Z",
    )
    _install_reader(monkeypatch, fresh, fresh)
    second = CliRunner().invoke(
        sn,
        _approval_args(
            _APPROVABLE_ROW,
            snapshot=fresh,
            expected_digest=load_dd_resolution_manifest().digest,
            revision=2,
        ),
    )

    assert second.exit_code != 0
    assert "active DD resolution" in second.output


def test_reused_evidence_and_non_increasing_revision_are_refused(
    tmp_path, monkeypatch
) -> None:
    _temporary_manifest(tmp_path, monkeypatch)
    snapshot = _snapshot()
    _install_reader(monkeypatch, snapshot, snapshot)
    runner = CliRunner()
    approved = runner.invoke(sn, _approval_args(_APPROVABLE_ROW, snapshot=snapshot))
    assert approved.exit_code == 0, approved.output
    resolution_id = load_dd_resolution_manifest().resolutions[0].id
    revoked = runner.invoke(
        sn,
        [
            "ddres",
            "revoke",
            resolution_id,
            "--actor",
            "catalog-review-board",
            "--reason",
            "Evidence no longer supports local authority.",
            "--expected-manifest-digest",
            load_dd_resolution_manifest().digest,
        ],
    )
    assert revoked.exit_code == 0, revoked.output

    _install_reader(monkeypatch, snapshot, snapshot)
    reused = runner.invoke(
        sn,
        _approval_args(
            _APPROVABLE_ROW,
            snapshot=snapshot,
            expected_digest=load_dd_resolution_manifest().digest,
            revision=2,
        ),
    )
    assert reused.exit_code != 0
    assert "already used" in reused.output

    fresh = _snapshot(
        observations=[{"id": f"dd_gap_observation:{'e' * 64}"}],
        last_seen_at="2026-08-16T11:00:00Z",
    )
    _install_reader(monkeypatch, fresh, fresh)
    stale_revision = runner.invoke(
        sn,
        _approval_args(
            _APPROVABLE_ROW,
            snapshot=fresh,
            expected_digest=load_dd_resolution_manifest().digest,
            revision=1,
        ),
    )
    assert stale_revision.exit_code != 0
    assert "must exceed prior revision" in stale_revision.output


def test_blocked_overlap_does_not_inherit_other_candidate_approval(
    tmp_path, monkeypatch
) -> None:
    _temporary_manifest(tmp_path, monkeypatch)
    path = "edge_profiles/ggd/ion/state/ionisation_potential"
    snapshot = _snapshot(
        id=f"dd_gap:{path}:unit_defect",
        path=path,
        observed_value="e",
        expected_value="eV",
        source_paths=[path],
    )
    _install_reader(monkeypatch, snapshot, snapshot)
    approved = CliRunner().invoke(
        sn,
        _approval_args("O17", path=path, snapshot=snapshot),
    )
    assert approved.exit_code == 0, approved.output

    blocked = CliRunner().invoke(sn, ["ddres", "show", "U19"])

    assert blocked.exit_code == 0
    assert "approved: no" in blocked.output


def test_second_revocation_is_refused(tmp_path, monkeypatch) -> None:
    _temporary_manifest(tmp_path, monkeypatch)
    snapshot = _snapshot()
    _install_reader(monkeypatch, snapshot, snapshot)
    runner = CliRunner()
    approved = runner.invoke(sn, _approval_args(_APPROVABLE_ROW, snapshot=snapshot))
    assert approved.exit_code == 0, approved.output
    resolution_id = load_dd_resolution_manifest().resolutions[0].id
    digest = load_dd_resolution_manifest().digest
    args = [
        "ddres",
        "revoke",
        resolution_id,
        "--actor",
        "catalog-review-board",
        "--reason",
        "Evidence no longer supports local authority.",
        "--expected-manifest-digest",
        digest,
    ]
    first = runner.invoke(sn, args)
    assert first.exit_code == 0, first.output

    args[-1] = load_dd_resolution_manifest().digest
    second = runner.invoke(sn, args)

    assert second.exit_code != 0
    assert "not effective active authority" in second.output
