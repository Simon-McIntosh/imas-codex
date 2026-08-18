"""Public CLI inspection of graph-backed DD resolution authority."""

from datetime import UTC, datetime

from click.testing import CliRunner

from imas_codex.cli.sn import sn
from imas_codex.standard_names import dd_resolutions
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionManifest,
    DDResolutionRecord,
    DDResolutionValue,
)

_ID = "dd_resolution:" + "a" * 64
_PATH = "camera_ir/channel/camera/direction/x"


def _manifest() -> DDResolutionManifest:
    return DDResolutionManifest(
        resolutions=(
            DDResolutionRecord(
                id=_ID,
                gap_id=f"dd_gap:{_PATH}:unit_defect",
                path=_PATH,
                dd_version="4.1.1",
                field="unit",
                observed=DDResolutionValue(kind="string", value="m"),
                effective=DDResolutionValue(kind="string", value="1"),
                reason="The published unit contradicts a direction component.",
                recorded_by="standard-names-maintainer",
                recorded_at=datetime(2026, 8, 17, tzinfo=UTC),
                upstream_reference="none-yet",
                retiring_release="none-yet",
            ),
        )
    )


def test_list_reports_graph_resolution(monkeypatch) -> None:
    monkeypatch.setattr(dd_resolutions, "load_dd_resolution_manifest", _manifest)
    result = CliRunner().invoke(sn, ["ddres", "list"])
    assert result.exit_code == 0, result.output
    assert "ID\tPATH\tDD_VERSION\tFIELD\tPUBLISHED\tEFFECTIVE" in result.output
    assert f"{_ID}\t{_PATH}\t4.1.1\tunit\tm\t1" in result.output


def test_show_reports_two_gates_and_plain_audit(monkeypatch) -> None:
    monkeypatch.setattr(dd_resolutions, "load_dd_resolution_manifest", _manifest)
    result = CliRunner().invoke(sn, ["ddres", "show", _ID])
    assert result.exit_code == 0, result.output
    assert f"evidence: dd_gap:{_PATH}:unit_defect" in result.output
    assert "upstream_reference: none-yet" in result.output
    assert "recorded_by: standard-names-maintainer" in result.output
    assert "recorded_at: 2026-08-17T00:00:00+00:00" in result.output
    assert "reason: The published unit" in result.output


def test_removed_mutation_commands_are_not_exposed() -> None:
    runner = CliRunner()
    for command in ("approve", "revoke"):
        result = runner.invoke(sn, ["ddres", command])
        assert result.exit_code != 0
        assert f"No such command '{command}'" in result.output
