"""Runtime behavior of graph-backed DD resolution authority."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from imas_codex.graph.models import DDResolutionField, DDResolutionValueKind
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionCollision,
    DDResolutionEvidenceMismatch,
    DDResolutionManifest,
    DDResolutionManifestInvalid,
    DDResolutionRecord,
    DDResolutionStale,
    DDResolutionValue,
    RawDDContext,
    load_dd_resolution_manifest,
    resolve_dd_context,
    resolve_dd_field,
)

_PATH = "camera_ir/channel/camera/direction/x"


def _value(value: str | None) -> DDResolutionValue:
    kind = DDResolutionValueKind.null if value is None else DDResolutionValueKind.string
    return DDResolutionValue(kind=kind, value=value)


def _record(**updates: object) -> DDResolutionRecord:
    values = {
        "id": "dd_resolution:" + "a" * 64,
        "gap_id": f"dd_gap:{_PATH}:unit_defect",
        "path": _PATH,
        "dd_version": "4.1.1",
        "field": "unit",
        "observed": _value("m"),
        "effective": _value("1"),
        "reason": "The published unit contradicts the dimensionless direction component.",
        "recorded_by": "standard-names-maintainer",
        "recorded_at": datetime(2026, 8, 17, tzinfo=UTC),
        "upstream_reference": "https://github.com/iterorganization/IMAS-Data-Dictionary/pull/242",
        "upstream_commit_reference": "cb0d86de388dbbdf62acca36de7b7f8c62bb9889",
        "retiring_release": "none-yet",
        "state": "active",
    }
    values.update(updates)
    return DDResolutionRecord.model_validate(values)


def _graph_row(**updates: object) -> dict[str, object]:
    record = _record()
    properties = {
        "id": record.id,
        "path": record.path,
        "dd_version": record.dd_version,
        "field": record.field.value,
        "published_kind": record.observed.kind.value,
        "published_value": json.dumps(record.observed.value),
        "effective_kind": record.effective.kind.value,
        "effective_value": json.dumps(record.effective.value),
        "reason": record.reason,
        "recorded_by": record.recorded_by,
        "recorded_at": record.recorded_at.isoformat(),
        "upstream_reference": record.upstream_reference,
        "upstream_commit_reference": record.upstream_commit_reference,
        "retiring_release": record.retiring_release,
        "status": record.state.value,
    }
    row: dict[str, object] = {
        "properties": properties,
        "source_paths": [record.path],
        "gap_ids": [record.gap_id],
        "version_ids": [record.dd_version],
    }
    row.update(updates)
    return row


class Reader:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def read_resolutions(self):
        return self.rows


def test_graph_loader_preserves_two_gates_and_plain_audit_trail() -> None:
    manifest = load_dd_resolution_manifest(graph_reader=Reader([_graph_row()]))
    [record] = manifest.resolutions

    assert record.gap_id == f"dd_gap:{_PATH}:unit_defect"
    assert record.upstream_reference.endswith("/pull/242")
    assert record.recorded_by == "standard-names-maintainer"
    assert record.recorded_at == datetime(2026, 8, 17, tzinfo=UTC)
    assert "published unit" in record.reason


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"gap_ids": []}, "EVIDENCED_BY"),
        ({"gap_ids": ["gap:a", "gap:b"]}, "EVIDENCED_BY"),
        ({"source_paths": []}, "one bridge"),
        ({"version_ids": []}, "DD-version"),
    ],
)
def test_graph_loader_refuses_missing_or_ambiguous_edges(updates, message) -> None:
    with pytest.raises(DDResolutionEvidenceMismatch, match=message):
        load_dd_resolution_manifest(graph_reader=Reader([_graph_row(**updates)]))


def test_graph_loader_refuses_missing_upstream_marker() -> None:
    row = _graph_row()
    row["properties"] = {**row["properties"], "upstream_reference": ""}
    with pytest.raises(DDResolutionEvidenceMismatch, match="upstream reference"):
        load_dd_resolution_manifest(graph_reader=Reader([row]))


def test_graph_loader_refuses_empty_or_unavailable_authority() -> None:
    with pytest.raises(DDResolutionManifestInvalid, match="empty"):
        load_dd_resolution_manifest(graph_reader=Reader([]))

    class Broken:
        def read_resolutions(self):
            raise OSError("unavailable")

    with pytest.raises(DDResolutionManifestInvalid, match="cannot read"):
        load_dd_resolution_manifest(graph_reader=Broken())


def test_exact_published_value_resolves_with_raw_provenance() -> None:
    record = _record()
    result = resolve_dd_field(
        path=_PATH,
        dd_version="4.1.1",
        field=DDResolutionField.unit,
        raw_value=_value("m"),
        manifest=DDResolutionManifest(resolutions=(record,)),
    )
    assert result.raw.value == "m"
    assert result.effective.value == "1"
    assert result.applied
    assert result.provenance == record


def test_graph_corrected_value_converges_without_overlay() -> None:
    record = _record()
    result = resolve_dd_field(
        path=_PATH,
        dd_version="4.1.1",
        field=DDResolutionField.unit,
        raw_value=_value("1"),
        manifest=DDResolutionManifest(resolutions=(record,)),
    )
    assert result.effective.value == "1"
    assert result.converged
    assert not result.applied


def test_third_value_is_stale() -> None:
    with pytest.raises(DDResolutionStale, match="neither"):
        resolve_dd_field(
            path=_PATH,
            dd_version="4.1.1",
            field=DDResolutionField.unit,
            raw_value=_value("kg"),
            manifest=DDResolutionManifest(resolutions=(_record(),)),
        )


def test_duplicate_exact_key_is_refused() -> None:
    with pytest.raises(DDResolutionCollision, match="multiple active"):
        DDResolutionManifest(
            resolutions=(_record(), _record(id="dd_resolution:" + "b" * 64))
        )


def test_inactive_history_before_active_replacement_is_valid() -> None:
    inactive = _record(state="withdrawn")
    active = _record(id="dd_resolution:" + "b" * 64)

    manifest = DDResolutionManifest(resolutions=(inactive, active))

    assert manifest.resolutions == (inactive, active)


def test_active_replacement_before_inactive_history_is_valid() -> None:
    active = _record()
    inactive = _record(id="dd_resolution:" + "b" * 64, state="withdrawn")

    manifest = DDResolutionManifest(resolutions=(active, inactive))

    assert manifest.resolutions == (active, inactive)


def test_nested_context_retains_raw_and_graph_snapshot_marker() -> None:
    manifest = DDResolutionManifest(resolutions=(_record(),))
    result = resolve_dd_context(
        RawDDContext(path=_PATH, dd_version="4.1.1", unit="1"),
        manifest=manifest,
    )
    projected = result.as_pipeline_item()
    assert result.unit == "1"
    assert projected["raw_dd_context"]["unit"] == "1"
    assert projected["_dd_resolution_marker"] == "resolved-dd-context"
    assert projected["dd_resolution_manifest_digest"] == manifest.digest
