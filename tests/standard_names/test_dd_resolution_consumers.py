"""Production consumers use the typed graph resolution seam."""

from collections.abc import Callable

import pytest

from imas_codex.graph.models import DDResolutionField, DDResolutionValueKind
from imas_codex.standard_names import dd_resolutions
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionManifestInvalid,
    DDResolutionStale,
    DDResolutionValue,
    load_dd_resolution_manifest,
    resolve_dd_field,
    resolve_dd_row,
)


@pytest.mark.parametrize(
    ("path", "effective"),
    [
        ("camera_ir/channel/camera/direction/x", "1"),
        (
            "wall/description_ggd/ggd/energy_fluxes/kinetic/neutral/state/incident/values",
            "W.m^-2",
        ),
        ("edge_profiles/ggd/ion/state/ionisation_potential", "eV"),
        ("equilibrium/time_slice/constraints/pressure/reconstructed", "Pa"),
    ],
)
@pytest.mark.graph
def test_live_graph_authority_returns_effective_unit(path: str, effective: str) -> None:
    result = resolve_dd_field(
        path=path,
        dd_version="4.1.1",
        field=DDResolutionField.unit,
        raw_value=DDResolutionValue(kind=DDResolutionValueKind.string, value=effective),
    )
    assert result.effective.value == effective
    assert result.converged
    assert result.provenance is not None


@pytest.mark.graph
def test_live_graph_authority_is_complete() -> None:
    manifest = load_dd_resolution_manifest()
    assert len(manifest.resolutions) == 37
    assert (
        len({(item.path, item.dd_version, item.field) for item in manifest.resolutions})
        == 37
    )


def test_graph_context_reads_effective_value_and_reports_published_provenance() -> None:
    from tests.standard_names.dd_resolution_test_data import (
        SYNTHETIC_ACTIVE_DIRECTION_ROW,
        load_synthetic_resolution_authority,
    )

    manifest = load_synthetic_resolution_authority(SYNTHETIC_ACTIVE_DIRECTION_ROW)
    context = resolve_dd_row(
        {"path": "camera_ir/channel/camera/direction/x", "unit": "1"},
        dd_version="4.1.1",
        manifest=manifest,
    )

    assert context.unit == "1"
    assert context.graph.unit == "1"
    assert context.raw.unit == "1"
    assert context.published.unit == "m"
    assert context.as_pipeline_item()["published_dd_context"]["unit"] == "m"
    assert context.applied_resolution_ids == (
        "dd_resolution:synthetic-active-direction",
    )
    assert context.converged_resolution_ids == ()
    assert context.resolution_provenance[0].observed.value == "m"
    assert context.resolution_provenance[0].upstream_reference == "none-yet"

    with pytest.raises(DDResolutionStale, match="graph value"):
        resolve_dd_row(
            {"path": "camera_ir/channel/camera/direction/x", "unit": "m"},
            dd_version="4.1.1",
            manifest=manifest,
        )


def test_prior_version_bridge_is_converged_without_being_applied() -> None:
    from tests.standard_names.dd_resolution_test_data import (
        SYNTHETIC_ACTIVE_DIRECTION_ROW,
        load_synthetic_resolution_authority,
    )

    manifest = load_synthetic_resolution_authority(SYNTHETIC_ACTIVE_DIRECTION_ROW)
    context = resolve_dd_row(
        {"path": "camera_ir/channel/camera/direction/x", "unit": "1"},
        dd_version="4.2.0",
        manifest=manifest,
    )

    assert context.applied_resolution_ids == ()
    assert context.converged_resolution_ids == (
        "dd_resolution:synthetic-active-direction",
    )


def test_attachment_refuses_published_value_on_active_graph_bridge(monkeypatch) -> None:
    from imas_codex.standard_names.workers import _is_attachment_consistent
    from tests.standard_names.dd_resolution_test_data import (
        SYNTHETIC_ACTIVE_DIRECTION_ROW,
        load_synthetic_resolution_authority,
    )

    manifest = load_synthetic_resolution_authority(SYNTHETIC_ACTIVE_DIRECTION_ROW)
    monkeypatch.setattr(dd_resolutions, "load_dd_resolution_manifest", lambda: manifest)

    with pytest.raises(DDResolutionStale, match="graph value"):
        _is_attachment_consistent(
            "camera_ir/channel/camera/direction/x",
            "x_direction_unit_vector",
            dd_unit="m",
            sn_unit="m",
        )


def test_source_refresh_persists_published_bridge_provenance(monkeypatch) -> None:
    from imas_codex.standard_names.source_refresh import stamp_source_snapshots
    from tests.standard_names.dd_resolution_test_data import (
        SYNTHETIC_ACTIVE_DIRECTION_ROW,
        load_synthetic_resolution_authority,
    )

    manifest = load_synthetic_resolution_authority(SYNTHETIC_ACTIVE_DIRECTION_ROW)
    monkeypatch.setattr(dd_resolutions, "load_dd_resolution_manifest", lambda: manifest)

    class SourceRefreshGraph:
        def __init__(self) -> None:
            self.updates: list[dict] = []

        def query(self, query: str, **parameters):
            if "RETURN sn.id AS sn_id" in query:
                return [
                    {
                        "sn_id": "x_direction_unit_vector",
                        "path": "camera_ir/channel/camera/direction/x",
                        "unit": "1",
                        "documentation": "Direction component.",
                    }
                ]
            self.updates = parameters["updates"]
            return []

    graph = SourceRefreshGraph()
    assert stamp_source_snapshots(gc=graph) == 1
    assert graph.updates == [
        {
            "sn_id": "x_direction_unit_vector",
            "path": "camera_ir/channel/camera/direction/x",
            "unit": "1",
            "documentation": "Direction component.",
            "raw_unit": "m",
            "raw_documentation": "Direction component.",
            "published_dd_context": {
                "path": "camera_ir/channel/camera/direction/x",
                "dd_version": "4.1.1",
                "unit": "m",
                "documentation": "Direction component.",
                "data_type": None,
                "node_type": None,
                "physics_domain": None,
                "cocos_transformation_type": None,
                "cocos_transformation_expression": None,
                "coordinates": [],
                "lifecycle_status": None,
                "lifecycle_version": None,
                "parents": [],
                "members": [],
            },
            "resolution_ids": ["dd_resolution:synthetic-active-direction"],
            "converged_ids": [],
            "manifest_digest": manifest.digest,
            "resolution_marker": "resolved-dd-context",
        }
    ]


@pytest.mark.graph
def test_consumer_boundaries_call_typed_authority(monkeypatch) -> None:
    from imas_codex.standard_names import release_notes, source_refresh
    from imas_codex.standard_names.sources import dd as dd_source
    from imas_codex.standard_names.workers import _is_attachment_consistent

    calls: list[str] = []
    original_rows: Callable = dd_resolutions.resolve_dd_rows
    original_row: Callable = dd_resolutions.resolve_dd_row
    original_load: Callable = dd_resolutions.load_dd_resolution_manifest

    def rows(*args, **kwargs):
        calls.append("extraction")
        return original_rows(*args, **kwargs)

    def row(*args, **kwargs):
        calls.append("source-refresh")
        return original_row(*args, **kwargs)

    def load(*args, **kwargs):
        calls.append("release-caveat")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(dd_resolutions, "resolve_dd_rows", rows)
    monkeypatch.setattr(dd_resolutions, "resolve_dd_row", row)
    monkeypatch.setattr(dd_resolutions, "load_dd_resolution_manifest", load)

    values = [{"path": "camera_ir/channel/camera/direction/x", "unit": "1"}]
    dd_source._apply_typed_dd_resolutions(values, "4.1.1")
    source_refresh._resolved_source_context(
        "edge_profiles/ggd/ion/state/ionisation_potential", "eV", "Ion energy."
    )
    release_summary = release_notes.summarize_dd_gap_facts([])
    accepted, reason = _is_attachment_consistent(
        "camera_ir/channel/camera/direction/x",
        "x_direction_unit_vector",
        dd_unit="1",
        sn_unit="1",
    )
    assert accepted, reason
    direction_bridge = next(
        bridge
        for bridge in release_summary["dd_resolution_bridges"]
        if bridge["path"] == "camera_ir/channel/camera/direction/x"
    )
    assert direction_bridge["published"] == {"kind": "string", "value": "m"}
    assert direction_bridge["effective"] == {"kind": "string", "value": "1"}
    assert direction_bridge["upstream_reference"]
    assert set(calls) >= {
        "extraction",
        "source-refresh",
        "release-caveat",
    }


def test_public_consumer_refuses_graph_authority_failure(monkeypatch) -> None:
    from imas_codex.standard_names.sources import dd as dd_source

    def unavailable():
        raise DDResolutionManifestInvalid("graph authority unavailable")

    monkeypatch.setattr(dd_resolutions, "load_dd_resolution_manifest", unavailable)
    with pytest.raises(DDResolutionManifestInvalid, match="unavailable"):
        dd_source._apply_typed_dd_resolutions(
            [{"path": "camera_ir/channel/camera/direction/x", "unit": "1"}],
            "4.1.1",
        )
