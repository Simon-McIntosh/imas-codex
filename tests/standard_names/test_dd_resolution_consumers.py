from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from imas_codex.graph.models import DDResolutionField, DDResolutionValueKind
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionManifestInvalid,
    DDResolutionValue,
    load_dd_resolution_manifest,
    resolve_dd_field,
)


@pytest.mark.parametrize(
    "path,raw,effective",
    [
        ("camera_ir/channel/camera/direction/x", "m", "1"),
        (
            "wall/description_ggd/ggd/energy_fluxes/kinetic/neutral/state/incident/values",
            "m^-2.s^-1",
            "W.m^-2",
        ),
        ("edge_profiles/ggd/ion/state/ionisation_potential", "e", "eV"),
        ("equilibrium/time_slice/constraints/pressure/reconstructed", "1", "Pa"),
    ],
)
def test_packaged_authority_preserves_raw_and_returns_effective_unit(
    path: str, raw: str, effective: str
) -> None:
    receipt = resolve_dd_field(
        path=path,
        dd_version="4.1.1",
        field=DDResolutionField.unit,
        raw_value=DDResolutionValue(
            kind=DDResolutionValueKind.string,
            value=raw,
        ),
    )

    assert receipt.raw.value == raw
    assert receipt.effective.value == effective
    assert receipt.applied is True
    assert receipt.resolution_id
    assert receipt.provenance is not None


def test_consumer_boundaries_call_typed_authority(monkeypatch) -> None:
    from imas_codex.standard_names import dd_resolutions, release_notes, source_refresh
    from imas_codex.standard_names.sources import dd as dd_source
    from imas_codex.standard_names.workers import _is_attachment_consistent

    calls: list[str] = []
    original_rows: Callable = dd_resolutions.resolve_dd_rows
    original_row: Callable = dd_resolutions.resolve_dd_row
    original_field: Callable = dd_resolutions.resolve_dd_field
    original_load: Callable = dd_resolutions.load_dd_resolution_manifest

    def resolve_rows(*args, **kwargs):
        calls.append("extraction")
        return original_rows(*args, **kwargs)

    def resolve_row(*args, **kwargs):
        calls.append("source-refresh")
        return original_row(*args, **kwargs)

    def resolve_field(*args, **kwargs):
        calls.append("attachment")
        return original_field(*args, **kwargs)

    def load_manifest(*args, **kwargs):
        calls.append("release-caveat")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(dd_resolutions, "resolve_dd_rows", resolve_rows)
    monkeypatch.setattr(dd_resolutions, "resolve_dd_row", resolve_row)
    monkeypatch.setattr(dd_resolutions, "resolve_dd_field", resolve_field)
    monkeypatch.setattr(dd_resolutions, "load_dd_resolution_manifest", load_manifest)

    rows = [{"path": "camera_ir/channel/camera/direction/x", "unit": "m"}]
    dd_source._apply_typed_dd_resolutions(rows, "4.1.1")
    source_refresh._resolved_source_context(
        "edge_profiles/ggd/ion/state/ionisation_potential", "e", "Ion energy."
    )
    release_notes.summarize_dd_gap_facts([])
    accepted, reason = _is_attachment_consistent(
        "camera_ir/channel/camera/direction/x",
        "x_direction_unit_vector",
        dd_unit="m",
        sn_unit="1",
    )

    assert accepted, reason
    assert rows[0]["unit"] == "1"
    assert rows[0]["raw_dd_context"]["unit"] == "m"
    assert rows[0]["dd_resolution_ids"]
    assert set(calls) >= {
        "extraction",
        "source-refresh",
        "attachment",
        "release-caveat",
    }


def test_packaged_manifest_strict_load_exposes_governed_state() -> None:
    manifest = load_dd_resolution_manifest()

    assert len(manifest.resolutions) == 37
    assert manifest.digest.startswith("sha256:")


def test_extraction_refuses_invalid_behavior_authority(
    tmp_path: Path, monkeypatch
) -> None:
    from imas_codex.standard_names import dd_resolutions
    from imas_codex.standard_names.sources import dd as dd_source

    invalid = tmp_path / "dd_resolutions.yaml"
    invalid.write_text("schema_version: 1\nresolutions: [\n")
    monkeypatch.setattr(dd_resolutions, "dd_resolution_manifest_path", lambda: invalid)

    with pytest.raises(DDResolutionManifestInvalid, match="not valid YAML"):
        dd_source._apply_typed_dd_resolutions(
            [{"path": "camera_ir/channel/camera/direction/x", "unit": "m"}],
            "4.1.1",
        )


def test_semantic_authority_snapshot_uses_effective_unit() -> None:
    from imas_codex.standard_names.source_authority import authority_snapshot

    snapshot = authority_snapshot(
        "camera_ir/channel/camera/direction/x",
        {
            "properties": {
                "unit": "m",
                "documentation": "Direction component.",
                "data_type": "FLT_0D",
                "lifecycle_status": "active",
            },
            "units": [],
            "parents": [],
            "coordinates": [],
        },
        {"properties": {"id": "4.1.1"}},
    )

    assert snapshot["dd_unit"] == "1"
