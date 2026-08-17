from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from imas_codex.graph.models import DDResolutionField, DDResolutionValueKind
from imas_codex.standard_names.dd_resolutions import (
    DDResolutionError,
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
    assert snapshot["raw_dd_context"]["unit"] == "m"
    assert snapshot["dd_resolution_ids"]
    assert snapshot["dd_resolution_manifest_digest"].startswith("sha256:")
    assert snapshot["_dd_resolution_marker"] == "resolved-dd-context"


def test_extraction_candidates_preserve_graph_row_when_authority_is_empty(
    tmp_path: Path, monkeypatch
) -> None:
    from imas_codex.standard_names import dd_resolutions, graph_ops

    authority = tmp_path / "dd_resolutions.yaml"
    authority.write_text("schema_version: 1\nresolutions: []\n")
    row = {
        "path": "equilibrium/time_slice/global_quantities/ip",
        "description": "Plasma current.",
        "unit": "A",
        "data_type": "FLT_0D",
        "ids_name": "equilibrium",
        "cluster_label": "global quantities",
    }

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query(self, *_args, **_kwargs):
            return [dict(row)]

    monkeypatch.setattr(
        dd_resolutions, "dd_resolution_manifest_path", lambda: authority
    )
    monkeypatch.setattr(graph_ops, "GraphClient", FakeGraph)

    [result] = graph_ops.get_extraction_candidates_dd()

    for key, value in row.items():
        assert result[key] == value
    assert result["raw_dd_context"]["unit"] == "A"
    assert result["_dd_resolution_marker"] == "resolved-dd-context"


def _broken_authority(tmp_path: Path, kind: str) -> Path:
    path = tmp_path / f"{kind}.yaml"
    if kind == "malformed":
        path.write_text("schema_version: 1\nresolutions: [\n")
    return path


@pytest.mark.parametrize("kind", ["malformed", "absent"])
@pytest.mark.asyncio
async def test_review_entry_point_propagates_authority_errors(
    tmp_path: Path, monkeypatch, kind: str
) -> None:
    from imas_codex.standard_names import dd_resolutions
    from imas_codex.standard_names.review import pipeline

    authority = _broken_authority(tmp_path, kind)
    monkeypatch.setattr(
        dd_resolutions, "dd_resolution_manifest_path", lambda: authority
    )

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query(self, query: str, **_kwargs):
            if "RETURN n.id AS id" in query:
                return [
                    {
                        "id": "camera_ir/channel/camera/direction/x",
                        "unit": "m",
                        "description": "Direction.",
                        "documentation": "Direction.",
                    }
                ]
            return []

    batch = {
        "names": [
            {
                "id": "x_direction_unit_vector",
                "source_paths": ["camera_ir/channel/camera/direction/x"],
            }
        ]
    }
    state = SimpleNamespace(
        review_batches=[batch],
        all_names=[],
        neighborhood_k=1,
        audit_report=None,
        enrich_stats=SimpleNamespace(
            total=0, processed=0, record_batch=lambda _n: None
        ),
        enrich_phase=SimpleNamespace(mark_done=lambda: None),
        should_stop=lambda: False,
    )
    monkeypatch.setattr(pipeline, "GraphClient", FakeGraph, raising=False)
    monkeypatch.setattr("imas_codex.graph.client.GraphClient", FakeGraph)
    monkeypatch.setattr(
        "imas_codex.standard_names.review.enrichment.build_neighborhood_context",
        lambda *_args, **_kwargs: [],
    )

    with pytest.raises(DDResolutionError):
        await pipeline.enrich_review_worker(state)


@pytest.mark.asyncio
async def test_review_entry_point_propagates_compose_context_authority_error(
    monkeypatch,
) -> None:
    from imas_codex.standard_names import workers
    from imas_codex.standard_names.review import pipeline

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query(self, *_args, **_kwargs):
            return []

    batch = {
        "names": [
            {
                "id": "x_direction_unit_vector",
                "source_paths": ["camera_ir/channel/camera/direction/x"],
            }
        ]
    }
    state = SimpleNamespace(
        review_batches=[batch],
        all_names=[],
        neighborhood_k=1,
        audit_report=None,
        enrich_stats=SimpleNamespace(
            total=0, processed=0, record_batch=lambda _n: None
        ),
        enrich_phase=SimpleNamespace(mark_done=lambda: None),
        should_stop=lambda: False,
    )
    monkeypatch.setattr("imas_codex.graph.client.GraphClient", FakeGraph)
    monkeypatch.setattr(
        "imas_codex.standard_names.review.enrichment.build_neighborhood_context",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        workers,
        "_enrich_batch_items",
        MagicMock(side_effect=DDResolutionManifestInvalid("invalid authority")),
    )

    with pytest.raises(DDResolutionError):
        await pipeline.enrich_review_worker(state)


@pytest.mark.parametrize("kind", ["malformed", "absent"])
@pytest.mark.asyncio
async def test_refine_docs_entry_point_propagates_authority_errors(
    tmp_path: Path, monkeypatch, kind: str
) -> None:
    import asyncio

    from imas_codex.standard_names import dd_resolutions, workers

    authority = _broken_authority(tmp_path, kind)
    monkeypatch.setattr(
        dd_resolutions, "dd_resolution_manifest_path", lambda: authority
    )

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query(self, query: str, **_kwargs):
            if "HAS_STANDARD_NAME" in query:
                return [
                    {
                        "path": "camera_ir/channel/camera/direction/x",
                        "ids": "camera_ir",
                        "unit": "m",
                        "documentation": "Direction.",
                        "description": "Direction.",
                    }
                ]
            return []

    monkeypatch.setattr("imas_codex.graph.client.GraphClient", FakeGraph)
    manager = MagicMock()
    manager.reserve.return_value = None

    with pytest.raises(DDResolutionError):
        await workers.process_refine_docs_batch(
            [{"id": "x_direction_unit_vector"}], manager, asyncio.Event()
        )


def test_effective_context_is_rejected_as_raw_release_evidence() -> None:
    from imas_codex.standard_names.dd_gaps import (
        build_unit_release_facts,
        load_raw_unit_release_facts,
    )
    from imas_codex.standard_names.dd_resolutions import resolve_dd_row

    effective = resolve_dd_row(
        {"path": "camera_ir/channel/camera/direction/x", "unit": "m"},
        dd_version="4.1.1",
    ).as_pipeline_item()

    with pytest.raises(ValueError, match="effective DD context"):
        build_unit_release_facts([effective])
    with pytest.raises(ValueError, match="effective DD context"):
        load_raw_unit_release_facts({effective["path"]: effective})


@pytest.mark.parametrize(
    "extractor_name", ["extract_dd_candidates", "extract_specific_paths"]
)
def test_public_extractors_return_typed_active_and_pass_through_rows(
    monkeypatch, extractor_name: str
) -> None:
    from imas_codex.standard_names.sources import dd as dd_source
    from imas_codex.standard_names.sources.base import ExtractionBatch

    active_path = "camera_ir/channel/camera/direction/x"
    plain_path = "equilibrium/time_slice/global_quantities/ip"
    source_rows = {
        active_path: {
            "path": active_path,
            "unit": "m",
            "unit_from_rel": "m",
            "unit_relationships": ["m"],
            "description": "Direction component.",
            "documentation": "Direction component.",
            "ids_name": "camera_ir",
            "cluster_label": "camera direction",
        },
        plain_path: {
            "path": plain_path,
            "unit": "A",
            "unit_from_rel": "A",
            "unit_relationships": ["A"],
            "description": "Plasma current.",
            "documentation": "Plasma current.",
            "ids_name": "equilibrium",
            "cluster_label": "global quantities",
        },
    }

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query(self, query: str, **kwargs):
            if "DDVersion {is_current: true}" in query:
                return [
                    {"dd_version": "4.1.1", "cocos_version": None, "cocos_params": None}
                ]
            if "MATCH (n:IMASNode {id: $path})" in query:
                return [dict(source_rows[kwargs["path"]])]
            return [dict(row) for row in source_rows.values()]

    monkeypatch.setattr("imas_codex.graph.client.GraphClient", FakeGraph)
    monkeypatch.setattr(dd_source, "report_extract_breakdown", lambda: {})
    monkeypatch.setattr(
        dd_source, "_apply_unit_overrides", lambda rows, **_kwargs: rows
    )
    monkeypatch.setattr(dd_source, "_qualify_sources", lambda rows, **_kwargs: rows)
    monkeypatch.setattr(
        "imas_codex.standard_names.enrichment.enrich_paths", lambda rows: rows
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.enrichment.group_by_concept_and_unit",
        lambda rows, **_kwargs: [
            ExtractionBatch(source="dd", group_key="typed", items=rows, context={})
        ],
    )
    monkeypatch.setattr(
        "imas_codex.standard_names.enrichment.build_batch_context",
        lambda *_args, **_kwargs: {},
    )

    extractor = getattr(dd_source, extractor_name)
    if extractor_name == "extract_dd_candidates":
        batches = extractor(explicit_paths=list(source_rows), write_skipped=False)
    else:
        batches = extractor(list(source_rows), write_side_effects=False)
    items = {item["path"]: item for batch in batches for item in batch.items}

    assert items[active_path]["unit"] == "1"
    assert items[active_path]["raw_dd_context"]["unit"] == "m"
    assert items[active_path]["dd_resolution_ids"]
    assert items[plain_path]["unit"] == "A"
    assert items[plain_path]["raw_dd_context"]["unit"] == "A"
    assert items[plain_path]["dd_resolution_ids"] == []
    assert all(
        item["_dd_resolution_marker"] == "resolved-dd-context"
        for item in items.values()
    )


@pytest.mark.parametrize(
    "extractor_name", ["extract_dd_candidates", "extract_specific_paths"]
)
def test_public_extractors_refuse_malformed_authority(
    tmp_path: Path, monkeypatch, extractor_name: str
) -> None:
    from imas_codex.standard_names import dd_resolutions
    from imas_codex.standard_names.sources import dd as dd_source

    authority = _broken_authority(tmp_path, "malformed")
    monkeypatch.setattr(
        dd_resolutions, "dd_resolution_manifest_path", lambda: authority
    )
    row = {
        "path": "camera_ir/channel/camera/direction/x",
        "unit": "m",
        "unit_from_rel": "m",
        "unit_relationships": ["m"],
    }

    class FakeGraph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query(self, query: str, **_kwargs):
            if "DDVersion {is_current: true}" in query:
                return [
                    {"dd_version": "4.1.1", "cocos_version": None, "cocos_params": None}
                ]
            return [dict(row)]

    monkeypatch.setattr("imas_codex.graph.client.GraphClient", FakeGraph)
    monkeypatch.setattr(dd_source, "report_extract_breakdown", lambda: {})
    extractor = getattr(dd_source, extractor_name)

    with pytest.raises(DDResolutionError):
        if extractor_name == "extract_dd_candidates":
            extractor(explicit_paths=[row["path"]], write_skipped=False)
        else:
            extractor([row["path"]], write_side_effects=False)
