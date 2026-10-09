"""A bounded map run keeps mapped sources and resumes the remaining work."""

from __future__ import annotations

import asyncio
import time

import pytest

from imas_codex.ids import models, workers
from imas_codex.ids.models import (
    SignalMappingBatch,
    TargetAssignment,
    TargetAssignmentBatch,
    ValidatedMappingResult,
    ValidatedSignalMapping,
)


@pytest.mark.parametrize("stop_kind", ["deadline", "cost_limit"])
def test_bounded_run_persists_mapped_source_and_next_run_skips_it(
    monkeypatch, stop_kind
):
    source_ids = ("jet:first", "jet:second")
    bound: set[str] = set()
    mapping: dict = {}
    mapped_runs: list[list[str]] = []
    selected_reads: list[list[str]] = []
    released_claims: list[list[str]] = []

    class Graph:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def query(self, statement, **params):
            if "RETURN DISTINCT sg.id AS source_id" in statement:
                selected = [
                    source_id for source_id in source_ids if source_id not in bound
                ]
                selected_reads.append(selected)
                return [{"source_id": source_id} for source_id in selected]
            if "MERGE (m:IMASMapping" in statement:
                mapping.update(params)
            return []

    graph = Graph()
    monkeypatch.setattr(workers, "GraphClient", lambda: graph)
    monkeypatch.setattr(
        models,
        "write_mapping_binding",
        lambda binding, gc: bound.add(binding.source_id),
    )
    monkeypatch.setattr(workers, "refresh_mapping_status", lambda *_args: "validated")
    monkeypatch.setattr(
        workers,
        "release_mapping_claims_batch",
        lambda source_ids: released_claims.append(source_ids),
    )

    async def no_assembly(*_args, **_kwargs):
        return
        yield

    def validate(_facility, ids_name, _dd_version, sections, _batches, **_kwargs):
        assert len(sections.assignments) == 1
        assignment = sections.assignments[0]
        return ValidatedMappingResult(
            facility="jet",
            ids_name=ids_name,
            dd_version="4.1.1",
            sections=sections.assignments,
            bindings=[
                ValidatedSignalMapping(
                    source_id=assignment.source_id,
                    target_id=assignment.imas_target_path,
                    confidence=0.9,
                    mapping_type="direct",
                )
            ],
        )

    async def stop_after_one_map(state, _specs, **_kwargs):
        remaining = [source_id for source_id in source_ids if source_id not in bound]
        mapped_runs.append(remaining[:1])
        assignment = TargetAssignment(
            source_id=remaining[0],
            imas_target_path="magnetics/flux_loop/flux/data",
            confidence=0.9,
            reasoning="selected",
        )
        state.assignments["magnetics"] = TargetAssignmentBatch(
            ids_name="magnetics", assignments=[assignment]
        )
        state.mapping_batches["magnetics"] = [
            (
                assignment,
                SignalMappingBatch(
                    ids_name="magnetics",
                    target_path=assignment.imas_target_path,
                    mappings=[],
                ),
            )
        ]
        if len(remaining) > 1:
            state.mapping_claims.add(remaining[-1])
        if stop_kind == "deadline":
            state.deadline = time.time() - 1
        else:
            state.cost.add("map", state.cost_limit, 0)
        state.stop_requested = True

    monkeypatch.setattr(workers, "run_discovery_engine", stop_after_one_map)
    monkeypatch.setattr("imas_codex.ids.mapping.adiscover_assembly", no_assembly)
    monkeypatch.setattr("imas_codex.ids.mapping.validate_mappings", validate)

    first = workers.MappingDiscoveryState(
        facility="jet", target_ids_list=["magnetics"], skip_errors=True
    )
    asyncio.run(workers.run_mapping_engine(first))

    assert bound == {"jet:first"}
    assert selected_reads[0] == list(source_ids)
    assert released_claims == [["jet:second"]]
    assert mapping["status"] == "generated"
    assert mapping["partial"] is True
    assert mapping["unmapped_sources"] == ["jet:second"]
    assert mapping["stop_reason"] == stop_kind

    second = workers.MappingDiscoveryState(
        facility="jet", target_ids_list=["magnetics"], skip_errors=True
    )
    asyncio.run(workers.run_mapping_engine(second))

    assert mapped_runs == [["jet:first"], ["jet:second"]]
    assert selected_reads[1] == ["jet:second"]
    assert bound == set(source_ids)
    assert mapping["partial"] is False
    assert mapping["unmapped_sources"] == []
