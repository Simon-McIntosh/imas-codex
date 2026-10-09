"""The assign phase ends once only sources this run already handled remain.

A choice that raises, or a source with no shortlist to choose from, is released
with its route left as ``escalated``. The assign phase's pending-work check must
exclude the sources it has already handled, or an unbounded run never reaches
done: the map loop waits on the assign phase forever and validation never runs.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from imas_codex.ids.models import (
    SignalMappingBatch,
    SignalMappingEntry,
    ValidatedMappingResult,
    ValidatedSignalMapping,
)
from imas_codex.ids.workers import (
    MappingDiscoveryState,
    assign_worker,
    map_worker,
    validate_worker,
)

ESCALATED = "facility:escalated"
SELECTED = "facility:selected"
TARGET = "magnetics/flux_loop/flux"
IDS = "magnetics"
TEST_TIMEOUT = 20.0


def _edge(path: str, ids: str = IDS) -> dict:
    return {
        "path": path,
        "ids": ids,
        "rank": 1,
        "route": True,
        "section": "/".join(path.split("/")[:2]),
        "data_type": "STRUCT_ARRAY",
        "timebasepath": None,
        "ndim": None,
        "arms": [],
    }


def _pending(facility, ids_names=None, handled_source_ids=None):
    """Stand-in for the graph check: the escalated source stays pending until
    it is excluded as handled, mirroring the released-source graph state."""
    return bool({ESCALATED} - set(handled_source_ids or ()))


async def _run_pipeline(state) -> None:
    graph = MagicMock()
    graph.__enter__.return_value = graph

    # --- assign: one escalated source whose choice raises ---
    with (
        patch("imas_codex.ids.workers.GraphClient", return_value=graph),
        patch(
            "imas_codex.ids.workers.get_mapping_route_thresholds",
            return_value=SimpleNamespace(shortlist_size=5),
        ),
        patch(
            "imas_codex.ids.workers.claim_sources_for_escalated",
            return_value=[{"id": ESCALATED, "physics_domain": "magnetics"}],
        ),
        patch(
            "imas_codex.ids.graph_ops.read_candidates",
            return_value={ESCALATED: [_edge(TARGET)]},
        ),
        patch(
            "imas_codex.ids.mapping.achoose_targets",
            new_callable=AsyncMock,
            side_effect=RuntimeError("model named a path outside its shortlist"),
        ),
        patch("imas_codex.ids.workers.release_mapping_claim"),
        patch("imas_codex.ids.workers.has_pending_assignment_work", _pending),
        patch("imas_codex.ids.workers.asyncio.sleep", new_callable=AsyncMock),
    ):
        await assign_worker(state)
    assert state.assign_phase.done, "assign phase did not reach done"
    assert state.assign_stats.errors == 1

    # --- map: the other, already selected source is mapped ---
    claim_seen: list[str] = []

    def _claim(_facility, ids_name, **_kw):
        if claim_seen:
            return []
        claim_seen.append(ids_name)
        return [{"id": SELECTED, "physics_domain": "magnetics"}]

    batch = SignalMappingBatch(
        ids_name=IDS,
        target_path=TARGET,
        mappings=[
            SignalMappingEntry(
                source_id=SELECTED,
                target_id=TARGET,
                confidence=1,
                transform_expression="value",
            )
        ],
    )
    with (
        patch("imas_codex.ids.workers.claim_sources_for_mapping", _claim),
        patch("imas_codex.ids.workers.GraphClient", return_value=graph),
        patch(
            "imas_codex.ids.graph_ops.read_candidates",
            return_value={SELECTED: [_edge(TARGET)]},
        ),
        patch(
            "imas_codex.ids.mapping._prepare_section_context",
            return_value={"prompt": "probe"},
        ),
        patch("imas_codex.ids.mapping._build_messages", return_value=[]),
        patch(
            "imas_codex.ids.mapping._acall_llm",
            new_callable=AsyncMock,
            return_value=batch,
        ),
        patch("imas_codex.ids.workers.refresh_mapping_status", return_value="mapped"),
        patch("imas_codex.ids.workers.has_pending_mapping_work", return_value=False),
        patch("imas_codex.ids.workers.asyncio.sleep", new_callable=AsyncMock),
    ):
        await map_worker(state)
    assert state.map_phase.done, "map phase did not complete"
    assert state.sources_mapped == 1

    # --- validate: the mapped source's binding is persisted ---
    async def _no_assembly(*_a, **_kw):
        return
        yield  # pragma: no cover

    validated = ValidatedMappingResult(
        facility="facility",
        ids_name=IDS,
        dd_version="4.1.1",
        sections=[],
        bindings=[
            ValidatedSignalMapping(
                source_id=SELECTED,
                target_id=TARGET,
                confidence=1,
                mapping_type="direct",
            )
        ],
    )
    persisted: list[str] = []

    def _persist(result, **_kw):
        persisted.extend(b.source_id for b in result.bindings)
        return "mapping-id"

    with (
        patch("imas_codex.ids.mapping.adiscover_assembly", _no_assembly),
        patch("imas_codex.ids.mapping.validate_mappings", return_value=validated),
        patch("imas_codex.ids.models.persist_mapping_result", side_effect=_persist),
        patch("imas_codex.ids.workers.GraphClient", return_value=graph),
        patch("imas_codex.ids.workers.refresh_mapping_status"),
    ):
        await validate_worker(state)
    assert SELECTED in persisted, "validation did not persist the mapped source"


def test_assign_phase_reaches_done_with_one_escalated_source_that_raises():
    state = MappingDiscoveryState(facility="facility", target_ids_list=[IDS])
    state.context_phase.mark_done()
    # Mirror the wired phase check: the released source still reads as escalated
    # pending work, so the phase's own done flag stays False and the worker's
    # handled exclusion is the only thing that can end the pass.
    state.assign_phase.set_has_work_fn(lambda: True)
    state.dd_version = "4.1.1"
    state.skip_errors = True

    asyncio.run(asyncio.wait_for(_run_pipeline(state), timeout=TEST_TIMEOUT))

    assert state.assign_phase.done
    assert state.map_phase.done
