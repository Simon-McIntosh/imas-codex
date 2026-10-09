"""Selected candidate homes remain claimable across assignment and mapping."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from imas_codex.ids.models import SignalMappingBatch, SignalMappingEntry, TargetChoice
from imas_codex.ids.workers import MappingDiscoveryState, assign_worker, map_worker


def test_assignment_releases_selected_source_for_mapping():
    source_id = "facility:probe"
    target = "magnetics/b_field_pol_probe/field"
    state = MappingDiscoveryState(facility="facility", target_ids_list=["magnetics"])
    state.assign_phase.mark_done()
    graph = MagicMock()
    graph.__enter__.return_value = graph
    choice = TargetChoice(
        source_id=source_id,
        paths=[target],
        confidence=1,
        reasoning="selected probe home",
    )
    with (
        patch("imas_codex.ids.workers.GraphClient", return_value=graph),
        patch(
            "imas_codex.ids.workers.get_mapping_route_thresholds",
            return_value=SimpleNamespace(shortlist_size=5),
        ),
        patch(
            "imas_codex.ids.workers.claim_sources_for_escalated",
            side_effect=[[{"id": source_id}], []],
        ),
        patch(
            "imas_codex.ids.graph_ops.read_candidates",
            return_value={source_id: [{"path": target}]},
        ),
        patch(
            "imas_codex.ids.mapping.achoose_targets",
            new_callable=AsyncMock,
            return_value=choice,
        ),
        patch("imas_codex.ids.graph_ops.select_candidates", return_value=1),
        patch("imas_codex.ids.workers.release_mapping_claim") as release,
    ):
        asyncio.run(assign_worker(state))

    release.assert_called_once_with(source_id)
    assert state.sources_assigned == 1


def test_map_worker_waits_for_selected_source_with_assign_claim():
    source_id = "facility:probe"
    target = "magnetics/b_field_pol_probe/field"
    state = MappingDiscoveryState(facility="facility", target_ids_list=["magnetics"])
    state.assign_phase.mark_done()
    claim_calls = 0
    mapped = False

    def claim(*_args, **_kwargs):
        nonlocal claim_calls
        claim_calls += 1
        # An assign claim still owns the selected source on the first poll.
        return [{"id": source_id}] if claim_calls == 2 else []

    def refresh(*_args):
        nonlocal mapped
        mapped = True
        return "mapped"

    edges = {
        source_id: [
            {
                "path": target,
                "ids": "magnetics",
                "route": True,
                "section": "magnetics/b_field_pol_probe",
            }
        ]
    }
    batch = SignalMappingBatch(
        ids_name="magnetics",
        target_path="magnetics/b_field_pol_probe",
        mappings=[
            SignalMappingEntry(
                source_id=source_id,
                target_id=target,
                confidence=1,
                transform_expression="value",
            )
        ],
    )
    graph = MagicMock()
    graph.__enter__.return_value = graph
    with (
        patch("imas_codex.ids.workers.claim_sources_for_mapping", side_effect=claim),
        patch(
            "imas_codex.ids.workers.has_pending_mapping_work",
            side_effect=lambda *_a: not mapped,
        ),
        patch("imas_codex.ids.workers.GraphClient", return_value=graph),
        patch("imas_codex.ids.graph_ops.read_candidates", return_value=edges),
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
        patch("imas_codex.ids.workers.refresh_mapping_status", side_effect=refresh),
        patch("imas_codex.ids.workers.asyncio.sleep", new_callable=AsyncMock),
    ):
        asyncio.run(map_worker(state))

    assert mapped
    assert state.sources_mapped == 1
    assert state.bindings_total == 1
    assert state.map_phase.done
