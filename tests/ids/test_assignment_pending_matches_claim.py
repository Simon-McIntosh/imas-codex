"""Assignment completion must see the sources the assign worker can claim."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.ids.workers import (
    MappingDiscoveryState,
    claim_sources_for_escalated,
    has_pending_assignment_work,
    run_mapping_engine,
)


@pytest.mark.parametrize(
    ("candidate_ids", "expected"), [(["magnetics"], True), (["summary"], False)]
)
def test_pending_matches_claim_for_target_candidate(candidate_ids, expected):
    source = {"physics_domain": "equilibrium", "candidate_ids": candidate_ids}

    def matches(**kwargs):
        predicate = kwargs["status_predicate"]
        params = kwargs["status_params"]
        assert "n.candidate_route = 'escalated'" in predicate
        assert "n.mapping_disposition IS NULL" in predicate
        assert "WHERE r.ids IN $ids_names" in predicate
        assert "domains" not in params
        return bool(set(source["candidate_ids"]) & set(params.get("ids_names", [])))

    def claim(*_args, **kwargs):
        return [source] if matches(**kwargs) else []

    with (
        patch("imas_codex.ids.workers.claim_batch", side_effect=claim) as claim_mock,
        patch(
            "imas_codex.ids.workers.has_pending",
            side_effect=lambda *_a, **kw: matches(**kw),
        ) as pending_mock,
    ):
        assert bool(claim_sources_for_escalated("jet", ["magnetics"])) is expected
        assert has_pending_assignment_work("jet", ["magnetics"]) is expected
    assert (
        claim_mock.call_args.kwargs["status_predicate"]
        == pending_mock.call_args.kwargs["status_predicate"]
    )


def test_mapping_engine_passes_target_ids_to_pending_check():
    state = MappingDiscoveryState(
        facility="jet",
        target_ids_list=["magnetics"],
        target_info=[{"ids_name": "magnetics", "domains": ["magnetic_field"]}],
    )
    with (
        patch("imas_codex.ids.workers.run_discovery_engine", new_callable=AsyncMock),
        patch(
            "imas_codex.ids.workers.has_pending_assignment_work", return_value=False
        ) as pending,
    ):
        asyncio.run(run_mapping_engine(state))
        state.assign_phase._has_work_fn()
    pending.assert_called_once_with("jet", ["magnetics"])


def test_claim_and_pending_pass_shared_predicate_unchanged():
    params = {"marker": "magnetics"}
    predicate = "n.assignment_marker = $marker"
    with (
        patch(
            "imas_codex.ids.workers._escalated_source_predicate",
            return_value=(predicate, params),
        ) as builder,
        patch("imas_codex.ids.workers.claim_batch", return_value=[]) as claim,
        patch("imas_codex.ids.workers.has_pending", return_value=False) as pending,
    ):
        claim_sources_for_escalated("jet", ["magnetics"], domains=["equilibrium"])
        has_pending_assignment_work("jet", ["magnetics"])

    assert builder.call_args_list == [
        ((["magnetics"], ["equilibrium"]), {}),
        ((["magnetics"],), {}),
    ]
    for graph_call in (claim.call_args, pending.call_args):
        assert graph_call.kwargs["status_predicate"] is predicate
        assert graph_call.kwargs["status_params"] is params
