"""Rescore keeps identity fixed and terminates when review is unaffordable."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from imas_codex.standard_names.budget import BudgetManager
from imas_codex.standard_names.pools import PoolSpec, _budget_saturation_watchdog
from imas_codex.standard_names.rescore import (
    apply_rescore_pool_contract,
    classify_rescore_budget_stop,
)


@dataclass
class _IdentityState:
    identity: str
    refined_from: list[tuple[str, str]]


def _pool(name: str, state: _IdentityState) -> PoolSpec:
    async def claim() -> None:
        return None

    async def process(_batch: object) -> int:
        if name == "refine_name":
            successor = f"{state.identity}_rewritten"
            state.refined_from.append((successor, state.identity))
            state.identity = successor
        return 1

    return PoolSpec(
        name=name,
        weight=1.0,
        claim=claim,
        process=process,
    )


@pytest.mark.asyncio
async def test_rescore_keeps_exact_identity_and_mints_no_successor() -> None:
    prior_identity = "particle_convection_velocity"
    state = _IdentityState(identity=prior_identity, refined_from=[])
    pools = [_pool("review_name", state), _pool("refine_name", state)]

    selected = apply_rescore_pool_contract(
        pools,
        scope_run_id="sn-rescore-20260901T135204Z",
    )
    assert [pool.name for pool in selected] == ["review_name"]

    for pool in selected:
        await pool.process({})

    assert state.identity == prior_identity
    assert state.refined_from == []


@pytest.mark.asyncio
async def test_unaffordable_rescore_reviewer_terminates_budget_exhausted() -> None:
    manager = BudgetManager(total_budget=0.01)
    review = _pool(
        "review_name",
        _IdentityState(identity="plasma_current", refined_from=[]),
    )
    review.health.pending_count = 1
    manager._consecutive_reserve_failures["review_name"] = manager.SATURATION_THRESHOLD
    stop_event = asyncio.Event()
    saturated_event = asyncio.Event()

    await asyncio.wait_for(
        _budget_saturation_watchdog(
            manager,
            [review],
            stop_event,
            saturated_event,
            poll=0.001,
        ),
        timeout=1.0,
    )

    assert stop_event.is_set()
    assert saturated_event.is_set()
    assert (
        classify_rescore_budget_stop(
            scope_run_id="sn-rescore-20260901T135204Z",
            budget_saturated=saturated_event.is_set(),
        )
        == "budget_exhausted"
    )
