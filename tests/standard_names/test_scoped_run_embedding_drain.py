"""Scoped runs drain embeddings before reporting successful completion."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_GO = "imas_codex.standard_names.graph_ops"
_LOOP = "imas_codex.standard_names.loop"


@pytest.mark.asyncio
async def test_scoped_run_drains_newly_described_accepted_name() -> None:
    """A scoped idle exit embeds its accepted described name before completion."""
    state = {"embedded": False}
    item = {
        "id": "intensity_at_spectral_line",
        "description": "Spectral-line intensity.",
        "claim_token": "embed-claim",
    }

    def graph_query(query: str, **_params):
        if "MATCH (rr:SNRun" in query and "count(rr) AS cnt" in query:
            return [{"cnt": 1}]
        if "sn.embedding IS NULL" in query and "sn.name_stage = 'accepted'" in query:
            return (
                [{"gap_count": 0, "gap_ids": []}]
                if state["embedded"]
                else [{"gap_count": 1, "gap_ids": [item["id"]]}]
            )
        return []

    graph = MagicMock()
    graph.query.side_effect = graph_query
    graph_context = MagicMock()
    graph_context.__enter__.return_value = graph
    graph_context.__exit__.return_value = False

    claim = MagicMock(side_effect=[[item], []])

    async def process(items, *_args, **_kwargs):
        assert items == [item]
        state["embedded"] = True
        return 1

    async def idle_run(*_args, **kwargs):
        kwargs["idle_exhausted_event"].set()
        return {}

    with (
        patch(f"{_LOOP}._build_pool_specs", return_value=[]),
        patch("imas_codex.standard_names.pools.run_pools", side_effect=idle_run),
        patch(f"{_GO}.create_sn_run_open"),
        patch(f"{_GO}.finalize_sn_run"),
        patch(f"{_GO}.persist_outcome_snapshot", return_value={}),
        patch(f"{_GO}.reset_persist_outcomes"),
        patch(
            f"{_GO}.scoped_terminal_residue",
            return_value={"total": 0, "names": [], "sources": []},
        ),
        patch(f"{_GO}.claim_embed_batch", claim),
        patch(
            "imas_codex.standard_names.workers.process_embed_batch",
            new=AsyncMock(side_effect=process),
        ) as process_mock,
        patch(
            "imas_codex.standard_names.budget.BudgetManager.start",
            new_callable=AsyncMock,
        ),
        patch(
            "imas_codex.standard_names.budget.BudgetManager.drain_pending",
            new_callable=AsyncMock,
            return_value=True,
        ),
        patch(
            "imas_codex.standard_names.budget.BudgetManager._get_total_spent_sync",
            return_value=0.0,
        ),
        patch("imas_codex.graph.client.GraphClient", return_value=graph_context),
    ):
        from imas_codex.standard_names.loop import run_sn_pools

        summary = await run_sn_pools(
            cost_limit=5.0,
            scope_run_id="bounded-run",
            skip_global_maintenance=True,
        )

    assert state["embedded"] is True
    assert claim.call_count == 2
    assert claim.call_args_list[0].kwargs == {
        "limit": 100,
        "scope_run_id": "bounded-run",
    }
    process_mock.assert_awaited_once()
    assert summary.stop_reason == "no_eligible_work"


@pytest.mark.asyncio
async def test_scoped_run_refuses_partial_embedding_drain() -> None:
    """A claimed row that is not persisted prevents successful completion."""
    item = {
        "id": "frequency_of_wave_diagnostic_channel",
        "description": "Wave frequency for a diagnostic channel.",
        "claim_token": "embed-claim",
    }
    release = MagicMock(return_value=1)

    with (
        patch(f"{_GO}.claim_embed_batch", return_value=[item]),
        patch(f"{_GO}.release_embed_claims", release),
        patch(
            "imas_codex.standard_names.workers.process_embed_batch",
            new=AsyncMock(return_value=0),
        ),
    ):
        from imas_codex.standard_names.loop import (
            _drain_scoped_standard_name_embeddings,
        )

        with pytest.raises(
            RuntimeError,
            match="embedding drain persisted 0 of 1 claimed rows",
        ):
            await _drain_scoped_standard_name_embeddings(
                "bounded-run",
                MagicMock(),
            )

    release.assert_called_once_with([item["id"]], item["claim_token"])
