"""Focused pool seeding acquires exact source ownership before compose."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_explicit_paths_return_only_claim_winners_with_fences() -> None:
    from imas_codex.standard_names.pool_adapter import _seed_explicit_paths

    winner = "equilibrium/time_slice/global_quantities/ip"
    loser = "equilibrium/time_slice/global_quantities/q_axis"
    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.__exit__.return_value = False
    gc.query.side_effect = [
        [
            {
                "path": winner,
                "description": "Plasma current.",
                "physics_domain": "magnetics",
                "data_type": "FLT_0D",
                "unit": "A",
            }
        ],
        [{"dd_version": "4.0.0", "cocos_version": 11}],
    ]

    with (
        patch(
            "imas_codex.standard_names.graph_ops.merge_standard_name_sources",
            return_value=2,
        ),
        patch(
            "imas_codex.standard_names.graph_ops.claim_explicit_standard_name_sources",
            return_value=[
                {
                    "id": f"dd:{winner}",
                    "source_id": winner,
                    "source_type": "dd",
                    "status": "extracted",
                    "claim_token": "focus-winner",
                    "claim_seq": 5,
                    "attempt_count": 2,
                }
            ],
        ),
        patch("imas_codex.graph.client.GraphClient", return_value=gc),
        patch("imas_codex.settings.get_dd_version", return_value="4.0.0"),
    ):
        items = await _seed_explicit_paths([winner, loser])

    assert [item["path"] for item in items] == [winner]
    assert items[0]["claim_token"] == "focus-winner"
    assert items[0]["claim_seq"] == 5
    assert items[0]["status"] == "extracted"
    assert items[0]["attempt_count"] == 2
    assert gc.query.call_args_list[0].kwargs["paths"] == [winner]


@pytest.mark.asyncio
async def test_explicit_paths_without_claim_winner_skip_graph_reads() -> None:
    from imas_codex.standard_names.pool_adapter import _seed_explicit_paths

    with (
        patch(
            "imas_codex.standard_names.graph_ops.merge_standard_name_sources",
            return_value=1,
        ),
        patch(
            "imas_codex.standard_names.graph_ops.claim_explicit_standard_name_sources",
            return_value=[],
        ),
        patch("imas_codex.graph.client.GraphClient") as graph_client,
        patch("imas_codex.settings.get_dd_version", return_value="4.0.0"),
    ):
        items = await _seed_explicit_paths(["equilibrium/time_slice/q"])

    assert items == []
    graph_client.assert_not_called()
