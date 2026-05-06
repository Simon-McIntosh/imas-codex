"""Tests for the dedicated embed worker pool."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

# ── test_compute_embed_hash ─────────────────────────────────────────


def test_compute_embed_hash_deterministic():
    """Hash is deterministic for the same input."""
    from imas_codex.standard_names.graph_ops import _compute_embed_hash

    h1 = _compute_embed_hash("electron_temperature", "Temperature of electrons")
    h2 = _compute_embed_hash("electron_temperature", "Temperature of electrons")
    assert h1 == h2
    assert len(h1) == 16  # truncated to 16 hex chars


def test_compute_embed_hash_differs_on_change():
    """Hash changes when description changes."""
    from imas_codex.standard_names.graph_ops import _compute_embed_hash

    h1 = _compute_embed_hash("electron_temperature", "Temperature of electrons")
    h2 = _compute_embed_hash("electron_temperature", "Electron thermal energy")
    assert h1 != h2


def test_compute_embed_hash_name_only():
    """Hash works when description is None."""
    from imas_codex.standard_names.graph_ops import _compute_embed_hash

    h = _compute_embed_hash("electron_temperature", None)
    assert len(h) == 16
    # Should differ from name+description hash
    h2 = _compute_embed_hash("electron_temperature", "some desc")
    assert h != h2


# ── test_claim_embed_batch_mock ─────────────────────────────────────


def test_claim_embed_batch_mock():
    """claim_embed_batch issues correct Cypher and returns items."""
    from imas_codex.standard_names.graph_ops import claim_embed_batch

    mock_gc = MagicMock()
    # First call: SET claim fields (no return needed)
    # Second call: RETURN claimed items
    mock_gc.query.side_effect = [
        None,  # SET query
        [
            {"id": "electron_temperature", "description": "T_e", "claim_token": "tok"},
        ],
    ]
    mock_gc.__enter__ = MagicMock(return_value=mock_gc)
    mock_gc.__exit__ = MagicMock(return_value=False)

    with patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=mock_gc):
        items = claim_embed_batch(limit=10)

    assert len(items) == 1
    assert items[0]["id"] == "electron_temperature"
    assert mock_gc.query.call_count == 2


# ── test_process_embed_batch_mock ───────────────────────────────────


@pytest.mark.asyncio
async def test_process_embed_batch_mock():
    """process_embed_batch embeds items and persists results."""
    from imas_codex.standard_names.workers import process_embed_batch

    items = [
        {"id": "electron_temperature", "description": "T_e", "claim_token": "tok"},
        {"id": "plasma_current", "description": "I_p", "claim_token": "tok"},
    ]

    def fake_embed(batch, text_field="_embed_text"):
        for it in batch:
            it["embedding"] = [0.1, 0.2, 0.3]

    events: list[dict] = []

    with (
        patch(
            "imas_codex.embeddings.description.embed_descriptions_batch",
            side_effect=fake_embed,
        ),
        patch(
            "imas_codex.standard_names.graph_ops.persist_embed_batch",
            return_value=2,
        ) as mock_persist,
    ):
        mgr = MagicMock()
        stop = asyncio.Event()
        written = await process_embed_batch(
            items, mgr, stop, on_event=lambda ev: events.append(ev)
        )

    assert written == 2
    mock_persist.assert_called_once()
    persist_args = mock_persist.call_args[0][0]
    assert len(persist_args) == 2
    for item in persist_args:
        assert "embed_text_hash" in item
        assert item["embedding"] == [0.1, 0.2, 0.3]

    assert len(events) == 2
    assert all(ev["pool"] == "embed_name" for ev in events)


@pytest.mark.asyncio
async def test_process_embed_batch_stops_on_event():
    """process_embed_batch returns 0 when stop_event is set."""
    from imas_codex.standard_names.workers import process_embed_batch

    items = [{"id": "foo", "description": "bar", "claim_token": "tok"}]
    mgr = MagicMock()
    stop = asyncio.Event()
    stop.set()  # Already stopped

    written = await process_embed_batch(items, mgr, stop)
    assert written == 0


# ── test_persist_clears_hash ────────────────────────────────────────


def test_persist_generated_docs_clears_hash():
    """persist_generated_docs clears embed_text_hash so embed worker picks it up."""
    from imas_codex.standard_names.graph_ops import persist_generated_docs

    mock_gc = MagicMock()
    mock_gc.query.side_effect = [
        # Main persist query result
        [{"docs_stage": "drafted"}],
        # embed_text_hash clearing query
        None,
    ]
    mock_gc.__enter__ = MagicMock(return_value=mock_gc)
    mock_gc.__exit__ = MagicMock(return_value=False)

    with (
        patch("imas_codex.standard_names.graph_ops.GraphClient", return_value=mock_gc),
        patch(
            "imas_codex.standard_names.graph_ops.bump_sn_run_counter",
        ),
    ):
        result = persist_generated_docs(
            sn_id="electron_temperature",
            description="T_e desc",
            documentation="docs text",
            model="test-model",
            run_id="run-1",
            claim_token="test-token",
        )

    assert result == "drafted"
    # Second query should be the embed_text_hash clearing
    calls = mock_gc.query.call_args_list
    assert len(calls) >= 2
    hash_clear_query = calls[1][0][0]
    assert "embed_text_hash" in hash_clear_query


# ── test_display_merge_constants ────────────────────────────────────


def test_display_rows_cover_all_pools():
    """Every internal pool is covered by exactly one display row."""
    from imas_codex.standard_names.display import (
        DISPLAY_POOL_MAP,
        DISPLAY_ROWS,
        POOL_ORDER,
    )

    # Every pool in POOL_ORDER must appear in exactly one display row.
    covered = set()
    for row in DISPLAY_ROWS:
        sub_pools = DISPLAY_POOL_MAP[row]
        for p in sub_pools:
            assert p in POOL_ORDER, f"{p} not in POOL_ORDER"
            assert p not in covered, f"{p} in multiple display rows"
            covered.add(p)

    assert covered == set(POOL_ORDER)


def test_display_rows_count():
    """5 display rows, 7 internal pools."""
    from imas_codex.standard_names.display import DISPLAY_ROWS, POOL_ORDER

    assert len(DISPLAY_ROWS) == 5
    assert len(POOL_ORDER) == 7


def test_pool_weights_sum_to_one():
    """Pool weights must sum to approximately 1.0."""
    from imas_codex.standard_names.pools import POOL_WEIGHTS

    total = sum(POOL_WEIGHTS.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"
    assert "embed_name" in POOL_WEIGHTS
