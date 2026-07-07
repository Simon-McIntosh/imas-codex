"""``sn run --only <phase>`` must actually skip the generate phase.

``--only link`` (and any ``--only`` selection that excludes the generate
phase) sets ``skip_generate=True``.  That flag must reach the pool
orchestrator: the ``generate_name`` pool is dropped and the domain
auto-seed sweep is skipped — otherwise ``--only link`` silently composes
new names and seeds new sources instead of running link resolution only.

All graph interaction is mocked (no live Neo4j).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.standard_names.budget import BudgetManager

_GO = "imas_codex.standard_names.graph_ops"
_BM = "imas_codex.standard_names.budget.BudgetManager"
_LOOP = "imas_codex.standard_names.loop"


# ---------------------------------------------------------------------------
# --only → skip flags
# ---------------------------------------------------------------------------


def test_only_link_maps_to_skip_generate() -> None:
    from imas_codex.standard_names.turn import skip_flags_from_only

    flags = skip_flags_from_only("link")
    assert flags["skip_generate"] is True
    assert flags["skip_review"] is True


def test_only_review_maps_to_skip_generate() -> None:
    from imas_codex.standard_names.turn import skip_flags_from_only

    flags = skip_flags_from_only("review")
    assert flags["skip_generate"] is True
    # review phase keeps review pools running
    assert flags["skip_review"] is False


# ---------------------------------------------------------------------------
# pool specs
# ---------------------------------------------------------------------------


def _build_specs(**kwargs) -> list:
    from imas_codex.standard_names.loop import _build_pool_specs

    mgr = BudgetManager(total_budget=10.0)
    stop = asyncio.Event()
    return _build_pool_specs(mgr, stop, **kwargs)


def test_build_pool_specs_skip_generate_drops_generate_name() -> None:
    """skip_generate removes the generate_name pool but keeps the rest."""
    names = {s.name for s in _build_specs(skip_generate=True)}
    assert "generate_name" not in names
    # review / refine / docs pools still run so link/review work drains
    assert "review_name" in names
    assert "refine_name" in names
    assert "generate_docs" in names


def test_build_pool_specs_default_keeps_generate_name() -> None:
    names = {s.name for s in _build_specs()}
    assert "generate_name" in names


# ---------------------------------------------------------------------------
# auto-seed sweep
# ---------------------------------------------------------------------------


def _run_sn_pools_patches(seed_mock: AsyncMock):
    """Patch every graph-backed startup call so run_sn_pools reaches (or
    skips) the auto-seed branch without touching a live graph."""
    mock_gc_ctx = MagicMock()
    mock_gc_inst = MagicMock()
    mock_gc_inst.query.return_value = [{"cnt": 1}]
    mock_gc_ctx.__enter__ = MagicMock(return_value=mock_gc_inst)
    mock_gc_ctx.__exit__ = MagicMock(return_value=False)

    return [
        patch(f"{_GO}.reconcile_standard_name_sources", return_value={}),
        patch(f"{_GO}.reconcile_vocab_gaps", return_value={}),
        patch(f"{_GO}.reconcile_provenance", return_value={}),
        patch(f"{_GO}.create_sn_run_open"),
        patch(f"{_GO}.finalize_sn_run"),
        patch(f"{_GO}.release_all_orphan_claims", return_value={"sn": 0, "sns": 0}),
        patch(f"{_GO}.rederive_structural_edges", return_value={}),
        patch(f"{_GO}.seed_parent_sources", return_value=0),
        patch(f"{_GO}.normalize_derived_parent_lifecycle", return_value=0),
        patch(f"{_GO}.structural_accept_derived_parents", return_value=0),
        patch(f"{_GO}.resolve_doc_links", return_value={}),
        patch(f"{_LOOP}._seed_all_domains", new=seed_mock),
        patch(
            "imas_codex.standard_names.pools.run_pools",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(f"{_BM}.start", new_callable=AsyncMock),
        patch(f"{_BM}.drain_pending", new_callable=AsyncMock, return_value=True),
        patch(f"{_BM}.get_total_spent", new_callable=AsyncMock, return_value=0.0),
        patch(f"{_BM}.exhausted", return_value=True),
        patch(f"{_BM}.phase_spent", new_callable=lambda: property(lambda self: {})),
        patch("imas_codex.graph.client.GraphClient", return_value=mock_gc_ctx),
    ]


async def _run(skip_generate: bool) -> AsyncMock:
    from imas_codex.standard_names.loop import run_sn_pools

    seed_mock = AsyncMock(return_value=0)
    patches = _run_sn_pools_patches(seed_mock)
    for p in patches:
        p.start()
    try:
        stop = asyncio.Event()
        stop.set()  # immediate stop — we only care about the startup path
        await run_sn_pools(
            cost_limit=5.0,
            domains=(),  # empty → auto-seed unless suppressed
            stop_event=stop,
            skip_generate=skip_generate,
        )
    finally:
        for p in patches:
            p.stop()
    return seed_mock


@pytest.mark.asyncio
async def test_run_sn_pools_skip_generate_skips_autoseed() -> None:
    """With skip_generate, the domain auto-seed sweep must not run."""
    seed_mock = await _run(skip_generate=True)
    assert seed_mock.await_count == 0


@pytest.mark.asyncio
async def test_run_sn_pools_without_skip_generate_autoseeds() -> None:
    """Control: without skip_generate, the auto-seed sweep runs (domains=())."""
    seed_mock = await _run(skip_generate=False)
    assert seed_mock.await_count == 1
