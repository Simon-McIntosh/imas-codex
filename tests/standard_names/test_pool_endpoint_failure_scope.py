"""A dead model seat degrades the run instead of stopping it.

A pool whose configured endpoint is unreachable fails only that pool.  The
run must not abort before other pools claim, pools bound to reachable seats
must claim and complete, and the surfaced failure text must name the
configuration key holding the wrong address rather than the endpoint URL.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_GO = "imas_codex.standard_names.graph_ops"
_AA = "imas_codex.standard_names.attachment_audit"
_DGO = "imas_codex.graph.dd_graph_ops"
_BM = "imas_codex.standard_names.budget.BudgetManager"
_WK = "imas_codex.standard_names.workers"

#: A batch each usable pool can claim exactly once, then the graph is empty.
_ONE_BATCH = [
    {
        "id": "sn-1",
        "source_id": "dd:equilibrium/time_slice/profiles_1d/psi",
        "path": "equilibrium/time_slice/profiles_1d/psi",
        "claim_token": "tok-1",
        "source_type": "dd",
        "name": "poloidal_flux",
    }
]


@pytest.fixture(autouse=True)
def _stub_parent_lifecycle_startup():
    """Keep graph-backed startup maintenance inside the test mock boundary."""
    startup_patches = (
        patch(f"{_GO}.reconcile_vocab_gaps", return_value={}),
        patch(
            f"{_GO}.revive_unit_skipped_sources",
            return_value={"checked": 0, "revived": 0},
        ),
        patch(
            f"{_GO}.retry_vocab_gap_sources_on_grammar_change",
            return_value={"checked": 0, "revived": 0},
        ),
        patch(f"{_GO}.reconcile_provenance", return_value={}),
        patch(f"{_GO}.reconcile_source_status_liveness", return_value={}),
        patch(f"{_GO}.retire_unreachable_hint_edits", return_value=0),
        patch(f"{_GO}.reconcile_grammar_segments", return_value={}),
        patch(f"{_GO}.reconcile_catalog_status", return_value={}),
        patch(f"{_GO}.reconcile_reviewable_name_stage", return_value={}),
        patch(f"{_GO}.reconcile_standard_name_cocos_links", return_value={}),
        patch(
            f"{_GO}.reconcile_standard_name_unit_edges",
            return_value={
                "names_realigned": 0,
                "edges_dropped": 0,
                "edges_created": 0,
            },
        ),
        patch(
            f"{_GO}.reconcile_standard_name_dd_edges",
            return_value={"edges_created": 0, "pairs_dropped": 0},
        ),
        patch(
            f"{_GO}.reconcile_standard_name_source_paths",
            return_value={"names_reconciled": 0},
        ),
        patch(f"{_GO}.rederive_structural_edges", return_value={}),
        patch(f"{_GO}.seed_parent_sources", return_value=0),
        patch(f"{_GO}.normalize_derived_parent_lifecycle", return_value=0),
        patch(f"{_GO}.structural_accept_derived_parents", return_value=0),
        patch(f"{_GO}.reconcile_orphan_parent_sources", return_value=0),
        patch(f"{_GO}.promote_stranded_reviewed", return_value={"name": 0, "docs": 0}),
        patch(f"{_GO}.mark_orphaned_standard_name_runs_stale", return_value=0),
        patch(f"{_AA}.reconcile_attachment_consistency", return_value=MagicMock()),
        patch(f"{_DGO}.reconcile_dd_unit_corrections", return_value={}),
        patch(
            "imas_codex.discovery.base.embed_worker.embed_description_worker",
            new_callable=AsyncMock,
        ),
    )
    with ExitStack() as stack:
        for startup_patch in startup_patches:
            stack.enter_context(startup_patch)
        yield


def _fast_idle_run_pools_patch():
    """Patch ``run_pools`` to compress idle exhaustion to ~1s.

    The original is captured at construction time — before the patch takes
    effect — so the wrapper delegates to the real harness rather than to its
    own mock.
    """
    import imas_codex.standard_names.pools as pools_module

    real_run_pools = pools_module.run_pools

    async def _fast(*args, **kwargs):
        kwargs.setdefault("idle_exhaustion_poll", 0.2)
        kwargs.setdefault("idle_exhaustion_polls", 4)
        kwargs.setdefault("stall_seconds", 5.0)
        return await real_run_pools(*args, **kwargs)

    return patch("imas_codex.standard_names.pools.run_pools", side_effect=_fast)


class _HealthRecorder:
    """Stand-in for the CLI display state that captures per-pool health."""

    def __init__(self) -> None:
        self.health: dict[str, object] = {}

    def set_pool_health(self, name: str, health: object) -> None:
        self.health[name] = health


def _claim_once_then_empty():
    """Yield one batch on the first claim, then always an empty graph.

    A function side effect (rather than a list) never exhausts, so a pool
    that keeps claiming past the batch sees an empty graph instead of a
    StopIteration being counted as a claim error.
    """
    state = {"n": 0}

    def _claim(**kwargs) -> list[dict]:
        state["n"] += 1
        return _ONE_BATCH if state["n"] == 1 else []

    return _claim


def _mock_graph():
    """A mock GraphClient whose SNRun-existence check returns one row."""
    ctx = MagicMock()
    inst = MagicMock()
    inst.query.return_value = [{"cnt": 1}]
    ctx.__enter__ = MagicMock(return_value=inst)
    ctx.__exit__ = MagicMock(return_value=False)
    return ctx


def _run_scaffold_patches():
    """The graph/budget/finalize scaffolding run_sn_pools needs to stay local."""
    return (
        patch(
            f"{_GO}.reconcile_standard_name_sources",
            return_value={"relinked": 0, "stale_marked": 0, "revived": 0},
        ),
        patch(f"{_GO}.create_sn_run_open"),
        patch(f"{_GO}.finalize_sn_run"),
        patch(f"{_GO}.release_all_orphan_claims", return_value={"sn": 0, "sns": 0}),
        patch(f"{_GO}.rederive_structural_edges", return_value={}),
        patch(f"{_GO}.seed_parent_sources", return_value=0),
        patch(f"{_GO}.normalize_derived_parent_lifecycle", return_value=0),
        patch(f"{_GO}.resolve_doc_links", return_value={}),
        patch(f"{_BM}.start", new_callable=AsyncMock),
        patch(f"{_BM}.drain_pending", new_callable=AsyncMock, return_value=True),
        patch(f"{_BM}.get_total_spent", new_callable=AsyncMock, return_value=0.0),
        patch(
            f"{_GO}.claim_generate_name_batch",
            side_effect=_claim_once_then_empty(),
        ),
        patch(f"{_GO}.claim_generate_docs_batch", return_value=[]),
        patch(
            f"{_GO}.claim_review_name_batch",
            side_effect=_claim_once_then_empty(),
        ),
        patch(f"{_GO}.claim_review_docs_batch", return_value=[]),
        patch(f"{_GO}.claim_refine_name_batch", return_value=[]),
        patch(f"{_GO}.claim_refine_docs_batch", return_value=[]),
        patch(f"{_GO}.claim_enrich_parents_batch", return_value=[]),
        patch(
            "imas_codex.standard_names.source_refresh.refresh_drifted_sources",
            return_value={},
        ),
        _fast_idle_run_pools_patch(),
        # The post-create SNRun assertion opens its own GraphClient; a
        # default-tier mock graph keeps it out of the live database.
        patch("imas_codex.graph.client.GraphClient", return_value=_mock_graph()),
    )


async def _drive_run():
    """Run the pool launcher with one dead and one live seat, return state.

    ``generate_name``'s batch processor raises a connection error (its
    configured seat cannot be reached); ``review_name``'s processor succeeds.
    Both pools claim exactly one batch, then the graph is empty so the run
    falls through to natural idle exhaustion.
    """
    from imas_codex.standard_names.loop import run_sn_pools

    recorder = _HealthRecorder()
    with ExitStack() as stack:
        for p in _run_scaffold_patches():
            stack.enter_context(p)
        stack.enter_context(
            patch(
                f"{_WK}.process_generate_name_batch",
                side_effect=ConnectionError(
                    "Failed to connect to http://localhost:18800/v1: "
                    "[Errno 111] Connection refused"
                ),
            )
        )
        review_process = stack.enter_context(
            patch(f"{_WK}.process_review_name_batch", new_callable=AsyncMock)
        )
        review_process.return_value = 1
        stack.enter_context(
            patch(f"{_WK}.process_generate_docs_batch", new_callable=AsyncMock)
        )
        stack.enter_context(
            patch(f"{_WK}.process_review_docs_batch", new_callable=AsyncMock)
        )
        stack.enter_context(
            patch(f"{_WK}.process_refine_name_batch", new_callable=AsyncMock)
        )
        stack.enter_context(
            patch(f"{_WK}.process_refine_docs_batch", new_callable=AsyncMock)
        )
        stack.enter_context(
            patch(f"{_WK}.process_enrich_parents_batch", new_callable=AsyncMock)
        )

        summary = await run_sn_pools(cost_limit=5.0, loop_state=recorder)

    return summary, recorder


async def test_unreachable_compose_seat_degrades_run_not_stops_it() -> None:
    """An unreachable compose seat fails the generate pool only.

    Review (bound to a reachable seat) still claims and completes, the run
    reaches a terminal state rather than aborting first, and the surfaced
    message names ``[tool.imas-codex.sn-compose]`` rather than the dead URL.
    """
    summary, recorder = await _drive_run()

    # Pools bound to a reachable seat claimed and completed.
    review_health = recorder.health.get("review_name")
    assert review_health is not None
    assert getattr(review_health, "total_processed", 0) >= 1

    # The run reached a terminal state rather than aborting before claiming,
    # and the dead seat only degraded it — it is not a whole-run failure.
    assert summary.stop_reason == "degraded"

    # The unreachable compose seat failed only the generate pool, whose
    # surfaced text names the configuration key, never the dead URL.
    gen_health = recorder.health.get("generate_name")
    assert gen_health is not None
    assert getattr(gen_health, "error_count", 0) >= 1
    last_error = getattr(gen_health, "last_error", "")
    assert "sn-compose" in last_error
    assert "18800" not in last_error


async def test_reachable_seat_pool_commits_work_while_compose_is_dead() -> None:
    """Review work still lands even though generate could not reach its seat.

    The same scenario, asserting the run's own accounting reflects the review
    pool's completed work rather than an un-started run.
    """
    summary, recorder = await _drive_run()

    assert summary.stop_reason == "degraded"
    assert summary.names_reviewed >= 1
    assert getattr(recorder.health.get("review_name"), "total_processed", 0) >= 1


def test_unreachable_classification_degrades_failed_otherwise() -> None:
    """The stop-reason classifier separates a dead seat from a real bug."""
    from imas_codex.standard_names.loop import _pool_error_stop_reason

    seat_health = SimpleNamespace(error_count=1, last_error="Connection refused")
    classified_health = SimpleNamespace(
        error_count=1,
        last_error="generate_name pool could not reach the endpoint "
        "configured for its seat; the address is set in "
        "[tool.imas-codex.sn-compose] (api-base / model-route)",
    )
    bug_health = SimpleNamespace(
        error_count=1,
        last_error="ValueError: unexpected condition while composing",
    )

    reason, count = _pool_error_stop_reason(
        "no_eligible_work", {"generate_name": seat_health}
    )
    assert (reason, count) == ("degraded", 1)

    reason, count = _pool_error_stop_reason(
        "no_eligible_work", {"generate_name": classified_health}
    )
    assert (reason, count) == ("degraded", 1)

    reason, count = _pool_error_stop_reason(
        "no_eligible_work", {"generate_name": bug_health}
    )
    assert (reason, count) == ("failed", 1)
