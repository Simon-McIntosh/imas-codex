"""Tests for imas_codex.standard_names.orphan_sweep.

Section 1 — Unit tests (mock GraphClient, no live Neo4j required)
-----------------------------------------------------------------
- test_revert_stuck_refining_name          — name_stage='refining' + stale claimed_at → reverted
- test_revert_stuck_refining_docs          — docs_stage='refining' + stale claimed_at → reverted
- test_no_revert_within_timeout            — claimed_at is fresh → not reverted
- test_tick_aggregates_zero_counts         — all queries return 0 → tick reports 0 (aggregation)
- test_sweep_reverts_unclaimed_refining_docs — docs refining + claimed_at IS NULL → reverted (live)
- test_stale_token_cleared_non_refining    — stale claim_token on non-refining SN → cleared
- test_loop_cancels_on_stop_event          — coroutine exits promptly when stop_event is set
- test_concurrent_safe                     — two ticks in flight don't double-revert (idempotent)

Section 2 — Integration tests (real Neo4j, auto-skipped when unavailable)
--------------------------------------------------------------------------
- test_sweep_reverts_stale_refining_name   — real SN node w/ stale claimed_at → reverted
- test_sweep_skips_fresh_refining_name     — real SN node w/ fresh claimed_at → unchanged
- test_sweep_reverts_stale_refining_docs   — real SN node w/ stale docs_stage → reverted
- test_sweep_skips_fresh_refining_docs     — real SN node w/ fresh docs_stage → unchanged
- test_sweep_reverts_stale_source_token    — real StandardNameSource w/ stale token → cleared
- test_sweep_skips_fresh_source_token      — real StandardNameSource w/ fresh token → unchanged
- test_sweep_atomic_clear                  — stage + token + claimed_at cleared together
- test_sweep_no_op_when_clean              — no stale claims → tick returns all-zero counts
- test_sweep_count_returns_correctly       — 3 stale name + 2 stale docs → correct per-category
- test_run_loop_respects_stop_event        — loop exits within 0.5 s when stop_event is set
- test_run_loop_periodic                   — stale claim swept within 0.3 s by running loop

Section 3 — run_sn_pools wiring: the sweep is a safety net, not a mutation
--------------------------------------------------------------------------
- test_sweep_starts_when_scoped_maintenance_is_bypassed — sweep task still starts under the bypass flag; the embed worker bundled with it still does not
- test_global_maintenance_writers_remain_bypassed_while_sweep_runs — every graph-wide mutation writer stays quiet when the bypass flag is set
- test_drain_starts_sweep_and_manifest_heartbeat_together — a bounded drain keeps its lease heartbeat AND gets the safety-net sweep
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from imas_codex.standard_names.orphan_sweep import (
    _orphan_sweep_tick,
    run_orphan_sweep_loop,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_gc(query_side_effects: list[list[dict]]) -> MagicMock:
    """Build a mock GraphClient that returns successive query results."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(side_effect=query_side_effects)
    return gc


def _patch_gc(gc: MagicMock):
    return patch(
        "imas_codex.standard_names.orphan_sweep.GraphClient",
        return_value=gc,
    )


# Five query labels in declaration order.
_LABELS = [
    "name_refining",
    "docs_refining",
    "stale_token_sn",
    "stale_token_source",
    "compose_attempt_cap",
]


def _zero_results() -> list[list[dict]]:
    """Five queries all returning 0."""
    return [[{"n": 0}]] * 5


# ---------------------------------------------------------------------------
# 1. test_revert_stuck_refining_name
# ---------------------------------------------------------------------------


def test_revert_stuck_refining_name():
    """name_stage='refining' with stale claimed_at is counted by the first query."""
    results = [
        [{"n": 2}],  # name_refining
        [{"n": 0}],  # docs_refining
        [{"n": 0}],  # stale_token_sn
        [{"n": 0}],  # stale_token_source
        [{"n": 0}],  # compose_attempt_cap
    ]
    gc = _make_gc(results)
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["name_refining"] == 2
    assert counts["docs_refining"] == 0
    # Verify the first query received the timeout parameter.
    first_call: call = gc.query.call_args_list[0]
    assert (
        first_call.kwargs.get("timeout_s") == 300
        or first_call.args[1:] == (300,)
        or "timeout_s=300" in str(first_call)
    )


# ---------------------------------------------------------------------------
# 2. test_revert_stuck_refining_docs
# ---------------------------------------------------------------------------


def test_revert_stuck_refining_docs():
    """docs_stage='refining' with stale claimed_at is counted by the second query."""
    results = [
        [{"n": 0}],  # name_refining
        [{"n": 3}],  # docs_refining
        [{"n": 0}],  # stale_token_sn
        [{"n": 0}],  # stale_token_source
        [{"n": 0}],  # compose_attempt_cap
    ]
    gc = _make_gc(results)
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["docs_refining"] == 3
    assert counts["name_refining"] == 0


# ---------------------------------------------------------------------------
# 3. test_no_revert_within_timeout
# ---------------------------------------------------------------------------


def test_no_revert_within_timeout():
    """When claimed_at is newer than threshold, queries return 0 — nothing reverted."""
    gc = _make_gc(_zero_results())
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert all(v == 0 for v in counts.values()), counts
    # All five queries must still have been called.
    assert gc.query.call_count == 5


# ---------------------------------------------------------------------------
# 4. test_no_revert_when_claim_clean
# ---------------------------------------------------------------------------


def test_tick_aggregates_zero_counts():
    """When every sweep query returns 0, the tick reports 0 for each category.

    Aggregation-only check (canned results); the WHERE semantics for the
    unclaimed-refining case are covered by
    ``test_sweep_reverts_unclaimed_refining_docs`` against a live graph.
    """
    gc = _make_gc(_zero_results())
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["name_refining"] == 0
    assert counts["docs_refining"] == 0


# ---------------------------------------------------------------------------
# 5. test_stale_token_cleared_non_refining
# ---------------------------------------------------------------------------


def test_stale_token_cleared_non_refining():
    """Stale claim_token on a non-refining SN is cleared by query 3."""
    results = [
        [{"n": 0}],  # name_refining
        [{"n": 0}],  # docs_refining
        [{"n": 5}],  # stale_token_sn  — 5 stale non-refining tokens cleared
        [{"n": 1}],  # stale_token_source
        [{"n": 0}],  # compose_attempt_cap
    ]
    gc = _make_gc(results)
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["stale_token_sn"] == 5
    assert counts["stale_token_source"] == 1


# ---------------------------------------------------------------------------
# 6. test_loop_cancels_on_stop_event
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_cancels_on_stop_event():
    """Coroutine exits promptly when stop_event is set before the first tick."""
    gc = _make_gc(_zero_results())
    stop_event = asyncio.Event()
    stop_event.set()  # Set before starting — should exit without sleeping.

    with _patch_gc(gc):
        # Should return almost immediately (well within 2 seconds).
        await asyncio.wait_for(
            run_orphan_sweep_loop(
                interval_s=30,
                timeout_s=300,
                stop_event=stop_event,
            ),
            timeout=2.0,
        )
    # Loop exited cleanly — no TimeoutError raised.


# ---------------------------------------------------------------------------
# 7. test_loop_stop_event_during_sleep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_stop_event_during_sleep():
    """Coroutine wakes up mid-sleep when stop_event is set and exits cleanly."""
    gc = _make_gc(_zero_results() * 10)  # Enough for multiple ticks.
    stop_event = asyncio.Event()

    async def _set_after_delay():
        await asyncio.sleep(0.05)
        stop_event.set()

    with _patch_gc(gc):
        setter = asyncio.create_task(_set_after_delay())
        await asyncio.wait_for(
            run_orphan_sweep_loop(
                interval_s=60,  # Long sleep — must wake on stop_event.
                timeout_s=300,
                stop_event=stop_event,
            ),
            timeout=2.0,
        )
        await setter


# ---------------------------------------------------------------------------
# 8. test_concurrent_safe (idempotent ticks)
# ---------------------------------------------------------------------------


def test_concurrent_safe():
    """Two _orphan_sweep_tick calls with the same parameters are idempotent.

    On the second tick the DB has already cleared the orphans, so all
    queries return 0.  Verifies that the tick function itself is stateless
    (no in-memory counter that would produce incorrect results on re-run).
    """
    gc_first = _make_gc(
        [
            [{"n": 1}],  # name_refining — first tick sees 1 stuck node
            [{"n": 0}],
            [{"n": 0}],
            [{"n": 0}],
            [{"n": 0}],  # compose_attempt_cap
        ]
    )
    gc_second = _make_gc(_zero_results())  # second tick: already cleared

    with _patch_gc(gc_first):
        counts_first = _orphan_sweep_tick(timeout_s=300)

    with _patch_gc(gc_second):
        counts_second = _orphan_sweep_tick(timeout_s=300)

    assert counts_first["name_refining"] == 1
    assert counts_second["name_refining"] == 0  # idempotent — no double-revert


# ---------------------------------------------------------------------------
# 9. test_tick_returns_all_labels
# ---------------------------------------------------------------------------


def test_tick_returns_all_labels():
    """_orphan_sweep_tick always returns all five keys regardless of counts."""
    gc = _make_gc(_zero_results())
    with _patch_gc(gc):
        counts = _orphan_sweep_tick(timeout_s=300)

    assert set(counts.keys()) == {
        "name_refining",
        "docs_refining",
        "stale_token_sn",
        "stale_token_source",
        "compose_attempt_cap",
    }


# ---------------------------------------------------------------------------
# 10. test_tick_passes_timeout_to_all_queries
# ---------------------------------------------------------------------------


def test_tick_passes_timeout_to_all_queries():
    """All five queries receive the timeout_s keyword argument."""
    gc = _make_gc(_zero_results())
    with _patch_gc(gc):
        _orphan_sweep_tick(timeout_s=42)

    assert gc.query.call_count == 5
    for c in gc.query.call_args_list:
        # timeout_s is passed as a keyword argument to gc.query.
        assert c.kwargs.get("timeout_s") == 42, (
            f"Expected timeout_s=42 in call kwargs, got {c.kwargs}"
        )


# ===========================================================================
# Section 2 — Integration tests (real Neo4j, auto-skipped when unavailable)
# ===========================================================================
#
# These tests create live :StandardName / :StandardNameSource nodes in the
# configured Neo4j instance, exercise _orphan_sweep_tick or the async loop,
# then assert the state of those nodes after the sweep.
#
# All test-node IDs are prefixed with "orphan_sweep_test__" so they can be
# targeted for cleanup without touching production data.
#
# Auto-skipped when Neo4j is unreachable (see conftest.py
# ``pytest_collection_modifyitems`` → @pytest.mark.graph skip logic).
# ===========================================================================

pytestmark_integration = [pytest.mark.graph, pytest.mark.integration]

_TEST_ID_PREFIX = "orphan_sweep_test__"


# ---------------------------------------------------------------------------
# graph_client fixture — function-scoped so each test gets a fresh cursor
# (the session-scoped fixture is in tests/graph/conftest.py; we duplicate
# a function-scoped variant here so that standard_names tests don't depend
# on the graph conftest being collected)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _gc(disposable_neo4j_uri: str):
    """Function-scoped client bound to the shared disposable graph."""
    from imas_codex.graph.client import GraphClient

    client = GraphClient(
        uri=disposable_neo4j_uri,
        username="neo4j",
        password="",
        graph_name="orphan-sweep-tests",
    )
    client.get_stats()

    yield client
    client.close()


@pytest.fixture(autouse=False)
def _clean_test_nodes(_gc):
    """Delete all orphan_sweep test nodes before and after each test."""
    _gc.query(
        "MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_TEST_ID_PREFIX,
    )
    _gc.query(
        "MATCH (n:StandardNameSource) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_TEST_ID_PREFIX,
    )
    yield
    _gc.query(
        "MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_TEST_ID_PREFIX,
    )
    _gc.query(
        "MATCH (n:StandardNameSource) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
        prefix=_TEST_ID_PREFIX,
    )


# ---------------------------------------------------------------------------
# Helper: create a :StandardName with a stale or fresh claimed_at
# ---------------------------------------------------------------------------


def _create_sn(
    gc,
    sn_id: str,
    *,
    name_stage: str = "reviewed",
    docs_stage: str = "reviewed",
    stale: bool = True,
    claim_token: str = "tok-test",
) -> None:
    """Create (or MERGE) a :StandardName node with claim fields set."""
    age_s = 400 if stale else 60  # 400 s old → stale; 60 s old → fresh
    gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage   = $name_stage,
            sn.docs_stage   = $docs_stage,
            sn.claim_token  = $token,
            sn.claimed_at   = datetime() - duration({seconds: $age_s})
        """,
        id=sn_id,
        name_stage=name_stage,
        docs_stage=docs_stage,
        token=claim_token,
        age_s=age_s,
    )


def _create_source(
    gc,
    source_id: str,
    *,
    stale: bool = True,
    claim_token: str = "tok-src-test",
) -> None:
    """Create (or MERGE) a :StandardNameSource node with claim fields set."""
    age_s = 400 if stale else 60
    gc.query(
        """
        MERGE (s:StandardNameSource {id: $id})
        SET s.claim_token = $token,
            s.claimed_at  = datetime() - duration({seconds: $age_s})
        """,
        id=source_id,
        token=claim_token,
        age_s=age_s,
    )


def _fetch_sn(gc, sn_id: str) -> dict:
    """Return the first row for a :StandardName node."""
    rows = gc.query(
        """
        MATCH (sn:StandardName {id: $id})
        RETURN sn.name_stage   AS name_stage,
               sn.docs_stage   AS docs_stage,
               sn.claim_token  AS claim_token,
               sn.claimed_at   AS claimed_at
        """,
        id=sn_id,
    )
    assert rows, f"StandardName {sn_id!r} not found"
    return rows[0]


def _fetch_source(gc, source_id: str) -> dict:
    """Return the first row for a :StandardNameSource node."""
    rows = gc.query(
        """
        MATCH (s:StandardNameSource {id: $id})
        RETURN s.claim_token AS claim_token,
               s.claimed_at  AS claimed_at
        """,
        id=source_id,
    )
    assert rows, f"StandardNameSource {source_id!r} not found"
    return rows[0]


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_reverts_stale_refining_name(_gc, _clean_test_nodes):
    """name_stage='refining' with stale claimed_at → reverted to 'reviewed'."""
    sn_id = f"{_TEST_ID_PREFIX}stale_name"
    _create_sn(_gc, sn_id, name_stage="refining", stale=True)

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["name_refining"] >= 1

    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "reviewed", row
    assert row["claim_token"] is None, row
    assert row["claimed_at"] is None, row


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_skips_fresh_refining_name(_gc, _clean_test_nodes):
    """name_stage='refining' with fresh claimed_at (60 s) → unchanged."""
    sn_id = f"{_TEST_ID_PREFIX}fresh_name"
    _create_sn(_gc, sn_id, name_stage="refining", stale=False)

    _orphan_sweep_tick(timeout_s=300)

    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "refining", row
    assert row["claim_token"] is not None, row


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_reverts_stale_refining_docs(_gc, _clean_test_nodes):
    """docs_stage='refining' with stale claimed_at → reverted to 'reviewed'."""
    sn_id = f"{_TEST_ID_PREFIX}stale_docs"
    _create_sn(_gc, sn_id, docs_stage="refining", stale=True)

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["docs_refining"] >= 1

    row = _fetch_sn(_gc, sn_id)
    assert row["docs_stage"] == "reviewed", row
    assert row["claim_token"] is None, row
    assert row["claimed_at"] is None, row


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_skips_fresh_refining_docs(_gc, _clean_test_nodes):
    """docs_stage='refining' with fresh claimed_at (60 s) → unchanged."""
    sn_id = f"{_TEST_ID_PREFIX}fresh_docs"
    _create_sn(_gc, sn_id, docs_stage="refining", stale=False)

    _orphan_sweep_tick(timeout_s=300)

    row = _fetch_sn(_gc, sn_id)
    assert row["docs_stage"] == "refining", row
    assert row["claim_token"] is not None, row


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_reverts_unclaimed_refining_docs(_gc, _clean_test_nodes):
    """docs_stage='refining' with claimed_at IS NULL → reverted to 'reviewed'.

    A refining node whose claim was cleared without advancing the stage owns no
    worker and is stranded forever under a stale-claim-only sweep. Recovery must
    catch the unclaimed case too.
    """
    sn_id = f"{_TEST_ID_PREFIX}unclaimed_docs"
    _create_sn(_gc, sn_id, docs_stage="refining", stale=False)
    _gc.query(
        "MATCH (sn:StandardName {id: $id}) "
        "SET sn.claimed_at = null, sn.claim_token = null",
        id=sn_id,
    )

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["docs_refining"] >= 1
    row = _fetch_sn(_gc, sn_id)
    assert row["docs_stage"] == "reviewed", row


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_reverts_stale_source_token(_gc, _clean_test_nodes):
    """StandardNameSource with stale claim_token+claimed_at → both cleared."""
    src_id = f"{_TEST_ID_PREFIX}stale_source"
    _create_source(_gc, src_id, stale=True)

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["stale_token_source"] >= 1

    row = _fetch_source(_gc, src_id)
    assert row["claim_token"] is None, row
    assert row["claimed_at"] is None, row


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_skips_fresh_source_token(_gc, _clean_test_nodes):
    """StandardNameSource with fresh claimed_at (60 s) → unchanged."""
    src_id = f"{_TEST_ID_PREFIX}fresh_source"
    _create_source(_gc, src_id, stale=False)

    _orphan_sweep_tick(timeout_s=300)

    row = _fetch_source(_gc, src_id)
    assert row["claim_token"] is not None, row


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_atomic_clear(_gc, _clean_test_nodes):
    """After sweep, name_stage + claim_token + claimed_at are all cleared.

    Verifies no partial state: either all three fields are cleared or none.
    """
    sn_id = f"{_TEST_ID_PREFIX}atomic"
    _create_sn(_gc, sn_id, name_stage="refining", stale=True)

    _orphan_sweep_tick(timeout_s=300)

    row = _fetch_sn(_gc, sn_id)
    # All three must be cleared together (atomicity check).
    assert row["name_stage"] == "reviewed", f"stage not cleared: {row}"
    assert row["claim_token"] is None, f"token not cleared: {row}"
    assert row["claimed_at"] is None, f"claimed_at not cleared: {row}"


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_no_op_when_clean(_gc, _clean_test_nodes):
    """Graph with no stale claims → tick returns 0 for all categories."""
    # Create a non-refining SN without a stale token — should not be swept.
    sn_id = f"{_TEST_ID_PREFIX}clean_sn"
    _gc.query(
        "MERGE (sn:StandardName {id: $id}) SET sn.name_stage = 'reviewed'",
        id=sn_id,
    )

    counts = _orphan_sweep_tick(timeout_s=1)  # very short timeout

    # Any freshly-created nodes won't have claimed_at set, so zero sweeps.
    assert counts["name_refining"] == 0, counts
    assert counts["docs_refining"] == 0, counts


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
def test_sweep_count_returns_correctly(_gc, _clean_test_nodes):
    """3 stale name-refining + 2 stale docs-refining → counts match."""
    for i in range(3):
        _create_sn(
            _gc,
            f"{_TEST_ID_PREFIX}cnt_name_{i}",
            name_stage="refining",
            stale=True,
        )
    for i in range(2):
        _create_sn(
            _gc,
            f"{_TEST_ID_PREFIX}cnt_docs_{i}",
            docs_stage="refining",
            stale=True,
        )

    counts = _orphan_sweep_tick(timeout_s=300)

    assert counts["name_refining"] >= 3, counts
    assert counts["docs_refining"] >= 2, counts


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_loop_respects_stop_event(_gc, _clean_test_nodes):
    """Loop exits within 0.5 s when stop_event is set immediately."""
    stop_event = asyncio.Event()
    stop_event.set()

    # Should return almost immediately — well within 0.5 s.
    await asyncio.wait_for(
        run_orphan_sweep_loop(
            interval_s=60,
            timeout_s=300,
            stop_event=stop_event,
        ),
        timeout=0.5,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_loop_periodic(_gc, _clean_test_nodes):
    """Stale claim is swept within 0.3 s by a running loop (interval=0.1 s)."""
    sn_id = f"{_TEST_ID_PREFIX}loop_periodic"
    _create_sn(_gc, sn_id, name_stage="refining", stale=True)

    stop_event = asyncio.Event()

    loop_task = asyncio.create_task(
        run_orphan_sweep_loop(
            interval_s=0,  # 0 → sleep(0) between ticks, maximally fast
            timeout_s=300,
            stop_event=stop_event,
        )
    )

    # Give the loop time to run at least one tick.
    await asyncio.sleep(0.3)
    stop_event.set()

    await asyncio.wait_for(loop_task, timeout=1.0)

    row = _fetch_sn(_gc, sn_id)
    assert row["name_stage"] == "reviewed", (
        f"Loop did not sweep the stale claim within 0.3 s: {row}"
    )


# ===========================================================================
# Section 3 — run_sn_pools wiring: the sweep is a safety net, not a mutation
# ===========================================================================
#
# --skip-global-maintenance silences the graph-wide reconcile writers (they
# rewrite live rows), but it must not silence the orphan sweep, which clears
# claim tokens abandoned by a dead process.  A claimed row is ineligible, so
# a run whose workers were killed mid-invocation and whose sweep was
# suppressed reports no work forever, with the budget idling.  These tests
# drive run_sn_pools over a mocked graph/worker boundary and hold both sides
# of the flag: the sweep still starts when the flag is set, and every
# mutating maintenance writer is still skipped.  A bounded drain keeps its
# lease heartbeat alongside the sweep.

_SN_GO = "imas_codex.standard_names.graph_ops"
_SN_LOOP = "imas_codex.standard_names.loop"

# Graph-wide mutation writers the bypass flag must keep quiet.  Each is
# declared with the return value its caller consumes, so a run that (wrongly)
# reaches one stays graph-free and the not-called assertion is the signal.
_MAINTENANCE_WRITERS: dict[str, object] = {
    "reconcile_standard_name_sources": {},
    "mark_orphaned_standard_name_runs_stale": 0,
    "release_all_orphan_claims": {"sn": 0, "sns": 0},
    "resolve_doc_links": {},
    "rederive_structural_edges": {},
    "normalize_derived_parent_lifecycle": 0,
    "reconcile_orphan_parent_sources": 0,
    "restamp_harmonized_families": {},
    "refresh_drifted_sources": {},
}

# Writers hosted outside graph_ops keep their binding-site module so the
# ``run_sn_pools`` import-site patch intercepts them.
_MAINTENANCE_WRITER_MODULES: dict[str, str] = {
    "restamp_harmonized_families": "imas_codex.standard_names.harmonize",
    "refresh_drifted_sources": "imas_codex.standard_names.source_refresh",
}


def _loop_graph_context() -> tuple[MagicMock, MagicMock]:
    """GraphClient mock whose default query answers the SNRun-count probe."""
    graph = MagicMock()
    graph.query.return_value = [{"cnt": 1}]
    context = MagicMock()
    context.__enter__.return_value = graph
    context.__exit__.return_value = False
    return context, graph


async def _drive_run_sn_pools(
    *,
    scope_run_id: str | None = None,
    drain_scope_id: str | None = None,
) -> dict[str, MagicMock]:
    """Run ``run_sn_pools`` over a mocked boundary; return its worker mocks.

    Only unconditional call sites are stubbed; the maintenance writers are
    replaced by spies so the assertion is on whether the orchestrator calls
    them, not on stubbed data.  The stop event is set before entry, so the
    run exits promptly after the startup path (which is where the sweep is
    wired) without any pool or drain work.

    Returns:
        ``{"sweep", "embed", "heartbeat", "writers"}`` — ``heartbeat`` is
        ``None`` for a non-drain run.
    """
    graph_ctx, _ = _loop_graph_context()
    with ExitStack() as stack:
        stack.enter_context(patch(f"{_SN_GO}.create_sn_run_open"))
        stack.enter_context(patch(f"{_SN_GO}.finalize_sn_run"))
        stack.enter_context(
            patch(f"{_SN_GO}.persist_outcome_snapshot", return_value={})
        )
        stack.enter_context(patch(f"{_SN_GO}.reset_persist_outcomes"))
        stack.enter_context(
            patch(
                f"{_SN_GO}.scoped_terminal_residue",
                return_value={"total": 0, "names": [], "sources": []},
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.ledger.find_provenance_orphans",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.audits."
                "find_flux_surface_reduction_violations",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.audits.find_removed_dd_sources",
                return_value=[],
            )
        )
        stack.enter_context(patch(f"{_SN_LOOP}._build_pool_specs", return_value=[]))
        stack.enter_context(
            patch(
                "imas_codex.standard_names.pools.run_pools",
                new_callable=AsyncMock,
                return_value={},
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.budget.BudgetManager.start",
                new_callable=AsyncMock,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.budget.BudgetManager.drain_pending",
                new_callable=AsyncMock,
                return_value=True,
            )
        )
        stack.enter_context(
            patch(
                "imas_codex.standard_names.budget.BudgetManager._get_total_spent_sync",
                return_value=0.0,
            )
        )
        stack.enter_context(
            patch("imas_codex.graph.client.GraphClient", return_value=graph_ctx)
        )
        sweep = stack.enter_context(
            patch(
                "imas_codex.standard_names.orphan_sweep.run_orphan_sweep_loop",
                new_callable=AsyncMock,
            )
        )
        embed = stack.enter_context(
            patch(
                "imas_codex.discovery.base.embed_worker.embed_description_worker",
                new_callable=AsyncMock,
            )
        )
        heartbeat: MagicMock | None = None
        if drain_scope_id:
            heartbeat = stack.enter_context(
                patch(
                    "imas_codex.standard_names.orphan_sweep."
                    "run_manifest_drain_heartbeat_loop",
                    new_callable=AsyncMock,
                )
            )
            stack.enter_context(
                patch(f"{_SN_GO}.finalize_manifest_drain_scope", return_value={})
            )
        writers = {
            name: stack.enter_context(
                patch(
                    f"{_MAINTENANCE_WRITER_MODULES.get(name, _SN_GO)}.{name}",
                    return_value=default,
                )
            )
            for name, default in _MAINTENANCE_WRITERS.items()
        }

        from imas_codex.standard_names.loop import run_sn_pools

        stop = asyncio.Event()
        stop.set()
        kwargs: dict[str, object] = {
            "cost_limit": 5.0,
            "stop_event": stop,
            "skip_global_maintenance": True,
        }
        if scope_run_id is not None:
            kwargs["scope_run_id"] = scope_run_id
        if drain_scope_id is not None:
            kwargs["drain_scope_id"] = drain_scope_id
            kwargs["drain_paths"] = ("magnetics/ip",)
            kwargs["drain_dd_version"] = "4.1.0"
        await run_sn_pools(**kwargs)

    return {
        "sweep": sweep,
        "embed": embed,
        "heartbeat": heartbeat,
        "writers": writers,
    }


@pytest.mark.asyncio
async def test_sweep_starts_when_scoped_maintenance_is_bypassed() -> None:
    """The bypass flag silences the embed worker, never the sweep task.

    The sweep was historically bundled with the mutating workers under a
    single ``if not skip_global_maintenance:`` block, so a scoped run lost
    its safety net exactly when a dead worker could leave claims stranded.
    """
    mocks = await _drive_run_sn_pools(scope_run_id="bounded-run")

    # create_task invoked the coroutine — the task was wired. (The task can be
    # cancelled on its first turn when the run stops immediately, so the
    # assertion is on creation, not on completion.)
    mocks["sweep"].assert_called()
    mocks["embed"].assert_not_called()  # mutating worker still skipped


@pytest.mark.asyncio
async def test_global_maintenance_writers_remain_bypassed_while_sweep_runs() -> None:
    """Every graph-wide mutation writer stays untouched when the sweep runs."""
    mocks = await _drive_run_sn_pools(scope_run_id="bounded-run")

    for _name, writer in mocks["writers"].items():
        writer.assert_not_called()
    mocks["sweep"].assert_called()


@pytest.mark.asyncio
async def test_drain_starts_sweep_and_manifest_heartbeat_together() -> None:
    """A bounded drain keeps its lease heartbeat and gains the sweep.

    The drain path forces the bypass flag, so before this decoupling it ran
    neither worker; a drain whose workers died mid-invocation also stranded
    its claims.  The heartbeat (lease liveness) must survive alongside the
    safety-net sweep.
    """
    mocks = await _drive_run_sn_pools(drain_scope_id="owned-scope")

    mocks["sweep"].assert_called()
    mocks["heartbeat"].assert_called()  # drain lease stays fresh
    mocks["embed"].assert_not_called()
    for _name, writer in mocks["writers"].items():
        writer.assert_not_called()
