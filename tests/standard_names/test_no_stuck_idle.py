"""
The idle-exhaustion watchdog must count only genuinely eligible work.

Counting on completion timestamps (``enriched_at``, ``reviewed_name_at``,
``reviewed_docs_at``) reports phantom pending rows, because those fields stay
set on fully-processed nodes; the watchdog then never fires and the loop spins
long after the backlog is empty.  ``_compute_pool_pending`` counts stage fields
instead, so terminal-state nodes contribute zero.  A graph populated with only
superseded + exhausted nodes must exit cleanly within 30 s with
``no_eligible_work``.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from imas_codex.cli.sn import _compute_pool_pending

# Module paths used for patching (lazy imports inside loop.py / graph_ops.py).
_GO = "imas_codex.standard_names.graph_ops"
_LOOP = "imas_codex.standard_names.loop"

# Unique prefix for all test nodes — prevents interference with production data
# and is cleaned up before and after each test.
_TEST_PREFIX = "stuck_idle_regtest__"

# Synthetic physics domain that only our test nodes carry.  Scoping
# _compute_pool_pending to this domain isolates the pre-flight assertion from
# any real drafted/reviewed nodes that may exist in the graph.
_TEST_DOMAIN = "regression_test_no_stuck_idle"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _gc():
    """Function-scoped real GraphClient; skip test if Neo4j is unreachable."""
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:
        pytest.skip(f"Neo4j not available: {exc}")

    yield client
    client.close()


@pytest.fixture()
def _clean(_gc):
    """Delete all regression-test nodes before and after each test."""

    def _wipe() -> None:
        for label in ("StandardName", "StandardNameSource"):
            _gc.query(
                f"MATCH (n:{label}) WHERE n.id STARTS WITH $p DETACH DELETE n",
                p=_TEST_PREFIX,
            )

    _wipe()
    yield
    _wipe()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.graph
@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_does_not_spin_when_all_work_complete(_gc, _clean) -> None:
    """
    Populate graph with ONLY superseded + exhausted nodes (zero eligible work).
    Run the pool loop with a generous budget.
    Assert it exits with stop_reason='no_eligible_work' within 30 seconds.

    Regression guard
    ----------------
    A completion-timestamp count (``sn.enriched_at IS NOT NULL`` and
    ``sn.reviewed_name_at IS NOT NULL``) matches every fully-processed node,
    so the watchdog sees phantom pending rows even when every StandardName is
    superseded or exhausted.  With ``pending_count > 0`` the idle-exhaustion
    watchdog never fires and the loop spins indefinitely.

    ``_compute_pool_pending`` counts on ``name_stage='drafted'`` etc. instead,
    so terminal-state nodes contribute 0, the watchdog fires, and the run
    exits cleanly in < 30 s.

    A count taken over timestamps rather than stages is caught twice over: the
    pre-flight assertion returns phantom rows before run_sn_pools even starts,
    and the 30 s timeout fires.
    """
    # --- Insert terminal-state StandardName nodes with legacy fields set ------
    # Fully processed (superseded / exhausted) but still carrying the legacy
    # timestamp fields, i.e. exactly the shape a timestamp-based count
    # misreports as pending.
    node_ids: list[str] = []
    for i in range(6):
        node_id = f"{_TEST_PREFIX}{uuid.uuid4().hex[:8]}"
        node_ids.append(node_id)
        stage = "superseded" if i % 2 == 0 else "exhausted"
        _gc.query(
            """
            MERGE (sn:StandardName {id: $id})
            SET sn.name_stage        = $stage,
                sn.docs_stage        = 'pending',
                sn.chain_length      = 3,
                sn.docs_chain_length = 0,
                sn.description       = 'Regression test node — terminal state',
                sn.documentation     = '',
                sn.kind              = 'scalar',
                sn.unit              = 'eV',
                sn.physics_domain    = $domain,
                sn.enriched_at       = '2024-01-01T00:00:00Z',
                sn.reviewed_name_at  = '2024-01-01T00:00:00Z',
                sn.reviewed_docs_at  = '2024-01-01T00:00:00Z'
            """,
            id=node_id,
            stage=stage,
            domain=_TEST_DOMAIN,
        )

    # Also insert some StandardNameSource nodes in terminal states — none
    # with status='extracted', which would trigger generate_name.
    for i in range(3):
        src_id = f"{_TEST_PREFIX}src_{uuid.uuid4().hex[:8]}"
        status = ["failed", "attached", "composed"][i % 3]
        _gc.query(
            """
            MERGE (s:StandardNameSource {id: $id})
            SET s.status         = $status,
                s.source_type    = 'dd',
                s.source_id      = 'regression/test/path',
                s.physics_domain = $domain
            """,
            id=src_id,
            status=status,
            domain=_TEST_DOMAIN,
        )

    # --- Pre-flight: _compute_pool_pending must return 0 for all pools --------
    # This is the direct regression assertion.
    # If the count were taken over completion timestamps instead of stage
    # fields, nodes with enriched_at / reviewed_name_at set would appear as
    # pending and this assertion would fail before run_sn_pools is even
    # called.
    from imas_codex.graph.client import GraphClient

    with GraphClient() as probe_gc:
        pre_counts = _compute_pool_pending(
            probe_gc,
            domains=[_TEST_DOMAIN],
            rotation_cap=3,
            min_score=0.75,
        )

    assert all(v == 0 for v in pre_counts.values()), (
        f"Pre-flight: _compute_pool_pending returned non-zero counts for "
        f"domain={_TEST_DOMAIN!r}: {pre_counts}.\n"
        "This is the stuck-idle regression: terminal-state nodes are being "
        "counted as pending work.  The watchdog query uses dead legacy fields "
        "(enriched_at / reviewed_name_at) instead of name_stage predicates."
    )

    # --- Build pending_fn (real graph query, domain-scoped) -------------------
    def _pending_fn() -> dict[str, int]:
        """Real _compute_pool_pending scoped to test domain."""
        with GraphClient() as gc:
            return _compute_pool_pending(
                gc,
                domains=[_TEST_DOMAIN],
                rotation_cap=3,
                min_score=0.75,
            )

    # --- Run run_sn_pools with mocked claims and seeding ----------------------
    # Claim functions are mocked to prevent any real LLM work from being
    # claimed from the graph (other domains may have genuine drafted nodes).
    # _seed_all_domains is mocked to prevent creating new StandardNameSource
    # nodes from the DD, which would generate real work.
    # reconcile_standard_name_sources is also mocked for test isolation.
    # The pending_fn intentionally uses the REAL graph query — this is what
    # makes the test an effective regression guard.
    #
    # run_pools is wrapped to shorten idle_exhaustion_poll / idle_exhaustion_polls
    # so the watchdog fires in < 2 s instead of the default 30 s (1.0 s × 30).
    # The wrapper does NOT change any correctness invariants — it only reduces
    # the consecutive-idle-poll count from 30 to 3 for test speed.
    from imas_codex.standard_names import pools as _pools_mod
    from imas_codex.standard_names.loop import run_sn_pools

    _orig_run_pools = _pools_mod.run_pools

    async def _fast_run_pools(*args: object, **kwargs: object) -> object:
        kwargs.setdefault("idle_exhaustion_poll", 0.1)
        kwargs.setdefault("idle_exhaustion_polls", 3)
        return await _orig_run_pools(*args, **kwargs)  # type: ignore[arg-type]

    with (
        patch(f"{_GO}.claim_generate_name_batch", return_value=[]),
        patch(f"{_GO}.claim_review_name_batch", return_value=[]),
        patch(f"{_GO}.claim_refine_name_batch", return_value=[]),
        patch(f"{_GO}.claim_generate_docs_batch", return_value=[]),
        patch(f"{_GO}.claim_review_docs_batch", return_value=[]),
        patch(f"{_GO}.claim_refine_docs_batch", return_value=[]),
        patch(
            f"{_GO}.reconcile_standard_name_sources",
            return_value={"relinked": 0, "stale_marked": 0, "revived": 0},
        ),
        patch(
            f"{_LOOP}._seed_all_domains",
            new=AsyncMock(return_value=0),
        ),
        # Inject tighter idle thresholds so the watchdog fires in ~0.3 s.
        patch("imas_codex.standard_names.pools.run_pools", new=_fast_run_pools),
    ):
        result = await asyncio.wait_for(
            run_sn_pools(
                cost_limit=10.0,
                domains=(),  # empty → auto-seed, which is mocked to return 0
                pending_fn=_pending_fn,
            ),
            timeout=30.0,
        )

    # --- Assertions -----------------------------------------------------------
    assert result.stop_reason == "no_eligible_work", (
        f"Expected stop_reason='no_eligible_work', got {result.stop_reason!r}.\n"
        "If the idle watchdog did not fire, the pending_fn may be returning "
        "phantom rows from terminal-state nodes — the stuck-idle regression."
    )
    assert result.cost_spent == 0.0, (
        f"Expected cost_spent=0.0, got {result.cost_spent}.\n"
        "No LLM calls should have fired with only superseded/exhausted nodes."
    )
