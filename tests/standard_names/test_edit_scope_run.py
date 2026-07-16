"""Tests for the edit-scope (``--edits``) pipeline mode.

An edit-scoped ``sn run`` restricts every pool claim to the pending
successors of a bulk ``sn edit`` — StandardName nodes carrying
``edit_status = 'open'`` — without needing ``--focus`` or a DD path.
The scope predicate is threaded through the SAME wiring as the
``scope_run_id`` (``--focus``) predicate: ``_scope_kwargs`` in
``_build_pool_specs`` → each pool's claim adapter → ``_claim_sn_atomic`` /
``claim_generate_name_batch``.

Section 1 — Unit (mock GraphClient, no live Neo4j)
--------------------------------------------------
- ``_claim_sn_atomic(edits_only=True)`` composes the ``edit_status='open'``
  fragment into the seed AND expand Cypher.
- ``claim_review_name_batch(edits_only=True)`` propagates the fragment.
- A plain call (``edits_only=False``) is unchanged — no edit_status filter.
- ``_build_pool_specs(edits_only=True)`` threads ``edits_only`` to the
  review claim adapter; the default (``False``) does not.

Section 2 — Integration (real Neo4j, auto-skipped when unavailable)
-------------------------------------------------------------------
- Two synthetic drafted names, one ``edit_status='open'`` and one without;
  ``claim_review_name_batch(edits_only=True)`` claims ONLY the open one.

All synthetic graph nodes use a unique ``__editscopetest__`` id prefix and
are wiped before + after each integration test; a real name is never matched.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names import loop as loop_mod

_EDIT_FRAGMENT = "coalesce(sn.edit_status, '') = 'open'"
_TEST_ID_PREFIX = "__editscopetest__"


# ---------------------------------------------------------------------------
# Mock helpers (mirror test_focus_flag.py)
# ---------------------------------------------------------------------------


def _mock_gc_tx():
    """Build a mock GraphClient wired for single-transaction claim functions."""
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    # _verify_*_claim_winners re-opens GraphClient and calls .query(); return
    # an empty winners list so the (empty) claim path is a clean no-op.
    gc.query = MagicMock(return_value=[])

    tx = MagicMock()
    tx.closed = False
    tx.commit = MagicMock()
    tx.close = MagicMock()

    session = MagicMock()
    session.begin_transaction = MagicMock(return_value=tx)

    @contextmanager
    def _session_ctx():
        yield session

    gc.session = _session_ctx
    return gc, tx


def _patch_graph_ops_gc(mock_gc):
    return patch(
        "imas_codex.standard_names.graph_ops.GraphClient",
        return_value=mock_gc,
    )


# ---------------------------------------------------------------------------
# 1. _claim_sn_atomic composes the edit-scope predicate
# ---------------------------------------------------------------------------


class TestClaimAtomicEditScope:
    def test_edits_only_adds_where_clause(self):
        """edits_only=True → seed Cypher contains the edit_status='open' filter."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])  # no eligible seed → early return

        with _patch_graph_ops_gc(gc):
            items = _claim_sn_atomic(
                eligibility_where="sn.name_stage = 'drafted'",
                query_params={},
                batch_size=1,
                edits_only=True,
            )

        assert items == []
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert _EDIT_FRAGMENT in seed_cypher, (
            f"Expected {_EDIT_FRAGMENT!r} in seed Cypher:\n{seed_cypher}"
        )

    def test_edits_only_propagates_to_expand(self):
        """The edit_status filter appears in the expand Cypher too (cluster branch)."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(
            side_effect=[
                # seed: has cluster_id → cluster-only expand branch
                [{"_cluster_id": "c-1", "_unit": None, "_physics_domain": None}],
                # expand: no additional items
                [],
                # read-back
                [
                    {
                        "id": "sn-x",
                        "description": "d",
                        "documentation": None,
                        "kind": None,
                        "unit": None,
                        "cluster_id": "c-1",
                        "physics_domain": None,
                        "validation_status": "valid",
                    }
                ],
            ]
        )

        with _patch_graph_ops_gc(gc):
            _claim_sn_atomic(
                eligibility_where="sn.name_stage = 'drafted'",
                query_params={},
                batch_size=5,
                edits_only=True,
            )

        seed_cypher = tx.run.call_args_list[0].args[0]
        expand_cypher = tx.run.call_args_list[1].args[0]
        assert _EDIT_FRAGMENT in seed_cypher
        assert _EDIT_FRAGMENT in expand_cypher

    def test_edits_only_combines_with_scope_run_id(self):
        """Both predicates AND together when edits_only and scope_run_id are set."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_graph_ops_gc(gc):
            _claim_sn_atomic(
                eligibility_where="sn.name_stage = 'drafted'",
                query_params={},
                batch_size=1,
                scope_run_id="run-abc",
                edits_only=True,
            )

        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "sn.run_id = $scope_run_id" in seed_cypher
        assert _EDIT_FRAGMENT in seed_cypher

    def test_default_has_no_edit_filter(self):
        """edits_only defaults to False → no edit_status fragment (backward compat)."""
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_graph_ops_gc(gc):
            _claim_sn_atomic(
                eligibility_where="sn.name_stage = 'drafted'",
                query_params={},
                batch_size=1,
            )

        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "edit_status" not in seed_cypher


# ---------------------------------------------------------------------------
# 2. claim_review_name_batch forwards edits_only
# ---------------------------------------------------------------------------


class TestReviewNameEditScope:
    def test_review_name_edits_only_adds_fragment(self):
        """claim_review_name_batch(edits_only=True) composes the fragment."""
        from imas_codex.standard_names.graph_ops import claim_review_name_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])  # nothing eligible

        with _patch_graph_ops_gc(gc):
            items = claim_review_name_batch(edits_only=True, batch_size=5)

        assert items == []
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert _EDIT_FRAGMENT in seed_cypher, (
            f"Expected {_EDIT_FRAGMENT!r} in seed Cypher:\n{seed_cypher}"
        )

    def test_review_name_default_no_fragment(self):
        """A plain claim_review_name_batch() call carries no edit_status filter."""
        from imas_codex.standard_names.graph_ops import claim_review_name_batch

        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_graph_ops_gc(gc):
            items = claim_review_name_batch(batch_size=5)

        assert items == []
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "edit_status" not in seed_cypher


class TestDocsPoolsEditScopeScoreGate:
    """edits_only drops the name-score gate on the docs claim pools.

    An open sn-edit is operator authorisation (like a curative scope_run_id):
    catalog-imported names carry no reviewer_score_name, so keeping the gate
    would make staged docs edits permanently unclaimable by ``sn run --edits``.
    """

    @pytest.mark.parametrize(
        "claim_name",
        [
            "claim_review_docs_batch",
            "claim_generate_docs_batch",
            "claim_refine_docs_batch",
        ],
    )
    def test_edits_only_drops_score_gate(self, claim_name):
        import imas_codex.standard_names.graph_ops as graph_ops

        claim = getattr(graph_ops, claim_name)
        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_graph_ops_gc(gc):
            items = claim(edits_only=True, batch_size=5)

        assert items == []
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "reviewer_score_name" not in seed_cypher, (
            f"{claim_name}: score gate must drop under edits_only:\n{seed_cypher}"
        )
        assert _EDIT_FRAGMENT in seed_cypher

    @pytest.mark.parametrize(
        "claim_name",
        [
            "claim_review_docs_batch",
            "claim_generate_docs_batch",
            "claim_refine_docs_batch",
        ],
    )
    def test_default_keeps_score_gate(self, claim_name):
        import imas_codex.standard_names.graph_ops as graph_ops

        claim = getattr(graph_ops, claim_name)
        gc, tx = _mock_gc_tx()
        tx.run = MagicMock(return_value=[])

        with _patch_graph_ops_gc(gc):
            items = claim(batch_size=5)

        assert items == []
        seed_cypher = tx.run.call_args_list[0].args[0]
        assert "reviewer_score_name" in seed_cypher, (
            f"{claim_name}: unscoped claims must keep the score gate:\n{seed_cypher}"
        )


# ---------------------------------------------------------------------------
# 3. _build_pool_specs threads edits_only to the claim adapters
# ---------------------------------------------------------------------------


class TestScopeKwargsThreading:
    def _capture_review_claim_kwargs(self, *, edits_only: bool) -> dict:
        """Build specs, invoke the review_name claim adapter, capture its kwargs.

        Patches ``claim_review_name_batch`` at its ``graph_ops`` source BEFORE
        building specs: ``_build_pool_specs`` imports it locally and binds the
        adapter closure to the current object, so the mock must be in place
        when the specs are constructed.
        """
        from imas_codex.standard_names import graph_ops

        captured: dict = {}

        def _fake_claim(**kw):
            captured.update(kw)
            return []

        with patch.object(graph_ops, "claim_review_name_batch", _fake_claim):
            specs = loop_mod._build_pool_specs(
                MagicMock(),
                asyncio.Event(),
                edits_only=edits_only,
            )
            review = next(s for s in specs if s.name == "review_name")
            asyncio.run(review.claim())

        return captured

    def test_edits_only_reaches_review_adapter(self):
        """edits_only=True threads through _scope_kwargs to the review claim fn."""
        captured = self._capture_review_claim_kwargs(edits_only=True)
        assert captured.get("edits_only") is True

    def test_default_omits_edits_only(self):
        """edits_only=False → the kwarg is absent (backward compat)."""
        captured = self._capture_review_claim_kwargs(edits_only=False)
        assert "edits_only" not in captured


# ---------------------------------------------------------------------------
# Section 2 — Integration (real Neo4j, auto-skipped when unavailable)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _gc():
    """Function-scoped GraphClient; skip if Neo4j is unreachable."""
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not available: {exc}")

    yield client
    client.close()


@pytest.fixture()
def _clean_test_nodes(_gc):
    """Wipe synthetic edit-scope test nodes before and after each test."""

    def _wipe():
        _gc.query(
            "MATCH (n) WHERE n.id STARTS WITH $prefix DETACH DELETE n",
            prefix=_TEST_ID_PREFIX,
        )

    _wipe()
    yield
    _wipe()


@pytest.mark.graph
@pytest.mark.integration
def test_edits_only_claims_only_open_edit(_gc, _clean_test_nodes):
    """claim_review_name_batch(edits_only=True) claims ONLY the open-edit name.

    The two synthetic nodes carry a unique ``physics_domain`` and the claim is
    scoped to that domain — so the random seed can ONLY land on this test's
    nodes, never on a real edit successor (a live migration may leave many
    real ``edit_status='open'`` names in the graph).
    """
    from imas_codex.standard_names.graph_ops import claim_review_name_batch

    open_id = f"{_TEST_ID_PREFIX}open"
    plain_id = f"{_TEST_ID_PREFIX}plain"
    test_domain = f"{_TEST_ID_PREFIX}domain"

    # Two drafted names in an isolated synthetic domain, sharing no cluster/unit;
    # only one has edit_status='open'. Both satisfy the review eligibility gate
    # (real description, not derived).
    _gc.query(
        """
        MERGE (o:StandardName {id: $open_id})
        SET o.name = 'editscope_open_name',
            o.description = 'synthetic edit-scope open successor',
            o.name_stage = 'drafted',
            o.docs_stage = 'pending',
            o.physics_domain = $domain,
            o.edit_status = 'open',
            o.claim_token = NULL,
            o.claimed_at = NULL
        MERGE (p:StandardName {id: $plain_id})
        SET p.name = 'editscope_plain_name',
            p.description = 'synthetic non-edit drafted name',
            p.name_stage = 'drafted',
            p.docs_stage = 'pending',
            p.physics_domain = $domain,
            p.edit_status = NULL,
            p.claim_token = NULL,
            p.claimed_at = NULL
        """,
        open_id=open_id,
        plain_id=plain_id,
        domain=test_domain,
    )

    items = claim_review_name_batch(edits_only=True, domain=test_domain, batch_size=10)
    claimed_ids = {it["id"] for it in items}

    assert claimed_ids == {open_id}, (
        f"edits_only should claim ONLY the open-edit name; got {claimed_ids}"
    )
