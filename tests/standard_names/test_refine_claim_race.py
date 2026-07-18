"""Tests for claim-race waste containment in the refine/review pools.

Covers the two mechanisms that stop concurrent replicas racing a tiny
eligible set into a storm of paid-but-discarded LLM calls:

1. The seed stamps a strictly-increasing per-node ``claim_seq``; the winner
   verifier keeps only the item whose committed ``claim_seq`` still matches the
   one it was assigned, so exactly one racer proceeds to the paid LLM call.
2. Scoped drains (``--focus``) cap docs-pool replicas at ~half the scope size,
   so N replicas never contend for a 1-2-name eligible set in the first place.

Also covers the wasted-paid-call tripwire ledger surfaced in the run summary.

All tests mock :class:`GraphClient` — no live Neo4j required.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

_GC = "imas_codex.standard_names.graph_ops.GraphClient"


def _committed_graph(node_id: str, winner_token: str, winner_seq: int):
    """Mock GraphClient whose committed state is one node held by *winner_token*.

    The verifier's re-read MATCHes ``WHERE sn.claim_token = $token`` — so a query
    on any losing token returns no rows (that racer's token was overwritten by
    the winner), and the winner's query returns the node with its committed seq.
    """

    def _query(_cypher: str, **kw):
        if kw.get("token") == winner_token and node_id in kw.get("ids", []):
            return [{"id": node_id, "claim_seq": winner_seq}]
        return []

    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(side_effect=_query)
    return gc


class TestVerifierClosesRace:
    """N concurrent claimers racing 1 eligible node → exactly 1 survivor."""

    @pytest.mark.parametrize("n", [2, 5, 20])
    def test_exactly_one_paid_call_proceeds(self, n: int):
        from imas_codex.standard_names.graph_ops import _verify_docs_claim_winners

        node = "perturbed_particle_energy"
        # The lock-serialised claim burst commits in order; the LAST committer
        # (seq == n) holds the node once the burst settles.
        winner_token = f"tok-{n}"
        winner_seq = n

        survivors = 0
        for i in range(1, n + 1):
            item = [{"id": node, "claim_token": f"tok-{i}", "claim_seq": i}]
            with patch(
                _GC, return_value=_committed_graph(node, winner_token, winner_seq)
            ):
                kept = _verify_docs_claim_winners(
                    item, eligible_stage="refining", settle_seconds=0
                )
            survivors += len(kept)

        assert survivors == 1, f"{survivors} racers passed verify — expected 1"

    def test_names_axis_uses_same_mechanism(self):
        from imas_codex.standard_names.graph_ops import _verify_name_claim_winners

        node = "n"
        survivors = 0
        for i in range(1, 6):
            item = [{"id": node, "claim_token": f"tok-{i}", "claim_seq": i}]
            with patch(_GC, return_value=_committed_graph(node, "tok-5", 5)):
                kept = _verify_name_claim_winners(
                    item, eligible_stage="drafted", settle_seconds=0
                )
            survivors += len(kept)
        assert survivors == 1

    def test_seq_mismatch_drops_even_when_token_matches(self):
        """Defensive: our token is on the node but a higher committed seq means
        a later claim superseded us — we must NOT proceed."""
        from imas_codex.standard_names.graph_ops import _verify_docs_claim_winners

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(return_value=[{"id": "x", "claim_seq": 5}])

        item = [{"id": "x", "claim_token": "tok", "claim_seq": 3}]
        with patch(_GC, return_value=gc):
            kept = _verify_docs_claim_winners(
                item, eligible_stage="refining", settle_seconds=0
            )
        assert kept == []

    def test_matching_seq_survives(self):
        from imas_codex.standard_names.graph_ops import _verify_docs_claim_winners

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(return_value=[{"id": "x", "claim_seq": 7}])

        item = [{"id": "x", "claim_token": "tok", "claim_seq": 7}]
        with patch(_GC, return_value=gc):
            kept = _verify_docs_claim_winners(
                item, eligible_stage="refining", settle_seconds=0
            )
        assert [it["id"] for it in kept] == ["x"]

    def test_missing_claim_seq_falls_back_to_token_only(self):
        """Legacy claims / mocked readbacks without claim_seq keep the prior
        token-only behaviour (never over-drop on missing data)."""
        from imas_codex.standard_names.graph_ops import _verify_docs_claim_winners

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(return_value=[{"id": "x"}])  # no claim_seq column

        item = [{"id": "x", "claim_token": "tok"}]  # no claim_seq on the item
        with patch(_GC, return_value=gc):
            kept = _verify_docs_claim_winners(
                item, eligible_stage="refining", settle_seconds=0
            )
        assert [it["id"] for it in kept] == ["x"]

    def test_empty_batch_short_circuits(self):
        from imas_codex.standard_names.graph_ops import _verify_docs_claim_winners

        assert _verify_docs_claim_winners([], eligible_stage="refining") == []


class TestSeedStampsClaimSeq:
    """The shared seed must emit a monotonic claim_seq for the verifier."""

    def test_seed_and_readback_cypher_carry_claim_seq(self):
        from imas_codex.standard_names.graph_ops import _claim_sn_atomic

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        tx = MagicMock()
        tx.closed = False
        # seed returns a node, expand no-op (batch_size=1), readback returns it.
        tx.run = MagicMock(
            side_effect=[
                [{"_cluster_id": None, "_unit": None, "_physics_domain": None}],
                [{"id": "x", "claim_token": "t", "claim_seq": 4}],
            ]
        )
        session = MagicMock()
        session.begin_transaction = MagicMock(return_value=tx)
        gc.session = MagicMock(return_value=_CtxYielding(session))

        with patch(_GC, return_value=gc):
            items = _claim_sn_atomic(
                eligibility_where="sn.docs_stage = 'reviewed'",
                query_params={},
                batch_size=1,
                stage_field="docs_stage",
                to_stage="refining",
            )

        seed_cypher = tx.run.call_args_list[0][0][0]
        readback_cypher = tx.run.call_args_list[1][0][0]
        assert "sn.claim_seq = coalesce(sn.claim_seq, 0) + 1" in seed_cypher
        assert "sn.claim_seq AS claim_seq" in readback_cypher
        assert items and items[0]["claim_seq"] == 4


class _CtxYielding:
    """Minimal context manager returning *obj* from ``__enter__``."""

    def __init__(self, obj):
        self._obj = obj

    def __enter__(self):
        return self._obj

    def __exit__(self, *a):
        return False


class TestPersistOutcomeTripwire:
    """The wasted-paid-call ledger feeding the run-summary tripwire."""

    def test_records_attempts_and_waste(self):
        from imas_codex.standard_names.graph_ops import (
            _record_persist_outcome,
            persist_outcome_snapshot,
            reset_persist_outcomes,
        )

        run = "run-tripwire"
        reset_persist_outcomes(run)
        # Pilot forensics: 39 of 59 refine calls no-oped on a claim-race.
        for _ in range(20):
            _record_persist_outcome(run, "refine_docs", persisted=True)
        for _ in range(39):
            _record_persist_outcome(run, "refine_docs", persisted=False)

        snap = persist_outcome_snapshot(run)
        assert snap["refine_docs"] == {"attempts": 59, "wasted": 39}
        ratio = snap["refine_docs"]["wasted"] / snap["refine_docs"]["attempts"]
        assert ratio > 0.02  # trips the >2% warning

        reset_persist_outcomes(run)
        assert persist_outcome_snapshot(run) == {}

    def test_runs_are_isolated(self):
        from imas_codex.standard_names.graph_ops import (
            _record_persist_outcome,
            persist_outcome_snapshot,
            reset_persist_outcomes,
        )

        reset_persist_outcomes()
        _record_persist_outcome("a", "refine_docs", persisted=False)
        _record_persist_outcome("b", "refine_docs", persisted=True)
        assert persist_outcome_snapshot("a") == {
            "refine_docs": {"attempts": 1, "wasted": 1}
        }
        assert persist_outcome_snapshot("b") == {
            "refine_docs": {"attempts": 1, "wasted": 0}
        }
        reset_persist_outcomes()


class TestScopedReplicaCap:
    """Scoped drains cap docs-pool replicas at ~half the scope size."""

    def _build(self, scope_run_id, scope_size, configured=20, **kw):
        from imas_codex.standard_names import loop as loopmod

        mgr = MagicMock()
        stop = asyncio.Event()
        with (
            patch.object(loopmod, "_count_scope_names", return_value=scope_size),
            patch("imas_codex.settings.get_pool_replicas", return_value=configured),
        ):
            specs = loopmod._build_pool_specs(
                mgr, stop, scope_run_id=scope_run_id, docs_only=True, **kw
            )
        return {s.name: s for s in specs}

    def test_tiny_scope_collapses_docs_replicas_to_one(self):
        by = self._build(scope_run_id="run-x", scope_size=2, configured=20)
        assert by["refine_docs"].replicas == 1
        assert by["review_docs"].replicas == 1
        assert by["generate_docs"].replicas == 1

    def test_cap_is_half_scope_rounded_up(self):
        by = self._build(scope_run_id="run-x", scope_size=25, configured=20)
        # ceil(25/2) = 13, min(20, 13) = 13
        assert by["refine_docs"].replicas == 13

    def test_large_scope_keeps_configured(self):
        by = self._build(scope_run_id="run-x", scope_size=1000, configured=20)
        assert by["refine_docs"].replicas == 20

    def test_unscoped_run_is_unaffected(self):
        from imas_codex.standard_names import loop as loopmod

        mgr = MagicMock()
        stop = asyncio.Event()
        with (
            patch.object(loopmod, "_count_scope_names", return_value=2) as counter,
            patch("imas_codex.settings.get_pool_replicas", return_value=20),
        ):
            specs = loopmod._build_pool_specs(mgr, stop, docs_only=True)
        by = {s.name: s for s in specs}
        assert by["refine_docs"].replicas == 20
        counter.assert_not_called()


class TestCountScopeNames:
    def test_returns_count(self):
        from imas_codex.standard_names import loop as loopmod

        gc = MagicMock()
        gc.__enter__ = MagicMock(return_value=gc)
        gc.__exit__ = MagicMock(return_value=False)
        gc.query = MagicMock(return_value=[{"n": 7}])
        with patch("imas_codex.graph.client.GraphClient", return_value=gc):
            assert loopmod._count_scope_names("run-x") == 7

    def test_zero_on_error(self):
        from imas_codex.standard_names import loop as loopmod

        with patch(
            "imas_codex.graph.client.GraphClient", side_effect=RuntimeError("boom")
        ):
            assert loopmod._count_scope_names("run-x") == 0
