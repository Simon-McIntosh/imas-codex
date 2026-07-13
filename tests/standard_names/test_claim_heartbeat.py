"""Claim-lease heartbeat: a held claim is kept alive under a long batch.

A quorum-consensus review (or an enrich batch) can outrun the claim lease
TTL. Without a heartbeat the lease expires mid-flight and a peer worker
re-claims the same names — duplicating the paid LLM spend. ``refresh_name_claims``
bumps ``claimed_at`` on the batch's names (compare-and-set on the claim token)
so a legitimately-held claim never goes stale while it is being processed.

All graph interaction is mocked (no live Neo4j). The reclaim predicate is
modelled from the claim functions' eligibility clause
(``claimed_at IS NULL OR claimed_at < now - TTL``) so the three lease
invariants can be asserted without a live database.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.graph_ops import (
    _CLAIM_TIMEOUT_SECONDS,
    refresh_name_claims,
)

TTL = timedelta(seconds=_CLAIM_TIMEOUT_SECONDS)
T0 = datetime(2026, 7, 13, 12, 0, 0)


class _FakeGraph:
    """A minimal stateful GraphClient standing in for Neo4j.

    Holds ``{id: {"claim_token", "claimed_at"}}`` and interprets exactly the
    ``refresh_name_claims`` compare-and-set query. ``now`` is the mock clock
    used as ``datetime()`` for the refresh SET.
    """

    def __init__(self, nodes: dict[str, dict], now: datetime) -> None:
        self.nodes = nodes
        self.now = now

    def __enter__(self) -> _FakeGraph:
        return self

    def __exit__(self, *_a) -> None:
        return None

    def query(self, cypher: str, **params):
        if "SET sn.claimed_at = datetime()" in cypher and "claim_token: $token" in cypher:
            refreshed = 0
            for nid in params["ids"]:
                node = self.nodes.get(nid)
                # compare-and-set: only nodes still holding our token
                if node is not None and node.get("claim_token") == params["token"]:
                    node["claimed_at"] = self.now
                    refreshed += 1
            return [{"refreshed": refreshed}]
        raise AssertionError(f"unexpected query: {cypher}")


def _reclaimable(node: dict, now: datetime, ttl: timedelta = TTL) -> bool:
    """The claim functions' staleness predicate, in Python.

    A node is eligible for re-claim when it has no lease or its lease is
    older than the TTL.
    """
    ca = node.get("claimed_at")
    return ca is None or ca < now - ttl


def _patch_gc(fake: _FakeGraph):
    return patch.object(graph_ops, "GraphClient", return_value=fake)


class TestRefreshShape:
    def test_returns_zero_without_hitting_graph_when_empty(self) -> None:
        gc = MagicMock()
        with patch.object(graph_ops, "GraphClient", return_value=gc):
            assert refresh_name_claims([], "tok") == 0
            assert refresh_name_claims(["a"], "") == 0
        gc.query.assert_not_called()

    def test_compare_and_set_on_token(self) -> None:
        """A refresh only touches nodes STILL holding our token — a lease lost
        to a peer (token overwritten) is never stolen back."""
        nodes = {
            "mine": {"claim_token": "TOK", "claimed_at": T0},
            "stolen": {"claim_token": "OTHER", "claimed_at": T0},
        }
        fake = _FakeGraph(nodes, now=T0 + timedelta(seconds=100))
        with _patch_gc(fake):
            n = refresh_name_claims(["mine", "stolen"], "TOK")
        assert n == 1
        assert nodes["mine"]["claimed_at"] == T0 + timedelta(seconds=100)
        assert nodes["stolen"]["claimed_at"] == T0  # untouched


class TestLeaseInvariants:
    def test_heartbeat_refreshes_within_ttl(self) -> None:
        """(a) A refresh inside the TTL bumps claimed_at to now."""
        nodes = {"n": {"claim_token": "TOK", "claimed_at": T0}}
        beat_at = T0 + timedelta(seconds=_CLAIM_TIMEOUT_SECONDS // 3)
        fake = _FakeGraph(nodes, now=beat_at)
        with _patch_gc(fake):
            assert refresh_name_claims(["n"], "TOK") == 1
        assert nodes["n"]["claimed_at"] == beat_at

    def test_stale_unbeaten_claim_is_reclaimable(self) -> None:
        """(b) With no heartbeat, a claim older than the TTL is reclaimable."""
        node = {"claim_token": "TOK", "claimed_at": T0}
        now = T0 + TTL + timedelta(seconds=1)
        assert _reclaimable(node, now) is True

    def test_heartbeated_claim_is_not_reclaimable(self) -> None:
        """(c) A live-heartbeated claim stays fresh — NOT reclaimable even
        long after the original claim, because each beat resets claimed_at."""
        nodes = {"n": {"claim_token": "TOK", "claimed_at": T0}}
        # Two heartbeats at TTL/3 and 2*TTL/3.
        third = _CLAIM_TIMEOUT_SECONDS // 3
        for elapsed in (third, 2 * third):
            fake = _FakeGraph(nodes, now=T0 + timedelta(seconds=elapsed))
            with _patch_gc(fake):
                refresh_name_claims(["n"], "TOK")
        # A peer checks staleness shortly after the last beat.
        now = T0 + timedelta(seconds=2 * third + 10)
        assert _reclaimable(nodes["n"], now) is False
        # Sanity: had it NOT been beaten, the same wall-clock WOULD be stale
        # once past the TTL from the original claim.
        never_beaten = {"claimed_at": T0}
        assert _reclaimable(never_beaten, T0 + TTL + timedelta(seconds=1)) is True
