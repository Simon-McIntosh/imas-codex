"""Pinned renames must never be rewritten or exhausted by refine_name.

A rename edit (``edit_mode='rename'``) carries an operator-chosen name string.
When a borderline pinned rename scores below threshold, the old refine path
tried to *reword* it — re-emitting the identical name tripped the
self-referential-refine guard and decomposing a lexicalised base tripped
grammar validation, either way wrongly marking a CORRECT name ``exhausted`` and
silently dropping the (already-superseded) predecessor's quantity from export.

The fix: refine resubmits a pinned rename to a fresh review quorum (bounded),
never rewrites it; and ``sn requeue`` recovers an already-stranded name. Graph
interaction is mocked (no live Neo4j) with a small stateful fake.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from imas_codex.standard_names import graph_ops


class _NodeGraph:
    """Models a single StandardName node across the two-query recovery/resubmit
    paths, dispatching on Cypher-text substrings."""

    def __init__(self, node: dict[str, Any]) -> None:
        self.node = node

    def __enter__(self) -> _NodeGraph:
        return self

    def __exit__(self, *_a: Any) -> None:
        return None

    def query(self, cypher: str, **p: Any) -> list[dict[str, Any]]:
        n = self.node
        # ── resubmit_pinned_rename_for_review (single MATCH+SET+RETURN) ──
        if "RETURN CASE WHEN n < $cap THEN 'resubmitted'" in cypher:
            if n.get("claim_token") != p["token"] or n.get("name_stage") != "refining":
                return []
            count = n.get("review_resubmit_count", 0) or 0
            if count < p["cap"]:
                n["name_stage"] = "drafted"
                n["review_resubmit_count"] = count + 1
                n["reviewer_score_name"] = None
                n["claim_token"] = None
                n["claimed_at"] = None
                return [{"outcome": "resubmitted"}]
            n["name_stage"] = "reviewed"
            n["claim_token"] = None
            n["claimed_at"] = None
            return [{"outcome": "capped"}]
        # ── requeue_name_for_review probe ──
        if "RETURN sn.name_stage AS stage" in cypher:
            return [{"stage": n.get("name_stage")}] if n else []
        # ── requeue_name_for_review write ──
        if (
            "SET sn.name_stage = 'drafted'" in cypher
            and "review_resubmit_count = 0" in cypher
        ):
            if n.get("name_stage") in ("exhausted", "reviewed"):
                n["name_stage"] = "drafted"
                n["reviewer_score_name"] = None
                n["review_resubmit_count"] = 0
                n["claim_token"] = None
                n["claimed_at"] = None
            return []
        raise AssertionError(f"unexpected query: {cypher}")


def _resubmit(node: dict[str, Any], token: str, cap: int = 4):
    fake = _NodeGraph(node)
    with patch.object(graph_ops, "GraphClient", return_value=fake):
        return graph_ops.resubmit_pinned_rename_for_review(
            sn_id=node["id"], token=token, rotation_cap=cap
        )


def _requeue(node: dict[str, Any], dry_run: bool = False):
    fake = _NodeGraph(node)
    with patch.object(graph_ops, "GraphClient", return_value=fake):
        return graph_ops.requeue_name_for_review(node["id"], dry_run=dry_run)


class TestResubmitPinnedRename:
    def test_under_cap_resubmits_to_drafted(self) -> None:
        node = {
            "id": "n",
            "name_stage": "refining",
            "claim_token": "tok",
            "review_resubmit_count": 0,
            "reviewer_score_name": 0.83,
        }
        out = _resubmit(node, "tok", cap=4)
        assert out == "resubmitted"
        assert node["name_stage"] == "drafted"
        assert node["review_resubmit_count"] == 1
        # Score cleared so the fresh quorum re-scores from scratch.
        assert node["reviewer_score_name"] is None
        assert node["claim_token"] is None

    def test_at_cap_rests_at_reviewed_never_exhausted(self) -> None:
        node = {
            "id": "n",
            "name_stage": "refining",
            "claim_token": "tok",
            "review_resubmit_count": 4,
            "reviewer_score_name": 0.83,
        }
        out = _resubmit(node, "tok", cap=4)
        assert out == "capped"
        # Never exhausted — left reviewable for operator resolution.
        assert node["name_stage"] == "reviewed"
        assert node["review_resubmit_count"] == 4

    def test_token_mismatch_is_noop(self) -> None:
        node = {"id": "n", "name_stage": "refining", "claim_token": "other"}
        out = _resubmit(node, "tok")
        assert out == ""
        assert node["name_stage"] == "refining"


class TestRequeueNameForReview:
    def test_requeues_exhausted(self) -> None:
        node = {"id": "n", "name_stage": "exhausted", "reviewer_score_name": 0.8}
        res = _requeue(node)
        assert res["ok"] is True and res["prior_stage"] == "exhausted"
        assert node["name_stage"] == "drafted"
        assert node["reviewer_score_name"] is None
        assert node["review_resubmit_count"] == 0

    def test_requeues_reviewed(self) -> None:
        node = {"id": "n", "name_stage": "reviewed"}
        res = _requeue(node)
        assert res["ok"] is True
        assert node["name_stage"] == "drafted"

    def test_refuses_accepted(self) -> None:
        node = {"id": "n", "name_stage": "accepted"}
        res = _requeue(node)
        assert res["ok"] is False and "accepted" in res["reason"]
        assert node["name_stage"] == "accepted"

    def test_refuses_superseded(self) -> None:
        node = {"id": "n", "name_stage": "superseded"}
        res = _requeue(node)
        assert res["ok"] is False
        assert node["name_stage"] == "superseded"

    def test_refuses_missing(self) -> None:
        fake = _NodeGraph({})
        with patch.object(graph_ops, "GraphClient", return_value=fake):
            res = graph_ops.requeue_name_for_review("missing")
        assert res["ok"] is False and "not found" in res["reason"]

    def test_dry_run_does_not_write(self) -> None:
        node = {"id": "n", "name_stage": "exhausted"}
        res = _requeue(node, dry_run=True)
        assert res["ok"] is True and res["dry_run"] is True
        assert node["name_stage"] == "exhausted"  # untouched


class TestRefineClaimExcludesCappedPinnedRenames:
    def test_claim_where_excludes_capped_pinned_renames(self) -> None:
        """The refine eligibility clause must skip pinned renames that have
        spent their re-review budget, so they rest at 'reviewed' instead of
        re-looping through refine."""
        captured: dict[str, str] = {}

        def _fake_claim(*, eligibility_where: str, **_kw: Any) -> list:
            captured["where"] = eligibility_where
            return []

        with (
            patch.object(graph_ops, "_claim_sn_atomic", side_effect=_fake_claim),
            patch.object(graph_ops, "_verify_name_claim_winners", return_value=[]),
        ):
            graph_ops.claim_refine_name_batch()

        where = captured["where"]
        assert "coalesce(sn.edit_mode, '') = 'rename'" in where
        assert "review_resubmit_count" in where
