"""Docs-axis reconciliation: the lifecycle mirror is re-derived from surviving reviews.

A lifecycle scalar can be reset by unrelated work while the docs-axis review
history that justified it survives. ``reconcile_docs_axis_from_reviews``
restores ``docs_stage`` and ``reviewer_score_docs`` from the winning surviving
docs review — and refuses a name that has no docs-axis review at all, so it
cannot manufacture a false-acceptance projection.

Behavioural test: a stateful fake graph applies the statement's predicate in
Python, so the restore, the refusal and idempotency are asserted without a live
Neo4j.
"""

from __future__ import annotations

from unittest.mock import patch

from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.graph_ops import reconcile_docs_axis_from_reviews


class _FakeDocsAxisGraph:
    """Stateful stand-in whose ``query`` applies the statement's predicate."""

    def __init__(self, nodes: list[dict]) -> None:
        self.nodes = nodes

    def __enter__(self) -> _FakeDocsAxisGraph:
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def _scope(self, ids: list[str] | None) -> list[dict]:
        return [
            n
            for n in self.nodes
            if (ids is None or n["id"] in ids)
            and n.get("name_stage") == "accepted"
            and (n.get("validation_status") or "") != "quarantined"
        ]

    @staticmethod
    def _has_winner(node: dict) -> bool:
        return node.get("_winner_score") is not None

    @staticmethod
    def _agrees(node: dict) -> bool:
        score = node.get("reviewer_score_docs")
        if node.get("docs_stage") != "accepted" or score is None:
            return False
        return abs(score - node["_winner_score"]) <= 1e-9

    def query(self, cypher: str, **params):
        scope = self._scope(params.get("ids"))
        if "sum(CASE WHEN" in cypher:
            with_winner = [n for n in scope if self._has_winner(n)]
            return [
                {
                    "with_winner": len(with_winner),
                    "repair": sum(1 for n in with_winner if not self._agrees(n)),
                    "agree": sum(1 for n in with_winner if self._agrees(n)),
                }
            ]
        if "SET sn." not in cypher:
            raise AssertionError(f"unexpected statement: {cypher}")
        hits = [n for n in scope if self._has_winner(n) and not self._agrees(n)]
        for node in hits:
            node["docs_stage"] = "accepted"
            node["reviewer_score_docs"] = node["_winner_score"]
        return [{"repair": len(hits)}]


def _run(nodes, *, dry_run: bool, ids: list[str] | None = None):
    fake = _FakeDocsAxisGraph(nodes)
    with patch.object(graph_ops, "GraphClient", return_value=fake):
        result = reconcile_docs_axis_from_reviews(dry_run=dry_run, ids=ids, gc=None)
    return result


def _node(name_id: str, *, winner: float | None, stage: str = "pending", score=None):
    return {
        "id": name_id,
        "name_stage": "accepted",
        "docs_stage": stage,
        "reviewer_score_docs": score,
        "_winner_score": winner,
    }


def test_restores_a_regressed_row_from_its_surviving_docs_review():
    """A reset docs scalar is re-derived from the winning surviving review."""
    nodes = [_node("effective_charge", winner=0.9375)]
    result = _run(nodes, dry_run=False, ids=["effective_charge"])
    assert result == {"repaired": 1}
    assert nodes[0]["docs_stage"] == "accepted"
    assert nodes[0]["reviewer_score_docs"] == 0.9375


def test_refuses_a_row_with_no_docs_axis_review():
    """Without a docs review the function must not invent an acceptance."""
    nodes = [_node("alpha_critical_parameter", winner=None)]
    result = _run(nodes, dry_run=False, ids=["alpha_critical_parameter"])
    assert result == {"repaired": 0}
    assert nodes[0]["docs_stage"] == "pending"
    assert nodes[0]["reviewer_score_docs"] is None
    assert nodes[0]["_winner_score"] is None


def test_dry_run_separates_the_repair_from_the_refusal():
    nodes = [
        _node("effective_charge", winner=0.9375),
        _node("alpha_critical_parameter", winner=None),
    ]
    result = _run(
        nodes, dry_run=True, ids=["effective_charge", "alpha_critical_parameter"]
    )
    assert result == {
        "agree": 0,
        "repair": 1,
        "with_winner": 1,
        "refused_no_review": 1,
    }


def test_rerun_is_idempotent_once_the_scalar_agrees():
    nodes = [_node("safety_factor", winner=1.0)]
    assert _run(nodes, dry_run=False, ids=["safety_factor"]) == {"repaired": 1}
    assert _run(nodes, dry_run=False, ids=["safety_factor"]) == {"repaired": 0}
    assert _run(nodes, dry_run=True, ids=["safety_factor"]) == {
        "agree": 1,
        "repair": 0,
        "with_winner": 1,
        "refused_no_review": 0,
    }
