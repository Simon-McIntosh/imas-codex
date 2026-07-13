"""Stranded-reviewed promotion: names whose stored score clears the CURRENT
threshold but are stuck at ``'reviewed'`` are promoted to ``'accepted'``.

A name is scored once and staged against the threshold in force at review
time. Lowering the acceptance threshold later strands names that scored
between the old and new thresholds — refine only claims below-threshold
names, so a stored score that already clears the current threshold is never
re-touched. ``promote_stranded_reviewed`` is the idempotent startup pass that
flips those to accepted on both axes.

Behavioural test: a small stateful fake graph applies the promotion query's
WHERE predicate in Python, so promote / guard / idempotency invariants are
asserted without a live Neo4j.
"""

from __future__ import annotations

from unittest.mock import patch

from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.graph_ops import promote_stranded_reviewed


class _FakeGraph:
    def __init__(self, nodes: list[dict]) -> None:
        self.nodes = nodes

    def __enter__(self) -> _FakeGraph:
        return self

    def __exit__(self, *_a) -> None:
        return None

    def _name_eligible(self, n: dict, min_score: float) -> bool:
        return (
            n.get("name_stage") == "reviewed"
            and n.get("reviewer_score_name", 0.0) >= min_score
            and (n.get("edit_status") or "") != "open"
            and (n.get("validation_status") or "") != "quarantined"
        )

    def _docs_eligible(self, n: dict, min_score: float) -> bool:
        return (
            n.get("docs_stage") == "reviewed"
            and n.get("reviewer_score_docs", 0.0) >= min_score
            and n.get("name_stage") == "accepted"
            and (n.get("validation_status") or "") != "quarantined"
        )

    def query(self, cypher: str, **params):
        ms = params["min_score"]
        is_name = "sn.name_stage = 'reviewed'" in cypher
        is_docs = "sn.docs_stage = 'reviewed'" in cypher
        mutate = "SET sn." in cypher
        pred = self._name_eligible if is_name else self._docs_eligible
        assert is_name ^ is_docs, cypher
        hits = [n for n in self.nodes if pred(n, ms)]
        if mutate:
            for n in hits:
                if is_name:
                    n["name_stage"] = "accepted"
                else:
                    n["docs_stage"] = "accepted"
        return [{"n": len(hits)}]


def _run(nodes: list[dict], min_score: float = 0.7, dry_run: bool = False):
    fake = _FakeGraph(nodes)
    with patch.object(graph_ops, "GraphClient", return_value=fake):
        return promote_stranded_reviewed(min_score, dry_run=dry_run)


class TestNameAxis:
    def test_reviewed_at_or_above_threshold_promotes(self) -> None:
        nodes = [{"id": "a", "name_stage": "reviewed", "reviewer_score_name": 0.72}]
        out = _run(nodes, min_score=0.7)
        assert out["name"] == 1
        assert nodes[0]["name_stage"] == "accepted"

    def test_below_threshold_not_promoted(self) -> None:
        nodes = [{"id": "b", "name_stage": "reviewed", "reviewer_score_name": 0.65}]
        out = _run(nodes, min_score=0.7)
        assert out["name"] == 0
        assert nodes[0]["name_stage"] == "reviewed"

    def test_open_edit_not_promoted(self) -> None:
        """A name carrying an unapplied edit must go through the normal accept
        path (which applies the rename / descendant cascade), never a bare
        stage flip."""
        nodes = [
            {
                "id": "c",
                "name_stage": "reviewed",
                "reviewer_score_name": 0.9,
                "edit_status": "open",
            }
        ]
        out = _run(nodes, min_score=0.7)
        assert out["name"] == 0
        assert nodes[0]["name_stage"] == "reviewed"

    def test_quarantined_not_promoted(self) -> None:
        nodes = [
            {
                "id": "d",
                "name_stage": "reviewed",
                "reviewer_score_name": 0.9,
                "validation_status": "quarantined",
            }
        ]
        out = _run(nodes, min_score=0.7)
        assert out["name"] == 0


class TestDocsAxis:
    def test_reviewed_docs_on_accepted_name_promotes(self) -> None:
        nodes = [
            {
                "id": "e",
                "name_stage": "accepted",
                "docs_stage": "reviewed",
                "reviewer_score_docs": 0.8,
            }
        ]
        out = _run(nodes, min_score=0.7)
        assert out["docs"] == 1
        assert nodes[0]["docs_stage"] == "accepted"

    def test_below_threshold_docs_not_promoted(self) -> None:
        nodes = [
            {
                "id": "f",
                "name_stage": "accepted",
                "docs_stage": "reviewed",
                "reviewer_score_docs": 0.5,
            }
        ]
        out = _run(nodes, min_score=0.7)
        assert out["docs"] == 0


class TestIdempotency:
    def test_second_run_is_noop(self) -> None:
        nodes = [
            {"id": "a", "name_stage": "reviewed", "reviewer_score_name": 0.9},
            {
                "id": "e",
                "name_stage": "accepted",
                "docs_stage": "reviewed",
                "reviewer_score_docs": 0.9,
            },
        ]
        first = _run(nodes, min_score=0.7)
        assert first == {"name": 1, "docs": 1}
        second = _run(nodes, min_score=0.7)
        assert second == {"name": 0, "docs": 0}

    def test_dry_run_counts_without_mutating(self) -> None:
        nodes = [{"id": "a", "name_stage": "reviewed", "reviewer_score_name": 0.9}]
        out = _run(nodes, min_score=0.7, dry_run=True)
        assert out["name"] == 1
        assert nodes[0]["name_stage"] == "reviewed"  # unchanged
