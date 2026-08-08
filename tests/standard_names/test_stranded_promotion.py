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
            and n.get("review_quorum_shortfall") is None
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


class TestQuorumShortfallIsHonoured:
    """A name the quorum gate parked must not be promoted by this pass.

    The two states are indistinguishable by score alone. A name stranded by a
    lowered threshold and a name held because its reviewer seats never reached
    a verdict both sit at 'reviewed' carrying a score above the bar — and a
    non-quorate mean can clear the bar while the seats behind it disagreed
    sharply per dimension. Promoting on the score alone publishes exactly what
    the gate withheld, so the marker decides.
    """

    def test_quorum_shortfall_blocks_promotion(self) -> None:
        nodes = [
            {
                "id": "shortfall",
                "name_stage": "reviewed",
                "reviewer_score_name": 0.925,
                "review_quorum_shortfall": (
                    "blind seats disagreed and the escalator seat did not "
                    "resolve them (method=max_cycles_reached)"
                ),
            }
        ]
        out = _run(nodes, min_score=0.85)
        assert out["name"] == 0
        assert nodes[0]["name_stage"] == "reviewed"

    def test_cleared_shortfall_promotes_again(self) -> None:
        """A quorate re-review nulls the marker, restoring ordinary eligibility."""
        nodes = [
            {
                "id": "cleared",
                "name_stage": "reviewed",
                "reviewer_score_name": 0.925,
                "review_quorum_shortfall": None,
            }
        ]
        out = _run(nodes, min_score=0.85)
        assert out["name"] == 1
        assert nodes[0]["name_stage"] == "accepted"

    def test_parked_name_survives_a_later_maintenance_pass(self) -> None:
        """The observed two-run sequence, end to end.

        Run one reviews the name: the blind seats split (0.85 and 1.0 overall,
        but 0.70 against 1.0 on one dimension), no escalator answers, and the
        gate parks it at 'reviewed' with the marker and a 0.925 mean. Run two
        starts and its maintenance pass sees a 'reviewed' name whose score
        clears the bar. It must leave it alone.
        """
        from imas_codex.standard_names.graph_ops import persist_reviewed_name

        gc_calls: list[dict] = []

        class _Recorder:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_a):
                return None

            def query(self_inner, _cypher, **params):
                gc_calls.append(params)
                if "target_stage" in params:
                    return [{"id": params["id"]}]
                return [{"chain_length": 0}]

        with patch.object(graph_ops, "GraphClient", return_value=_Recorder()):
            stage = persist_reviewed_name(
                sn_id="radial_coordinate_of_shunt",
                claim_token="tok",
                score=0.925,
                model="openrouter/x-ai/grok-4.5",
                min_score=0.85,
                rotation_cap=3,
                resolution_method="max_cycles_reached",
                reviewer_chain_size=3,
            )

        assert stage == "reviewed"
        write = next(p for p in gc_calls if "target_stage" in p)
        assert write["quorum_shortfall"]

        # Run two: the node as the gate left it, through the maintenance pass.
        nodes = [
            {
                "id": "radial_coordinate_of_shunt",
                "name_stage": write["target_stage"],
                "reviewer_score_name": 0.925,
                "review_quorum_shortfall": write["quorum_shortfall"],
            }
        ]
        out = _run(nodes, min_score=0.85)
        assert out["name"] == 0
        assert nodes[0]["name_stage"] == "reviewed"


class TestFakeMirrorsProduction:
    """The fake reimplements the production WHERE clause, so it can drift.

    That drift is how a promotion path kept accepting names after the quorum
    gate landed: the gate was tested, the pass was tested, and neither test
    knew the other clause existed. Pin the mirror to the real predicate so a
    future clause cannot be invisible here.
    """

    def test_every_production_name_clause_is_consulted_by_the_fake(self) -> None:
        import re

        from imas_codex.standard_names.graph_ops import PROMOTE_STRANDED_NAME_WHERE

        referenced = set(re.findall(r"sn\.(\w+)", PROMOTE_STRANDED_NAME_WHERE))

        class _RecordingNode(dict):
            """Records reads. Pre-populated so no clause short-circuits.

            The predicate is a chain of ``and``s, so probing with an empty node
            stops at the first clause and every later field looks unread.
            """

            def __init__(self) -> None:
                super().__init__(
                    name_stage="reviewed",
                    reviewer_score_name=1.0,
                    edit_status=None,
                    validation_status=None,
                    review_quorum_shortfall=None,
                )
                self.read: set[str] = set()

            def get(self, key, default=None):
                self.read.add(key)
                return super().get(key, default)

        node = _RecordingNode()
        assert _FakeGraph([])._name_eligible(node, 0.85), (
            "the probe node must satisfy every clause, or short-circuiting "
            "hides the fields this test is meant to observe"
        )

        missing = referenced - node.read
        assert not missing, (
            f"the fake ignores production predicate field(s) {sorted(missing)} — "
            "update _name_eligible so this harness still mirrors the real clause"
        )
