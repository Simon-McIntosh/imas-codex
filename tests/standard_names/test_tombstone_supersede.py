"""Tombstone-id supersede: fold a name INTO an already-existing accepted name.

``sn edit --rename`` refuses a rename onto an existing id, and the
source-keyed supersede only retires predecessors sharing one source — so
folding a name into an existing canonical name has no supported path.
``tombstone_supersede_into`` is that operation: it stamps the old name
``superseded`` with ``superseded_from_stage='accepted'`` and threads a
``REFINED_FROM`` lineage to the live successor so the P1 export emits a
``status: deprecated`` stub pointing at it.

All graph interaction is mocked (no live Neo4j). A small stateful fake models
the lookup + write so the stamp/merge and refusal guards are asserted
directly.
"""

from __future__ import annotations

from unittest.mock import patch

from imas_codex.standard_names import graph_ops
from imas_codex.standard_names.edit import supersede_into


class _FakeGraph:
    """Models {id: node} with the two queries tombstone_supersede_into issues:
    a lookup (OPTIONAL MATCH ... RETURN) and the write (SET + MERGE)."""

    def __init__(self, nodes: dict[str, dict]) -> None:
        self.nodes = nodes
        self.refined_from: set[tuple[str, str]] = set()  # (successor, predecessor)

    def __enter__(self) -> _FakeGraph:
        return self

    def __exit__(self, *_a) -> None:
        return None

    def _descends(self, a: str, b: str) -> bool:
        """True if a reaches b along REFINED_FROM (a -*-> b)."""
        seen, frontier = set(), [a]
        while frontier:
            cur = frontier.pop()
            for succ, pred in self.refined_from:
                if succ == cur and pred not in seen:
                    if pred == b:
                        return True
                    seen.add(pred)
                    frontier.append(pred)
        return False

    def query(self, cypher: str, **p):
        if "RETURN old.id AS old_id" in cypher:  # lookup
            old = self.nodes.get(p["old_id"])
            into = self.nodes.get(p["into_id"])
            return [
                {
                    "old_id": old["id"] if old else None,
                    "old_stage": old.get("name_stage") if old else None,
                    "old_sfs": old.get("superseded_from_stage") if old else None,
                    "into_id": into["id"] if into else None,
                    "into_stage": into.get("name_stage") if into else None,
                    # cycle: old already reaches into along REFINED_FROM, so
                    # threading into→old would close a loop.
                    "cycle": self._descends(p["old_id"], p["into_id"]),
                }
            ]
        if "SET old.name_stage = 'superseded'" in cypher:  # write
            old = self.nodes[p["old_id"]]
            old["name_stage"] = "superseded"
            old["superseded_from_stage"] = (
                old.get("superseded_from_stage") or "accepted"
            )
            old["claim_token"] = None
            old["claimed_at"] = None
            self.refined_from.add((p["into_id"], p["old_id"]))
            return []
        if "RETURN size(moved) AS moved" in cypher:
            return [{"moved": 0}]
        if "CREATE (change:StandardNameChange" in cypher:
            return []
        raise AssertionError(f"unexpected query: {cypher}")


def _run(nodes: dict[str, dict], old: str, into: str, dry_run: bool = False):
    fake = _FakeGraph(nodes)
    with patch.object(graph_ops, "GraphClient", return_value=fake):
        return supersede_into(old, into, dry_run=dry_run), fake


class TestSupersedeIntoAccepted:
    def test_stamps_superseded_sfs_and_refined_from(self) -> None:
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "accepted"},
        }
        res, fake = _run(nodes, "old", "into")
        assert res["ok"] is True
        assert nodes["old"]["name_stage"] == "superseded"
        assert nodes["old"]["superseded_from_stage"] == "accepted"
        assert nodes["old"]["claim_token"] is None
        # successor lineage: (into)-[:REFINED_FROM]->(old)
        assert ("into", "old") in fake.refined_from

    def test_allows_approved_target(self) -> None:
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "approved"},
        }
        res, fake = _run(nodes, "old", "into")
        assert res["ok"] is True
        assert nodes["old"]["name_stage"] == "superseded"
        assert ("into", "old") in fake.refined_from

    def test_dry_run_does_not_write(self) -> None:
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "accepted"},
        }
        res, fake = _run(nodes, "old", "into", dry_run=True)
        assert res["ok"] is True and res["dry_run"] is True
        assert nodes["old"]["name_stage"] == "accepted"  # untouched
        assert fake.refined_from == set()


class TestRefusals:
    def test_refuses_non_accepted_target(self) -> None:
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "reviewed"},
        }
        res, fake = _run(nodes, "old", "into")
        assert res["ok"] is False
        assert "not 'accepted' or 'approved'" in res["reason"]
        assert nodes["old"]["name_stage"] == "accepted"  # nothing written

    def test_refuses_missing_old(self) -> None:
        nodes = {"into": {"id": "into", "name_stage": "accepted"}}
        res, _ = _run(nodes, "old", "into")
        assert res["ok"] is False and "not found" in res["reason"]

    def test_refuses_missing_target(self) -> None:
        nodes = {"old": {"id": "old", "name_stage": "accepted"}}
        res, _ = _run(nodes, "old", "into")
        assert res["ok"] is False and "not found" in res["reason"]

    def test_refuses_self_fold(self) -> None:
        nodes = {"old": {"id": "old", "name_stage": "accepted"}}
        res, _ = _run(nodes, "old", "old")
        assert res["ok"] is False and "same" in res["reason"]

    def test_refuses_cycle(self) -> None:
        """old already descends from into (old is a successor of into) —
        threading into→old would form a REFINED_FROM cycle."""
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "accepted"},
        }
        fake = _FakeGraph(nodes)
        fake.refined_from.add(("old", "into"))  # old already a successor of into
        with patch.object(graph_ops, "GraphClient", return_value=fake):
            res = supersede_into("old", "into")
        assert res["ok"] is False and "cycle" in res["reason"]


class TestIdempotent:
    def test_second_run_restamps_and_reports_already(self) -> None:
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "accepted"},
        }
        fake = _FakeGraph(nodes)
        with patch.object(graph_ops, "GraphClient", return_value=fake):
            first = supersede_into("old", "into")
            second = supersede_into("old", "into")
        assert first["ok"] and first["already_superseded"] is False
        assert second["ok"] and second["already_superseded"] is True
        assert nodes["old"]["name_stage"] == "superseded"
        assert nodes["old"]["superseded_from_stage"] == "accepted"
        assert ("into", "old") in fake.refined_from  # single merged edge
