"""Tombstone-id supersede: fold a name INTO an already-existing accepted name.

``sn edit --rename`` refuses a rename onto an existing id, and the
source-keyed supersede only retires predecessors sharing one source — so
folding a name into an existing canonical name has no supported path.
``tombstone_supersede_into`` is that operation: it stamps the old name
``superseded`` with ``superseded_from_stage='accepted'`` and threads a
``REFINED_FROM`` lineage to the live successor so the P1 export emits a
``status: deprecated`` stub pointing at it.

The fold also carries the folded name's sources onto the target: a fold asserts
the two names denote one quantity, so a source whose only name was the folded
one must not be left un-named.

All graph interaction is mocked (no live Neo4j). A small stateful fake models
the lookup + write so the stamp/merge, the carry-over, and the refusal guards
are asserted directly.
"""

from __future__ import annotations

from unittest.mock import patch

from imas_codex.standard_names import attachment_audit, graph_ops
from imas_codex.standard_names.attachment_audit import AttachmentAuditResult
from imas_codex.standard_names.edit import supersede_into


class _FakeGraph:
    """Models {id: node} plus the source→name edges the fold has to carry.

    Answers the four queries ``tombstone_supersede_into`` issues: the lookup,
    the carry-over census, the tombstone write, and the source carry-over.
    """

    def __init__(
        self, nodes: dict[str, dict], produced: dict[str, set[str]] | None = None
    ) -> None:
        self.nodes = nodes
        self.refined_from: set[tuple[str, str]] = set()  # (successor, predecessor)
        #: source id → the set of names it produces
        self.produced: dict[str, set[str]] = produced or {}

    def _sources_of(self, name: str) -> list[str]:
        return sorted(s for s, names in self.produced.items() if name in names)

    def _live(self, name: str) -> bool:
        stage = self.nodes.get(name, {}).get("name_stage", "")
        return stage not in ("superseded", "exhausted")

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
        if "RETURN count(src) AS sources" in cypher:  # carry-over census
            srcs = self._sources_of(p["old_id"])
            strand = [
                s
                for s in srcs
                if not any(n != p["old_id"] and self._live(n) for n in self.produced[s])
            ]
            return [{"sources": len(srcs), "would_strand": len(strand)}]
        if "MERGE (src)-[:PRODUCED_NAME]->(into)" in cypher:  # carry-over write
            for s in self._sources_of(p["old_id"]):
                self.produced[s].add(p["into_id"])
            return []
        if "SET old.name_stage = 'superseded'" in cypher:  # write
            old = self.nodes[p["old_id"]]
            old["name_stage"] = "superseded"
            old["superseded_from_stage"] = (
                old.get("superseded_from_stage") or "accepted"
            )
            old["claim_token"] = None
            old["claimed_at"] = None
            # Mirror the Cypher CASE that closes an open edit on the fold.
            if old.get("edit_status") == "open":
                old["edit_status"] = "applied"
            self.refined_from.add((p["into_id"], p["old_id"]))
            return []
        if "RETURN size(moved) AS moved" in cypher:
            return [{"moved": 0}]
        if "CREATE (change:StandardNameChange" in cypher:
            return []
        raise AssertionError(f"unexpected query: {cypher}")


def _run(
    nodes: dict[str, dict],
    old: str,
    into: str,
    dry_run: bool = False,
    produced: dict[str, set[str]] | None = None,
):
    fake = _FakeGraph(nodes, produced)
    # The consistency gate on the carried-over pairings opens its own client
    # against the live graph; these are mocked unit tests, so stub it out. Its
    # own behaviour is covered in test_attachment_audit.
    with (
        patch.object(graph_ops, "GraphClient", return_value=fake),
        patch.object(
            attachment_audit,
            "gate_migrated_attachments",
            return_value=AttachmentAuditResult(),
        ),
    ):
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

    def test_closes_open_edit_on_fold(self) -> None:
        """A predecessor folded with a still-open edit must not stay 'open' —
        it becomes terminal (unreviewable), so the edit is reconciled to
        'applied' rather than orphaned."""
        nodes = {
            "old": {"id": "old", "name_stage": "accepted", "edit_status": "open"},
            "into": {"id": "into", "name_stage": "accepted"},
        }
        res, _ = _run(nodes, "old", "into")
        assert res["ok"] is True
        assert nodes["old"]["name_stage"] == "superseded"
        assert nodes["old"]["edit_status"] == "applied"

    def test_leaves_closed_edit_untouched(self) -> None:
        """A predecessor with no open edit keeps whatever edit_status it had."""
        nodes = {
            "old": {"id": "old", "name_stage": "accepted", "edit_status": "rejected"},
            "into": {"id": "into", "name_stage": "accepted"},
        }
        res, _ = _run(nodes, "old", "into")
        assert res["ok"] is True
        assert nodes["old"]["edit_status"] == "rejected"

    def test_dry_run_does_not_write(self) -> None:
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "accepted"},
        }
        res, fake = _run(nodes, "old", "into", dry_run=True)
        assert res["ok"] is True and res["dry_run"] is True
        assert nodes["old"]["name_stage"] == "accepted"  # untouched
        assert fake.refined_from == set()


class TestSourceCarryOver:
    """A fold declares two names one quantity, so the sources must follow.

    Without the carry-over a source whose only name was the folded one is left
    with no live name at all — silently un-named, and rewound to paid
    re-composition by the next drain.
    """

    def test_sources_follow_the_fold(self) -> None:
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "accepted"},
        }
        produced = {"dd:a": {"old"}, "dd:b": {"old", "into"}, "dd:c": {"into"}}
        res, _ = _run(nodes, "old", "into", produced=produced)
        assert res["ok"] is True
        assert produced["dd:a"] == {"old", "into"}, "a source's only name was folded"
        assert produced["dd:b"] == {"old", "into"}, "already on both — no change"
        assert produced["dd:c"] == {"into"}

    def test_reports_how_many_sources_the_fold_carries(self) -> None:
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "accepted"},
        }
        produced = {"dd:a": {"old"}, "dd:b": {"old", "into"}}
        res, _ = _run(nodes, "old", "into", produced=produced)
        assert res["sources_carried"] == 2
        # Only dd:a had no other live name, so only it would have been stranded.
        assert res["sources_would_strand"] == 1

    def test_dry_run_reports_the_blast_radius_without_writing(self) -> None:
        """The operator must see how many sources hang on the fold before it runs."""
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "accepted"},
        }
        produced = {"dd:a": {"old"}, "dd:b": {"old"}}
        res, _ = _run(nodes, "old", "into", dry_run=True, produced=produced)
        assert res["sources_carried"] == 2
        assert res["sources_would_strand"] == 2
        assert produced == {"dd:a": {"old"}, "dd:b": {"old"}}, "dry run wrote"

    def test_a_source_already_named_elsewhere_does_not_count_as_stranded(self) -> None:
        nodes = {
            "old": {"id": "old", "name_stage": "accepted"},
            "into": {"id": "into", "name_stage": "accepted"},
            "other": {"id": "other", "name_stage": "accepted"},
            "dead": {"id": "dead", "name_stage": "superseded"},
        }
        produced = {"dd:a": {"old", "other"}, "dd:b": {"old", "dead"}}
        res, _ = _run(nodes, "old", "into", produced=produced)
        # dd:a is covered by a live name; dd:b's only other name is a tombstone.
        assert res["sources_would_strand"] == 1


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
