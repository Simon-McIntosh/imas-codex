"""Minting the standard-name review set from DD paths.

Pure logic over an injected graph view (stubbed ``gc``) covers the base join,
immediate-family closure union, deterministic sort, unmatched reporting.
Live-graph cases exercise the same behavior with synthetic nodes.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.minting import MintResult, mint_sn_list

# ── Pure behavior over a stubbed graph ─────────────────────────────────────


class _FakeGC:
    def __init__(self, base_rows, fam_rows):
        self._base = base_rows
        self._fam = fam_rows
        self.calls = 0
        self.base_params = None

    def query(self, cypher, **params):
        self.calls += 1
        if "PRODUCED_NAME" in cypher and "HAS_PARENT" not in cypher:
            self.base_params = params
            return self._base
        if "HAS_PARENT" in cypher:
            return self._fam
        return []


def test_mint_empty_does_not_touch_graph():
    fake = _FakeGC([], [])
    res = mint_sn_list([], gc=fake)
    assert res == MintResult(names=[], unmatched_paths=[])
    assert fake.calls == 0


def test_mint_base_join_family_and_unmatched():
    base = [
        {"path": "ids/p1", "ids": ["b_name"]},
        {"path": "ids/p2", "ids": ["a_name"]},
    ]
    fam = [{"fam_ids": ["parent_z", "sibling_y"]}]
    fake = _FakeGC(base, fam)

    res = mint_sn_list(["ids/p1", "ids/p2", "ids/p3"], gc=fake)

    # Base ∪ family, sorted and de-duplicated.
    assert res.names == ["a_name", "b_name", "parent_z", "sibling_y"]
    # p1/p2 matched; p3 has no linked name → reported, not dropped.
    assert res.unmatched_paths == ["ids/p3"]


def test_mint_joins_over_the_produced_name_edge_on_bare_paths():
    """The join key is the bare DD path, resolved to the ``dd:``-prefixed source.

    The name's ``source_paths`` projection stores prefixed ids, so joining on it
    against bare input paths matches nothing — the edge is the authority.
    """
    fake = _FakeGC([{"path": "ids/p1", "ids": ["n"]}], [{"fam_ids": []}])
    mint_sn_list(["ids/p1"], gc=fake)
    assert fake.base_params["paths"] == ["ids/p1"]


def test_mint_reports_a_path_whose_only_names_are_dead():
    """A row with no live names is unmatched, not a silent match."""
    fake = _FakeGC([{"path": "ids/p1", "ids": []}], [{"fam_ids": []}])
    res = mint_sn_list(["ids/p1"], gc=fake)
    assert res.names == []
    assert res.unmatched_paths == ["ids/p1"]


def test_mint_collects_every_name_a_path_produced() -> None:
    """One DD path may produce several live names; all belong to the batch."""
    fake = _FakeGC([{"path": "ids/p1", "ids": ["n2", "n1"]}], [{"fam_ids": []}])
    res = mint_sn_list(["ids/p1"], gc=fake)
    assert res.names == ["n1", "n2"]
    assert res.unmatched_paths == []


def test_mint_dedups_input_paths():
    fake = _FakeGC([{"path": "ids/p1", "ids": ["n"]}], [{"fam_ids": []}])
    res = mint_sn_list(["ids/p1", "ids/p1"], gc=fake)
    assert res.names == ["n"]
    assert res.unmatched_paths == []


# ── Live graph behavior ────────────────────────────────────────────────────

PREFIX = "__minttest__"
SOURCE_PREFIX = f"dd:{PREFIX}/"
LEAF1 = f"{PREFIX}/leaf1"
LEAF2 = f"{PREFIX}/leaf2"
LEAF_DEAD = f"{PREFIX}/leaf_dead"


def _cleanup():
    with GraphClient() as gc:
        gc.query(
            "MATCH (n) WHERE n.id STARTS WITH $name_prefix "
            "OR n.id STARTS WITH $source_prefix DETACH DELETE n",
            name_prefix=PREFIX,
            source_prefix=SOURCE_PREFIX,
        )


@pytest.fixture
def mint_graph():
    _cleanup()
    try:
        with GraphClient() as gc:
            gc.query(
                """
                MERGE (child_a:StandardName {id: $child_a})
                  SET child_a.name_stage='accepted', child_a.source_paths=['dd:'+$leaf1]
                MERGE (parent:StandardName {id: $parent})
                  SET parent.name_stage='accepted', parent.source_paths=[]
                MERGE (child_b:StandardName {id: $child_b})
                  SET child_b.name_stage='accepted', child_b.source_paths=[]
                MERGE (child_dead:StandardName {id: $child_dead})
                  SET child_dead.name_stage='superseded', child_dead.source_paths=[]
                MERGE (grandchild:StandardName {id: $grandchild})
                  SET grandchild.name_stage='accepted', grandchild.source_paths=[]
                MERGE (unrelated:StandardName {id: $unrelated})
                  SET unrelated.name_stage='accepted', unrelated.source_paths=[]
                MERGE (child_a)-[:HAS_PARENT]->(parent)
                MERGE (child_b)-[:HAS_PARENT]->(parent)
                MERGE (child_dead)-[:HAS_PARENT]->(parent)
                MERGE (grandchild)-[:HAS_PARENT]->(child_a)
                // The source node and its PRODUCED_NAME edge are what the base join
                // reads; source_paths above is only the projection of it. A dead
                // name also carries a source, to prove the stage filter bites.
                MERGE (src1:StandardNameSource {id: 'dd:'+$leaf1})
                  SET src1.source_id=$leaf1, src1.source_type='dd',
                      src1.status='composed'
                MERGE (src1)-[:PRODUCED_NAME]->(child_a)
                MERGE (src_dead:StandardNameSource {id: 'dd:'+$leaf_dead})
                  SET src_dead.source_id=$leaf_dead, src_dead.source_type='dd',
                      src_dead.status='composed'
                MERGE (src_dead)-[:PRODUCED_NAME]->(child_dead)
                """,
                child_a=f"{PREFIX}_child_a",
                parent=f"{PREFIX}_parent",
                child_b=f"{PREFIX}_child_b",
                child_dead=f"{PREFIX}_child_dead",
                grandchild=f"{PREFIX}_grandchild",
                unrelated=f"{PREFIX}_unrelated",
                leaf1=LEAF1,
                leaf_dead=LEAF_DEAD,
            )
        yield
    finally:
        _cleanup()


@pytest.mark.graph
def test_mint_cleanup_removes_fixture_sources():
    with GraphClient() as gc:
        gc.query(
            "MERGE (s:StandardNameSource {id: $id}) "
            "SET s.source_id=$source_id, s.source_type='dd', s.status='composed'",
            id=f"dd:{LEAF1}",
            source_id=LEAF1,
        )

    _cleanup()

    with GraphClient() as gc:
        remaining = gc.query(
            "MATCH (s:StandardNameSource) "
            "WHERE s.id STARTS WITH $source_prefix RETURN count(s) AS count",
            source_prefix=SOURCE_PREFIX,
        )[0]["count"]
    assert remaining == 0


@pytest.mark.graph
def test_mint_live_closure(mint_graph):
    res = mint_sn_list([LEAF1, LEAF2, LEAF_DEAD])

    assert res.names == sorted(
        [
            f"{PREFIX}_child_a",  # base join
            f"{PREFIX}_parent",  # parent (one hop up)
            f"{PREFIX}_child_b",  # sibling (parent's other child)
            f"{PREFIX}_grandchild",  # own child (one hop down)
        ]
    )
    # superseded sibling and the unrelated (no HAS_PARENT) name are excluded.
    assert f"{PREFIX}_child_dead" not in res.names
    assert f"{PREFIX}_unrelated" not in res.names
    # leaf2 has no source at all; leaf_dead has one, but it produced only a
    # superseded name — both are reported rather than silently matched.
    assert res.unmatched_paths == [LEAF2, LEAF_DEAD]
