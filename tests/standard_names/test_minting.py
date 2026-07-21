"""Minting the standard-name review set from DD paths.

Section 1 — pure logic over an injected graph view (stubbed ``gc``): base join,
immediate-family closure union, deterministic sort, unmatched reporting.
Section 2 — the same over a live graph with synthetic nodes.
"""

from __future__ import annotations

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.minting import MintResult, mint_sn_list

# ── Section 1 — pure (stubbed graph) ──────────────────────────────────────


class _FakeGC:
    def __init__(self, base_rows, fam_rows):
        self._base = base_rows
        self._fam = fam_rows
        self.calls = 0

    def query(self, cypher, **params):
        self.calls += 1
        if "source_paths" in cypher and "HAS_PARENT" not in cypher:
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
        {"id": "b_name", "source_paths": ["ids/p1", "ids/other"]},
        {"id": "a_name", "source_paths": ["ids/p2"]},
    ]
    fam = [{"fam_ids": ["parent_z", "sibling_y"]}]
    fake = _FakeGC(base, fam)

    res = mint_sn_list(["ids/p1", "ids/p2", "ids/p3"], gc=fake)

    # Base ∪ family, sorted and de-duplicated.
    assert res.names == ["a_name", "b_name", "parent_z", "sibling_y"]
    # p1/p2 matched; p3 has no linked name → reported, not dropped.
    assert res.unmatched_paths == ["ids/p3"]


def test_mint_dedups_input_paths():
    fake = _FakeGC([{"id": "n", "source_paths": ["ids/p1"]}], [{"fam_ids": []}])
    res = mint_sn_list(["ids/p1", "ids/p1"], gc=fake)
    assert res.names == ["n"]
    assert res.unmatched_paths == []


# ── Section 2 — live graph ────────────────────────────────────────────────

PREFIX = "__minttest__"
LEAF1 = f"{PREFIX}/leaf1"
LEAF2 = f"{PREFIX}/leaf2"


def _cleanup():
    with GraphClient() as gc:
        gc.query("MATCH (n) WHERE n.id STARTS WITH $p DETACH DELETE n", p=PREFIX)


@pytest.fixture
def mint_graph():
    _cleanup()
    with GraphClient() as gc:
        gc.query(
            """
            MERGE (child_a:StandardName {id: $child_a})
              SET child_a.name_stage='accepted', child_a.source_paths=[$leaf1]
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
            """,
            child_a=f"{PREFIX}_child_a",
            parent=f"{PREFIX}_parent",
            child_b=f"{PREFIX}_child_b",
            child_dead=f"{PREFIX}_child_dead",
            grandchild=f"{PREFIX}_grandchild",
            unrelated=f"{PREFIX}_unrelated",
            leaf1=LEAF1,
        )
    yield
    _cleanup()


@pytest.mark.graph
def test_mint_live_closure(mint_graph):
    res = mint_sn_list([LEAF1, LEAF2])

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
    # leaf2 has no linked name → reported.
    assert res.unmatched_paths == [LEAF2]
