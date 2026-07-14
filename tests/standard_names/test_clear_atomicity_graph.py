"""Live-Neo4j integration test for the atomic scoped ``clear_standard_names``.

The unit suite (``test_clear_atomicity.py``) pins the cypher SHAPE against a
mocked graph but cannot prove the one behavioural assumption the single-
statement rewrite relies on: that within one statement,
``DELETE rel … WITH DISTINCT sn WHERE NOT EXISTS {()-[:HAS_STANDARD_NAME]->(sn)}``
sees POST-delete state (Neo4j's Eager read-after-write barrier), so the
orphan-guard protects a name still attached to an out-of-scope producer path.
This exercises that against a real database.

SAFETY — surgically self-contained:
  * Every node it creates carries the unique test-only id prefix
    ``__cleartest__`` (StandardName / StandardNameReview / IMASNode). No real
    id can collide.
  * The scoped clear's ``path_allowlist`` contains ONLY this test's synthetic
    IMASNode ids, so the delete matches only this test's synthetic
    StandardNames — never any real StandardName.
  * A before/after fixture wipes the ``__cleartest__`` subgraph, so the test
    leaves no residue even if an assertion fails mid-run.
  * The test creates NO StandardNameSource nodes, so the clear's global
    orphan-source reset (Step E) has nothing of this test's to touch; its
    global orphan-review sweep (Step C) only ever removes already-parentless
    review nodes (this test's reviews are deleted atomically WITH their parent,
    never orphaned).

Run only when Neo4j is reachable:
    uv run pytest tests/standard_names/test_clear_atomicity_graph.py -m graph -v
"""

from __future__ import annotations

import uuid

import pytest

from imas_codex.standard_names.graph_ops import clear_standard_names

_TEST_ID_PREFIX = "__cleartest__"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _gc():
    """Function-scoped real GraphClient; skip if Neo4j is unreachable."""
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Neo4j not available: {exc}")

    yield client
    client.close()


@pytest.fixture()
def _clean(_gc):
    """Delete every ``__cleartest__`` node before and after each test."""

    def _wipe() -> None:
        for label in ("StandardName", "StandardNameReview", "IMASNode"):
            _gc.query(
                f"MATCH (n:{label}) WHERE n.id STARTS WITH $p DETACH DELETE n",
                p=_TEST_ID_PREFIX,
            )

    _wipe()
    yield
    _wipe()


# ---------------------------------------------------------------------------
# Graph helpers (synthetic, prefixed)
# ---------------------------------------------------------------------------


def _uid(tag: str) -> str:
    return f"{_TEST_ID_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


def _create_imasnode(gc, node_id: str) -> None:
    gc.query(
        "MERGE (n:IMASNode {id: $id}) SET n.description = 'clear-atomicity test path'",
        id=node_id,
    )


def _create_sn(gc, sn_id: str, *, name_stage: str = "drafted") -> None:
    gc.query(
        """
        MERGE (sn:StandardName {id: $id})
        SET sn.name_stage = $name_stage,
            sn.description = 'clear-atomicity test quantity',
            sn.kind = 'scalar'
        """,
        id=sn_id,
        name_stage=name_stage,
    )


def _create_review(gc, review_id: str, sn_id: str) -> None:
    gc.query(
        """
        MERGE (r:StandardNameReview {id: $rid})
        WITH r
        MATCH (sn:StandardName {id: $sn_id})
        MERGE (sn)-[:HAS_REVIEW]->(r)
        """,
        rid=review_id,
        sn_id=sn_id,
    )


def _link_producer(gc, node_id: str, sn_id: str) -> None:
    gc.query(
        """
        MATCH (n:IMASNode {id: $node_id})
        MATCH (sn:StandardName {id: $sn_id})
        MERGE (n)-[:HAS_STANDARD_NAME]->(sn)
        """,
        node_id=node_id,
        sn_id=sn_id,
    )


def _sn_exists(gc, sn_id: str) -> bool:
    rows = gc.query("MATCH (sn:StandardName {id: $id}) RETURN count(sn) AS n", id=sn_id)
    return bool(rows and rows[0]["n"] > 0)


def _review_exists(gc, review_id: str) -> bool:
    rows = gc.query(
        "MATCH (r:StandardNameReview {id: $id}) RETURN count(r) AS n", id=review_id
    )
    return bool(rows and rows[0]["n"] > 0)


def _incoming_producer_count(gc, sn_id: str) -> int:
    rows = gc.query(
        """
        MATCH (:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName {id: $id})
        RETURN count(*) AS n
        """,
        id=sn_id,
    )
    return rows[0]["n"] if rows else 0


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_scoped_clear_orphan_guard_and_survivor_review(_gc, _clean) -> None:
    """A scoped clear (allowlisting one in-scope path) must, in ONE atomic
    statement against a real Neo4j:

    * delete a name whose ONLY producer is the in-scope path — and its review;
    * SPARE a name also attached to an out-of-scope path (orphan-guard sees
      post-delete state) — and spare that survivor's review;
    * remove only the in-scope producer edge from the survivor.
    """
    path_in = _uid("path_in")
    path_out = _uid("path_out")
    sn_single = _uid("sn_single")  # only producer is the in-scope path
    sn_two = _uid("sn_two")  # also produced by an out-of-scope path
    rev_single = _uid("rev_single")
    rev_two = _uid("rev_two")

    _create_imasnode(_gc, path_in)
    _create_imasnode(_gc, path_out)
    _create_sn(_gc, sn_single)
    _create_sn(_gc, sn_two)
    _create_review(_gc, rev_single, sn_single)
    _create_review(_gc, rev_two, sn_two)

    # sn_single: in-scope path only. sn_two: in-scope AND out-of-scope.
    _link_producer(_gc, path_in, sn_single)
    _link_producer(_gc, path_in, sn_two)
    _link_producer(_gc, path_out, sn_two)

    # Precondition sanity.
    assert _sn_exists(_gc, sn_single)
    assert _sn_exists(_gc, sn_two)
    assert _incoming_producer_count(_gc, sn_two) == 2

    # Scoped clear: allowlist ONLY the synthetic in-scope path. This matches
    # only this test's synthetic names.
    deleted = clear_standard_names(path_allowlist=[path_in])

    # Exactly one name became a true orphan and was deleted.
    assert deleted == 1, f"expected 1 deleted, got {deleted}"

    # sn_single + its review are GONE (its only producer edge was in-scope).
    assert not _sn_exists(_gc, sn_single), "single-path name should be deleted"
    assert not _review_exists(_gc, rev_single), (
        "deleted name's review must be removed in the same atomic statement"
    )

    # sn_two SURVIVES: the out-of-scope producer keeps it alive (orphan-guard
    # saw the post-delete state). Its review survives too (survivor-review
    # handling — never pre-deleted for a name that lives on).
    assert _sn_exists(_gc, sn_two), (
        "name with a live out-of-scope producer must survive"
    )
    assert _review_exists(_gc, rev_two), "survivor's review must NOT be deleted"

    # Only the in-scope producer edge was stripped from the survivor.
    assert _incoming_producer_count(_gc, sn_two) == 1, (
        "the in-scope producer edge should be removed, the out-of-scope one kept"
    )
