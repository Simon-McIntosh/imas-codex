"""Focus/batch scoping binds names to a run without resetting them by default.

A ``--focus``/``--batch`` drain binds the focused paths to a scope ``run_id`` so
the scoped pools claim only those names. The default must be **no-reset**:
resetting in-flight items to ``pending`` churns the hard tail (each pass
re-shoots exhausted names), so the blunt re-stage lives behind ``--reseed`` and
targeted resets behind ``--reset-to``. ``scope_focus_names`` implements both
modes; these pin the behaviour.
"""

from __future__ import annotations

import uuid

import pytest

_PREFIX = "test_focus_scope__"
_SOURCE_PREFIX = f"dd:{_PREFIX}"


@pytest.fixture()
def _gc():
    try:
        from imas_codex.graph.client import GraphClient

        client = GraphClient()
        client.get_stats()
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"Neo4j not available: {exc}")
    yield client
    client.close()


def _wipe_fixture_nodes(gc) -> None:
    gc.query(
        "MATCH (n:StandardName) WHERE n.id STARTS WITH $p DETACH DELETE n",
        p=_PREFIX,
    )
    gc.query(
        "MATCH (n:StandardNameSource) WHERE n.id STARTS WITH $p DETACH DELETE n",
        p=_SOURCE_PREFIX,
    )


@pytest.fixture()
def _clean(_gc):
    _wipe_fixture_nodes(_gc)
    try:
        yield
    finally:
        _wipe_fixture_nodes(_gc)


def _uid(tag: str) -> str:
    return f"{_PREFIX}{tag}_{uuid.uuid4().hex[:8]}"


def _seed(gc, src: str, name: str, *, stage: str, score: float) -> None:
    gc.query(
        """
        MERGE (sns:StandardNameSource {id: $src})
        SET sns.source_id = substring($src, 3), sns.status = 'composed',
            sns.source_type = 'dd'
        MERGE (sn:StandardName {id: $name})
        SET sn.name_stage = $stage, sn.docs_stage = 'drafted',
            sn.reviewer_score_name = $score, sn.origin = 'pipeline',
            sn.run_id = null
        MERGE (sns)-[:PRODUCED_NAME]->(sn)
        """,
        src=src,
        name=name,
        stage=stage,
        score=score,
    )


def _read(gc, name: str) -> dict:
    return gc.query(
        "MATCH (sn:StandardName {id:$n}) RETURN sn.name_stage AS stage, "
        "sn.reviewer_score_name AS score, sn.run_id AS run_id",
        n=name,
    )[0]


@pytest.mark.graph
def test_cleanup_removes_fixture_names_and_sources(_gc):
    src, name = f"dd:{_uid('cleanup_s')}", _uid("cleanup_nm")
    try:
        _seed(_gc, src, name, stage="reviewed", score=0.7)
        _wipe_fixture_nodes(_gc)
        counts = _gc.query(
            """
            MATCH (n)
            WHERE (n:StandardName AND n.id STARTS WITH $name_prefix)
               OR (n:StandardNameSource AND n.id STARTS WITH $source_prefix)
            RETURN count(n) AS count
            """,
            name_prefix=_PREFIX,
            source_prefix=_SOURCE_PREFIX,
        )[0]
        assert counts["count"] == 0
    finally:
        _wipe_fixture_nodes(_gc)


@pytest.mark.graph
def test_no_reset_binds_run_id_and_preserves_stage(_gc, _clean):
    """Default scoping stamps run_id but leaves stage and score intact."""
    from imas_codex.standard_names.graph_ops import scope_focus_names

    src, name = f"dd:{_uid('s')}", _uid("nm")
    _seed(_gc, src, name, stage="reviewed", score=0.7)

    scope_focus_names([src], "scope-abc", reset=False)

    r = _read(_gc, name)
    assert r["stage"] == "reviewed"  # not reset
    assert r["score"] == 0.7  # score kept
    assert r["run_id"] == "scope-abc"  # bound to the scope


@pytest.mark.graph
def test_reset_restages_to_pending_and_clears_score(_gc, _clean):
    """The reset mode (opt-in) re-stages to pending and clears the score."""
    from imas_codex.standard_names.graph_ops import scope_focus_names

    src, name = f"dd:{_uid('s')}", _uid("nm")
    _seed(_gc, src, name, stage="reviewed", score=0.7)

    scope_focus_names([src], "scope-abc", reset=True)

    r = _read(_gc, name)
    assert r["stage"] == "pending"
    assert r["score"] is None
    assert r["run_id"] == "scope-abc"
