"""A source returned to the pool retires the exhausted name it leaves behind.

Only an exhausted live name releases its source. Recycling its last producer
supersedes the name in the same statement and preserves a change receipt.
Superseded history releases sources without another retirement, while sources
of live names awaiting review or regeneration stay bound.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.graph

_PREFIX = "test_recycled_source__"


@pytest.fixture()
def _gc(disposable_neo4j_empty: str):
    from iter_codex.graph.client import GraphClient

    client = GraphClient(uri=disposable_neo4j_empty, username="neo4j", password="")
    yield client
    client.close()


def _plant(gc, name_id: str, stage: str, source_id: str) -> None:
    gc.query(
        """
        MERGE (sn:StandardName {id: $name})
        SET sn.name_stage = $stage, sn.origin = 'pipeline',
            sn.source_paths = [$source]
        MERGE (src:StandardNameSource {id: $source})
        SET src.source_type = 'dd', src.source_id = $source_id,
            src.status = 'composed', src.produced_sn_id = $name
        MERGE (src)-[:PRODUCED_NAME]->(sn)
        """,
        name=name_id,
        stage=stage,
        source=source_id,
        source_id=source_id.removeprefix("dd:"),
    )


def _source_status(gc, source_id: str) -> str | None:
    rows = gc.query(
        "MATCH (s:StandardNameSource {id: $id}) RETURN s.status AS status",
        id=source_id,
    )
    return rows[0]["status"] if rows else None


def _name_stage(gc, name_id: str) -> str | None:
    rows = gc.query(
        "MATCH (sn:StandardName {id: $id}) RETURN sn.name_stage AS stage",
        id=name_id,
    )
    return rows[0]["stage"] if rows else None


def _sourceless(gc, name_id: str) -> bool:
    from iter_codex.standard_names.source_invariant import (
        find_sourceless_live_name_rows,
    )

    return name_id in {
        row["id"]
        for row in find_sourceless_live_name_rows(
            gc, [name_id], derived_parents_exempt=False
        )
    }


def test_exhausted_name_is_retired_when_its_source_is_recycled(_gc):
    from iter_codex.standard_names.graph_store.graph_audits import (
        reconcile_source_status_liveness,
    )

    name = _PREFIX + "exhausted"
    source = _PREFIX + "exhausted_source"
    _plant(_gc, name, "exhausted", source)

    result = reconcile_source_status_liveness(gc=_gc)

    assert _source_status(_gc, source) == "extracted"
    assert result["orphaned_reset"] == 1
    assert _name_stage(_gc, name) == "superseded"
    assert result["names_retired"] == 1
    assert not _sourceless(_gc, name)
    changes = _gc.query(
        """
        MATCH (sn:StandardName {id: $id})-[:HAS_INTERNAL_CHANGE]->(c)
        RETURN c.operation AS operation, c.origin AS origin
        """,
        id=name,
    )
    assert changes == [
        {"operation": "retire_recycled_source_name", "origin": "source_recycling"}
    ]
    assert reconcile_source_status_liveness(gc=_gc)["names_retired"] == 0
    assert _gc.query(
        "MATCH (:StandardName {id: $id})-[:HAS_INTERNAL_CHANGE]->(c) "
        "RETURN count(c) AS n",
        id=name,
    ) == [{"n": 1}]


@pytest.mark.parametrize(
    "stage", ["contested", "accepted", "drafted", "pending", "refining"]
)
def test_source_of_a_non_exhausted_live_name_is_not_taken(_gc, stage):
    from iter_codex.standard_names.graph_store.graph_audits import (
        reconcile_source_status_liveness,
    )

    name = _PREFIX + stage
    source = _PREFIX + stage + "_source"
    _plant(_gc, name, stage, source)

    result = reconcile_source_status_liveness(gc=_gc)

    assert _source_status(_gc, source) == "composed"
    assert result["orphaned_reset"] == 0
    assert result["names_retired"] == 0
    assert _name_stage(_gc, name) == stage
    edges = _gc.query(
        "MATCH (:StandardNameSource {id: $s})-[:PRODUCED_NAME]->(:StandardName) "
        "RETURN count(*) AS n",
        s=source,
    )
    assert edges[0]["n"] == 1


def test_source_of_a_claimed_refining_name_is_not_taken(_gc):
    from iter_codex.standard_names.graph_store.graph_audits import (
        reconcile_source_status_liveness,
    )

    name = _PREFIX + "claimed_refining"
    source = name + "_source"
    _plant(_gc, name, "refining", source)
    _gc.query(
        "MATCH (n:StandardName {id:$id}) "
        "SET n.claimed_at=datetime(), n.claim_token='refine-owner'",
        id=name,
    )
    before = _gc.query(
        "MATCH (n:StandardName {id:$id}) RETURN properties(n) AS properties", id=name
    )
    assert reconcile_source_status_liveness(gc=_gc)["names_retired"] == 0
    assert (
        _gc.query(
            "MATCH (n:StandardName {id:$id}) RETURN properties(n) AS properties",
            id=name,
        )
        == before
    )
    assert _gc.query(
        "MATCH (s:StandardNameSource {id:$id})-[:PRODUCED_NAME]->(n) "
        "RETURN s.status AS status, n.id AS name",
        id=source,
    ) == [{"status": "composed", "name": name}]


def test_superseded_history_still_releases_its_source(_gc):
    from iter_codex.standard_names.graph_store.graph_audits import (
        reconcile_source_status_liveness,
    )

    name = _PREFIX + "superseded"
    source = _PREFIX + "superseded_source"
    _plant(_gc, name, "superseded", source)

    result = reconcile_source_status_liveness(gc=_gc)

    assert _source_status(_gc, source) == "extracted"
    assert result["orphaned_reset"] == 1
    assert result["names_retired"] == 0
    assert _name_stage(_gc, name) == "superseded"


def test_name_that_keeps_another_producer_is_not_retired(_gc):
    from iter_codex.standard_names.graph_store.graph_audits import (
        reconcile_source_status_liveness,
    )

    name = _PREFIX + "shared"
    released = _PREFIX + "shared_released"
    kept = _PREFIX + "shared_kept"
    live = _PREFIX + "shared_live_target"
    _plant(_gc, name, "exhausted", released)
    _gc.query(
        """
        MATCH (sn:StandardName {id: $name})
        MERGE (live:StandardName {id: $live})
        SET live.name_stage = 'pending', live.origin = 'pipeline'
        MERGE (src:StandardNameSource {id: $kept})
        SET src.source_type = 'dd', src.source_id = $kept,
            src.status = 'attached', src.produced_sn_id = $name
        MERGE (src)-[:PRODUCED_NAME]->(sn)
        MERGE (src)-[:PRODUCED_NAME]->(live)
        """,
        name=name,
        live=live,
        kept=kept,
    )

    result = reconcile_source_status_liveness(gc=_gc)

    assert _source_status(_gc, released) == "extracted"
    assert _source_status(_gc, kept) == "attached"
    assert result["names_retired"] == 0
    assert _name_stage(_gc, name) == "exhausted"


def test_recycling_all_producers_retires_name_once(_gc):
    from iter_codex.standard_names.graph_store.graph_audits import (
        reconcile_source_status_liveness,
    )

    name = _PREFIX + "multiple_sources"
    sources = [name + "_left", name + "_right"]
    for source in sources:
        _plant(_gc, name, "exhausted", source)

    result = reconcile_source_status_liveness(gc=_gc)

    assert result["orphaned_reset"] == 2
    assert result["names_retired"] == 1
    assert _name_stage(_gc, name) == "superseded"
    assert all(_source_status(_gc, source) == "extracted" for source in sources)
    assert _gc.query(
        "MATCH (:StandardName {id: $id})-[:HAS_INTERNAL_CHANGE]->(c) "
        "RETURN count(c) AS n",
        id=name,
    ) == [{"n": 1}]


def test_stale_producer_does_not_keep_a_recycled_name_live(_gc):
    from iter_codex.standard_names.graph_store.graph_audits import (
        reconcile_source_status_liveness,
    )

    name = _PREFIX + "stale_producer"
    released = name + "_released"
    stale = name + "_stale"
    _plant(_gc, name, "exhausted", released)
    _plant(_gc, name, "exhausted", stale)
    _gc.query(
        "MATCH (s:StandardNameSource {id: $id}) SET s.status = 'stale'",
        id=stale,
    )

    result = reconcile_source_status_liveness(gc=_gc)

    assert result["names_retired"] == 1
    assert _name_stage(_gc, name) == "superseded"
    assert _source_status(_gc, stale) == "stale"
    assert not _sourceless(_gc, name)


@pytest.mark.parametrize("protection", ["derived", "catalog"])
def test_source_stays_bound_when_exhausted_name_cannot_be_retired(_gc, protection):
    from iter_codex.standard_names.graph_store.graph_audits import (
        reconcile_source_status_liveness,
    )

    name = _PREFIX + protection
    source = _PREFIX + protection + "_source"
    _plant(_gc, name, "exhausted", source)
    _gc.query(
        "MATCH (sn:StandardName {id: $id}) "
        "SET sn.origin = $origin, sn.catalog_pr_number = $catalog",
        id=name,
        origin="derived" if protection == "derived" else "pipeline",
        catalog=1 if protection == "catalog" else None,
    )

    result = reconcile_source_status_liveness(gc=_gc)

    assert _source_status(_gc, source) == "composed"
    assert result["orphaned_reset"] == 0
    assert result["names_retired"] == 0
    assert _name_stage(_gc, name) == "exhausted"
    assert not _sourceless(_gc, name)
