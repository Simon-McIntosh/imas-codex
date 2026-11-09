"""A rich-write refusal cannot strand its claim's pending name."""

from unittest.mock import MagicMock

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.standard_names.graph_store.graph_catalog import (
    persist_generated_name_winners,
)
from tests.standard_names.test_persist_refusal_per_source import (
    REFUSED,
    REFUSED_PATH,
    SEQ,
    TOKEN,
    _candidate,
    _services,
)

pytestmark = pytest.mark.graph


@pytest.fixture
def graph(disposable_neo4j_empty):
    with GraphClient(
        uri=disposable_neo4j_empty,
        username="neo4j",
        password="",
        graph_name="persist-refusal-tests",
    ) as client:
        client.query(
            """
            CREATE (source:StandardNameSource {
                id: $source_id, source_type: 'dd', source_id: $path,
                status: 'extracted', claim_token: $token, claim_seq: $seq,
                attempt_count: 2
            })
            """,
            source_id=f"dd:{REFUSED_PATH}",
            path=REFUSED_PATH,
            token=TOKEN,
            seq=SEQ,
        )
        yield client


def _refuse(graph, *, other_producer=False, cost=None, approved=False):
    services = _services(graph, [])
    # Each graph context owns its connection; every write still uses the
    # transaction handle supplied by the persister.
    services.GraphClient = GraphClient

    def writer(candidates, *, gc, return_written_ids, refusal_reasons):
        assert return_written_ids
        assert gc.query(
            """MATCH (:StandardNameSource)-[r:PRODUCED_NAME]->(sn:StandardName)
            WHERE sn.id = $name AND r.provisional = true
            RETURN sn.name_stage AS stage""",
            name=REFUSED,
        ) == [{"stage": "pending"}]
        if other_producer:
            gc.query(
                """MATCH (sn:StandardName {id: $name})
                CREATE (:StandardNameSource {id: 'dd:other', status: 'composed'})
                    -[:PRODUCED_NAME]->(sn)""",
                name=REFUSED,
            )
        if cost is not None:
            gc.query(
                """CREATE (:LLMCost {id: 'test-cost', llm_cost: $cost,
                                     standard_name_ids: [$name]})""",
                name=REFUSED,
                cost=cost,
            )
        if approved:
            gc.query(
                """MATCH (sn:StandardName {id: $name})
                SET sn.catalog_commit_sha = 'catalog-proof'""",
                name=REFUSED,
            )
        refusal_reasons[REFUSED] = "grammar parse failed: refused test candidate"
        return []

    services.write_standard_names = MagicMock(side_effect=writer)
    result = persist_generated_name_winners(
        [_candidate(REFUSED, REFUSED_PATH)], compose_model="test", services=services
    )
    assert result == []
    source = graph.query(
        """MATCH (source:StandardNameSource {id: $id})
        RETURN source.status AS status, source.claim_token AS token,
               source.last_error AS error""",
        id=f"dd:{REFUSED_PATH}",
    )[0]
    assert source["status"] == "failed"
    assert source["token"] is None
    assert "grammar parse failed: refused test candidate" in source["error"]


def _pending_names(graph):
    return graph.query(
        """MATCH (sn:StandardName {id: $name})
        RETURN sn.name_stage AS stage,
               [(source:StandardNameSource)-[:PRODUCED_NAME]->(sn) |
                 source.id] AS producers""",
        name=REFUSED,
    )


@pytest.mark.parametrize("cost", [None, 0.0])
def test_refused_winner_leaves_no_pending_orphan(graph, cost):
    _refuse(graph, cost=cost)

    assert _pending_names(graph) == [], "refusal left a pending name without a producer"
    receipts = graph.query(
        """MATCH (change:StandardNameChange {to_name: $name})
        RETURN change.operation AS operation""",
        name=REFUSED,
    )
    assert receipts == [{"operation": "remove_provenance_orphan"}]


def test_refused_winner_keeps_name_produced_elsewhere(graph):
    _refuse(graph, other_producer=True)

    assert _pending_names(graph) == [{"stage": "pending", "producers": ["dd:other"]}]


@pytest.mark.parametrize("protection", [{"cost": 0.25}, {"approved": True}])
def test_refused_winner_keeps_producer_for_protected_name(graph, protection):
    _refuse(graph, **protection)

    assert _pending_names(graph) == [
        {"stage": "pending", "producers": [f"dd:{REFUSED_PATH}"]}
    ]


def test_placeholder_retirement_rolls_back_with_batch(graph, monkeypatch):
    from imas_codex.standard_names import provenance_lifecycle

    retire = provenance_lifecycle.retire_unrecoverable_provenance_orphans

    def abort_after_retirement(gc, ids):
        assert retire(gc, ids) == [REFUSED]
        raise RuntimeError("retirement transaction aborted")

    monkeypatch.setattr(
        provenance_lifecycle,
        "retire_unrecoverable_provenance_orphans",
        abort_after_retirement,
    )
    with pytest.raises(RuntimeError, match="retirement transaction aborted"):
        _refuse(graph)

    assert _pending_names(graph) == []
    assert graph.query(
        """MATCH (source:StandardNameSource {id: $id})
        RETURN source.claim_token AS token, source.status AS status""",
        id=f"dd:{REFUSED_PATH}",
    ) == [{"token": TOKEN, "status": "extracted"}]
    assert graph.query("MATCH (c:StandardNameChange) RETURN count(c) AS count") == [
        {"count": 0}
    ]


@pytest.mark.parametrize(
    "name",
    [
        "first_local_tangential_width_of_fiber_bundle",
        "second_local_tangential_width_of_fiber_bundle",
    ],
)
def test_grammar_refusal_records_writer_reason_without_leaving_placeholder(graph, name):
    from imas_codex.standard_names.graph_store import graph_catalog, graph_foundation

    services = _services(graph, [])
    services.GraphClient = GraphClient
    services.write_standard_names = lambda *args, **kwargs: (
        graph_catalog.write_standard_names(
            *args, **kwargs, services=graph_foundation.get_services()
        )
    )
    candidate = {**_candidate(name, REFUSED_PATH), "unit": "m"}

    assert (
        persist_generated_name_winners(
            [candidate], compose_model="test", services=services
        )
        == []
    )
    assert (
        graph.query("MATCH (sn:StandardName {id: $name}) RETURN sn.id AS id", name=name)
        == []
    )
    source = graph.query(
        """MATCH (source:StandardNameSource {id: $id})
        RETURN source.last_error AS reason, source.claim_token AS token""",
        id=f"dd:{REFUSED_PATH}",
    )[0]
    assert source["token"] is None
    assert f"rich standard-name write refused candidate {name!r}" in source["reason"]
    assert "grammar parse failed" in source["reason"]
    assert "does not match any physical_base or geometry_carrier" in source["reason"]
