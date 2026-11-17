"""Description retrieval distinguishes a quantity from a repeated identifier."""

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.llm.search_tools import _text_search_signals, _vector_search_signals

pytestmark = pytest.mark.graph


def test_name_only_descriptions_are_name_matches(disposable_graph):
    with GraphClient() as graph:
        graph.query(
            "UNWIND $names AS name CREATE (:FacilitySignal "
            "{id: name, facility_id: 'description-fixture', name: name, "
            "accessor: name, description: name, source_description: name})",
            names=["DEMO:TEMP_ALPHA", "DEMO:TEMP_BETA", "DEMO:TEMP_GAMMA"],
        )
        graph.ensure_fulltext_indexes()
        graph.query("CALL db.awaitIndexes(60)")
        for name in ["DEMO:TEMP_ALPHA", "DEMO:TEMP_BETA", "DEMO:TEMP_GAMMA"]:
            hits = _text_search_signals(graph, name, "description-fixture", 5)
            hit = next(row for row in hits if row["id"] == name)
            assert hit["match_fields"] == ["name"]
        graph.query("DROP INDEX facility_signal_text IF EXISTS")
        for name in ["DEMO:TEMP_ALPHA", "DEMO:TEMP_BETA", "DEMO:TEMP_GAMMA"]:
            hits = _text_search_signals(graph, name, "description-fixture", 5)
            hit = next(row for row in hits if row["id"] == name)
            assert hit["match_fields"] == ["name"]


def test_informative_description_is_searchable_with_other_facility(disposable_graph):
    with GraphClient() as graph:
        graph.query(
            "UNWIND $rows AS row CREATE (s:FacilitySignal) SET s = row",
            rows=[
                {
                    "id": "demo:temp",
                    "facility_id": "description-fixture",
                    "name": "DEMO:TEMP",
                    "accessor": "DEMO:TEMP",
                    "description": "Cryostat helium outlet temperature",
                },
                {
                    "id": "other:current",
                    "facility_id": "other-fixture",
                    "name": "current",
                    "accessor": "current",
                    "description": "Plasma current measurement",
                },
            ],
        )
        graph.ensure_fulltext_indexes()
        graph.query("CALL db.awaitIndexes(60)")
        hits = _text_search_signals(graph, "helium outlet", "description-fixture", 5)
        assert hits[0]["id"] == "demo:temp"
        assert hits[0]["match_fields"] == ["description"]
        other = _text_search_signals(graph, "plasma current", "other-fixture", 5)
        assert other[0]["id"] == "other:current"
        assert "description" in other[0]["match_fields"]


def test_vector_retrieval_refuses_name_only_descriptions(disposable_graph):
    with GraphClient() as graph:
        graph.query(
            "UNWIND $rows AS row CREATE (s:FacilitySignal) SET s = row",
            rows=[
                {
                    "id": "demo:opaque",
                    "facility_id": "description-fixture",
                    "name": "DEMO:TEMP",
                    "accessor": "DEMO:TEMP",
                    "description": "TEMP",
                    "embedding": [1.0] * 256,
                },
                {
                    "id": "demo:quantity",
                    "facility_id": "description-fixture",
                    "name": "DEMO:OUTLET",
                    "accessor": "DEMO:OUTLET",
                    "description": "Helium outlet temperature",
                    "embedding": [1.0] * 256,
                },
            ],
        )
        graph.ensure_vector_indexes()
        graph.query("CALL db.awaitIndexes(60)")
        ids, _ = _vector_search_signals(
            graph, [1.0] * 256, "description-fixture", 5, None, None
        )
        assert ids == ["demo:quantity"]
