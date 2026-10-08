"""Exercise code reference statements on the live graph in a rolled-back transaction."""

from uuid import uuid4

import pytest

from imas_codex.graph import GraphClient
from imas_codex.ingestion.graph import (
    link_chunks_to_edas_signals,
    link_chunks_to_ids_roots,
    link_chunks_to_imas_paths,
)
from imas_codex.llm.search_tools import (
    _enrich_code_chunks,
    _reference_search_code_chunks,
)


class _TransactionGraph:
    def __init__(self, transaction):
        self.transaction = transaction

    def query(self, cypher, **params):
        return [dict(row) for row in self.transaction.run(cypher, **params)]


@pytest.mark.graph
def test_real_code_reference_links_and_missing_count_guard():
    suffix = uuid4().hex
    example = f"reference-check:{suffix}"
    known = f"known{suffix}"
    missing = f"missing{suffix}"
    category = f"TEST{suffix}"
    path = f"{category}/{known}"
    ids_id = f"ids:{suffix}"
    imas_id = f"equilibrium/test/{suffix}"
    signal_id = f"jt-60sa:general/{suffix}"
    chunk_id = f"{example}:chunk"
    missing_chunk_id = f"{example}:missing"

    with GraphClient() as client, client.session() as session:
        transaction = session.begin_transaction()
        graph = _TransactionGraph(transaction)
        try:
            assert (
                graph.query("MATCH (f:Facility {id: 'jt-60sa'}) RETURN count(f) AS n")[
                    0
                ]["n"]
                == 1
            )
            graph.query(
                """
                CREATE (:IDS {id: $ids_id}), (:IMASNode {id: $imas_id}),
                       (:FacilitySignal {id: $signal_id, facility_id: 'jt-60sa',
                         data_source_path: $path}),
                       (:CodeChunk {id: $chunk_id, facility_id: 'jt-60sa',
                         code_example_id: $example, source_file: '/test/edas.py',
                         text: $text, related_ids: [$ids_id],
                         imas_paths: [$imas_id, $unresolved_imas]}),
                       (:CodeChunk {id: $missing_chunk_id, facility_id: 'jt-60sa',
                         code_example_id: $example, source_file: '/test/edas.py',
                         text: $missing_text})
                """,
                ids_id=ids_id,
                imas_id=imas_id,
                signal_id=signal_id,
                path=path,
                chunk_id=chunk_id,
                missing_chunk_id=missing_chunk_id,
                example=example,
                text=f"db.eddbreadTime('E101173', '{category}', '{known}', t1, t2)",
                missing_text=f"db.eddbreadTime('E101173', '{category}', '{missing}', t1, t2)",
                unresolved_imas=f"equilibrium/absent/{suffix}",
            )

            assert link_chunks_to_imas_paths(graph, [example]) == 1
            assert link_chunks_to_ids_roots(graph, [example]) == 1
            assert graph.query(
                "MATCH (c:CodeChunk {id: $id})-[:REFERENCES_IMAS]->(p:IMASNode) "
                "RETURN collect(p.id) AS paths",
                id=chunk_id,
            )[0]["paths"] == [imas_id]
            assert graph.query(
                "MATCH (c:CodeChunk {id: $id})-[:REFERENCES_IDS]->(p:IDS) "
                "RETURN collect(p.id) AS roots",
                id=chunk_id,
            )[0]["roots"] == [ids_id]

            counts = link_chunks_to_edas_signals(graph, [example])
            assert counts == {"chunks": 2, "references": 2, "resolved": 1}
            assert graph.query(
                "MATCH (c:CodeChunk {id: $id})-[:CONTAINS_REF]->"
                "(d:DataReference)-[:RESOLVES_TO_FACILITY_SIGNAL]->(s:FacilitySignal) "
                "RETURN d.edas_category AS category, d.edas_data_name AS name, s.id AS signal",
                id=chunk_id,
            ) == [{"category": category, "name": known, "signal": signal_id}]
            assert (
                graph.query(
                    "MATCH (c:CodeChunk {id: $id})-[:CONTAINS_REF]->"
                    "(d:DataReference)-[:RESOLVES_TO_FACILITY_SIGNAL]->(s) "
                    "RETURN count(s) AS n",
                    id=missing_chunk_id,
                )[0]["n"]
                == 0
            )

            hits = _reference_search_code_chunks(
                graph, f"Find code reading {category} {known}", "jt-60sa", 10
            )
            assert [hit["id"] for hit in hits] == [chunk_id]
            assert (
                _reference_search_code_chunks(
                    graph, f"{category} absent", "jt-60sa", 10
                )
                == []
            )
            enriched = _enrich_code_chunks(graph, [chunk_id])
            assert signal_id in str(enriched[0]["data_refs"])

            graph.query(
                "MATCH (c:CodeChunk {id: $id}) SET c.related_ids = [$known, $missing]",
                id=chunk_id,
                known=ids_id,
                missing=f"ids:absent:{suffix}",
            )
            with pytest.raises(ValueError, match="1 of 2 named references"):
                link_chunks_to_ids_roots(graph, [example])
        finally:
            transaction.rollback()
