"""Document lexical search uses the fields stored by graph ingestion."""

from uuid import uuid4

import pytest

from imas_codex.graph.client import GraphClient
from imas_codex.ids.slot_context import _text_search_documents


class _TransactionGraph:
    def __init__(self, transaction):
        self.transaction = transaction

    def query(self, cypher, **params):
        return [dict(row) for row in self.transaction.run(cypher, **params)]


@pytest.mark.graph
def test_document_search_finds_declared_text_and_chunk_content():
    marker = uuid4().hex
    facility = f"slot-doc-search-{marker}"
    fields = ("filename", "preview_text", "document_purpose", "path", "chunk")

    with GraphClient() as client, client.session() as session:
        transaction = session.begin_transaction()
        graph = _TransactionGraph(transaction)
        try:
            for field in fields:
                term = f"needle{marker}{field}"
                document_id = f"{facility}:{field}"
                values = {
                    "filename": "ordinary.pdf",
                    "preview_text": None,
                    "document_purpose": None,
                    "path": None,
                }
                if field != "chunk":
                    values[field] = term
                graph.query(
                    "CREATE (d:Document {id: $id, facility_id: $facility, "
                    "filename: $filename, preview_text: $preview_text, "
                    "document_purpose: $document_purpose, path: $path, "
                    "url: $url})",
                    id=document_id,
                    facility=facility,
                    url=f"https://example.org/{field}.pdf",
                    **values,
                )
                if field == "chunk":
                    graph.query(
                        "MATCH (d:Document {id: $id}) "
                        "CREATE (d)-[:HAS_CHUNK]->(:WikiChunk {id: $chunk_id, text: $text})",
                        id=document_id,
                        chunk_id=f"{document_id}:chunk",
                        text=term,
                    )

                hits = _text_search_documents(graph, term, facility, 10)
                assert [hit["id"] for hit in hits] == [document_id]
                assert hits[0]["title"] == values["filename"]
                if field != "filename":
                    assert term in hits[0]["description"]
        finally:
            transaction.rollback()
