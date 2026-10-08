"""Search and rank evidence for one source-to-target transform decision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from imas_codex.embeddings.encoder import Encoder
from imas_codex.graph.client import GraphClient
from imas_codex.llm.search_tools import (
    _embed,
    _enrich_code_chunks,
    _enrich_signals,
    _enrich_wiki_chunks,
    _text_search_code_chunks,
    _text_search_signals,
    _text_search_wiki_chunks,
    _vector_search_code_chunks,
    _vector_search_documents,
    _vector_search_signals,
    _vector_search_wiki_chunks,
    rerank_candidates,
)


def _text_search_documents(
    gc: GraphClient, query: str, facility: str, limit: int
) -> list[dict[str, Any]]:
    """Find facility documents by declared metadata or attached chunk text."""
    return gc.query(
        """
        MATCH (d:Document)
        WHERE d.facility_id = $facility
          AND (toLower(d.filename) CONTAINS $term
               OR toLower(d.preview_text) CONTAINS $term
               OR toLower(d.document_purpose) CONTAINS $term
               OR toLower(d.path) CONTAINS $term
               OR EXISTS {
                   MATCH (d)-[:HAS_CHUNK]->(chunk:WikiChunk)
                   WHERE toLower(chunk.text) CONTAINS $term
               })
        OPTIONAL MATCH (d)-[:HAS_CHUNK]->(chunk:WikiChunk)
        WHERE toLower(chunk.text) CONTAINS $term
        WITH d, head(collect(chunk.text)) AS matching_chunk
        OPTIONAL MATCH (p:WikiPage)-[:HAS_DOCUMENT]->(d)
        RETURN d.id AS id, coalesce(d.filename, d.path, d.id) AS title,
               coalesce(matching_chunk, d.preview_text, d.document_purpose,
                        d.path, '') AS description,
               d.url AS url, head(collect(p.title)) AS page_title, 0.5 AS score
        LIMIT $limit
        """,
        facility=facility,
        term=query.lower(),
        limit=limit,
    )


def _hybrid_ids(
    ids: list[str],
    scores: dict[str, float],
    text_hits: list[dict[str, Any]],
    limit: int,
) -> tuple[list[str], dict[str, float]]:
    """Blend lexical and vector scores using the search tools' hybrid rule."""
    scores = dict(scores)
    for hit in text_hits:
        item_id = hit["id"]
        text_score = round(hit["score"], 3)
        if item_id in scores:
            scores[item_id] = round(scores[item_id] * 0.7 + text_score * 0.3 + 0.1, 3)
        else:
            scores[item_id] = text_score
            ids.append(item_id)
    return sorted(set(ids), key=lambda item_id: scores[item_id], reverse=True)[
        :limit
    ], scores


def _item(kind: str, row: Mapping[str, Any], score: float) -> dict[str, Any]:
    """Keep the text, kind and source locator together for the decision state."""
    origin_fields = {
        "code": ("source_file",),
        "wiki": ("page_url", "page_id"),
        "document": ("url", "id"),
        "signal": ("node_path", "id"),
    }
    origin = next(
        (str(row[field]) for field in origin_fields[kind] if row.get(field)),
        str(row["id"]),
    )
    text = row.get("text") or row.get("description") or ""
    return {
        "id": row["id"],
        "kind": kind,
        "origin": origin,
        "title": row.get("function_name")
        or row.get("page_title")
        or row.get("title")
        or row.get("name")
        or "",
        "text": str(text),
        "score": score,
        "path": origin,
    }


def build_slot_context(
    facility: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    slot: str,
    *,
    gc: GraphClient | None = None,
    encoder: Encoder | None = None,
    limit: int = 8,
    candidates_per_kind: int = 12,
) -> list[dict[str, Any]]:
    """Return the most relevant evidence for one binding slot.

    The search query names the source and destination. Jev sees the slot as
    well, so a sign decision can prefer direction evidence over a general
    mention of the same signal. Empty source kinds add no placeholder.
    """
    if limit <= 0 or candidates_per_kind <= 0:
        return []
    if gc is None:
        gc = GraphClient()
    if encoder is None:
        encoder = Encoder()

    retrieval_query = " ".join(
        str(value)
        for value in (source.get("name"), source.get("description"), source.get("id"))
        if value
    )
    decision_query = " ".join(
        str(value)
        for value in (
            retrieval_query,
            target.get("id"),
            target.get("documentation"),
            f"{slot} convention",
        )
        if value
    )
    embedding = _embed(encoder, retrieval_query)
    pool: list[dict[str, Any]] = []

    code_ids, code_scores = _vector_search_code_chunks(
        gc, embedding, facility, candidates_per_kind
    )
    code_ids, code_scores = _hybrid_ids(
        code_ids,
        code_scores,
        _text_search_code_chunks(gc, retrieval_query, facility, candidates_per_kind),
        candidates_per_kind,
    )
    if code_ids:
        code_rows = {row["id"]: row for row in _enrich_code_chunks(gc, code_ids)}
        pool.extend(
            _item("code", code_rows[item_id], code_scores[item_id])
            for item_id in code_ids
            if item_id in code_rows
        )

    wiki_ids, wiki_scores = _vector_search_wiki_chunks(
        gc, embedding, facility, candidates_per_kind
    )
    wiki_ids, wiki_scores = _hybrid_ids(
        wiki_ids,
        wiki_scores,
        _text_search_wiki_chunks(gc, retrieval_query, facility, candidates_per_kind),
        candidates_per_kind,
    )
    if wiki_ids:
        wiki_rows = {row["id"]: row for row in _enrich_wiki_chunks(gc, wiki_ids)}
        pool.extend(
            _item("wiki", wiki_rows[item_id], wiki_scores[item_id])
            for item_id in wiki_ids
            if item_id in wiki_rows
        )

    document_rows, document_scores = _vector_search_documents(
        gc, embedding, facility, candidates_per_kind
    )
    document_rows = [row for row in document_rows if "url" in row]
    document_term = str(
        source.get("name") or source.get("description") or source.get("id") or ""
    )
    text_documents = _text_search_documents(
        gc, document_term, facility, candidates_per_kind
    )
    document_ids, document_scores = _hybrid_ids(
        [row["id"] for row in document_rows],
        document_scores,
        text_documents,
        candidates_per_kind,
    )
    documents = {row["id"]: row for row in [*document_rows, *text_documents]}
    pool.extend(
        _item("document", documents[item_id], document_scores[item_id])
        for item_id in document_ids
        if item_id in documents
    )

    signal_ids, signal_scores = _vector_search_signals(
        gc, embedding, facility, candidates_per_kind, None, None
    )
    signal_ids, signal_scores = _hybrid_ids(
        signal_ids,
        signal_scores,
        _text_search_signals(gc, retrieval_query, facility, candidates_per_kind),
        candidates_per_kind,
    )
    if signal_ids:
        signals = {row["id"]: row for row in _enrich_signals(gc, signal_ids)}
        pool.extend(
            _item("signal", signals[item_id], signal_scores[item_id])
            for item_id in signal_ids
            if item_id in signals
        )

    pool.sort(key=lambda item: item["score"], reverse=True)
    ranked, _note = rerank_candidates(decision_query, pool)
    return ranked[:limit]
