from unittest.mock import MagicMock

from imas_codex.discovery.wiki.graph_ops import CONTENT_INGEST_THRESHOLD
from imas_codex.discovery.wiki.parallel import get_wiki_discovery_stats
from imas_codex.discovery.wiki.progress import DocsItem, WikiProgressDisplay


def test_pending_ingest_uses_content_cutoff(monkeypatch):
    client = MagicMock()

    def query(statement, **params):
        if "AS pending_ingest" in statement:
            assert "wp.score_composite >= $min_score" in statement
            assert params["min_score"] == CONTENT_INGEST_THRESHOLD
            return [{"pending_ingest": 1}]
        return []

    client.query.side_effect = query
    graph = MagicMock()
    graph.return_value.__enter__.return_value = client
    monkeypatch.setattr("imas_codex.discovery.wiki.parallel.GraphClient", graph)
    stats = get_wiki_discovery_stats("jet")
    assert stats["pending_ingest"] == 1


def test_document_label_uses_content_cutoff(monkeypatch):
    display = WikiProgressDisplay(facility="jet", cost_limit=1.0)
    monkeypatch.setattr(display, "_get_embed_indicator", lambda: None)
    display.state.current_docs = DocsItem("accepted.pdf", "pdf", score_composite=0.2)
    assert "accepted.pdf (skipped)" not in display._build_pipeline_section().plain

    display.state.current_docs = DocsItem("rejected.pdf", "pdf", score_composite=0.1)
    assert "rejected.pdf (skipped)" in display._build_pipeline_section().plain


def test_pending_document_ingest_uses_content_cutoff(monkeypatch):
    client = MagicMock()

    def query(statement, **params):
        if "WITH wa.status AS status" in statement:
            return [
                {
                    "status": "scored",
                    "score": score,
                    "atype": "pdf",
                    "exempt": False,
                    "cnt": 1,
                }
                for score in (0.2, 0.1)
            ]
        return []

    client.query.side_effect = query
    graph = MagicMock()
    graph.return_value.__enter__.return_value = client
    monkeypatch.setattr("imas_codex.discovery.wiki.parallel.GraphClient", graph)

    stats = get_wiki_discovery_stats("jet")
    assert stats["total_documents"] == 2
    assert stats["pending_document_ingest"] == 1
