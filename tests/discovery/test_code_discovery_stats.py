"""Code progress queries and refresh error visibility."""

from unittest.mock import MagicMock, patch

from imas_codex.discovery.code.parallel import get_code_discovery_stats


def test_code_progress_queries_have_single_cypher_braces():
    queries = []
    graph = MagicMock()

    def capture(query, **_params):
        queries.append(query)
        return []

    graph.query.side_effect = capture
    context = MagicMock()
    context.__enter__.return_value = graph

    with patch("imas_codex.discovery.code.parallel.GraphClient", return_value=context):
        get_code_discovery_stats(
            "jt-60sa",
            min_relevance=0.5,
            min_ingest_relevance=0.5,
            min_facet_relevance=0.5,
        )

    assert len(queries) == 8
    for query in queries:
        assert "{{" not in query
        assert "}}" not in query
