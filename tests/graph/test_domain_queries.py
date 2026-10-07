"""Tests for domain query functions.

These tests validate the domain query functions (find_signals, find_wiki,
find_dd_paths, find_code, find_data_nodes, map_signals_to_imas, facility_overview)
without requiring Neo4j — they test structure and error handling using mocks.
"""

from unittest.mock import MagicMock, patch

import pytest

from imas_codex.graph.domain_queries import (
    facility_overview,
    find_code,
    find_data_nodes,
    find_dd_paths,
    find_signals,
    find_wiki,
    map_signals_to_imas,
    wiki_page_chunks,
)


@pytest.fixture
def mock_gc():
    """Mock GraphClient for testing without Neo4j."""
    gc = MagicMock()
    gc.query.return_value = []
    return gc


@pytest.fixture
def mock_embed():
    """Mock embed function."""
    return MagicMock(return_value=[0.1] * 256)


class TestFindSignals:
    """Test find_signals domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_signals(facility="tcv", gc=mock_gc, embed_fn=mock_embed)
        assert isinstance(result, list)

    def test_calls_query_with_facility(self, mock_gc, mock_embed):
        find_signals(facility="tcv", gc=mock_gc, embed_fn=mock_embed)
        assert mock_gc.query.called
        # Should have facility parameter
        call_kwargs = mock_gc.query.call_args
        assert "tcv" in str(call_kwargs)

    def test_semantic_search(self, mock_gc, mock_embed):
        mock_gc.query.return_value = [
            {
                "id": "tcv:ip",
                "name": "ip",
                "description": "Plasma current",
                "score": 0.9,
            }
        ]
        result = find_signals(
            query="plasma current",
            facility="tcv",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        assert mock_embed.called
        assert len(result) == 1
        assert result[0]["id"] == "tcv:ip"

    def test_facility_predicate_inside_search(self, mock_gc, mock_embed):
        """Facility predicate is an in-index pre-filter, inside SEARCH."""
        find_signals(
            query="plasma current",
            facility="jt-60sa",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "facility_signal_desc_embedding" in cypher
        open_idx = cypher.index("SEARCH")
        close_idx = cypher.index(") SCORE AS")
        pred_idx = cypher.index("s.facility_id = $facility")
        assert open_idx < pred_idx < close_idx, cypher
        assert "AT_FACILITY" not in cypher

    def test_requires_facility(self, mock_gc, mock_embed):
        with pytest.raises(ValueError, match="facility"):
            find_signals(gc=mock_gc, embed_fn=mock_embed)

    def test_include_access_true(self, mock_gc, mock_embed):
        find_signals(
            facility="tcv", include_access=True, gc=mock_gc, embed_fn=mock_embed
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "DataAccess" in cypher

    def test_include_access_false(self, mock_gc, mock_embed):
        find_signals(
            facility="tcv",
            include_access=False,
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "DataAccess" not in cypher


class TestFindWiki:
    """Test find_wiki domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_wiki(query="equilibrium", gc=mock_gc, embed_fn=mock_embed)
        assert isinstance(result, list)

    def test_requires_query_or_keyword(self, mock_gc, mock_embed):
        with pytest.raises(ValueError, match="query.*keyword"):
            find_wiki(gc=mock_gc, embed_fn=mock_embed)

    def test_calls_embedding(self, mock_gc, mock_embed):
        find_wiki(query="plasma", gc=mock_gc, embed_fn=mock_embed)
        mock_embed.assert_called_once_with("plasma")

    def test_includes_page_context(self, mock_gc, mock_embed):
        mock_gc.query.return_value = [
            {
                "text": "content",
                "page_title": "Test Page",
                "page_url": "http://...",
                "score": 0.8,
            }
        ]
        find_wiki(query="test", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "WikiPage" in cypher

    def test_facility_predicate_inside_search(self, mock_gc, mock_embed):
        """Wiki facility predicate is an in-index pre-filter, inside SEARCH."""
        find_wiki(
            query="equilibrium", facility="jt-60sa", gc=mock_gc, embed_fn=mock_embed
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "wiki_chunk_embedding" in cypher
        open_idx = cypher.index("SEARCH")
        close_idx = cypher.index(") SCORE AS")
        pred_idx = cypher.index("c.facility_id = $facility")
        assert open_idx < pred_idx < close_idx, cypher
        assert "AT_FACILITY" not in cypher

    def test_text_contains_keyword_only(self, mock_gc, mock_embed):
        """Keyword-only search without semantic query."""
        find_wiki(text_contains="fishbone", gc=mock_gc, embed_fn=mock_embed)
        mock_embed.assert_not_called()
        cypher = mock_gc.query.call_args[0][0]
        assert "CONTAINS" in cypher

    def test_page_title_contains(self, mock_gc, mock_embed):
        """Filter by page title substring."""
        find_wiki(page_title_contains="fishbone", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "title" in cypher.lower()
        assert "CONTAINS" in cypher

    def test_semantic_with_text_filter(self, mock_gc, mock_embed):
        """Combined semantic + keyword filtering."""
        find_wiki(
            query="instabilities",
            text_contains="fishbone",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        mock_embed.assert_called_once()
        cypher = mock_gc.query.call_args[0][0]
        assert "SEARCH" in cypher
        assert "CONTAINS" in cypher

    def test_semantic_with_title_filter(self, mock_gc, mock_embed):
        """Semantic search filtered by page title."""
        find_wiki(
            query="kink mode",
            page_title_contains="fishbone",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "SEARCH" in cypher
        assert "title" in cypher.lower()


class TestWikiPageChunks:
    """Test wiki_page_chunks helper."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = wiki_page_chunks("fishbone", gc=mock_gc, embed_fn=mock_embed)
        assert isinstance(result, list)

    def test_filters_by_title(self, mock_gc, mock_embed):
        wiki_page_chunks("fishbone", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "title" in cypher.lower()
        assert "CONTAINS" in cypher

    def test_with_facility(self, mock_gc, mock_embed):
        wiki_page_chunks("fishbone", facility="jet", gc=mock_gc, embed_fn=mock_embed)
        call_kwargs = mock_gc.query.call_args
        assert "jet" in str(call_kwargs)

    def test_with_text_contains(self, mock_gc, mock_embed):
        wiki_page_chunks(
            "fishbone",
            text_contains="team",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        # Should have two CONTAINS conditions
        assert cypher.count("CONTAINS") >= 2

    def test_returns_page_context(self, mock_gc, mock_embed):
        mock_gc.query.return_value = [
            {
                "page_title": "Controlling fishbones",
                "page_url": "http://...",
                "facility": "jet",
                "section": "Team",
                "text": "content",
            }
        ]
        result = wiki_page_chunks("fishbone", gc=mock_gc, embed_fn=mock_embed)
        assert result[0]["page_title"] == "Controlling fishbones"


class TestFindImas:
    """Test find_dd_paths domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_dd_paths(
            query="electron temperature", gc=mock_gc, embed_fn=mock_embed
        )
        assert isinstance(result, list)

    def test_semantic_search(self, mock_gc, mock_embed):
        find_dd_paths(query="electron temperature", gc=mock_gc, embed_fn=mock_embed)
        mock_embed.assert_called_once()
        assert mock_gc.query.called

    def test_filters_deprecated(self, mock_gc, mock_embed):
        find_dd_paths(query="test", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "DEPRECATED_IN" in cypher


class TestFindCode:
    """Test find_code domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_code(
            query="equilibrium reconstruction", gc=mock_gc, embed_fn=mock_embed
        )
        assert isinstance(result, list)

    def test_with_facility_filter(self, mock_gc, mock_embed):
        find_code(query="test", facility="tcv", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "facility_id" in cypher

    def test_facility_predicate_inside_search(self, mock_gc, mock_embed):
        """Code facility predicate is an in-index pre-filter, inside SEARCH."""
        find_code(
            query="equilibrium reconstruction",
            facility="jt-60sa",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "code_chunk_embedding" in cypher
        open_idx = cypher.index("SEARCH")
        close_idx = cypher.index(") SCORE AS")
        pred_idx = cypher.index("cc.facility_id = $facility")
        assert open_idx < pred_idx < close_idx, cypher
        assert "CodeFile" not in cypher

    def test_renders_source_through_has_chunk_edge(self, mock_gc, mock_embed):
        """find_code resolves source_file via the writer's HAS_CHUNK edge."""
        find_code(query="test", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert (
            "OPTIONAL MATCH (ce:CodeExample)-[:HAS_CHUNK]->(cc)\n"
            "WITH *, coalesce(ce.source_file, cc.source_file) AS source_file"
        ) in cypher

    def test_retired_edge_names_absent(self, mock_gc, mock_embed):
        """The dead reversed and undeclared edges are gone."""
        find_code(query="test", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "CODE_EXAMPLE_ID" not in cypher
        assert "CodeFile" not in cypher

    def test_pools_beyond_the_limit(self, mock_gc, mock_embed):
        """The embedding retrieves a pool larger than the limit."""
        from imas_codex.llm.search_tools import RERANK_POOL

        find_code(query="test", facility="tcv", limit=5, gc=mock_gc, embed_fn=mock_embed)

        assert mock_gc.query.call_args[1]["k"] == RERANK_POOL

    def test_reranks_the_returned_rows_to_the_limit(
        self, mock_gc, mock_embed, monkeypatch
    ):
        """Rows come back in rerank order, cut to the limit."""
        from imas_codex.discovery.base import judgment

        mock_gc.query.return_value = [
            {"text": "a", "function_name": "fa", "source_file": "a.py", "score": 0.9},
            {"text": "b", "function_name": "fb", "source_file": "b.py", "score": 0.8},
            {"text": "c", "function_name": "fc", "source_file": "c.py", "score": 0.7},
        ]

        async def fake(model, state, questions, *, service=None, **_kwargs):
            score = {"a.py": 1, "b.py": 5, "c.py": 3}[
                state["candidate"]["locator"]["path"]
            ]
            return {"relevance_grade": {"type": "score", "score": score}}, 0.0

        monkeypatch.setattr(judgment, "acall_decisions", fake)

        result = find_code(
            query="q", facility="tcv", limit=2, gc=mock_gc, embed_fn=mock_embed
        )

        assert [row["source_file"] for row in result] == ["b.py", "c.py"]


class TestFindDataNodes:
    """Test find_data_nodes domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = find_data_nodes(facility="tcv", gc=mock_gc, embed_fn=mock_embed)
        assert isinstance(result, list)

    def test_with_tree_filter(self, mock_gc, mock_embed):
        find_data_nodes(
            facility="tcv", data_source_name="results", gc=mock_gc, embed_fn=mock_embed
        )
        call_kwargs = mock_gc.query.call_args
        assert "results" in str(call_kwargs)

    def test_with_semantic_search(self, mock_gc, mock_embed):
        find_data_nodes(
            query="electron density",
            facility="tcv",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        # Should use vector search when query is provided
        cypher = mock_gc.query.call_args[0][0]
        # Should have vector search or WHERE clause
        assert "embedding" in cypher.lower() or "description" in cypher.lower()


class TestMapSignalsToImas:
    """Test map_signals_to_imas domain query."""

    def test_returns_list(self, mock_gc, mock_embed):
        result = map_signals_to_imas(facility="tcv", gc=mock_gc)
        assert isinstance(result, list)

    def test_includes_imas_paths(self, mock_gc, mock_embed):
        map_signals_to_imas(facility="tcv", gc=mock_gc)
        cypher = mock_gc.query.call_args[0][0]
        assert "MAPS_TO_IMAS" in cypher
        assert "MEMBER_OF" in cypher
        assert "IMASNode" in cypher


class TestFacilityOverview:
    """Test facility_overview domain query."""

    def test_returns_dict(self, mock_gc, mock_embed):
        mock_gc.query.return_value = [
            {
                "diagnostics": 5,
                "trees": 3,
                "signals": 100,
                "wiki_pages": 20,
                "code_files": 50,
            }
        ]
        result = facility_overview(facility="tcv", gc=mock_gc)
        assert isinstance(result, dict)

    def test_has_facility_key(self, mock_gc, mock_embed):
        mock_gc.query.return_value = [
            {
                "diagnostics": 0,
                "trees": 0,
                "signals": 0,
                "wiki_pages": 0,
                "code_files": 0,
            }
        ]
        result = facility_overview(facility="tcv", gc=mock_gc)
        assert result["facility"] == "tcv"


class TestFunctionSignatures:
    """Verify function signatures and help text."""

    def test_find_signals_has_docstring(self):
        assert find_signals.__doc__ is not None
        assert "facility" in find_signals.__doc__

    def test_find_wiki_has_docstring(self):
        assert find_wiki.__doc__ is not None

    def test_find_imas_has_docstring(self):
        assert find_dd_paths.__doc__ is not None

    def test_all_functions_importable(self):
        from imas_codex.graph.domain_queries import (
            facility_overview,
            find_code,
            find_data_nodes,
            find_dd_paths,
            find_signals,
            find_wiki,
            map_signals_to_imas,
        )
