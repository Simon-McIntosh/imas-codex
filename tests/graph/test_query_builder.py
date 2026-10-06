"""Tests for graph_search query builder."""

from unittest.mock import MagicMock

import pytest

from imas_codex.graph.query_builder import (
    graph_search,
    render_chunk_source,
    render_path_prefix_clause,
)


@pytest.fixture
def mock_gc():
    gc = MagicMock()
    gc.query.return_value = []
    return gc


@pytest.fixture
def mock_embed():
    return MagicMock(return_value=[0.1] * 256)


class TestGraphSearchBasic:
    """Basic query builder tests."""

    def test_simple_label_query(self, mock_gc, mock_embed):
        graph_search("Facility", gc=mock_gc, embed_fn=mock_embed)
        assert mock_gc.query.called
        cypher = mock_gc.query.call_args[0][0]
        assert "Facility" in cypher

    def test_returns_list(self, mock_gc, mock_embed):
        result = graph_search("Facility", gc=mock_gc, embed_fn=mock_embed)
        assert isinstance(result, list)

    def test_invalid_label_raises(self, mock_gc, mock_embed):
        with pytest.raises(ValueError, match="Unknown label"):
            graph_search("NonExistentLabel", gc=mock_gc, embed_fn=mock_embed)

    def test_limit_parameter(self, mock_gc, mock_embed):
        graph_search("Facility", limit=5, gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "5" in cypher


class TestGraphSearchWhere:
    """Test property filters."""

    def test_where_dict_generates_where_clause(self, mock_gc, mock_embed):
        graph_search(
            "FacilitySignal",
            where={"diagnostic": "ip"},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "diagnostic" in cypher

    def test_where_validates_properties(self, mock_gc, mock_embed):
        with pytest.raises(ValueError, match="property"):
            graph_search(
                "Facility",
                where={"nonexistent_prop": "value"},
                gc=mock_gc,
                embed_fn=mock_embed,
            )

    def test_multiple_where_conditions(self, mock_gc, mock_embed):
        graph_search(
            "FacilitySignal",
            where={"diagnostic": "ip", "physics_domain": "magnetics"},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "diagnostic" in cypher
        assert "physics_domain" in cypher


class TestGraphSearchSemantic:
    """Test semantic search integration."""

    def test_semantic_triggers_vector_search(self, mock_gc, mock_embed):
        graph_search(
            "FacilitySignal",
            semantic="plasma current",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        mock_embed.assert_called_once_with("plasma current")
        cypher = mock_gc.query.call_args[0][0]
        assert "vector" in cypher.lower() or "SEARCH" in cypher

    def test_semantic_on_label_without_index_raises(self, mock_gc, mock_embed):
        with pytest.raises(ValueError, match="vector index"):
            graph_search(
                "Facility",
                semantic="test query",
                gc=mock_gc,
                embed_fn=mock_embed,
            )

    def test_semantic_with_where_combined(self, mock_gc, mock_embed):
        graph_search(
            "FacilitySignal",
            semantic="electron density",
            where={"diagnostic": "interferometer"},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "SEARCH" in cypher
        assert "diagnostic" in cypher


class TestGraphSearchTraverse:
    """Test relationship traversal."""

    def test_traverse_adds_match(self, mock_gc, mock_embed):
        graph_search(
            "FacilitySignal",
            traverse=["DATA_ACCESS>DataAccess"],
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "DATA_ACCESS" in cypher
        assert "DataAccess" in cypher


class TestGraphSearchReturnProps:
    """Test property projection."""

    def test_return_props_limits_output(self, mock_gc, mock_embed):
        graph_search(
            "Facility",
            return_props=["id", "name"],
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "n.id" in cypher
        assert "n.name" in cypher

    def test_default_return_props_uses_key_props(self, mock_gc, mock_embed):
        graph_search("Facility", gc=mock_gc, embed_fn=mock_embed)
        cypher = mock_gc.query.call_args[0][0]
        assert "n.id" in cypher


class TestGraphSearchOrderBy:
    """Test ordering."""

    def test_order_by_property(self, mock_gc, mock_embed):
        graph_search(
            "Facility",
            order_by="name",
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "ORDER BY" in cypher
        assert "name" in cypher


class TestGraphSearchFilterOps:
    """Test text and comparison filter operators."""

    def test_contains_operator(self, mock_gc, mock_embed):
        graph_search(
            "WikiChunk",
            where={"text__contains": "fishbone"},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "CONTAINS" in cypher

    def test_starts_with_operator(self, mock_gc, mock_embed):
        graph_search(
            "SignalNode",
            where={"path__starts_with": "\\RESULTS"},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "STARTS WITH" in cypher

    def test_ends_with_operator(self, mock_gc, mock_embed):
        graph_search(
            "FacilityPath",
            where={"path__ends_with": ".py"},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "ENDS WITH" in cypher

    def test_in_operator(self, mock_gc, mock_embed):
        graph_search(
            "FacilityPath",
            where={"status__in": ["discovered", "explored"]},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "IN" in cypher

    def test_gt_operator(self, mock_gc, mock_embed):
        graph_search(
            "FacilityPath",
            where={"score_composite__gt": 0.7},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert ">" in cypher

    def test_ne_operator(self, mock_gc, mock_embed):
        graph_search(
            "FacilityPath",
            where={"status__ne": "failed"},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "<>" in cypher

    def test_operator_validates_base_property(self, mock_gc, mock_embed):
        with pytest.raises(ValueError, match="property"):
            graph_search(
                "Facility",
                where={"nonexistent__contains": "x"},
                gc=mock_gc,
                embed_fn=mock_embed,
            )

    def test_contains_with_semantic(self, mock_gc, mock_embed):
        """Operators should combine with semantic search."""
        graph_search(
            "WikiChunk",
            semantic="fishbone instabilities",
            where={"text__contains": "fishbone"},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert "SEARCH" in cypher
        assert "CONTAINS" in cypher

    def test_multiple_operators(self, mock_gc, mock_embed):
        graph_search(
            "FacilityPath",
            where={"score_composite__gte": 0.5, "status__ne": "failed"},
            gc=mock_gc,
            embed_fn=mock_embed,
        )
        cypher = mock_gc.query.call_args[0][0]
        assert ">=" in cypher
        assert "<>" in cypher


class TestRenderChunkSource:
    """The rendered chunk-source resolution is pinned so readers cannot diverge."""

    def test_pins_rendered_text(self):
        assert render_chunk_source("cc") == (
            "OPTIONAL MATCH (ce:CodeExample)-[:HAS_CHUNK]->(cc)\n"
            "WITH *, coalesce(ce.source_file, cc.source_file) AS source_file"
        )

    def test_parameterises_alias_and_output(self):
        assert render_chunk_source("node", "src") == (
            "OPTIONAL MATCH (ce:CodeExample)-[:HAS_CHUNK]->(node)\n"
            "WITH *, coalesce(ce.source_file, node.source_file) AS src"
        )


class TestRenderPathPrefixClause:
    """The prefix clause renderer shared by path claims and scoped resets."""

    def test_alias_p_root_filter(self):
        assert render_path_prefix_clause("p", "root_filter") == (
            "AND any(prefix IN $root_filter WHERE p.path STARTS WITH prefix)"
        )

    def test_alias_n_root_prefixes(self):
        assert render_path_prefix_clause("n", "root_prefixes") == (
            "AND any(prefix IN $root_prefixes WHERE n.path STARTS WITH prefix)"
        )

    def test_prefixes_parameter(self):
        assert render_path_prefix_clause("p", "prefixes") == (
            "AND any(prefix IN $prefixes WHERE p.path STARTS WITH prefix)"
        )

    def test_uses_starts_with(self):
        clause = render_path_prefix_clause("n", "root_prefixes")
        assert "STARTS WITH" in clause
