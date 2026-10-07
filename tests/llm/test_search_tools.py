"""Tests for unified MCP search tools.

Each test class covers one search_* MCP tool. Tests mock the GraphClient
and Encoder to avoid requiring a running Neo4j/embedding server.

Uses a query routing helper to dispatch gc.query mock results based on
Cypher query content, making tests resilient to changes in call ordering
or the addition of new query stages (text search, hybrid merge, etc.).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import ClientError, ServiceUnavailable

from imas_codex.graph.query_builder import render_chunk_source
from imas_codex.llm.search_formatters import (
    _interpolate_template,
    format_code_report,
    format_docs_report,
    format_fetch_report,
    format_imas_report,
    format_signals_report,
)
from imas_codex.llm.search_tools import (
    _CHUNK_DOCUMENTS_CAP,
    RERANK_POOL,
    _enrich_code_chunks,
    _enrich_wiki_chunks,
    _fetch,
    _fetch_wiki_page,
    _search_code,
    _search_docs,
    _search_signals,
    _text_search_code_chunks,
    _text_search_signals,
    _text_search_wiki_chunks,
    rerank_candidates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _route_query(routes: dict[str, list[dict[str, Any]]]) -> Any:
    """Create a gc.query side_effect that dispatches by Cypher content.

    The first matching pattern wins; unmatched queries return [].
    Pattern matching is simple substring (``pattern in cypher``).
    """

    def handler(cypher: str, **kwargs: Any) -> list[dict[str, Any]]:
        for pattern, result in routes.items():
            if pattern in cypher:
                return result
        return []

    return handler


# ---------------------------------------------------------------------------
# Canned mock data
# ---------------------------------------------------------------------------

# Signals — uses new access_methods array format
_SIGNAL_ENRICHMENT_IP = {
    "id": "tcv:magnetics/ip",
    "name": "ip",
    "description": "Plasma current",
    "physics_domain": "magnetics",
    "unit": "A",
    "checked": True,
    "example_shot": 84000,
    "node_path": "\\RESULTS::I_P",
    "accessor": "MDSplus",
    "diagnostic_name": "magnetics",
    "diagnostic_category": "magnetics",
    "tree_path": "\\RESULTS::I_P",
    "data_source_name": "tcv_shot",
    "access_methods": [
        {
            "access_template": "t = MDSplus.Tree('tcv_shot', shot)\nnode = t.getNode('\\\\results::i_p')",
            "access_type": "mdsplus",
            "imports_template": "import MDSplus",
            "connection_template": None,
            "imas_path": "magnetics.ip.0d[:].value",
            "imas_docs": "Plasma current positive sign",
            "imas_unit": "A",
        }
    ],
}

_SIGNAL_ENRICHMENT_BPOL = {
    "id": "tcv:magnetics/bpol",
    "name": "bpol",
    "description": "Poloidal magnetic field",
    "physics_domain": "magnetics",
    "unit": "T",
    "checked": False,
    "example_shot": None,
    "node_path": None,
    "accessor": None,
    "diagnostic_name": "magnetics",
    "diagnostic_category": "magnetics",
    "tree_path": None,
    "data_source_name": None,
    "access_methods": [
        {
            "access_template": None,
            "access_type": None,
            "imports_template": None,
            "connection_template": None,
            "imas_path": None,
            "imas_docs": None,
            "imas_unit": None,
        }
    ],
}

_SIGNAL_VECTOR_RESULTS = [
    {"id": "tcv:magnetics/ip", "score": 0.92},
    {"id": "tcv:magnetics/bpol", "score": 0.85},
]

_DATA_NODE_RESULTS = [
    {
        "id": "tcv:\\RESULTS::I_P",
        "path": "\\RESULTS::I_P",
        "data_source_name": "tcv_shot",
        "description": "Plasma current",
        "unit": "A",
    }
]

# IMAS — new enrichment format with lifecycle_status, structure_reference, etc.
_IMAS_ENRICHMENT_TEMP = {
    "id": "core_profiles.profiles_1d[:].electrons.temperature",
    "name": "temperature",
    "ids": "core_profiles",
    "documentation": "Electron temperature profile",
    "data_type": "FLT_1D",
    "ndim": 1,
    "node_type": "leaf",
    "physics_domain": "core_transport",
    "cocos_transformation_type": None,
    "lifecycle_status": "active",
    "lifecycle_version": None,
    "structure_reference": None,
    "unit": "eV",
    "clusters": ["Electron Temperature Profiles"],
    "coordinates": ["core_profiles.profiles_1d[:].grid.rho_tor_norm"],
    "introduced_in": "3.0.0",
}

_IMAS_ENRICHMENT_IP = {
    "id": "magnetics.ip.0d[:].value",
    "name": "value",
    "ids": "magnetics",
    "documentation": "Plasma current",
    "data_type": "FLT_0D",
    "ndim": 0,
    "node_type": "leaf",
    "physics_domain": "magnetics",
    "cocos_transformation_type": "ip_like",
    "lifecycle_status": "active",
    "lifecycle_version": None,
    "structure_reference": None,
    "unit": "A",
    "clusters": [],
    "coordinates": [],
    "introduced_in": "3.0.0",
}


# ---------------------------------------------------------------------------
# search_signals
# ---------------------------------------------------------------------------


class TestSearchSignals:
    """Unit tests for search_signals tool."""

    @pytest.fixture()
    def mock_gc(self):
        """GraphClient mock with default empty routing."""
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        """Encoder mock that returns a fixed embedding."""
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def test_empty_results(self, mock_gc, mock_encoder):
        """Empty vector + text + tree results produce a descriptive message."""
        result = _search_signals(
            query="plasma current",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "No signals found" in result

    def test_signal_results_formatted(self, mock_gc, mock_encoder):
        """Signal results are formatted into a readable report."""
        mock_gc.query.side_effect = _route_query(
            {
                "facility_signal_desc_embedding": _SIGNAL_VECTOR_RESULTS,
                "UNWIND $signal_ids": [
                    _SIGNAL_ENRICHMENT_IP,
                    _SIGNAL_ENRICHMENT_BPOL,
                ],
                "signal_node_desc_embedding": _DATA_NODE_RESULTS,
            }
        )

        result = _search_signals(
            query="plasma current",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )

        assert "## Signals" in result
        assert "tcv:magnetics/ip" in result
        assert "Plasma current" in result
        assert "magnetics" in result
        # New access_methods format: access template shown
        assert "Data access" in result

    def test_diagnostic_filter_passed(self, mock_gc, mock_encoder):
        """diagnostic parameter is included in the vector search query."""
        _search_signals(
            query="current",
            facility="tcv",
            diagnostic="magnetics",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        # Find the vector search call (facility_signal_desc_embedding)
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "facility_signal_desc_embedding" in cypher:
                assert "diagnostic" in cypher
                break
        else:
            pytest.fail("No vector search call found")

    def test_physics_domain_filter_passed(self, mock_gc, mock_encoder):
        """physics_domain parameter is included in the vector search query."""
        _search_signals(
            query="current",
            facility="tcv",
            physics_domain="magnetics",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "facility_signal_desc_embedding" in cypher:
                assert "physics_domain" in cypher
                break
        else:
            pytest.fail("No vector search call found")

    def test_facility_filter_inside_search(self, mock_gc, mock_encoder):
        """Facility predicate is an in-index pre-filter, inside SEARCH."""
        _search_signals(
            query="plasma current",
            facility="jt-60sa",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "facility_signal_desc_embedding" in cypher:
                open_idx = cypher.index("SEARCH")
                close_idx = cypher.index(") SCORE AS")
                pred_idx = cypher.index("s.facility_id = $facility")
                assert open_idx < pred_idx < close_idx, cypher
                break
        else:
            pytest.fail("No facility_signal_desc_embedding vector search call found")

    def test_overfetch_survives_non_facility_postfilter(self, mock_gc, mock_encoder):
        """A diagnostic or domain post-filter keeps the widened ANN cut."""
        _search_signals(
            query="plasma current",
            facility="jt-60sa",
            physics_domain="magnetics",
            k=10,
            gc=mock_gc,
            encoder=mock_encoder,
        )
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "facility_signal_desc_embedding" in cypher:
                assert call[1]["k"] == 50, call[1]
                assert call[1]["limit"] == 10, call[1]
                assert "LIMIT $k" in cypher
                break
        else:
            pytest.fail("No facility_signal_desc_embedding vector search call found")

    def test_no_overfetch_without_postfilter(self, mock_gc, mock_encoder):
        """With the facility pre-filter as the only filter, the cut is k."""
        _search_signals(
            query="plasma current",
            facility="jt-60sa",
            k=10,
            gc=mock_gc,
            encoder=mock_encoder,
        )
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "facility_signal_desc_embedding" in cypher:
                assert call[1]["k"] == 10, call[1]
                assert "LIMIT $k" in cypher
                break
        else:
            pytest.fail("No facility_signal_desc_embedding vector search call found")

    def test_no_data_nodes_section_omitted(self, mock_gc, mock_encoder):
        """When data node search returns empty, section is omitted."""
        mock_gc.query.side_effect = _route_query(
            {
                "facility_signal_desc_embedding": [
                    {"id": "tcv:magnetics/ip", "score": 0.92}
                ],
                "UNWIND $signal_ids": [
                    {
                        **_SIGNAL_ENRICHMENT_IP,
                        "access_methods": [
                            {
                                "access_template": None,
                                "access_type": None,
                                "imports_template": None,
                                "connection_template": None,
                                "imas_path": None,
                                "imas_docs": None,
                                "imas_unit": None,
                            }
                        ],
                    }
                ],
            }
        )

        result = _search_signals(
            query="plasma current",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "Related Data Nodes" not in result

    def test_embedding_unavailable(self, mock_gc):
        """When encoder raises, return helpful error message."""
        from imas_codex.embeddings.encoder import EmbeddingBackendError

        bad_encoder = MagicMock()
        bad_encoder.embed_texts.side_effect = EmbeddingBackendError("unavailable")

        result = _search_signals(
            query="current",
            facility="tcv",
            gc=mock_gc,
            encoder=bad_encoder,
        )
        assert "Embedding" in result or "unavailable" in result.lower()

    def test_neo4j_unavailable(self, mock_encoder):
        """When Neo4j is down, return helpful error message."""
        bad_gc = MagicMock()
        bad_gc.query.side_effect = ServiceUnavailable("Connection refused")

        result = _search_signals(
            query="current",
            facility="tcv",
            gc=bad_gc,
            encoder=mock_encoder,
        )
        assert "not running" in result.lower() or "neo4j" in result.lower()

    def test_hybrid_text_boost(self, mock_gc, mock_encoder):
        """Signals found by both vector and text search get boosted."""
        mock_gc.query.side_effect = _route_query(
            {
                "facility_signal_desc_embedding": [
                    {"id": "tcv:magnetics/ip", "score": 0.80}
                ],
                "s.name) CONTAINS": [{"id": "tcv:magnetics/ip", "score": 0.6}],
                "UNWIND $signal_ids": [_SIGNAL_ENRICHMENT_IP],
            }
        )

        result = _search_signals(
            query="ip",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "tcv:magnetics/ip" in result
        # The boosted score should be different from the raw vector score
        assert "0.80" not in result  # Score should be modified by boost

    def test_text_only_result(self, mock_gc, mock_encoder):
        """Signal found only by text search (not vector) is included."""
        mock_gc.query.side_effect = _route_query(
            {
                # No vector results
                "s.name) CONTAINS": [{"id": "tcv:magnetics/ip", "score": 0.6}],
                "UNWIND $signal_ids": [_SIGNAL_ENRICHMENT_IP],
            }
        )

        result = _search_signals(
            query="ip",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "tcv:magnetics/ip" in result


# ---------------------------------------------------------------------------
# Formatter unit tests
# ---------------------------------------------------------------------------


class TestFormatSignalsReport:
    """Unit tests for the signals report formatter."""

    def test_empty_signals(self):
        """Empty signal list produces empty-result message."""
        result = format_signals_report([], [], {})
        assert "No signals found" in result

    def test_basic_signal_formatting(self):
        """A single signal with access_methods is formatted with all sections."""
        signals = [_SIGNAL_ENRICHMENT_IP]
        scores = {"tcv:magnetics/ip": 0.92}

        result = format_signals_report(signals, [], scores)

        assert "tcv:magnetics/ip" in result
        assert "0.92" in result
        assert "Plasma current" in result
        assert "magnetics" in result
        assert "Data access" in result
        assert "IMAS mapping" in result
        assert "Data node" in result

    def test_legacy_flat_format_backward_compat(self):
        """Legacy flat access format still works (backward compat)."""
        signals = [
            {
                "id": "tcv:magnetics/ip",
                "name": "ip",
                "description": "Plasma current",
                "physics_domain": "magnetics",
                "unit": "A",
                "checked": True,
                "example_shot": 84000,
                "diagnostic_name": "magnetics",
                "diagnostic_category": "magnetics",
                "access_template": "tree.getNode('\\\\results::i_p')",
                "access_type": "mdsplus",
                "imports_template": "import MDSplus",
                "connection_template": None,
                "tree_path": "\\RESULTS::I_P",
                "data_source_name": "tcv_shot",
                "imas_path": "magnetics.ip.0d[:].value",
                "imas_docs": "Plasma current positive sign",
                "imas_unit": "A",
            }
        ]
        scores = {"tcv:magnetics/ip": 0.92}

        result = format_signals_report(signals, [], scores)

        assert "tcv:magnetics/ip" in result
        assert "Data access" in result
        assert "IMAS mapping" in result

    def test_data_nodes_section(self):
        """Data nodes appear in Related Data Nodes section."""
        data_nodes = [
            {
                "id": "tcv:\\RESULTS::I_P",
                "path": "\\RESULTS::I_P",
                "data_source_name": "tcv_shot",
                "description": "Plasma current",
                "unit": "A",
                "score": 0.95,
            }
        ]
        result = format_signals_report([], data_nodes, {})
        assert "Related Data Nodes" in result
        assert "\\RESULTS::I_P" in result

    def test_long_access_template_rendered_fully(self):
        """Long access templates are rendered without truncation."""
        long_template = "x = 1\n" * 100
        signals = [
            {
                "id": "tcv:test/sig",
                "name": "sig",
                "description": "Test signal",
                "physics_domain": None,
                "unit": None,
                "checked": False,
                "example_shot": None,
                "diagnostic_name": None,
                "diagnostic_category": None,
                "node_path": None,
                "accessor": None,
                "tree_path": None,
                "data_source_name": None,
                "access_methods": [
                    {
                        "access_template": long_template,
                        "access_type": "mdsplus",
                        "imports_template": None,
                        "connection_template": None,
                        "imas_path": None,
                        "imas_docs": None,
                        "imas_unit": None,
                    }
                ],
            }
        ]
        result = format_signals_report(signals, [], {"tcv:test/sig": 0.5})
        # Full template should be present — no truncation
        assert long_template.strip() in result

    def test_multiple_access_methods(self):
        """Multiple access methods per signal are all rendered."""
        signals = [
            {
                "id": "tcv:magnetics/ip",
                "name": "ip",
                "description": "Plasma current",
                "physics_domain": "magnetics",
                "unit": "A",
                "checked": True,
                "example_shot": 84000,
                "diagnostic_name": "magnetics",
                "diagnostic_category": None,
                "node_path": None,
                "accessor": None,
                "tree_path": None,
                "data_source_name": None,
                "access_methods": [
                    {
                        "access_template": "tree.getNode('\\\\ip').data()",
                        "access_type": "mdsplus",
                        "imports_template": "import MDSplus",
                        "connection_template": None,
                        "imas_path": "magnetics.ip.0d[:].value",
                        "imas_docs": None,
                        "imas_unit": None,
                    },
                    {
                        "access_template": "imas_entry.get('magnetics/ip')",
                        "access_type": "imas",
                        "imports_template": "import imas",
                        "connection_template": None,
                        "imas_path": "magnetics.ip.0d[:].value",
                        "imas_docs": None,
                        "imas_unit": None,
                    },
                ],
            }
        ]
        result = format_signals_report(signals, [], {"tcv:magnetics/ip": 0.9})
        # Both access methods should be shown
        assert "mdsplus" in result
        assert "imas" in result


# ---------------------------------------------------------------------------
# search_docs
# ---------------------------------------------------------------------------


class TestSearchDocs:
    """Unit tests for search_docs tool."""

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def test_empty_results(self, mock_gc, mock_encoder):
        """Empty results produce a descriptive message."""
        result = _search_docs(
            query="equilibrium",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "No documentation found" in result

    def test_wiki_chunks_formatted(self, mock_gc, mock_encoder):
        """Wiki chunks are grouped by page and formatted."""
        chunk_vector = [
            {"id": "jet:wiki:chunk:1", "score": 0.90},
            {"id": "jet:wiki:chunk:2", "score": 0.85},
        ]
        enrichment = [
            {
                "id": "jet:wiki:chunk:1",
                "text": "Fishbone instabilities are observed when...",
                "section": "Overview",
                "page_title": "Fishbone instabilities",
                "page_url": "https://wiki.jet.efda.org/fishbone",
                "linked_signals": ["jet:mhd/fishbone_amplitude"],
                "linked_data_nodes": [],
                "imas_refs": ["mhd_linear.time_slice[:].toroidal_mode[:].n_tor"],
            },
            {
                "id": "jet:wiki:chunk:2",
                "text": "Detection methods include Mirnov coil analysis...",
                "section": "Detection Methods",
                "page_title": "Fishbone instabilities",
                "page_url": "https://wiki.jet.efda.org/fishbone",
                "linked_signals": [],
                "linked_data_nodes": [],
                "imas_refs": [],
            },
        ]
        mock_gc.query.side_effect = _route_query(
            {
                "wiki_chunk_embedding": chunk_vector,
                "WikiChunk {id: cid}": enrichment,
            }
        )

        result = _search_docs(
            query="fishbone instabilities",
            facility="jet",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "Wiki Documentation" in result
        assert "Fishbone instabilities" in result
        assert "Overview" in result
        assert "Detection Methods" in result

    def test_page_grouping(self, mock_gc, mock_encoder):
        """Chunks from the same page are grouped together."""
        chunk_vector = [
            {"id": "c1", "score": 0.9},
            {"id": "c2", "score": 0.8},
        ]
        enrichment = [
            {
                "id": "c1",
                "text": "Section 1 content",
                "section": "Sec1",
                "page_title": "Same Page",
                "page_url": "http://wiki/same",
                "linked_signals": [],
                "linked_data_nodes": [],
                "imas_refs": [],
            },
            {
                "id": "c2",
                "text": "Section 2 content",
                "section": "Sec2",
                "page_title": "Same Page",
                "page_url": "http://wiki/same",
                "linked_signals": [],
                "linked_data_nodes": [],
                "imas_refs": [],
            },
        ]
        mock_gc.query.side_effect = _route_query(
            {
                "wiki_chunk_embedding": chunk_vector,
                "WikiChunk {id: cid}": enrichment,
            }
        )

        result = _search_docs(
            query="test", facility="tcv", gc=mock_gc, encoder=mock_encoder
        )
        # Page title should appear once as header, not twice
        assert result.count('### Page: "Same Page"') == 1

    def test_cosine_breaks_saturated_score_ties(self, mock_gc, mock_encoder):
        """Chunks tied at the 1.0 clamp come back in descending cosine order.

        The hybrid score ties at the same value for many chunks, and a title
        match then clamps every displayed score to 1.00, so nothing in the
        displayed score can order them. The raw cosine must, rather than the
        arbitrary set-iteration order the ranking previously inherited.
        """
        query = "equilibrium flux"
        n = 8
        cosines = [0.90 - 0.05 * i for i in range(n)]
        # Text hits chosen so every chunk's hybrid score rounds to the same
        # 0.82: 0.7*cosine + 0.3*text + 0.1. A title overlap of two query
        # terms then lifts each to 1.02, which clamps to 1.00.
        text_scores = [(0.72 - 0.7 * c) / 0.3 for c in cosines]
        chunk_vector = [{"id": f"c{i}", "score": c} for i, c in enumerate(cosines)]
        text_hits = [{"id": f"c{i}", "score": t} for i, t in enumerate(text_scores)]
        enrichment = [
            {
                "id": f"c{i}",
                "text": f"chunk body number {i}",
                "section": f"sec{i}",
                "page_title": "equilibrium flux",
                "page_url": None,
                "linked_signals": [],
                "linked_data_nodes": [],
                "imas_refs": [],
            }
            for i in range(n)
        ]

        def handler(cypher: str, **kwargs: Any) -> list[dict[str, Any]]:
            if "wiki_chunk_embedding" in cypher:
                return chunk_vector
            if "wiki_chunk_text" in cypher:
                return text_hits
            if "WikiChunk {id: cid}" in cypher:
                by_id = {e["id"]: e for e in enrichment}
                return [by_id[c] for c in kwargs["chunk_ids"]]
            return []

        mock_gc.query = MagicMock(side_effect=handler)

        result = _search_docs(
            query=query,
            facility="jt-60sa",
            k=30,
            gc=mock_gc,
            encoder=mock_encoder,
        )

        # Every chunk is displayed at the clamp, so the score alone cannot
        # order them.
        assert result.count("[score: 1.00]") == n
        positions = [result.index(f"**Section: sec{i}**") for i in range(n)]
        # Cosine order is c0 (0.90) .. c{n-1} (0.50); anything else means the
        # tie was broken arbitrarily.
        assert positions == sorted(positions)

    def test_cross_links_shown(self, mock_gc, mock_encoder):
        """Cross-links to signals and IMAS paths are shown."""
        chunk_vector = [{"id": "c1", "score": 0.9}]
        enrichment = [
            {
                "id": "c1",
                "text": "Content about plasma current",
                "section": "Signals",
                "page_title": "Magnetics",
                "page_url": None,
                "linked_signals": ["tcv:magnetics/ip"],
                "linked_data_nodes": ["\\RESULTS::I_P"],
                "imas_refs": ["magnetics.ip.0d[:].value"],
            },
        ]
        mock_gc.query.side_effect = _route_query(
            {
                "wiki_chunk_embedding": chunk_vector,
                "WikiChunk {id: cid}": enrichment,
            }
        )

        result = _search_docs(
            query="plasma current", facility="tcv", gc=mock_gc, encoder=mock_encoder
        )
        assert "tcv:magnetics/ip" in result
        assert "magnetics.ip.0d[:].value" in result

    def test_wiki_facility_filter_inside_search(self, mock_gc, mock_encoder):
        """Wiki-chunk facility predicate is an in-index pre-filter, inside SEARCH."""
        _search_docs(
            query="plasma current", facility="jt-60sa", gc=mock_gc, encoder=mock_encoder
        )
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "wiki_chunk_embedding" in cypher:
                open_idx = cypher.index("SEARCH")
                close_idx = cypher.index(") SCORE AS")
                pred_idx = cypher.index("c.facility_id = $facility")
                assert open_idx < pred_idx < close_idx, cypher
                break
        else:
            pytest.fail("No wiki_chunk_embedding vector search call found")

    def test_wiki_overfetch_survives_non_facility_postfilter(
        self, mock_gc, mock_encoder
    ):
        """A WikiPage post-filter keeps the widened ANN cut."""
        _search_docs(
            query="plasma current",
            facility="jt-60sa",
            physics_domain="magnetics",
            k=10,
            gc=mock_gc,
            encoder=mock_encoder,
        )
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "wiki_chunk_embedding" in cypher:
                assert call[1]["k"] == 2 * RERANK_POOL, call[1]
                assert call[1]["limit"] == RERANK_POOL, call[1]
                assert "LIMIT $k" in cypher
                break
        else:
            pytest.fail("No wiki_chunk_embedding vector search call found")

    def test_wiki_no_overfetch_without_postfilter(self, mock_gc, mock_encoder):
        """With the facility pre-filter as the only filter, the cut is k."""
        _search_docs(
            query="plasma current",
            facility="jt-60sa",
            k=10,
            gc=mock_gc,
            encoder=mock_encoder,
        )
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "wiki_chunk_embedding" in cypher:
                assert call[1]["k"] == RERANK_POOL, call[1]
                assert "LIMIT $k" in cypher
                break
        else:
            pytest.fail("No wiki_chunk_embedding vector search call found")

    def test_embedding_unavailable(self, mock_gc):
        """When encoder is unavailable, return helpful message."""
        from imas_codex.embeddings.encoder import EmbeddingBackendError

        bad_encoder = MagicMock()
        bad_encoder.embed_texts.side_effect = EmbeddingBackendError("unavailable")

        result = _search_docs(
            query="test", facility="tcv", gc=mock_gc, encoder=bad_encoder
        )
        assert "Embedding" in result or "unavailable" in result.lower()

    def test_hybrid_text_boost(self, mock_gc, mock_encoder):
        """Wiki chunks found by both vector and text search get boosted."""
        mock_gc.query.side_effect = _route_query(
            {
                "wiki_chunk_embedding": [{"id": "c1", "score": 0.70}],
                "c.text) CONTAINS": [{"id": "c1", "score": 0.5}],
                "WikiChunk {id: cid}": [
                    {
                        "id": "c1",
                        "text": "Fishbone content",
                        "section": "Overview",
                        "page_title": "Fishbone",
                        "page_url": None,
                        "linked_signals": [],
                        "linked_data_nodes": [],
                        "imas_refs": [],
                    }
                ],
            }
        )
        result = _search_docs(
            query="fishbone", facility="jet", gc=mock_gc, encoder=mock_encoder
        )
        assert "Fishbone" in result
        # Score should be boosted from 0.70
        assert "0.70" not in result


class TestFormatDocsReport:
    """Unit tests for the docs report formatter."""

    def test_empty_docs(self):
        result = format_docs_report([], [], {})
        assert "No documentation found" in result

    def test_documents_section(self):
        """Documents appear in Related Documents section."""
        documents = [
            {
                "id": "art1",
                "title": "Analysis Report.pdf",
                "description": "Detailed analysis",
                "page_title": "MHD diagnostics",
            },
        ]
        result = format_docs_report([], documents, {})
        assert "Related Documents" in result
        assert "Analysis Report.pdf" in result
        assert "MHD diagnostics" in result


# ---------------------------------------------------------------------------
# search_code
# ---------------------------------------------------------------------------


class TestSearchCode:
    """Unit tests for search_code tool."""

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def test_empty_results(self, mock_gc, mock_encoder):
        """Empty results produce a descriptive message."""
        result = _search_code(
            query="equilibrium reconstruction",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "No code examples found" in result

    def test_code_results_formatted(self, mock_gc, mock_encoder):
        """Code results are formatted with data references."""
        vector_results = [{"id": "chunk:1", "score": 0.89}]
        enrichment = [
            {
                "id": "chunk:1",
                "text": "def read_equilibrium(shot):\n    tree = MDSplus.Tree('tcv_shot', shot)\n    psi = tree.getNode('\\\\results::psi').data()",
                "function_name": "read_equilibrium",
                "source_file": "/home/codes/liuqe/liuqe_reader.py",
                "source_file_id": "tcv:/home/codes/liuqe/liuqe_reader.py",
                "facility_id": "tcv",
                "data_refs": [
                    {
                        "type": "mdsplus",
                        "raw": "\\RESULTS::PSI",
                        "tree": "\\RESULTS::PSI",
                        "imas": "equilibrium.time_slice[:].profiles_2d[:].psi",
                        "tdi": None,
                    }
                ],
                "directory": "/home/codes/liuqe",
                "dir_description": "LIUQE equilibrium reconstruction code",
            },
        ]
        mock_gc.query.side_effect = _route_query(
            {
                "code_chunk_embedding": vector_results,
                "CodeChunk {id: cid}": enrichment,
            }
        )

        result = _search_code(
            query="equilibrium",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "Code Examples" in result
        assert "read_equilibrium" in result
        assert "liuqe_reader.py" in result
        assert "Data references" in result

    def test_facility_filter(self, mock_gc, mock_encoder):
        """facility parameter filters code chunks via CodeExample."""
        _search_code(
            query="test",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        # Find the vector search call (code_chunk_embedding)
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "code_chunk_embedding" in cypher:
                assert "facility" in cypher.lower()
                break
        else:
            pytest.fail("No vector search call found")

    def test_facility_filter_inside_search(self, mock_gc, mock_encoder):
        """Facility predicate is an in-index pre-filter, inside SEARCH."""
        _search_code(
            query="SELENE equilibrium reconstruction",
            facility="jt-60sa",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "code_chunk_embedding" in cypher:
                open_idx = cypher.index("SEARCH")
                close_idx = cypher.index(") SCORE AS")
                pred_idx = cypher.index("cc.facility_id = $facility")
                assert open_idx < pred_idx < close_idx, cypher
                break
        else:
            pytest.fail("No code_chunk_embedding vector search call found")

    def test_code_example_facility_filter_inside_search(self, mock_gc, mock_encoder):
        """Code-example facility predicate is an in-index pre-filter, inside SEARCH."""
        from imas_codex.llm.search_tools import _vector_search_code_examples

        _vector_search_code_examples(mock_gc, [0.1] * 1024, facility="jt-60sa", k=10)
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "code_example_desc_embedding" in cypher:
                open_idx = cypher.index("SEARCH")
                close_idx = cypher.index(") SCORE AS")
                pred_idx = cypher.index("ce.facility_id = $facility")
                assert open_idx < pred_idx < close_idx, cypher
                break
        else:
            pytest.fail("No code_example_desc_embedding vector search call found")

    def test_code_example_missing_index_returns_empty(self, mock_gc):
        """A missing/unavailable index yields no examples rather than failing."""
        from neo4j.exceptions import ClientError

        from imas_codex.llm.search_tools import _vector_search_code_examples

        mock_gc.query.side_effect = ClientError("no such index")
        assert (
            _vector_search_code_examples(
                mock_gc, [0.1] * 1024, facility="jt-60sa", k=10
            )
            == []
        )

    def test_code_example_programming_error_propagates(self, mock_gc):
        """A non-client error is not swallowed by the narrowed except."""
        from imas_codex.llm.search_tools import _vector_search_code_examples

        mock_gc.query.side_effect = ValueError("boom")
        with pytest.raises(ValueError):
            _vector_search_code_examples(
                mock_gc, [0.1] * 1024, facility="jt-60sa", k=10
            )

    def test_no_facility_filter(self, mock_gc, mock_encoder):
        """Without facility, no facility filter in vector query."""
        _search_code(
            query="test",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "code_chunk_embedding" in cypher:
                assert "facility" not in call[1]  # kwargs
                break

    def test_embedding_unavailable(self, mock_gc):
        """When encoder is unavailable, return helpful message."""
        from imas_codex.embeddings.encoder import EmbeddingBackendError

        bad_encoder = MagicMock()
        bad_encoder.embed_texts.side_effect = EmbeddingBackendError("unavailable")

        result = _search_code(query="test", gc=mock_gc, encoder=bad_encoder)
        assert "Embedding" in result or "unavailable" in result.lower()

    def test_hybrid_text_boost(self, mock_gc, mock_encoder):
        """Code chunks found by both vector and text search get boosted."""
        enrichment = [
            {
                "id": "chunk:1",
                "text": "def solve(): pass",
                "function_name": "solve",
                "source_file": "/code/solver.py",
                "source_file_id": "tcv:/code/solver.py",
                "facility_id": "tcv",
                "data_refs": [],
                "directory": "/code",
                "dir_description": None,
            }
        ]
        mock_gc.query.side_effect = _route_query(
            {
                "code_chunk_embedding": [{"id": "chunk:1", "score": 0.75}],
                "cc.text) CONTAINS": [{"id": "chunk:1", "score": 0.5}],
                "CodeChunk {id: cid}": enrichment,
            }
        )

        result = _search_code(
            query="solve",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "solve" in result
        # Boosted score, not raw vector score
        assert "0.75" not in result


class TestFormatCodeReport:
    """Unit tests for the code report formatter."""

    def test_empty_code(self):
        result = format_code_report([], {})
        assert "No code examples found" in result

    def test_data_refs_shown(self):
        """Data references appear in output."""
        code_results = [
            {
                "id": "c1",
                "text": "tree.getNode('psi').data()",
                "function_name": "read_psi",
                "source_file": "/code/reader.py",
                "source_file_id": "tcv:/code/reader.py",
                "facility_id": "tcv",
                "data_refs": [
                    {
                        "type": "mdsplus",
                        "raw": "\\PSI",
                        "tree": "\\PSI",
                        "imas": None,
                        "tdi": None,
                    }
                ],
                "directory": "/code",
                "dir_description": "Analysis code",
            }
        ]
        result = format_code_report(code_results, {"c1": 0.85})
        assert "read_psi" in result
        assert "\\PSI" in result
        assert "Data references" in result

    def test_code_fence_uses_chunk_language(self):
        """Code fence uses the chunk's language, not hardcoded 'python'."""
        code_results = [
            {
                "id": "c1",
                "text": "SUBROUTINE PFCOIL(x)\n  WRITE(*,*) x\nEND SUBROUTINE",
                "function_name": "PFCOIL",
                "source_file": "/home/mkovari/process/pfcoil.f",
                "source_file_id": "jet:/home/mkovari/process/pfcoil.f",
                "facility_id": "jet",
                "language": "fortran",
                "data_refs": [],
                "directory": None,
                "dir_description": None,
            }
        ]
        result = format_code_report(code_results, {"c1": 0.89})
        assert "```fortran" in result
        assert "```python" not in result

    def test_code_fence_defaults_to_python(self):
        """Code fence defaults to python when language is missing."""
        code_results = [
            {
                "id": "c1",
                "text": "def foo(): pass",
                "function_name": "foo",
                "source_file": "/code/foo.py",
                "facility_id": "tcv",
                "data_refs": [],
            }
        ]
        result = format_code_report(code_results, {"c1": 0.8})
        assert "```python" in result


# ---------------------------------------------------------------------------
# search_dd_paths
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="search_dd_paths moved to shared GraphSearchTool; tests in tests/graph_mcp/"
)
class TestSearchImas:
    """Unit tests for search_dd_paths tool.

    Uses query routing because _text_search_dd_paths_by_query makes
    a variable number of sub-queries depending on the query words.
    """

    _search_imas = None  # stub: original function removed; class is skipped

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def test_empty_results(self, mock_gc, mock_encoder):
        """Empty vector + text + cluster results produce a descriptive message."""
        result = _search_imas(  # noqa: F821
            query="electron temperature",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "No IMAS paths found" in result

    def test_path_results_formatted(self, mock_gc, mock_encoder):
        """Path results are enriched and formatted into a report."""
        path_vector = [
            {
                "id": "core_profiles.profiles_1d[:].electrons.temperature",
                "score": 0.95,
            },
        ]
        cluster_vector = [
            {
                "id": "cluster:electron_temp",
                "label": "Electron Temperature Profiles",
                "scope": "core_profiles",
                "path_count": 12,
                "sample_paths": ["core_profiles.profiles_1d[:].electrons.temperature"],
                "score": 0.88,
            },
        ]
        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": path_vector,
                "cluster_description_embedding": cluster_vector,
                "UNWIND $path_ids": [_IMAS_ENRICHMENT_TEMP],
            }
        )

        result = _search_imas(  # noqa: F821
            query="electron temperature",
            gc=mock_gc,
            encoder=mock_encoder,
        )

        assert "IMAS Paths" in result
        assert "core_profiles.profiles_1d[:].electrons.temperature" in result
        assert "eV" in result
        assert "Electron Temperature Profiles" in result

    def test_ids_filter_passed(self, mock_gc, mock_encoder):
        """ids_filter parameter is passed to the vector search query."""
        _search_imas(  # noqa: F821
            query="temperature",
            ids_filter="core_profiles",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "imas_node_embedding" in cypher:
                assert "ids_filter" in cypher or "ids" in cypher
                break
        else:
            pytest.fail("No vector search call found")

    def test_facility_crossrefs(self, mock_gc, mock_encoder):
        """Facility cross-references are fetched and formatted."""
        path_vector = [
            {"id": "magnetics.ip.0d[:].value", "score": 0.92},
        ]
        crossrefs = [
            {
                "id": "magnetics.ip.0d[:].value",
                "facility_signals": ["tcv:magnetics/ip"],
                "wiki_mentions": ["Plasma Current"],
                "code_files": ["/home/codes/ip_analysis.py"],
            },
        ]
        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": path_vector,
                "FacilitySignal": crossrefs,
                "UNWIND $path_ids": [_IMAS_ENRICHMENT_IP],
            }
        )

        result = _search_imas(  # noqa: F821
            query="plasma current",
            facility="tcv",
            gc=mock_gc,
            encoder=mock_encoder,
        )

        assert "tcv:magnetics/ip" in result
        assert "Plasma Current" in result
        assert "ip_analysis.py" in result

    def test_version_context(self, mock_gc, mock_encoder):
        """Version context is fetched and formatted when requested."""
        path_vector = [
            {"id": "magnetics.ip.0d[:].value", "score": 0.92},
        ]
        version_ctx = [
            {
                "id": "magnetics.ip.0d[:].value",
                "change_count": 2,
                "notable_changes": [
                    {
                        "version": "3.20.0",
                        "type": "sign_convention",
                        "summary": "Sign convention clarified for COCOS",
                    },
                ],
            },
        ]
        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": path_vector,
                "IMASNodeChange": version_ctx,
                "UNWIND $path_ids": [_IMAS_ENRICHMENT_IP],
            }
        )

        result = _search_imas(  # noqa: F821
            query="plasma current",
            include_version_context=True,
            gc=mock_gc,
            encoder=mock_encoder,
        )

        assert "Version history" in result
        assert "sign_convention" in result

    def test_embedding_unavailable(self, mock_gc):
        """When encoder raises, return helpful error message."""
        from imas_codex.embeddings.encoder import EmbeddingBackendError

        bad_encoder = MagicMock()
        bad_encoder.embed_texts.side_effect = EmbeddingBackendError("unavailable")

        result = _search_imas(query="test", gc=mock_gc, encoder=bad_encoder)  # noqa: F821
        assert "Embedding" in result or "unavailable" in result.lower()

    def test_neo4j_unavailable(self, mock_encoder):
        """When Neo4j is down, return helpful error message."""
        bad_gc = MagicMock()
        bad_gc.query.side_effect = ServiceUnavailable("Connection refused")

        result = _search_imas(query="test", gc=bad_gc, encoder=mock_encoder)  # noqa: F821
        assert "not running" in result.lower() or "neo4j" in result.lower()

    def test_hybrid_text_imas(self, mock_gc, mock_encoder):
        """IMAS paths found by both vector and text search get boosted."""
        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [
                    {"id": "magnetics.ip.0d[:].value", "score": 0.80}
                ],
                "documentation) CONTAINS": [
                    {"id": "magnetics.ip.0d[:].value", "score": 0.7}
                ],
                "UNWIND $path_ids": [_IMAS_ENRICHMENT_IP],
            }
        )

        result = _search_imas(  # noqa: F821
            query="plasma current",
            gc=mock_gc,
            encoder=mock_encoder,
        )
        assert "magnetics.ip.0d[:].value" in result


class TestFormatImasReport:
    """Unit tests for the IMAS report formatter."""

    def test_empty_results(self):
        """Empty results produce a descriptive message."""
        result = format_imas_report([], [], {}, {}, {})
        assert "No IMAS paths found" in result

    def test_path_formatting(self):
        """A single IMAS path is formatted with all metadata."""
        paths = [_IMAS_ENRICHMENT_TEMP]
        scores = {"core_profiles.profiles_1d[:].electrons.temperature": 0.95}

        result = format_imas_report(paths, [], {}, {}, scores)

        assert "core_profiles.profiles_1d[:].electrons.temperature" in result
        assert "0.95" in result
        assert "eV" in result
        assert "core_transport" in result
        assert "Electron Temperature Profiles" in result
        assert "3.0.0" in result

    def test_lifecycle_status_shown(self):
        """Non-active lifecycle status is displayed."""
        paths = [
            {
                **_IMAS_ENRICHMENT_IP,
                "lifecycle_status": "obsolescent",
            }
        ]
        scores = {"magnetics.ip.0d[:].value": 0.9}

        result = format_imas_report(paths, [], {}, {}, scores)
        assert "obsolescent" in result

    def test_active_lifecycle_status_hidden(self):
        """Active lifecycle status is NOT displayed (default state)."""
        paths = [_IMAS_ENRICHMENT_IP]
        scores = {"magnetics.ip.0d[:].value": 0.9}

        result = format_imas_report(paths, [], {}, {}, scores)
        assert "Lifecycle" not in result

    def test_structure_reference_shown(self):
        """Structure reference URL is displayed when present."""
        paths = [
            {
                **_IMAS_ENRICHMENT_IP,
                "structure_reference": "https://imas.io/docs/magnetics/ip",
            }
        ]
        scores = {"magnetics.ip.0d[:].value": 0.9}

        result = format_imas_report(paths, [], {}, {}, scores)
        assert "Structure" in result
        assert "https://imas.io/docs/magnetics/ip" in result

    def test_cluster_formatting(self):
        """Clusters appear in Related Clusters section."""
        clusters = [
            {
                "id": "cluster:plasma_current",
                "label": "Plasma Current Measurements",
                "scope": "magnetics",
                "path_count": 8,
                "sample_paths": [
                    "magnetics.ip.0d[:].value",
                    "magnetics.ip.measure[:].value",
                ],
                "score": 0.88,
            },
        ]
        scores = {"cluster:plasma_current": 0.88}

        result = format_imas_report([], clusters, {}, {}, scores)

        assert "Related Clusters" in result
        assert "Plasma Current Measurements" in result
        assert "magnetics" in result

    def test_cluster_deduplication(self):
        """Duplicate clusters by label are deduplicated."""
        clusters = [
            {
                "id": "cluster:a",
                "label": "Electron Temp",
                "scope": "core_profiles",
                "path_count": 5,
                "sample_paths": [],
                "score": 0.9,
            },
            {
                "id": "cluster:b",
                "label": "Electron Temp",
                "scope": "global",
                "path_count": 3,
                "sample_paths": [],
                "score": 0.85,
            },
        ]
        scores = {"cluster:a": 0.9, "cluster:b": 0.85}

        result = format_imas_report([], clusters, {}, {}, scores)
        # Label should appear only once in the "Related Clusters" section
        cluster_section = result.split("## Related Clusters")[1]
        assert cluster_section.count('"Electron Temp"') == 1

    def test_facility_xrefs_shown(self):
        """Facility cross-references appear in path output."""
        paths = [_IMAS_ENRICHMENT_IP]
        xrefs = {
            "magnetics.ip.0d[:].value": {
                "id": "magnetics.ip.0d[:].value",
                "facility_signals": ["tcv:magnetics/ip"],
                "wiki_mentions": ["Current Measurements"],
                "code_files": ["/codes/ip.py"],
            },
        }
        scores = {"magnetics.ip.0d[:].value": 0.9}

        result = format_imas_report(paths, [], xrefs, {}, scores)

        assert "tcv:magnetics/ip" in result
        assert "Current Measurements" in result
        assert "/codes/ip.py" in result


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


class TestFetch:
    """Unit tests for the fetch tool."""

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    def test_wiki_page_fetch(self, mock_gc):
        """Fetch a wiki page by ID returns all chunks."""
        wiki_chunks = [
            {
                "source_type": "wiki_page",
                "title": "Equilibrium",
                "url": "https://wiki.example.com/Equilibrium",
                "source_id": "tcv:Equilibrium",
                "section": "Introduction",
                "text": "This page describes equilibrium reconstruction.",
                "chunk_index": 0,
                "mdsplus_paths": None,
                "imas_paths": None,
            },
            {
                "source_type": "wiki_page",
                "title": "Equilibrium",
                "url": "https://wiki.example.com/Equilibrium",
                "source_id": "tcv:Equilibrium",
                "section": "Methods",
                "text": "LIUQE is the main equilibrium code at TCV.",
                "chunk_index": 1,
                "mdsplus_paths": ["\\results::i_p"],
                "imas_paths": ["equilibrium.time_slice[:].global_quantities.ip"],
            },
        ]
        mock_gc.query.side_effect = _route_query({"WikiPage": wiki_chunks})

        result = _fetch("tcv:Equilibrium", gc=mock_gc)

        assert "Wiki Page: Equilibrium" in result
        assert "Chunks: 2" in result
        assert "equilibrium reconstruction" in result
        assert "LIUQE" in result
        assert "MDSplus paths" in result
        assert "IMAS paths" in result

    def test_code_file_fetch(self, mock_gc):
        """Fetch a code file by ID returns code chunks."""
        code_chunks = [
            {
                "source_type": "code",
                "title": "/home/codes/liuqe.py",
                "source_id": "tcv:/home/codes/liuqe.py",
                "url": "/home/codes/liuqe.py",
                "section": "solve_equilibrium",
                "text": "def solve_equilibrium(shot):\n    pass",
                "chunk_index": 10,
                "mdsplus_paths": None,
                "imas_paths": None,
            },
        ]
        mock_gc.query.side_effect = _route_query({"CodeExample": code_chunks})

        result = _fetch("tcv:/home/codes/liuqe.py", gc=mock_gc)

        assert "Code File" in result
        assert "solve_equilibrium" in result
        assert "def solve_equilibrium" in result

    def test_image_fetch(self, mock_gc):
        """Fetch an image returns description and metadata."""
        image_results = [
            {
                "source_type": "image",
                "title": "Magnetic field topology",
                "url": "https://wiki.example.com/images/topo.png",
                "source_id": "tcv:topo.png",
                "description": "Diagram showing the magnetic field line topology.",
                "ocr_text": "B_pol = 1.5 T",
                "mermaid": None,
                "keywords": ["magnetics", "topology"],
                "width": 800,
                "height": 600,
                "parent_pages": ["Magnetics Overview"],
            },
        ]
        mock_gc.query.side_effect = _route_query({"(img:Image)": image_results})

        result = _fetch("tcv:topo.png", gc=mock_gc)

        assert "Image: Magnetic field topology" in result
        assert "800×600" in result
        assert "Magnetics Overview" in result
        assert "magnetic field line topology" in result
        assert "B_pol = 1.5 T" in result

    def test_no_match(self, mock_gc):
        """No matches returns guidance message."""
        result = _fetch("nonexistent", gc=mock_gc)

        assert "No resource found" in result
        assert "search_docs" in result

    def test_neo4j_not_running(self):
        """ServiceUnavailable returns descriptive message."""
        with patch(
            "imas_codex.llm.search_tools.GraphClient",
            side_effect=ServiceUnavailable("Connection refused"),
        ):
            result = _fetch("anything")
            assert "Neo4j is not running" in result


class TestFormatFetchReport:
    """Tests for format_fetch_report formatter."""

    def test_empty_chunks(self):
        assert format_fetch_report([]) == "No content found."

    def test_wiki_page_format(self):
        chunks = [
            {
                "source_type": "wiki_page",
                "title": "Diagnostics",
                "url": "https://wiki.example.com/Diagnostics",
                "source_id": "tcv:Diagnostics",
                "section": "Overview",
                "text": "TCV has many diagnostics.",
                "chunk_index": 0,
                "mdsplus_paths": None,
                "imas_paths": ["diagnostics"],
            },
        ]
        result = format_fetch_report(chunks)
        assert "Wiki Page: Diagnostics" in result
        assert "ID: tcv:Diagnostics" in result
        assert "URL: https://wiki.example.com/Diagnostics" in result
        assert "TCV has many diagnostics" in result
        assert "IMAS paths: diagnostics" in result

    def test_code_format_with_line_numbers(self):
        chunks = [
            {
                "source_type": "code",
                "title": "/codes/eq.py",
                "source_id": "tcv:/codes/eq.py",
                "url": "/codes/eq.py",
                "section": "main",
                "text": "import sys",
                "chunk_index": 1,
                "mdsplus_paths": None,
                "imas_paths": None,
            },
        ]
        result = format_fetch_report(chunks)
        assert "Code File" in result
        assert "Line 1" in result
        assert "```\nimport sys\n```" in result

    def test_cross_references_aggregated(self):
        """MDSplus and IMAS paths are aggregated across all chunks."""
        chunks = [
            {
                "source_type": "wiki_page",
                "title": "Signals",
                "url": "",
                "source_id": "tcv:Signals",
                "section": "Section A",
                "text": "First chunk.",
                "chunk_index": 0,
                "mdsplus_paths": ["\\results::i_p"],
                "imas_paths": ["magnetics.ip"],
            },
            {
                "source_type": "wiki_page",
                "title": "Signals",
                "url": "",
                "source_id": "tcv:Signals",
                "section": "Section B",
                "text": "Second chunk.",
                "chunk_index": 1,
                "mdsplus_paths": ["\\results::b_tor"],
                "imas_paths": ["magnetics.ip"],
            },
        ]
        result = format_fetch_report(chunks)
        assert "Cross-references" in result
        assert "\\results::i_p" in result
        assert "\\results::b_tor" in result
        assert "magnetics.ip" in result


class TestDocsFetchHints:
    """Tests that search_docs results include fetch hints."""

    def test_page_includes_fetch_hint(self):
        chunks = [
            {
                "id": "tcv:Diagnostics:chunk_0",
                "text": "Some text",
                "section": "Overview",
                "page_id": "tcv:Diagnostics",
                "page_title": "Diagnostics",
                "page_url": "https://wiki.example.com",
                "linked_signals": [],
                "linked_data_nodes": [],
                "imas_refs": [],
            },
        ]
        scores = {"tcv:Diagnostics:chunk_0": 0.9}
        result = format_docs_report(chunks, [], scores)
        assert "fetch('tcv:Diagnostics')" in result

    def test_document_includes_fetch_hint(self):
        documents = [
            {
                "id": "jet:fishbone.ppt",
                "title": "Fishbone Presentation",
                "page_title": "Instabilities",
                "description": "A presentation",
            },
        ]
        scores = {"jet:fishbone.ppt": 0.88}
        result = format_docs_report([], documents, scores)
        assert "fetch('jet:fishbone.ppt')" in result


class TestCodeFetchHints:
    """Tests that search_code results include fetch hints."""

    def test_code_includes_source_file_fetch_hint(self):
        code_results = [
            {
                "id": "tcv:liuqe:chunk_0",
                "text": "def solve(): pass",
                "function_name": "solve",
                "source_file": "/codes/liuqe.py",
                "source_file_id": "tcv:/codes/liuqe.py",
                "facility_id": "tcv",
                "data_refs": [],
                "directory": "/codes",
                "dir_description": None,
            },
        ]
        scores = {"tcv:liuqe:chunk_0": 0.85}
        result = format_code_report(code_results, scores)
        assert "fetch('tcv:/codes/liuqe.py')" in result


# ---------------------------------------------------------------------------
# Template interpolation
# ---------------------------------------------------------------------------


class TestInterpolateTemplate:
    """Tests for _interpolate_template in the formatter."""

    def test_node_path_substitution(self):
        sig = {
            "node_path": "\\RESULTS::I_P",
            "accessor": None,
            "data_source_name": None,
        }
        result = _interpolate_template("getNode('{node_path}')", sig)
        assert result == "getNode('\\RESULTS::I_P')"

    def test_accessor_substitution(self):
        sig = {"node_path": None, "accessor": "MDSplus", "data_source_name": None}
        result = _interpolate_template("Use {accessor} library", sig)
        assert result == "Use MDSplus library"

    def test_data_source_from_data_source_name_primary(self):
        sig = {
            "node_path": None,
            "accessor": None,
            "data_source_name": "tcv_shot",
        }
        result = _interpolate_template("Tree('{data_source}', shot)", sig)
        assert result == "Tree('tcv_shot', shot)"

    def test_data_source_from_data_source_name(self):
        sig = {
            "node_path": None,
            "accessor": None,
            "data_source_name": "tcv_shot",
        }
        result = _interpolate_template("Tree('{data_source}', shot)", sig)
        assert result == "Tree('tcv_shot', shot)"

    def test_shot_placeholder_preserved(self):
        """The {shot} placeholder should NOT be substituted."""
        sig = {"node_path": None, "accessor": None, "data_source_name": None}
        result = _interpolate_template("Tree('tree', {shot})", sig)
        assert result == "Tree('tree', {shot})"

    def test_missing_values_kept(self):
        """Placeholders with no matching data are kept as-is."""
        sig = {"node_path": None, "accessor": None, "data_source_name": None}
        result = _interpolate_template("{node_path}", sig)
        assert result == "{node_path}"

    def test_multiple_substitutions(self):
        """Multiple placeholders are all substituted."""
        sig = {
            "node_path": "\\IP",
            "accessor": "MDSplus",
            "data_source_name": "tcv_shot",
        }
        template = (
            "t = {accessor}.Tree('{data_source}', shot)\nn = t.getNode('{node_path}')"
        )
        result = _interpolate_template(template, sig)
        assert "MDSplus.Tree('tcv_shot', shot)" in result
        assert "t.getNode('\\IP')" in result


# ---------------------------------------------------------------------------
# Schema guard: ensure search functions use correct property names
# ---------------------------------------------------------------------------


class TestSchemaGuard:
    """Tests that search queries reference correct schema properties.

    These tests verify that the Cypher queries generated by search tools
    match the actual graph schema, catching schema drift early.
    """

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def test_signal_enrichment_uses_access_methods(self, mock_gc, mock_encoder):
        """Signal enrichment aggregates DataAccess into access_methods."""
        mock_gc.query.side_effect = _route_query(
            {
                "facility_signal_desc_embedding": [{"id": "tcv:test/s", "score": 0.9}],
            }
        )
        _search_signals(query="test", facility="tcv", gc=mock_gc, encoder=mock_encoder)
        # Find the enrichment call
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "UNWIND $signal_ids" in cypher:
                # Should use collect(DISTINCT { ... }) AS access_methods
                assert "access_methods" in cypher
                # Should traverse DataAccess → IMASNode
                assert "DATA_ACCESS" in cypher
                assert "MAPS_TO_IMAS" in cypher
                break

    def test_code_enrichment_uses_code_example(self, mock_gc, mock_encoder):
        """Code enrichment traverses CodeExample -[:HAS_CHUNK]-> CodeChunk."""
        mock_gc.query.side_effect = _route_query(
            {
                "code_chunk_embedding": [{"id": "cc:1", "score": 0.9}],
            }
        )
        _search_code(query="test", facility="tcv", gc=mock_gc, encoder=mock_encoder)
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "CodeChunk {id: cid}" in cypher:
                # Uses HAS_CHUNK reversed (CodeExample -[:HAS_CHUNK]-> CodeChunk)
                assert "HAS_CHUNK" in cypher
                assert "CodeExample" in cypher
                # Data refs via CodeChunk
                assert "CONTAINS_REF" in cypher
                break

    def test_code_vector_uses_facility_id_property(self, mock_gc, mock_encoder):
        """Code vector search filters facility via CodeChunk.facility_id property."""
        _search_code(query="test", facility="tcv", gc=mock_gc, encoder=mock_encoder)
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "code_chunk_embedding" in cypher:
                assert "cc.facility_id" in cypher
                break
        else:
            pytest.fail("No code vector search call found")

    def test_imas_enrichment_includes_lifecycle_fields(self, mock_gc, mock_encoder):
        """IMAS enrichment returns lifecycle_status and structure_reference."""
        from imas_codex.tools.graph_search import GraphSearchTool

        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [
                    {"id": "magnetics.ip.0d[:].value", "score": 0.9}
                ],
            }
        )
        import asyncio

        tool = GraphSearchTool(mock_gc)
        tool._embed_query = lambda q: [0.1] * 1024
        asyncio.run(tool.search_dd_paths(query="ip"))
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "UNWIND $path_ids" in cypher:
                assert "lifecycle_status" in cypher
                assert "structure_reference" in cypher or "path_doc" in cypher
                break

    def test_facility_crossrefs_uses_property_match(self, mock_gc, mock_encoder):
        """Facility crossrefs use property-based matching, not relationship traversal."""
        from imas_codex.tools.graph_search import GraphSearchTool

        mock_gc.query.side_effect = _route_query(
            {
                "imas_node_embedding": [
                    {"id": "magnetics.ip.0d[:].value", "score": 0.9}
                ],
                "UNWIND $path_ids": [_IMAS_ENRICHMENT_IP],
            }
        )
        import asyncio

        tool = GraphSearchTool(mock_gc)
        tool._embed_query = lambda q: [0.1] * 1024
        asyncio.run(tool.search_dd_paths(query="ip", facility="tcv"))
        for call in mock_gc.query.call_args_list:
            cypher = call[0][0]
            if "FacilitySignal" in cypher and "WikiChunk" in cypher:
                # Uses property-based facility filter
                assert "facility_id" in cypher
                # CodeChunk matched via related_ids property
                assert "CodeChunk" in cypher
                break


# ---------------------------------------------------------------------------
# Dynamic handbook pages (WikiPage.fronts_database) and the DOCUMENTS edges
# ---------------------------------------------------------------------------


class TestHandbookDocumentEdges:
    """The MCP tools follow the dynamic handbook page -> signal edges both ways."""

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def test_fronts_database_page_lists_signals(self, mock_gc):
        """fetch of a dynamic page renders the catalogue it fronts, capped."""
        page_chunks = [
            {
                "source_type": "wiki_page",
                "title": "Category Information",
                "url": "https://wiki.jt60sa.org/CategoryInformation",
                "source_id": "jt-60sa:CategoryInformation",
                "fronts_database": ["EDDB"],
                "section": None,
                "text": "Dynamic table rendered from the EDDB catalogue.",
                "chunk_index": 0,
                "mdsplus_paths": None,
                "imas_paths": None,
            }
        ]
        fronted_rows = [
            {
                "category": "parameter",
                "total": 6298,
                "samples": [
                    {
                        "name": "PSRC/magFluxLp1",
                        "unit": "Wb",
                        "pid": None,
                        "access": "ok, rtn = db.eddbreadTime('{shot}', ...)",
                    }
                ],
            }
        ]
        mock_gc.query.side_effect = _route_query(
            {
                "RETURN 'wiki_page' AS source_type": page_chunks,
                "p.id IN $page_ids": fronted_rows,
            }
        )

        result = _fetch_wiki_page(mock_gc, "jt-60sa:CategoryInformation")

        assert "Fronted signal catalogue (EDDB)" in result
        assert "**parameter** — 6298 signals" in result
        assert "PSRC/magFluxLp1" in result
        # The stub text is still shown above the catalogue listing.
        assert "Dynamic table rendered" in result

    def test_signal_shows_documented_page_citation(self):
        """A signal reached by DOCUMENTS cites the page that fronts its catalogue."""
        sig = dict(
            _SIGNAL_ENRICHMENT_IP,
            documented_pages=[
                {"title": "Category Information", "id": "jt-60sa:CategoryInformation"}
            ],
        )
        result = format_signals_report([sig], [], {"tcv:magnetics/ip": 0.9})

        assert (
            'Documented in: "Category Information" (jt-60sa:CategoryInformation)'
            in result
        )

    def test_stub_chunk_enrichment_query_is_capped(self, mock_gc):
        """The chunk enrichment caps its DOCUMENTS collection per chunk.

        A dynamic stub chunk documents a whole catalogue, so the query returns
        a bounded sample plus the true count rather than every signal id.
        """
        _enrich_wiki_chunks(mock_gc, ["jt-60sa:chunk:stub"])

        cypher = mock_gc.query.call_args[0][0]
        kwargs = mock_gc.query.call_args[1]
        assert "size(all_signals) AS signal_count" in cypher
        assert "all_signals[..$documents_cap]" in cypher
        assert kwargs.get("documents_cap") == _CHUNK_DOCUMENTS_CAP

    def test_stub_chunk_report_shows_count_not_full_list(self):
        """A capped chunk renders a total and a sample, not thousands of ids."""
        rows = [
            {
                "id": "jt-60sa:chunk:stub",
                "text": "Dynamic table rendered from the EDDB catalogue.",
                "section": None,
                "page_id": "jt-60sa:CategoryInformation",
                "page_title": "Category Information",
                "page_url": None,
                "signal_count": 6298,
                "linked_signals": ["jt-60sa:general/psrc_magfluxlp1"],
                "linked_data_nodes": [],
                "imas_refs": [],
                "tool_mentions": None,
            }
        ]
        result = format_docs_report(rows, [], {})

        assert "Fronts 6298 signals; e.g. jt-60sa:general/psrc_magfluxlp1" in result

    def test_docs_report_lists_matching_signals(self):
        """Documentation search names the signal family it reaches.

        The signals section sits beside the wiki/document prose, so a search
        that names a signal family returns the signals and the pages together.
        """
        signals = [
            {
                "id": "jt-60sa:general/psrc_magfluxlp1",
                "name": "PSRC/magFluxLp1",
                "description": "Observation of Flux Loop (1-7)",
            }
        ]
        result = format_docs_report([], [], {}, signals=signals)

        assert "## Signals (1 matches)" in result
        assert "PSRC/magFluxLp1" in result
        assert "jt-60sa:general/psrc_magfluxlp1" in result

    def test_search_docs_includes_matching_signals(self, mock_gc, mock_encoder):
        """search_docs surfaces signals whose name/description match the query."""
        mock_gc.query.side_effect = _route_query(
            {
                "wiki_chunk_embedding": [{"id": "c1", "score": 0.9}],
                "WikiChunk {id: cid}": [
                    {
                        "id": "c1",
                        "text": "EDDB manual table excerpt",
                        "section": "General",
                        "page_title": "EDDB Manual",
                        "page_url": None,
                        "linked_signals": [],
                        "linked_data_nodes": [],
                        "imas_refs": [],
                    }
                ],
                "facility_signal_text": [
                    {"id": "jt-60sa:general/psrc_magfluxlp1", "score": 6.5}
                ],
                "FacilitySignal {id: sid}": [
                    {
                        "id": "jt-60sa:general/psrc_magfluxlp1",
                        "name": "PSRC/magFluxLp1",
                        "description": "Observation of Flux Loop (1-7)",
                    }
                ],
            }
        )

        result = _search_docs(
            query="flux loop",
            facility="jt-60sa",
            gc=mock_gc,
            encoder=mock_encoder,
        )

        assert "EDDB manual table excerpt" in result
        assert "## Signals" in result
        assert "PSRC/magFluxLp1" in result


class _FulltextSignatureClient:
    """A GraphClient whose query() mirrors Session.run's own signature.

    ``Session.run(query, **params)`` binds its first positional argument to a
    parameter named ``query``, so a caller that also binds a Cypher parameter
    named ``query`` raises ``TypeError`` before the query runs. A stub with a
    plain ``query(cypher, **params)`` signature cannot reproduce that collision
    and would let a regression pass, so this stub reproduces the real one.
    """

    def __init__(self, hits, marker):
        self._hits = hits
        self._marker = marker
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        if self._marker in query:
            return list(self._hits)
        return []


class TestFulltextBranchParameterBinding:
    """The fulltext branches bind search text without colliding with Session.run.

    A parameter named ``query`` collides with ``Session.run``'s own argument,
    so the call raised and the branch was silently dead. These tests drive each
    repaired branch through a stub with the real signature, so binding the text
    to ``query`` again fails rather than falling through unnoticed.
    """

    def test_signal_fulltext_branch_binds_search_query(self):
        hits = [{"id": "jt-60sa:general/psrc_magfluxlp1", "score": 6.5}]
        gc = _FulltextSignatureClient(hits, marker="facility_signal_text")

        result = _text_search_signals(gc, "flux loop", "jt-60sa", 8)

        assert result == hits
        cypher, params = gc.calls[0]
        assert "$search_query" in cypher
        assert params["search_query"] == "flux loop"
        assert "query" not in params

    def test_code_chunk_fulltext_branch_binds_search_query(self):
        hits = [{"id": "jt-60sa:code:c1", "score": 3.5}]
        gc = _FulltextSignatureClient(hits, marker="code_chunk_text")

        result = _text_search_code_chunks(gc, "flux loop", "jt-60sa", 8)

        assert result == hits
        cypher, params = gc.calls[0]
        assert "$search_query" in cypher
        assert params["search_query"] == "flux loop"
        assert "query" not in params

    def test_wiki_chunk_fulltext_branch_binds_search_query(self):
        hits = [{"id": "jt-60sa:wiki:c1", "score": 4.2}]
        gc = _FulltextSignatureClient(hits, marker="wiki_chunk_text")

        result = _text_search_wiki_chunks(gc, "flux loop", "jt-60sa", 8)

        assert result == hits
        cypher, params = gc.calls[0]
        assert "$search_query" in cypher
        assert params["search_query"] == "flux loop"
        assert "query" not in params


class _FulltextRaisingClient:
    """Raises a programming error on the fulltext call and answers elsewhere."""

    def __init__(self, marker):
        self._marker = marker

    def query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        if self._marker in query:
            raise TypeError("Session.run() got multiple values for argument 'query'")
        return []


class _AbsentIndexClient:
    """Raises a client error for the fulltext index, as an absent index does."""

    def __init__(self):
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def query(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        if "queryNodes" in query:
            raise ClientError(
                "There is no procedure with the name "
                "`db.index.fulltext.queryNodes` registered for this database."
            )
        return [{"id": "jt-60sa:general/psrc_magfluxlp1", "score": 0.6}]


class TestFulltextFailureHandling:
    """A programming error propagates; an absent index still falls back."""

    def test_signal_fulltext_programming_error_propagates(self):
        with pytest.raises(TypeError):
            _text_search_signals(
                _FulltextRaisingClient("facility_signal_text"),
                "flux loop",
                "jt-60sa",
                8,
            )

    def test_code_chunk_fulltext_programming_error_propagates(self):
        with pytest.raises(TypeError):
            _text_search_code_chunks(
                _FulltextRaisingClient("code_chunk_text"), "flux loop", "jt-60sa", 8
            )

    def test_wiki_chunk_fulltext_programming_error_propagates(self):
        with pytest.raises(TypeError):
            _text_search_wiki_chunks(
                _FulltextRaisingClient("wiki_chunk_text"), "flux loop", "jt-60sa", 8
            )

    def test_signal_missing_index_falls_back_to_contains(self):
        gc = _AbsentIndexClient()

        result = _text_search_signals(gc, "flux loop", "jt-60sa", 8)

        assert result == [{"id": "jt-60sa:general/psrc_magfluxlp1", "score": 0.6}]
        assert "CONTAINS" in gc.calls[-1][0]

    def test_code_chunk_missing_index_falls_back_to_contains(self):
        gc = _AbsentIndexClient()

        result = _text_search_code_chunks(gc, "flux loop", "jt-60sa", 8)

        assert result == [{"id": "jt-60sa:general/psrc_magfluxlp1", "score": 0.6}]
        assert "CONTAINS" in gc.calls[-1][0]

    def test_wiki_chunk_missing_index_falls_back_to_contains(self):
        gc = _AbsentIndexClient()

        result = _text_search_wiki_chunks(gc, "flux loop", "jt-60sa", 8)

        assert result == [{"id": "jt-60sa:general/psrc_magfluxlp1", "score": 0.6}]
        assert "CONTAINS" in gc.calls[-1][0]


class TestCodeChunkSourceResolution:
    """Each code-chunk reader resolves its source through the one declared edge.

    A chunk reaches the file it came from through the owning CodeExample's
    ``HAS_CHUNK`` edge, so every reader embeds ``render_chunk_source`` rather
    than an inline join or coalesce the schema does not back. These pin the
    attribute each reader reads so the readers cannot diverge again.
    """

    def test_enrich_code_chunks_embeds_renderer(self):
        gc = MagicMock()
        gc.query = MagicMock(return_value=[])

        _enrich_code_chunks(gc, ["jt-60sa:general/psrc_magflux:chunk-1"])

        cypher = gc.query.call_args[0][0]
        assert render_chunk_source("cc", "source_file") in cypher
        # exactly one coalesce over source_file: the renderer's, not a second arm
        assert cypher.count("coalesce(ce.source_file, cc.source_file)") == 1
        assert "coalesce(ce.source_file, cc.source_file) AS source_file" in cypher
        assert "source_file,\n" in cypher
        assert "CODE_EXAMPLE_ID" not in cypher

    def test_code_chunk_search_embeds_renderer(self):
        from imas_codex.llm.server import _code_chunk_search_query

        cypher = _code_chunk_search_query()
        assert render_chunk_source("node", "source_file") in cypher
        assert "source_file, node.facility_id AS source_facility" in cypher
        assert "CodeFile" not in cypher


# ---------------------------------------------------------------------------
# Rerank
# ---------------------------------------------------------------------------


def _score_answers(score: int) -> dict[str, Any]:
    return {
        "relevance_grade": {
            "type": "score",
            "score": score,
            "probabilities": {
                str(level): (1.0 if level == score else 0.0) for level in range(1, 6)
            },
            "confidence": 0.9,
        }
    }


def _fake_decisions(
    by_path: dict[str, int],
    *,
    cost: float = 0.001,
    stall: tuple[str, ...] = (),
    fail: tuple[str, ...] = (),
) -> Any:
    """Async stand-in for acall_decisions, keyed by the candidate's path."""

    async def fake(model, state, questions, *, service=None, **_kwargs):
        path = state["candidate"]["locator"]["path"]
        if path in stall:
            await asyncio.sleep(30)
        if path in fail:
            raise RuntimeError("the endpoint refused every attempt")
        return _score_answers(by_path[path]), cost

    return fake


class TestRerankCandidates:
    """The rerank reorders by Score and never fails the search it serves."""

    @pytest.fixture(autouse=True)
    def _decisions_key(self, monkeypatch):
        """Set a decisions key so the replaced judgement is actually consulted.

        The rerank skips every call when the key is absent, which would make
        these tests measure the skip rather than the ordering they replace.
        """
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-key")

    def _candidates(self) -> list[dict[str, Any]]:
        return [
            {"id": "c0", "path": "alpha.py", "function_name": "alpha", "text": "A"},
            {"id": "c1", "path": "beta.py", "function_name": "beta", "text": "B"},
            {"id": "c2", "path": "gamma.py", "function_name": "gamma", "text": "C"},
        ]

    def test_higher_score_ranks_first(self, monkeypatch):
        from imas_codex.discovery.base import judgment

        monkeypatch.setattr(
            judgment,
            "acall_decisions",
            _fake_decisions({"alpha.py": 2, "beta.py": 5, "gamma.py": 1}),
        )

        ordered, note = rerank_candidates("q", self._candidates())

        assert [c["path"] for c in ordered] == ["beta.py", "alpha.py", "gamma.py"]
        assert note is None

    def test_state_carries_the_locator_and_bounds_the_text(self, monkeypatch):
        from imas_codex.discovery.base import judgment

        seen: list[dict[str, Any]] = []

        async def fake(model, state, questions, *, service=None, **_kwargs):
            seen.append(state)
            return _score_answers(3), 0.0

        monkeypatch.setattr(judgment, "acall_decisions", fake)

        candidates = [
            {
                "id": "c0",
                "path": "alpha.py",
                "function_name": "alpha",
                "text": "x" * 2000,
            }
        ]

        rerank_candidates("how does the rerank work", candidates)

        assert len(seen) == 1
        state = seen[0]
        assert state["query"] == "how does the rerank work"
        assert state["candidate"]["locator"] == {
            "path": "alpha.py",
            "title": "alpha",
        }
        assert len(state["candidate"]["text"]) == 1500

    def test_no_candidates_returns_an_empty_list_without_a_note(self):
        ordered, note = rerank_candidates("q", [])

        assert ordered == []
        assert note is None

    def test_a_failed_decision_keeps_the_embedding_order_and_notes_it(
        self, monkeypatch
    ):
        from imas_codex.discovery.base import judgment

        monkeypatch.setattr(
            judgment,
            "acall_decisions",
            _fake_decisions({}, fail=("alpha.py", "beta.py", "gamma.py")),
        )

        ordered, note = rerank_candidates("q", self._candidates())

        assert [c["path"] for c in ordered] == ["alpha.py", "beta.py", "gamma.py"]
        assert note is not None
        assert "3 of 3" in note
        assert "embedding position" in note

    def test_a_candidate_unfinished_at_the_budget_keeps_its_position(self, monkeypatch):
        from imas_codex.discovery.base import judgment

        monkeypatch.setattr(
            judgment,
            "acall_decisions",
            _fake_decisions({"alpha.py": 1, "gamma.py": 5}, stall=("beta.py",)),
        )

        ordered, note = rerank_candidates("q", self._candidates(), budget_seconds=0.05)

        assert [c["path"] for c in ordered] == ["gamma.py", "beta.py", "alpha.py"]
        assert note is not None
        assert "1 of 3" in note

    def test_a_failed_batch_returns_the_embedding_order_with_a_note(self, monkeypatch):
        from imas_codex.llm import search_tools

        def boom(coro):
            coro.close()
            raise RuntimeError("no decisions transport")

        monkeypatch.setattr(search_tools, "_run_batch", boom)

        ordered, note = rerank_candidates("q", self._candidates())

        assert [c["path"] for c in ordered] == ["alpha.py", "beta.py", "gamma.py"]
        assert note is not None
        assert "embedding order returned" in note


class TestSearchToolsRerank:
    """The search tools score a pool larger than k, then rerank it."""

    @pytest.fixture(autouse=True)
    def _decisions_key(self, monkeypatch):
        """Set a decisions key so the search reaches the rerank it asserts on."""
        monkeypatch.setenv("OPENROUTER_API_KEY_IMAS_CODEX", "test-key")

    @pytest.fixture()
    def mock_gc(self):
        gc = MagicMock()
        gc.query = MagicMock(side_effect=_route_query({}))
        return gc

    @pytest.fixture()
    def mock_encoder(self):
        enc = MagicMock()
        enc.embed_texts = MagicMock(return_value=[[0.1] * 1024])
        return enc

    def _wiki_handler(self, ids: list[str]) -> Any:
        chunks = [
            {
                "id": cid,
                "text": f"body {cid}",
                "section": cid,
                "page_title": cid,
                "page_url": None,
                "linked_signals": [],
                "linked_signal_nodes": [],
                "linked_data_nodes": [],
                "imas_refs": [],
            }
            for cid in ids
        ]
        by_id = {chunk["id"]: chunk for chunk in chunks}
        vector = [{"id": cid, "score": 0.9 - 0.1 * i} for i, cid in enumerate(ids)]

        def handler(cypher: str, **kwargs: Any) -> list[dict[str, Any]]:
            if "wiki_chunk_embedding" in cypher:
                return vector
            if "MATCH (c:WikiChunk {id: cid})" in cypher:
                return [by_id[cid] for cid in kwargs["chunk_ids"]]
            return []

        return handler

    def _code_handler(self, ids: list[str]) -> Any:
        chunks = [
            {
                "id": cid,
                "text": f"def f_{cid}():\n    pass",
                "function_name": f"f_{cid}",
                "source_file": f"{cid}.py",
                "source_file_id": f"tcv:/code/{cid}.py",
                "facility_id": "tcv",
                "data_refs": [],
                "directory": None,
                "dir_description": None,
            }
            for cid in ids
        ]
        by_id = {chunk["id"]: chunk for chunk in chunks}
        vector = [{"id": cid, "score": 0.9 - 0.1 * i} for i, cid in enumerate(ids)]

        def handler(cypher: str, **kwargs: Any) -> list[dict[str, Any]]:
            if "code_chunk_embedding" in cypher:
                return vector
            if "MATCH (cc:CodeChunk {id: cid})" in cypher:
                return [by_id[cid] for cid in kwargs["chunk_ids"]]
            return []

        return handler

    def test_docs_report_follows_the_rerank_order(
        self, mock_gc, mock_encoder, monkeypatch
    ):
        from imas_codex.discovery.base import judgment

        mock_gc.query = MagicMock(side_effect=self._wiki_handler(["c0", "c1", "c2"]))
        monkeypatch.setattr(
            judgment,
            "acall_decisions",
            _fake_decisions({"c2": 5, "c0": 3, "c1": 1}),
        )

        result = _search_docs(
            query="equilibrium", facility="tcv", k=3, gc=mock_gc, encoder=mock_encoder
        )

        order = [result.index(f'### Page: "{cid}"') for cid in ("c2", "c0", "c1")]
        assert order == sorted(order)

    def test_docs_appends_a_note_when_the_rerank_fails(
        self, mock_gc, mock_encoder, monkeypatch
    ):
        from imas_codex.discovery.base import judgment

        mock_gc.query = MagicMock(side_effect=self._wiki_handler(["c0", "c1"]))
        monkeypatch.setattr(
            judgment,
            "acall_decisions",
            _fake_decisions({}, fail=("c0", "c1")),
        )

        result = _search_docs(
            query="equilibrium", facility="tcv", k=2, gc=mock_gc, encoder=mock_encoder
        )

        assert "2 of 2" in result
        assert result.index('### Page: "c0"') < result.index('### Page: "c1"')
        assert "embedding position" in result

    def test_code_report_follows_the_rerank_order(
        self, mock_gc, mock_encoder, monkeypatch
    ):
        from imas_codex.discovery.base import judgment

        mock_gc.query = MagicMock(side_effect=self._code_handler(["k0", "k1", "k2"]))
        monkeypatch.setattr(
            judgment,
            "acall_decisions",
            _fake_decisions({"k2.py": 5, "k0.py": 1, "k1.py": 3}),
        )

        result = _search_code(
            query="equilibrium", facility="tcv", k=3, gc=mock_gc, encoder=mock_encoder
        )

        order = [result.index(f"### f_{cid} —") for cid in ("k2", "k1", "k0")]
        assert order == sorted(order)

    def test_code_report_pools_beyond_k(self, mock_gc, mock_encoder):
        mock_gc.query = MagicMock(side_effect=self._code_handler(["k0", "k1"]))

        _search_code(
            query="equilibrium", facility="tcv", k=1, gc=mock_gc, encoder=mock_encoder
        )

        for call in mock_gc.query.call_args_list:
            if "code_chunk_embedding" in call[0][0]:
                assert call[1]["limit"] == RERANK_POOL, call[1]
                break
        else:
            pytest.fail("No code_chunk_embedding vector search call found")
