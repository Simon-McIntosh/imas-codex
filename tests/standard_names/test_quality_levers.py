"""Tests for Standard Names quality mechanisms."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# =========================================================================
# Domain-vocabulary pre-seeding
# =========================================================================


class TestDomainVocabularyPreseed:
    """Tests for build_domain_vocabulary_preseed."""

    def test_none_domain_returns_empty(self):
        """None domain returns empty string."""
        from imas_codex.standard_names.context import build_domain_vocabulary_preseed

        assert build_domain_vocabulary_preseed(None) == ""

    def test_empty_domain_returns_empty(self):
        """Empty string domain returns empty string."""
        from imas_codex.standard_names.context import build_domain_vocabulary_preseed

        assert build_domain_vocabulary_preseed("") == ""

    def test_returns_formatted_vocabulary(self):
        """With mocked graph, returns formatted vocabulary lines."""
        from imas_codex.standard_names.context import build_domain_vocabulary_preseed

        mock_rows = [
            {
                "name": "electron_temperature",
                "description": "Electron temperature profile. Used in transport.",
                "name_stage": "accepted",
            },
            {
                "name": "plasma_current",
                "description": "Total plasma current",
                "name_stage": "published",
            },
        ]

        with patch("imas_codex.graph.client.GraphClient") as MockGC:
            mock_gc = MagicMock()
            mock_gc.query.return_value = mock_rows
            mock_gc.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc.__exit__ = MagicMock(return_value=False)
            MockGC.return_value = mock_gc

            result = build_domain_vocabulary_preseed("transport")

        assert "electron_temperature" in result
        assert "plasma_current" in result
        # First sentence only
        assert "Used in transport" not in result

    def test_graceful_failure(self):
        """Returns empty string on graph failure."""
        from imas_codex.standard_names.context import build_domain_vocabulary_preseed

        with patch(
            "imas_codex.graph.client.GraphClient", side_effect=Exception("no graph")
        ):
            result = build_domain_vocabulary_preseed("magnetics")

        assert result == ""


# =========================================================================
# Reference standard-name few-shot retrieval
# =========================================================================


class TestReferenceSNRetrieval:
    """Tests for search_standard_names_with_documentation."""

    def test_empty_query_returns_empty(self):
        """Empty query returns empty list."""
        from imas_codex.standard_names.search import (
            search_standard_names_with_documentation,
        )

        assert search_standard_names_with_documentation("") == []
        assert search_standard_names_with_documentation("   ") == []

    def test_excludes_specified_ids(self):
        """Excluded IDs are filtered from results."""
        from imas_codex.standard_names.search import (
            search_standard_names_with_documentation,
        )

        mock_rows = [
            {
                "name": "electron_temperature",
                "description": "Te",
                "documentation": "Doc",
                "unit": "eV",
                "score": 0.9,
            },
            {
                "name": "ion_temperature",
                "description": "Ti",
                "documentation": "Doc2",
                "unit": "eV",
                "score": 0.8,
            },
        ]

        import numpy as np

        mock_embedding = np.zeros(384)

        with (
            patch("imas_codex.embeddings.config.EncoderConfig"),
            patch("imas_codex.embeddings.encoder.Encoder") as MockEnc,
            patch("imas_codex.graph.client.GraphClient") as MockGC,
        ):
            enc_instance = MagicMock()
            enc_instance.embed_texts.return_value = [mock_embedding]
            MockEnc.return_value = enc_instance

            mock_gc = MagicMock()
            mock_gc.query.return_value = mock_rows
            mock_gc.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc.__exit__ = MagicMock(return_value=False)
            MockGC.return_value = mock_gc

            results = search_standard_names_with_documentation(
                "electron temperature",
                k=5,
                exclude_ids=["electron_temperature"],
            )

        assert len(results) == 1
        assert results[0]["name"] == "ion_temperature"

    def test_returns_full_docs(self):
        """Results include documentation and tags."""
        from imas_codex.standard_names.search import (
            search_standard_names_with_documentation,
        )

        mock_rows = [
            {
                "name": "safety_factor",
                "description": "Safety factor q",
                "documentation": "The safety factor $q$ quantifies...",
                "unit": "1",
                "score": 0.95,
            },
        ]

        import numpy as np

        mock_embedding = np.zeros(384)

        with (
            patch("imas_codex.embeddings.config.EncoderConfig"),
            patch("imas_codex.embeddings.encoder.Encoder") as MockEnc,
            patch("imas_codex.graph.client.GraphClient") as MockGC,
        ):
            enc_instance = MagicMock()
            enc_instance.embed_texts.return_value = [mock_embedding]
            MockEnc.return_value = enc_instance

            mock_gc = MagicMock()
            mock_gc.query.return_value = mock_rows
            mock_gc.__enter__ = MagicMock(return_value=mock_gc)
            mock_gc.__exit__ = MagicMock(return_value=False)
            MockGC.return_value = mock_gc

            results = search_standard_names_with_documentation("safety factor")

        assert len(results) == 1
        assert "documentation" in results[0]
        assert results[0]["documentation"].startswith("The safety factor")


# =========================================================================
# Reviewer-theme extraction
# =========================================================================


class TestReviewerThemes:
    """Tests for extract_reviewer_themes."""

    def test_none_domain_returns_empty(self):
        """None domain returns empty list."""
        from imas_codex.standard_names.review.themes import extract_reviewer_themes

        assert extract_reviewer_themes(None) == []

    def test_returns_themes_from_comments(self):
        """Themes are extracted from mocked reviewer comments."""
        from imas_codex.standard_names.review.themes import _extract_themes_from_texts

        comments = [
            "Missing sign convention in documentation",
            "Sign convention not specified for COCOS quantity",
            "Documentation lacks sign convention paragraph",
            "Inconsistent boundary naming pattern",
            "Boundary naming inconsistent with rest of batch",
        ]
        themes = _extract_themes_from_texts(comments)
        assert isinstance(themes, list)
        assert len(themes) > 0

    def test_empty_comments_returns_empty(self):
        """No comments produces empty themes."""
        from imas_codex.standard_names.review.themes import _extract_themes_from_texts

        assert _extract_themes_from_texts([]) == []

    def test_graceful_graph_failure(self):
        """Returns empty list on graph failure."""
        from imas_codex.standard_names.review.themes import extract_reviewer_themes

        with patch(
            "imas_codex.graph.client.GraphClient", side_effect=Exception("no graph")
        ):
            result = extract_reviewer_themes("magnetics")

        assert result == []


# =========================================================================
# Grammar-failure re-prompt
# =========================================================================


class TestGrammarRetry:
    """Tests for the grammar-failure re-prompt."""

    @pytest.mark.asyncio
    async def test_retry_returns_revised_name(self):
        """Successful retry returns a revised name."""
        from pydantic import BaseModel, Field

        from imas_codex.standard_names.workers import _grammar_retry

        class MockResponse:
            revised_name = "electron_temperature"
            explanation = "Fixed grammar"

        async def mock_acall(*args, **kwargs):
            return MockResponse(), 0.01, 100

        result = await _grammar_retry(
            "electron_temp",
            "parse error: unknown base",
            "test-model",
            mock_acall,
        )
        assert result[0] == "electron_temperature"

    @pytest.mark.asyncio
    async def test_retry_returns_none_on_failure(self):
        """Failed retry returns None."""
        from imas_codex.standard_names.workers import _grammar_retry

        async def mock_acall(*args, **kwargs):
            raise Exception("LLM error")

        result = await _grammar_retry(
            "bad_name",
            "parse error",
            "test-model",
            mock_acall,
        )
        assert result[0] is None


# =========================================================================
# State fields
# =========================================================================


class TestStateFields:
    """Tests for new state tracking fields."""

    def test_state_has_quality_lever_fields(self):
        """StandardNameBuildState has retry, revision, and audit counters."""
        from imas_codex.standard_names.state import StandardNameBuildState

        state = StandardNameBuildState(facility="dd")
        assert state.grammar_retries == 0
        assert state.grammar_retries_succeeded == 0
        assert state.opus_revisions_attempted == 0
        assert state.opus_revisions_accepted == 0
        assert state.audits_run == 0
        assert state.audits_failed == 0

    def test_state_fields_are_mutable(self):
        """State tracking fields can be incremented."""
        from imas_codex.standard_names.state import StandardNameBuildState

        state = StandardNameBuildState(facility="dd")
        state.grammar_retries += 1
        state.audits_run += 5
        state.opus_revisions_accepted += 2
        assert state.grammar_retries == 1
        assert state.audits_run == 5
        assert state.opus_revisions_accepted == 2
