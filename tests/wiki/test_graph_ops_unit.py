"""Tests for wiki graph operations.

Covers retry_on_deadlock decorator, claim/mark functions, bulk create
functions, and type classification constants — all with mocked GraphClient.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import TransientError

from imas_codex.discovery.base.claims import retry_on_deadlock
from imas_codex.discovery.wiki.graph_ops import (
    CLAIM_TIMEOUT_SECONDS,
    IMAGE_DOCUMENT_TYPES,
    INGESTABLE_DOCUMENT_TYPES,
    SCORABLE_DOCUMENT_TYPES,
    _bulk_create_wiki_documents,
    _bulk_create_wiki_pages,
)

# =============================================================================
# Type classification constants
# =============================================================================


class TestTypeClassification:
    """Tests for document type classification constants."""

    def test_ingestable_types(self):
        """Ingestable types should include text-extractable formats."""
        expected = {
            "pdf",
            "text_document",
            "presentation",
            "spreadsheet",
            "notebook",
            "json",
        }
        assert INGESTABLE_DOCUMENT_TYPES == expected

    def test_image_types(self):
        """Image types should only include image."""
        assert IMAGE_DOCUMENT_TYPES == {"image"}

    def test_scorable_types(self):
        """Scorable types should be ingestable + metadata-only types."""
        assert INGESTABLE_DOCUMENT_TYPES.issubset(SCORABLE_DOCUMENT_TYPES)
        assert "data" in SCORABLE_DOCUMENT_TYPES
        assert "archive" in SCORABLE_DOCUMENT_TYPES
        assert "other" in SCORABLE_DOCUMENT_TYPES

    def test_images_not_scorable(self):
        """Image types should NOT be scorable (they use VLM pipeline)."""
        assert IMAGE_DOCUMENT_TYPES.isdisjoint(SCORABLE_DOCUMENT_TYPES)


# =============================================================================
# retry_on_deadlock decorator
# =============================================================================


class TestRetryOnDeadlock:
    """Tests for the retry_on_deadlock decorator."""

    def test_success_on_first_try(self):
        """Function succeeding on first call should work normally."""
        call_count = 0

        @retry_on_deadlock(max_attempts=3)
        def my_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = my_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_transient_error(self):
        """Should retry on TransientError."""
        call_count = 0

        @retry_on_deadlock(max_attempts=3, base_delay=0.001, max_delay=0.01)
        def my_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TransientError("Deadlock detected")
            return "success"

        result = my_func()
        assert result == "success"
        assert call_count == 3

    def test_exhaust_retries(self):
        """Should raise after exhausting all attempts."""

        @retry_on_deadlock(max_attempts=2, base_delay=0.001, max_delay=0.01)
        def my_func():
            raise TransientError("Persistent deadlock")

        with pytest.raises(TransientError, match="Persistent deadlock"):
            my_func()

    def test_non_transient_error_not_retried(self):
        """Non-TransientError should propagate immediately."""
        call_count = 0

        @retry_on_deadlock(max_attempts=3)
        def my_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Not transient")

        with pytest.raises(ValueError, match="Not transient"):
            my_func()
        assert call_count == 1

    def test_preserves_function_metadata(self):
        """Decorated function should preserve __name__."""

        @retry_on_deadlock()
        def my_named_func():
            pass

        assert my_named_func.__name__ == "my_named_func"


# =============================================================================
# _bulk_create_wiki_pages
# =============================================================================


class TestBulkCreateWikiPages:
    """Tests for _bulk_create_wiki_pages with mocked GraphClient."""

    def test_creates_pages(self):
        """Should create pages via graph query."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 3}]

        pages = [
            {"id": "tcv:Page1", "title": "Page 1", "url": "https://wiki/Page1"},
            {"id": "tcv:Page2", "title": "Page 2", "url": "https://wiki/Page2"},
            {"id": "tcv:Page3", "title": "Page 3", "url": "https://wiki/Page3"},
        ]

        result = _bulk_create_wiki_pages(gc, "tcv", pages)
        assert result == 3
        gc.query.assert_called_once()

    def test_batch_processing(self):
        """Should process in batches."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 2}]

        pages = [
            {"id": f"tcv:Page{i}", "title": f"Page {i}", "url": f"url{i}"}
            for i in range(5)
        ]

        result = _bulk_create_wiki_pages(gc, "tcv", pages, batch_size=2)
        # 5 pages / 2 per batch = 3 batches
        assert gc.query.call_count == 3
        assert result == 6  # 2 * 3 batches

    def test_empty_batch(self):
        """Empty batch should return 0."""
        gc = MagicMock()
        result = _bulk_create_wiki_pages(gc, "tcv", [])
        assert result == 0
        gc.query.assert_not_called()

    def test_progress_callback(self):
        """Should invoke progress callback."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 1}]

        progress_calls = []

        def on_progress(msg, stats):
            progress_calls.append(msg)

        pages = [{"id": "tcv:P1", "title": "P1", "url": "u1"}]
        _bulk_create_wiki_pages(gc, "tcv", pages, on_progress=on_progress)
        assert len(progress_calls) > 0
        assert "creating pages" in progress_calls[0]


# =============================================================================
# _bulk_create_wiki_documents
# =============================================================================


class TestBulkCreateDocuments:
    """Tests for _bulk_create_wiki_documents with mocked GraphClient."""

    def test_creates_documents(self):
        """Should create document nodes."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 2}]

        documents = [
            {
                "id": "tcv:report.pdf",
                "filename": "report.pdf",
                "url": "https://wiki/report.pdf",
                "document_type": "pdf",
            },
            {
                "id": "tcv:img.png",
                "filename": "img.png",
                "url": "https://wiki/img.png",
                "document_type": "image",
            },
        ]

        result = _bulk_create_wiki_documents(gc, "tcv", documents)
        assert result == 2

    def test_score_exempt_flag_set(self):
        """Image documents should get score_exempt=True."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 1}]

        documents = [
            {
                "id": "tcv:photo.png",
                "filename": "photo.png",
                "url": "https://wiki/photo.png",
                "document_type": "image",
            },
        ]

        _bulk_create_wiki_documents(gc, "tcv", documents)
        # Verify score_exempt was set to True for image document
        assert documents[0]["score_exempt"] is True

    def test_non_image_not_exempt(self):
        """Non-image documents should not be score_exempt."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 1}]

        documents = [
            {
                "id": "tcv:report.pdf",
                "filename": "report.pdf",
                "url": "x",
                "document_type": "pdf",
            },
        ]

        _bulk_create_wiki_documents(gc, "tcv", documents)
        assert documents[0]["score_exempt"] is False

    def test_linked_pages_relationships(self):
        """Should create HAS_DOCUMENT relationships for linked_pages."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 1}]

        documents = [
            {
                "id": "tcv:file.pdf",
                "filename": "file.pdf",
                "url": "x",
                "document_type": "pdf",
                "linked_pages": ["MainPage", "Reports"],
            },
        ]

        _bulk_create_wiki_documents(gc, "tcv", documents)
        # Should have 2 query calls: 1 for documents + 1 for page links
        assert gc.query.call_count == 2

    def test_no_linked_pages_skips_link_query(self):
        """No linked_pages should skip the relationship query."""
        gc = MagicMock()
        gc.query.return_value = [{"count": 1}]

        documents = [
            {
                "id": "tcv:file.pdf",
                "filename": "file.pdf",
                "url": "x",
                "document_type": "pdf",
            },
        ]

        _bulk_create_wiki_documents(gc, "tcv", documents)
        # Only 1 query (document creation, no page links)
        assert gc.query.call_count == 1


# =============================================================================
# Pending work checks (with mocked GraphClient)
# =============================================================================


class TestPendingWorkChecks:
    """Tests for has_pending_* functions with mocked GraphClient."""

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_has_pending_work_true(self, mock_gc_class):
        """Should return True when pending pages exist."""
        from imas_codex.discovery.wiki.graph_ops import has_pending_work

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"pending": 42}]

        result = has_pending_work("tcv")
        assert result is True

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_has_pending_work_false(self, mock_gc_class):
        """Should return False when no pending pages."""
        from imas_codex.discovery.wiki.graph_ops import has_pending_work

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"pending": 0}]

        result = has_pending_work("tcv")
        assert result is False

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_has_pending_document_work(self, mock_gc_class):
        """Should check document pending state."""
        from imas_codex.discovery.wiki.graph_ops import has_pending_document_work

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"pending": 5}]

        result = has_pending_document_work("tcv")
        assert result is True

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_has_pending_scan_work(self, mock_gc_class):
        """Should check for scanned pages awaiting scoring."""
        from imas_codex.discovery.wiki.graph_ops import has_pending_scan_work

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [{"pending": 0}]

        result = has_pending_scan_work("tcv")
        assert result is False


# =============================================================================
# Claim timeout constant
# =============================================================================


class TestClaimConfiguration:
    """Tests for claim configuration constants."""

    def test_timeout_value(self):
        """Claim timeout should be 5 minutes."""
        assert CLAIM_TIMEOUT_SECONDS == 300


# =============================================================================
# Document failure classification (deferral) and recoverable-failure patterns
# =============================================================================


class TestClassifyDocumentDeferral:
    """classify_document_deferral maps unsupported-input errors to reasons."""

    def test_no_error_returns_none(self):
        from imas_codex.discovery.wiki.graph_ops import classify_document_deferral

        assert classify_document_deferral(None) is None
        assert classify_document_deferral("") is None

    def test_wmf_loader_defers_with_format(self):
        from imas_codex.discovery.wiki.graph_ops import classify_document_deferral

        reason = classify_document_deferral(
            "cannot find loader for this WMF file", "presentation"
        )
        assert reason == "unsupported image format (WMF)"

    def test_unidentified_image_defers(self):
        from imas_codex.discovery.wiki.graph_ops import classify_document_deferral

        reason = classify_document_deferral(
            "UnidentifiedImageError: cannot identify image file", "image"
        )
        assert reason == "unsupported image format (image)"

    def test_missing_resource_dead_link(self):
        from imas_codex.discovery.wiki.graph_ops import classify_document_deferral

        reason = classify_document_deferral(
            "Failed to fetch https://nakasvr23.iferc.org/MISSING RESOURCE Code/x",
            "pdf",
        )
        assert reason == "dead link"

    def test_http_404_dead_link(self):
        from imas_codex.discovery.wiki.graph_ops import classify_document_deferral

        assert (
            classify_document_deferral("HTTP Error 404: Not Found", "pdf")
            == "dead link"
        )

    def test_word_file_type_mismatch(self):
        from imas_codex.discovery.wiki.graph_ops import classify_document_deferral

        reason = classify_document_deferral(
            "file '<_io.BytesIO object at 0x1>' is not a Word file", "text_document"
        )
        assert reason == "type mismatch: text_document"

    def test_unknown_error_stays_failed(self):
        from imas_codex.discovery.wiki.graph_ops import classify_document_deferral

        assert classify_document_deferral("Connection refused", "pdf") is None
        assert classify_document_deferral("some novel parse error", "pdf") is None


class TestRecoverFailedDocuments:
    """recover_failed_documents resets environment faults by prior state."""

    def _mock_gc(self, mock_gc_class, counts):
        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.side_effect = [[{"recovered": c}] for c in counts]
        return mock_gc

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_environment_fault_recovers_to_discovered_without_score(
        self, mock_gc_class
    ):
        from imas_codex.discovery.wiki.graph_ops import recover_failed_documents

        mock_gc = self._mock_gc(mock_gc_class, (4, 0))
        total = recover_failed_documents("jt-60sa")

        assert total == 4
        first_query = mock_gc.query.call_args_list[0].args[0]
        assert "No module named" in first_query
        assert "score_composite IS NULL" in first_query

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_environment_fault_recovers_to_scored_with_score(self, mock_gc_class):
        from imas_codex.discovery.wiki.graph_ops import recover_failed_documents

        mock_gc = self._mock_gc(mock_gc_class, (0, 516))
        total = recover_failed_documents("jt-60sa")

        assert total == 516
        second_query = mock_gc.query.call_args_list[1].args[0]
        assert "No module named" in second_query
        assert "score_composite IS NOT NULL" in second_query

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_returns_zero_on_graph_error(self, mock_gc_class):
        from imas_codex.discovery.wiki.graph_ops import recover_failed_documents

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.side_effect = RuntimeError("Neo4j unavailable")

        assert recover_failed_documents("jt-60sa") == 0


class TestDeferFailedDocuments:
    """defer_failed_documents reclassifies unsupported already-failed rows."""

    @patch("imas_codex.discovery.wiki.graph_ops.mark_document_deferred")
    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_reclassifies_three_classes_leaves_unknown(self, mock_gc_class, mock_defer):
        from imas_codex.discovery.wiki.graph_ops import defer_failed_documents

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [
            {
                "id": "d1",
                "error": "cannot find loader for this WMF file",
                "document_type": "presentation",
            },
            {
                "id": "d2",
                "error": "Failed to fetch https://x/MISSING RESOURCE Code/",
                "document_type": "pdf",
            },
            {
                "id": "d3",
                "error": "file '<_io.BytesIO>' is not a Word file",
                "document_type": "text_document",
            },
            {
                "id": "d4",
                "error": "Connection refused",
                "document_type": "pdf",
            },
        ]

        assert defer_failed_documents("jt-60sa") == 3

        calls = {c.args[0]: c.args[1] for c in mock_defer.call_args_list}
        assert calls["d1"] == "unsupported image format (WMF)"
        assert calls["d2"] == "dead link"
        assert calls["d3"] == "type mismatch: text_document"
        assert "d4" not in calls

    @patch("imas_codex.discovery.wiki.graph_ops.mark_document_deferred")
    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_returns_zero_on_graph_error(self, mock_gc_class, mock_defer):
        from imas_codex.discovery.wiki.graph_ops import defer_failed_documents

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.side_effect = RuntimeError("Neo4j unavailable")

        assert defer_failed_documents("jt-60sa") == 0
        mock_defer.assert_not_called()


class TestDocumentStatusWriterFailClosed:
    """A status write that does not reach the graph surfaces to the caller.

    A deferral or failure mark that never landed must not read as a success
    beside a warning: each writer raises, and each caller either propagates
    the failure or counts only the writes that landed.
    """

    @staticmethod
    def _raising_gc(mock_gc_class):
        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.side_effect = RuntimeError("Neo4j unavailable")
        return mock_gc

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_mark_document_deferred_raises_when_write_fails(self, mock_gc_class):
        from imas_codex.discovery.wiki.graph_ops import mark_document_deferred

        self._raising_gc(mock_gc_class)

        with pytest.raises(RuntimeError):
            mark_document_deferred("doc:1", "dead link")

    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_mark_document_failed_raises_when_write_fails(self, mock_gc_class):
        from imas_codex.discovery.wiki.graph_ops import mark_document_failed

        self._raising_gc(mock_gc_class)

        with pytest.raises(RuntimeError):
            mark_document_failed("doc:2", "boom")

    @patch("imas_codex.discovery.wiki.graph_ops.mark_document_deferred")
    def test_failed_or_deferred_propagates_deferred_write_failure(self, mock_defer):
        from imas_codex.discovery.wiki.graph_ops import (
            mark_document_failed_or_deferred,
        )

        mock_defer.side_effect = RuntimeError("Neo4j unavailable")

        with pytest.raises(RuntimeError):
            mark_document_failed_or_deferred(
                "doc:1", "cannot find loader for this WMF file", "presentation"
            )
        mock_defer.assert_called_once_with("doc:1", "unsupported image format (WMF)")

    @patch("imas_codex.discovery.wiki.graph_ops.mark_document_failed")
    def test_failed_or_deferred_propagates_failed_write_failure(self, mock_failed):
        from imas_codex.discovery.wiki.graph_ops import (
            mark_document_failed_or_deferred,
        )

        mock_failed.side_effect = RuntimeError("Neo4j unavailable")

        with pytest.raises(RuntimeError):
            mark_document_failed_or_deferred("doc:2", "Connection refused", "pdf")
        mock_failed.assert_called_once_with("doc:2", "Connection refused")

    @patch("imas_codex.discovery.wiki.graph_ops.mark_document_deferred")
    @patch("imas_codex.discovery.wiki.graph_ops.GraphClient")
    def test_defer_failed_documents_surfaces_write_failure(
        self, mock_gc_class, mock_defer
    ):
        from imas_codex.discovery.wiki.graph_ops import defer_failed_documents

        mock_gc = MagicMock()
        mock_gc_class.return_value.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc_class.return_value.__exit__ = MagicMock(return_value=False)
        mock_gc.query.return_value = [
            {"id": "d1", "error": "HTTP Error 404: Not Found", "document_type": "pdf"}
        ]
        mock_defer.side_effect = RuntimeError("Neo4j unavailable")

        with pytest.raises(RuntimeError):
            defer_failed_documents("jt-60sa")
        mock_defer.assert_called_once_with("d1", "dead link")
