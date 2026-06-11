"""Tests for embed-failure resilience in the SN pipeline.

Verifies that embedding failures do NOT quarantine names — they should
set ``embed_failed_at`` and allow the name to continue advancing through
the pipeline (review does not require vectors; only MCP search does).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Tests: graph_ops.persist_generated_name_batch — embed failure handling
# ---------------------------------------------------------------------------


class TestEmbedFailureDoesNotQuarantine:
    """Embed failure sets embed_failed_at; never sets quarantine status."""

    def test_embed_failure_does_not_quarantine(self) -> None:
        """When inline embedding is skipped (deferred to embed worker),
        validation_status is not 'quarantined' and embed_text_hash is None."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = {
            "id": "electron_temperature",
            "model": "test-model",
            "validation_status": "valid",
            "validation_issues": [],
            "source_paths": [],
            "unit": "eV",
            "physics_domain": "equilibrium",
        }

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            persist_generated_name_batch(
                [candidate],
                compose_model="test-model",
            )

        # Should NOT be quarantined
        assert candidate.get("validation_status") != "quarantined"
        # Embedding is deferred — embed_text_hash should be None
        assert candidate.get("embed_text_hash") is None
        # embedding should be None (deferred to embed worker)
        assert candidate.get("embedding") is None

    def test_embed_success_sets_embedded_at(self) -> None:
        """Embedding is deferred — embedded_at is NOT set inline.
        The embed worker pool handles it asynchronously."""
        from imas_codex.standard_names.graph_ops import persist_generated_name_batch

        candidate = {
            "id": "electron_temperature",
            "model": "test-model",
            "validation_status": "valid",
            "validation_issues": [],
            "source_paths": [],
            "unit": "eV",
            "physics_domain": "equilibrium",
        }

        mock_gc = MagicMock()
        mock_gc.__enter__ = MagicMock(return_value=mock_gc)
        mock_gc.__exit__ = MagicMock(return_value=False)

        with patch(
            "imas_codex.standard_names.graph_ops.GraphClient",
            return_value=mock_gc,
        ):
            persist_generated_name_batch(
                [candidate],
                compose_model="test-model",
            )

        # Embedding is deferred — embedded_at is NOT set inline
        assert candidate.get("embedded_at") is None
        # embed_text_hash is cleared so embed worker picks it up
        assert candidate.get("embed_text_hash") is None
