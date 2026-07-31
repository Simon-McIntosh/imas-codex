"""Unit tests for the pool-compose helpers.

Covers:

* ``_process_attachments_core`` — stateless attachment dispatch.
* ``_update_sources_after_skip`` — claim release + status='skipped'.
* ``_search_reference_exemplars`` — query synthesis + exclude_ids handling.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# _process_attachments_core
# ---------------------------------------------------------------------------


class TestProcessAttachmentsCore:
    def test_empty_input_returns_zero_counts(self) -> None:
        from imas_codex.standard_names.workers import _process_attachments_core

        wlog = logging.getLogger("test")
        out = _process_attachments_core([], wlog)
        assert out == {"accepted": 0, "rejected": 0}

    def test_inconsistent_tense_rejected(self) -> None:
        """Attachment whose source_id implies a different tense than the SN
        target gets rejected and never reaches the graph layer."""
        from imas_codex.standard_names.workers import _process_attachments_core

        wlog = logging.getLogger("test")

        # ``_is_attachment_consistent`` is the gate; mock it to reject all.
        with (
            patch(
                "imas_codex.standard_names.workers._is_attachment_consistent",
                return_value=(False, "tense_mismatch"),
            ),
            patch(
                "imas_codex.graph.client.GraphClient",
            ) as mock_gc,
        ):
            attach = SimpleNamespace(
                source_id="equilibrium/x",
                standard_name="some_name",
                reason="ok",
            )
            out = _process_attachments_core([attach], wlog)

        assert out == {"accepted": 0, "rejected": 1}
        # Graph layer never invoked when nothing is accepted.
        assert not mock_gc.called

    def test_accepted_writes_with_exact_source_fence(self) -> None:
        from imas_codex.standard_names.workers import _process_attachments_core

        wlog = logging.getLogger("test")
        with (
            patch(
                "imas_codex.standard_names.workers._is_attachment_consistent",
                return_value=(True, ""),
            ),
            patch(
                "imas_codex.standard_names.graph_ops.persist_claimed_attachments",
                return_value=["dd:equilibrium/x"],
            ) as persist,
        ):
            attach = SimpleNamespace(
                source_id="equilibrium/x",
                standard_name="some_name",
                reason="duplicate",
            )
            out = _process_attachments_core(
                [attach],
                wlog,
                source_claims=[
                    {
                        "path": "equilibrium/x",
                        "claim_token": "winner",
                        "claim_seq": 4,
                    }
                ],
            )

        assert out == {"accepted": 1, "rejected": 0}
        item = persist.call_args.args[0][0]
        assert item["sns_id"] == "dd:equilibrium/x"
        assert item["claim_token"] == "winner"
        assert item["claim_seq"] == 4

    def test_accepted_without_source_fence_is_rejected(self) -> None:
        from imas_codex.standard_names.workers import _process_attachments_core

        attach = SimpleNamespace(
            source_id="equilibrium/x",
            standard_name="some_name",
            reason="duplicate",
        )
        with patch(
            "imas_codex.standard_names.workers._is_attachment_consistent",
            return_value=(True, ""),
        ):
            out = _process_attachments_core([attach], logging.getLogger("test"))

        assert out == {"accepted": 0, "rejected": 1}


# ---------------------------------------------------------------------------
# _update_sources_after_skip
# ---------------------------------------------------------------------------


class TestUpdateSourcesAfterSkip:
    def test_empty_list_is_noop(self) -> None:
        from imas_codex.standard_names.workers import _update_sources_after_skip

        wlog = logging.getLogger("test")
        with patch("imas_codex.graph.client.GraphClient") as mock_gc:
            _update_sources_after_skip([], "dd", wlog)
        assert not mock_gc.called

    def test_dd_source_ids_without_fences_are_ignored(self) -> None:
        from imas_codex.standard_names.workers import _update_sources_after_skip

        wlog = logging.getLogger("test")
        with patch("imas_codex.graph.client.GraphClient") as mock_gc:
            _update_sources_after_skip(["a/b", "c/d"], "dd", wlog)

        assert not mock_gc.called

    def test_signal_source_ids_without_fences_are_ignored(self) -> None:
        from imas_codex.standard_names.workers import _update_sources_after_skip

        wlog = logging.getLogger("test")
        with patch("imas_codex.graph.client.GraphClient") as mock_gc:
            _update_sources_after_skip(["x"], "signals", wlog)
        assert not mock_gc.called

    def test_graph_failure_swallowed(self) -> None:
        """A graph error must not propagate (best-effort cleanup)."""
        from imas_codex.standard_names.workers import _update_sources_after_skip

        wlog = MagicMock()
        with patch("imas_codex.graph.client.GraphClient") as mock_gc:
            mock_gc.return_value.__enter__.return_value.query.side_effect = (
                RuntimeError("boom")
            )
            _update_sources_after_skip(["a"], "dd", wlog)
        # Logged a warning; did not raise.
        assert wlog.warning.called


# ---------------------------------------------------------------------------
# _search_reference_exemplars
# ---------------------------------------------------------------------------


class TestSearchReferenceExemplars:
    def test_empty_descriptions_returns_empty(self) -> None:
        from imas_codex.standard_names.workers import _search_reference_exemplars

        out = _search_reference_exemplars(
            [{"path": "a"}, {"path": "b"}],  # no descriptions
            domains=["magnetics"],
        )
        assert out == []

    def test_synthesises_query_from_first_three_items(self) -> None:
        from imas_codex.standard_names.workers import _search_reference_exemplars

        items = [
            {"description": "alpha", "existing_name": "ex_a"},
            {"description": "beta", "existing_name": "ex_b"},
            {"description": "gamma"},
            {"description": "delta_should_be_skipped"},
        ]

        with patch(
            "imas_codex.standard_names.search.search_standard_names_with_documentation",
            return_value=[{"id": "x", "description": "y"}],
        ) as mock_search:
            out = _search_reference_exemplars(items, ["magnetics"], k=5)

        assert out == [{"id": "x", "description": "y"}]
        mock_search.assert_called_once()
        call_args = mock_search.call_args
        # First three descriptions joined into the query.
        assert "alpha" in call_args.args[0]
        assert "beta" in call_args.args[0]
        assert "gamma" in call_args.args[0]
        assert "delta_should_be_skipped" not in call_args.args[0]
        # exclude_ids derived from real existing_name (not path-munged).
        assert call_args.kwargs["exclude_ids"] == ["ex_a", "ex_b"]
        assert call_args.kwargs["k"] == 5

    def test_search_failure_returns_empty(self) -> None:
        from imas_codex.standard_names.workers import _search_reference_exemplars

        with patch(
            "imas_codex.standard_names.search.search_standard_names_with_documentation",
            side_effect=RuntimeError("boom"),
        ):
            out = _search_reference_exemplars([{"description": "x"}], ["magnetics"])
        assert out == []

    def test_no_existing_name_passes_none_exclude(self) -> None:
        from imas_codex.standard_names.workers import _search_reference_exemplars

        with patch(
            "imas_codex.standard_names.search.search_standard_names_with_documentation",
            return_value=[],
        ) as mock_search:
            _search_reference_exemplars([{"description": "alpha"}], ["magnetics"])
        # When no existing_name fields are present, exclude_ids should be None
        # (not an empty list — the search backend interprets None vs []).
        assert mock_search.call_args.kwargs["exclude_ids"] is None
