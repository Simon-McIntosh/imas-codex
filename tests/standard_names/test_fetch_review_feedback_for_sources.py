"""The fetch_review_feedback_for_sources graph helper.

Mocks the ``GraphClient`` context manager and verifies the helper's
Cypher query, parameter binding, and result shaping.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from imas_codex.standard_names.graph_ops import fetch_review_feedback_for_sources


class TestFetchReviewFeedbackForSources:
    def test_empty_input_short_circuits(self) -> None:
        """Empty / None input returns {} without calling the graph."""
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            assert fetch_review_feedback_for_sources(None) == {}
            assert fetch_review_feedback_for_sources([]) == {}
            assert fetch_review_feedback_for_sources(set()) == {}
            MockGC.assert_not_called()

    def test_returns_feedback_mapping(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "source_id": "dd:equilibrium/time_slice/profiles_1d/psi",
                    "previous_name": "too_long_flux_name",
                    "previous_description": "desc",
                    "previous_documentation": "doc",
                    "reviewer_score": 0.32,
                    "review_tier": "poor",
                    "reviewer_comments": "Drop prefix",
                    "reviewer_scores_json": json.dumps(
                        {
                            "grammar": 15,
                            "semantic": 14,
                            "documentation": 10,
                            "convention": 5,
                            "completeness": 12,
                            "compliance": 6,
                        }
                    ),
                    "validation_status": "valid",
                }
            ]
        )
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            out = fetch_review_feedback_for_sources(
                [
                    "dd:equilibrium/time_slice/profiles_1d/psi",
                    "",  # filtered out
                ]
            )

        # Cypher assertions
        cypher = mock_gc.query.call_args[0][0]
        assert "HAS_STANDARD_NAME" in cypher
        assert "reviewer_score" in cypher
        assert "IS NOT NULL" in cypher
        assert "UNWIND $ids" in cypher
        # Params: sorted dedup'd list, no empty strings
        ids_param = mock_gc.query.call_args[1]["ids"]
        assert ids_param == ["dd:equilibrium/time_slice/profiles_1d/psi"]

        # Result shaping
        assert list(out) == ["dd:equilibrium/time_slice/profiles_1d/psi"]
        fb = out["dd:equilibrium/time_slice/profiles_1d/psi"]
        assert fb["previous_name"] == "too_long_flux_name"
        assert fb["reviewer_score"] == 0.32
        assert fb["review_tier"] == "poor"
        assert fb["reviewer_comments"] == "Drop prefix"
        assert fb["validation_status"] == "valid"
        # JSON parsed into dict
        assert isinstance(fb["reviewer_scores"], dict)
        assert fb["reviewer_scores"]["convention"] == 5

    def test_missing_reviewer_scores_json_tolerated(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "source_id": "dd:x",
                    "previous_name": "n",
                    "previous_description": None,
                    "previous_documentation": None,
                    "reviewer_score": 0.5,
                    "review_tier": "inadequate",
                    "reviewer_comments": "meh",
                    "reviewer_scores_json": None,
                    "validation_status": "valid",
                }
            ]
        )
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            out = fetch_review_feedback_for_sources(["dd:x"])
        assert out["dd:x"]["reviewer_scores"] is None

    def test_invalid_json_tolerated(self) -> None:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "source_id": "dd:x",
                    "previous_name": "n",
                    "previous_description": None,
                    "previous_documentation": None,
                    "reviewer_score": 0.5,
                    "review_tier": "inadequate",
                    "reviewer_comments": "meh",
                    "reviewer_scores_json": "not-json",
                    "validation_status": "valid",
                }
            ]
        )
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            out = fetch_review_feedback_for_sources(["dd:x"])
        assert out["dd:x"]["reviewer_scores"] is None

    def test_prefers_lowest_scoring_name_for_duplicate_source(self) -> None:
        """If a source_id maps to multiple SNs, pick the worse-scored one."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "source_id": "dd:x",
                    "previous_name": "mediocre",
                    "previous_description": None,
                    "previous_documentation": None,
                    "reviewer_score": 0.55,
                    "review_tier": "inadequate",
                    "reviewer_comments": "",
                    "reviewer_scores_json": None,
                    "validation_status": "valid",
                },
                {
                    "source_id": "dd:x",
                    "previous_name": "awful",
                    "previous_description": None,
                    "previous_documentation": None,
                    "reviewer_score": 0.20,
                    "review_tier": "poor",
                    "reviewer_comments": "",
                    "reviewer_scores_json": None,
                    "validation_status": "valid",
                },
            ]
        )
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            out = fetch_review_feedback_for_sources(["dd:x"])
        assert out["dd:x"]["previous_name"] == "awful"
        assert out["dd:x"]["reviewer_score"] == 0.20


class TestSourceToNameEdge:
    """The link is read over PRODUCED_NAME, the authority.

    ``HAS_STANDARD_NAME`` originates from the upstream DD node, which is keyed
    on the bare path, never from the ``StandardNameSource``, which is keyed
    ``dd:<path>``. Reading only that edge therefore matched NOTHING for the
    source ids every caller passes, so an operator's ``sn edit --hint`` was
    silently dropped before it reached the compose prompt.
    """

    @staticmethod
    def _captured_cypher() -> str:
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(return_value=[])
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            fetch_review_feedback_for_sources(["dd:x"])
        return mock_gc.query.call_args[0][0]

    def test_produced_name_is_traversed(self) -> None:
        assert "PRODUCED_NAME" in self._captured_cypher()

    def test_has_standard_name_is_still_accepted(self) -> None:
        """Callers passing an upstream IMASNode / FacilitySignal id still work."""
        cypher = self._captured_cypher()
        assert "HAS_STANDARD_NAME" in cypher
        assert "UNION" in cypher

    def test_an_open_edit_does_not_require_a_prior_review(self) -> None:
        """A hint is operator intent; gating it on a review score drops it.

        An ``sn edit`` stages a name for regeneration, so there is frequently no
        ``reviewer_score_name`` yet — exactly the case the steering is for.
        """
        cypher = self._captured_cypher()
        assert "edit_mode" in cypher
        assert "reviewer_score_name IS NOT NULL" in cypher

    def test_hint_fields_reach_the_caller(self) -> None:
        """An edit-staged name with no review score still delivers its hint."""
        mock_gc = MagicMock()
        mock_gc.query = MagicMock(
            return_value=[
                {
                    "source_id": "dd:x",
                    "previous_name": "wrong_name",
                    "previous_description": None,
                    "previous_documentation": None,
                    "reviewer_score": None,
                    "review_tier": None,
                    "reviewer_comments": None,
                    "reviewer_scores_json": None,
                    "validation_status": "valid",
                    "edit_mode": "hint",
                    "name_hint": "name it for the quantity, not the device",
                    "edit_reason": "collapsed onto an unrelated name",
                    "edit_origin": "operator",
                }
            ]
        )
        with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
            MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
            MockGC.return_value.__exit__ = MagicMock(return_value=False)
            out = fetch_review_feedback_for_sources(["dd:x"])
        assert out["dd:x"]["name_hint"] == "name it for the quantity, not the device"
        assert out["dd:x"]["edit_mode"] == "hint"
        assert out["dd:x"]["edit_reason"] == "collapsed onto an unrelated name"
        assert out["dd:x"]["reviewer_score"] is None
