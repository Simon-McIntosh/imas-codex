"""Tests for the VocabGap classifier gate in write_vocab_gaps.

Verifies that VocabGap nodes are classified correctly and that only
'absent' gaps are persisted — wrong_slot_placement, ambiguous_known_token,
and decomposable categories are filtered out at the persistence layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_segment_token_map(mapping: dict[str, tuple[str, ...]]) -> dict:
    """Build a mock SEGMENT_TOKEN_MAP from a simplified mapping."""
    return mapping


class TestVocabGapClassification:
    """Mock the gap-write call site; verify correct category assignment."""

    @pytest.fixture(autouse=True)
    def _clear_caches(self):
        """Clear lru_cache between tests so mock data takes effect."""
        from imas_codex.standard_names.segments import (
            _segment_token_index,
            known_segments,
            open_segments,
            resolved_base_segment,
        )

        # Clear ALL segment-lookup caches — a mocked SEGMENT_TOKEN_MAP must not
        # leak cached results (e.g. known_segments) into sibling test modules.
        _caches = (
            _segment_token_index,
            open_segments,
            known_segments,
            resolved_base_segment,
        )
        for _c in _caches:
            _c.cache_clear()
        yield
        for _c in _caches:
            _c.cache_clear()

    def _run_write_vocab_gaps(self, gaps, segment_token_map):
        """Run write_vocab_gaps with a mocked ISN and GraphClient."""
        mock_gc_instance = MagicMock()
        mock_gc_instance.query = MagicMock()
        mock_gc_instance.__enter__ = MagicMock(return_value=mock_gc_instance)
        mock_gc_instance.__exit__ = MagicMock(return_value=False)

        with (
            patch(
                "imas_codex.standard_names.segments.SEGMENT_TOKEN_MAP",
                segment_token_map,
                create=True,
            ),
            patch(
                "imas_standard_names.grammar.constants.SEGMENT_TOKEN_MAP",
                segment_token_map,
                create=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "imas_standard_names": MagicMock(),
                    "imas_standard_names.grammar": MagicMock(),
                    "imas_standard_names.grammar.constants": MagicMock(
                        SEGMENT_TOKEN_MAP=segment_token_map
                    ),
                },
            ),
            patch(
                "imas_codex.standard_names.graph_ops.GraphClient",
                return_value=mock_gc_instance,
            ),
        ):
            from imas_codex.standard_names.graph_ops import write_vocab_gaps

            write_vocab_gaps(gaps, "dd", skip_segment_filter=True)

        # Extract the batch passed to the first MERGE query (gap nodes)
        calls = mock_gc_instance.query.call_args_list
        assert calls, "Expected at least one query call"
        # First call is the MERGE for VocabGap nodes
        first_call = calls[0]
        batch = first_call.kwargs.get("batch", first_call[1].get("batch", []))
        return {node["id"]: node for node in batch}

    def test_absent_token(self):
        """Token not in any segment → category='absent'."""
        stm = {
            "subject": ("electron", "ion"),
            "process": ("heating", "cooling"),
            "physical_base": (),  # open
        }
        gaps = [
            {
                "source_id": "eq/ts/p1d/psi",
                "segment": "process",
                "token": "frobnicating",
                "reason": "test",
            }
        ]
        nodes = self._run_write_vocab_gaps(gaps, stm)
        gap_id = "vocab_gap:process:frobnicating"
        assert gap_id in nodes
        assert nodes[gap_id]["category"] == "absent"
        assert nodes[gap_id]["actual_segments"] == []

    def test_wrong_slot_placement(self):
        """Token in another segment → classified as wrong_slot_placement, NOT persisted."""
        stm = {
            "subject": ("electron", "ion", "particle"),
            "process": ("heating", "cooling"),
            "physical_base": (),
        }
        gaps = [
            {
                "source_id": "eq/ts/p1d/psi",
                "segment": "process",
                "token": "ion",
                "reason": "LLM placed ion in process",
            }
        ]
        nodes = self._run_write_vocab_gaps(gaps, stm)
        # wrong_slot_placement gaps are filtered out — NOT persisted
        assert len(nodes) == 0

    def test_ambiguous_known_token(self):
        """Token in multiple segments, reported on none → filtered out, NOT persisted."""
        stm = {
            "subject": ("parallel", "radial"),
            "orientation": ("parallel", "toroidal"),
            "process": ("heating",),
            "physical_base": (),
        }
        gaps = [
            {
                "source_id": "eq/ts/p1d/psi",
                "segment": "process",
                "token": "parallel",
                "reason": "LLM confused",
            }
        ]
        nodes = self._run_write_vocab_gaps(gaps, stm)
        # ambiguous_known_token gaps are filtered out — NOT persisted
        assert len(nodes) == 0

    def test_mixed_batch(self):
        """A batch with all categories: only 'absent' gaps persist."""
        stm = {
            "subject": ("electron", "ion"),
            "orientation": ("parallel", "radial"),
            "qualifier": ("parallel",),  # overlap with orientation
            "process": ("heating",),
            "physical_base": (),
        }
        gaps = [
            # absent — should persist
            {
                "source_id": "a",
                "segment": "process",
                "token": "turbulating",
                "reason": "test",
            },
            # wrong_slot (ion in subject, reported on process) — filtered
            {
                "source_id": "b",
                "segment": "process",
                "token": "ion",
                "reason": "test",
            },
            # ambiguous (parallel in orientation+qualifier, reported on process) — filtered
            {
                "source_id": "c",
                "segment": "process",
                "token": "parallel",
                "reason": "test",
            },
        ]
        nodes = self._run_write_vocab_gaps(gaps, stm)

        # Only the truly absent token should be persisted
        assert len(nodes) == 1
        assert nodes["vocab_gap:process:turbulating"]["category"] == "absent"
