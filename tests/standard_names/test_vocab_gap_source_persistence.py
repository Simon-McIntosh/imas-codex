"""Vocab-gap outcome is persisted via the VocabGap node, not a string.

write_vocab_gaps creates the canonical VocabGap node plus a direct
``StandardNameSource -[:HAS_STANDARD_NAME_VOCAB_GAP]-> VocabGap`` edge (the
one-hop "why is this source blocked?" link reconcile traverses). The status
marker then only flips the source status: retire to ``vocab_gap`` for a
genuinely-absent closed-segment gap, or keep retryable (attempt-count cap) for
a non-actionable composer mis-report, which gets no node at all.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from imas_codex.standard_names.workers import _update_sources_after_vocab_gap

_WLOG = logging.getLogger("test-vocab-gap")


def _run_marker(gaps, *, actionable_tokens, open_segments=()):
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(return_value=[])

    with (
        patch("imas_codex.graph.client.GraphClient", return_value=gc),
        patch(
            "imas_codex.standard_names.segments.is_actionable_gap",
            side_effect=lambda seg, tok: tok in actionable_tokens,
        ),
        patch(
            "imas_codex.standard_names.segments.is_open_segment",
            side_effect=lambda seg: seg in open_segments,
        ),
    ):
        _update_sources_after_vocab_gap(gaps, "dd", _WLOG)
    return [(c.args[0], c.kwargs) for c in gc.query.call_args_list]


def _find(calls, needle):
    return [(cy, kw) for cy, kw in calls if needle in cy]


def test_actionable_gap_retires_source_no_string():
    gaps = [
        {
            "source_id": "equilibrium/constraints/n_e_line/measured",
            "segment": "physical_base",
            "token": "line_integrated_density",
            "reason": "no base",
        }
    ]
    calls = _run_marker(gaps, actionable_tokens={"line_integrated_density"})
    retire = _find(calls, "status = 'vocab_gap'")
    assert retire, "actionable gap must retire the source to vocab_gap"
    cy, kw = retire[0]
    assert kw["ids"] == ["dd:equilibrium/constraints/n_e_line/measured"]
    # The string denormalization is gone — the VocabGap node/edge is the record.
    assert "skip_reason_detail" not in cy


def test_nonactionable_gap_kept_retryable():
    gaps = [
        {
            "source_id": "summary/plasma_duration/value",
            "segment": "physical_base",
            "token": "plasma_pulse_duration",
            "reason": "decomposable",
        }
    ]
    calls = _run_marker(gaps, actionable_tokens=set())
    assert not _find(calls, "status = 'vocab_gap'"), (
        "a non-actionable gap must not strand the source at vocab_gap"
    )
    retry = _find(calls, "attempt_count")
    assert retry, "non-actionable gap must keep the source retryable"
    assert retry[0][1]["ids"] == ["dd:summary/plasma_duration/value"]
    assert "skip_reason_detail" not in retry[0][0]


def test_open_segment_gap_ignored():
    gaps = [{"source_id": "x/y/z", "segment": "grammar_ambiguity", "token": "w"}]
    calls = _run_marker(
        gaps, actionable_tokens=set(), open_segments={"grammar_ambiguity"}
    )
    assert not _find(calls, "status = 'vocab_gap'")
    assert not _find(calls, "attempt_count")


def test_write_vocab_gaps_links_source_to_node():
    """write_vocab_gaps must MERGE a direct StandardNameSource→VocabGap edge."""
    from imas_codex.standard_names import graph_ops

    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(return_value=[])

    gaps = [
        {
            "source_id": "equilibrium/time_slice/profiles_1d/some_absent_base",
            "segment": "physical_base",
            "token": "some_absent_base_zzz",
            "reason": "genuinely absent",
        }
    ]
    with (
        patch.object(graph_ops, "GraphClient", return_value=gc),
        patch("imas_codex.standard_names.segments.is_valid_segment", return_value=True),
        patch(
            "imas_codex.standard_names.segments.classify_gap",
            return_value=("absent", []),
        ),
    ):
        graph_ops.write_vocab_gaps(gaps, "dd")

    queries = " || ".join(c.args[0] for c in gc.query.call_args_list)
    # A StandardNameSource-anchored MERGE onto the VocabGap must be issued.
    assert "StandardNameSource" in queries
    src_edge = [
        c
        for c in gc.query.call_args_list
        if "StandardNameSource" in c.args[0]
        and "HAS_STANDARD_NAME_VOCAB_GAP" in c.args[0]
    ]
    assert src_edge, "write_vocab_gaps must link the StandardNameSource to the VocabGap"
    # And the source id is built with the source-type prefix.
    assert src_edge[0].kwargs.get("prefix") == "dd:"
