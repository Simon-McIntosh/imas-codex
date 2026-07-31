"""Vocab-gap outcome is persisted via the VocabGap node, not a string.

write_vocab_gaps creates the canonical VocabGap node plus a direct
``StandardNameSource -[:HAS_STANDARD_NAME_VOCAB_GAP]-> VocabGap`` edge (the
one-hop "why is this source blocked?" link reconcile traverses). The status
marker then only flips the source status: retire to ``vocab_gap`` for a
genuinely-absent closed-segment gap, or keep retryable (attempt-count cap) for
a non-actionable composer mis-report, which gets no node at all.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.standard_names.workers import _consume_claimed_vocab_gaps


def _run_marker(gaps, *, actionable_tokens, open_segments=()):
    source_ids = {gap["source_id"] for gap in gaps}
    batch = [
        {
            "path": source_id,
            "claim_token": "winner",
            "claim_seq": 8,
            "attempt_count": 1,
        }
        for source_id in source_ids
    ]

    with (
        patch(
            "imas_codex.standard_names.graph_ops.persist_claimed_vocab_gaps",
            side_effect=lambda _gaps, outcomes, **_kwargs: [
                item["sns_id"] for item in outcomes
            ],
        ) as persist,
        patch(
            "imas_codex.standard_names.segments.is_actionable_gap",
            side_effect=lambda seg, tok: tok in actionable_tokens,
        ),
        patch(
            "imas_codex.standard_names.segments.is_open_segment",
            side_effect=lambda seg: seg in open_segments,
        ),
    ):
        winners = _consume_claimed_vocab_gaps(gaps, batch, source_type="dd")
    outcomes = persist.call_args.args[1] if persist.called else []
    return winners, outcomes


def test_actionable_gap_retires_source_no_string():
    gaps = [
        {
            "source_id": "equilibrium/constraints/n_e_line/measured",
            "segment": "physical_base",
            "token": "line_integrated_density",
            "reason": "no base",
        }
    ]
    winners, outcomes = _run_marker(gaps, actionable_tokens={"line_integrated_density"})
    assert winners == gaps
    assert outcomes == [
        {
            "sns_id": "dd:equilibrium/constraints/n_e_line/measured",
            "source_id": "equilibrium/constraints/n_e_line/measured",
            "claim_token": "winner",
            "claim_seq": 8,
            "status": "vocab_gap",
            "last_error": None,
            "skip_reason": None,
            "skip_reason_detail": None,
        }
    ]


def test_nonactionable_gap_kept_retryable():
    gaps = [
        {
            "source_id": "summary/plasma_duration/value",
            "segment": "physical_base",
            "token": "plasma_pulse_duration",
            "reason": "decomposable",
        }
    ]
    winners, outcomes = _run_marker(gaps, actionable_tokens=set())
    assert winners == gaps
    assert outcomes[0]["status"] == "extracted"
    assert outcomes[0]["claim_token"] == "winner"
    assert outcomes[0]["claim_seq"] == 8


def test_open_segment_gap_ignored():
    gaps = [{"source_id": "x/y/z", "segment": "grammar_ambiguity", "token": "w"}]
    winners, outcomes = _run_marker(
        gaps, actionable_tokens=set(), open_segments={"grammar_ambiguity"}
    )
    assert winners == []
    assert outcomes == []


def test_gap_writer_failure_rolls_back_source_outcome():
    """A gap-writer failure cannot strand a source in a terminal state."""
    from imas_codex.standard_names import graph_ops

    tx = MagicMock()
    tx.closed = False
    tx.run = MagicMock(return_value=[{"id": "dd:x/y", "source_id": "x/y"}])
    session = MagicMock()
    session.begin_transaction.return_value = tx

    @contextmanager
    def _session():
        yield session

    gc = MagicMock()
    gc.__enter__.return_value = gc
    gc.__exit__.return_value = False
    gc.session = _session
    gaps = [
        {
            "source_id": "x/y",
            "segment": "physical_base",
            "token": "missing_quantity",
            "reason": "missing",
        }
    ]
    outcomes = [
        {
            "sns_id": "dd:x/y",
            "source_id": "x/y",
            "claim_token": "winner",
            "claim_seq": 8,
            "status": "vocab_gap",
        }
    ]
    with (
        patch.object(graph_ops, "GraphClient", return_value=gc),
        patch.object(
            graph_ops,
            "write_vocab_gaps",
            side_effect=RuntimeError("gap write failed"),
        ) as writer,
        pytest.raises(RuntimeError, match="gap write failed"),
    ):
        graph_ops.persist_claimed_vocab_gaps(gaps, outcomes, source_type="dd")

    writer.assert_called_once()
    assert writer.call_args.kwargs["gc"]._transaction is tx
    tx.commit.assert_not_called()
    tx.close.assert_called_once()
    assert not any(
        "sns.status = b.status" in item.args[0] for item in tx.run.call_args_list
    )


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
