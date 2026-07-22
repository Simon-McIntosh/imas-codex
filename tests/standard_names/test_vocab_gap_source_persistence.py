"""The compose-time vocab-gap outcome must be persisted on each source.

A source blocked by a genuinely-absent closed-segment token is retired to
``vocab_gap`` with the blocking ``segment:token`` recorded in
``skip_reason_detail`` — the source is self-describing, no join required.

A source whose reported gap is non-actionable (decomposable / wrong-slot /
ambiguous / false positive) is NOT a vocabulary deficiency: the composer
erred and the source is still nameable, so it must stay retryable under the
attempt-count cap rather than be stranded at ``vocab_gap``.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

from imas_codex.standard_names.workers import _update_sources_after_vocab_gap

_WLOG = logging.getLogger("test-vocab-gap")


def _run(gaps, *, actionable_tokens, open_segments=()):
    """Invoke the marker with GraphClient + classification mocked.

    Returns the list of (cypher, kwargs) query calls the function issued.
    """
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(return_value=[])

    def _is_actionable(segment, token):
        return token in actionable_tokens

    def _is_open(segment):
        return segment in open_segments

    with (
        patch("imas_codex.graph.client.GraphClient", return_value=gc),
        patch(
            "imas_codex.standard_names.segments.is_actionable_gap",
            side_effect=_is_actionable,
        ),
        patch(
            "imas_codex.standard_names.segments.is_open_segment",
            side_effect=_is_open,
        ),
    ):
        _update_sources_after_vocab_gap(gaps, "dd", _WLOG)

    return [(c.args[0], c.kwargs) for c in gc.query.call_args_list]


def _find(calls, needle):
    return [(cy, kw) for cy, kw in calls if needle in cy]


def test_actionable_gap_retires_and_records_detail():
    gaps = [
        {
            "source_id": "equilibrium/constraints/n_e_line/measured",
            "segment": "physical_base",
            "token": "line_integrated_density",
            "reason": "no base for line-integrated density",
        }
    ]
    calls = _run(gaps, actionable_tokens={"line_integrated_density"})

    retire = _find(calls, "status = 'vocab_gap'")
    assert retire, "actionable gap must retire the source to vocab_gap"
    _, kw = retire[0]
    rows = kw["rows"]
    assert rows[0]["sns_id"] == "dd:equilibrium/constraints/n_e_line/measured"
    # The blocking segment:token MUST be persisted on the source.
    assert "physical_base:line_integrated_density" in rows[0]["detail"]
    # And the retire query must write skip_reason_detail.
    assert "skip_reason_detail" in retire[0][0]


def test_nonactionable_gap_kept_retryable_not_stranded():
    gaps = [
        {
            "source_id": "summary/plasma_duration/value",
            "segment": "physical_base",
            "token": "plasma_pulse_duration",  # decomposable → non-actionable
            "reason": "decomposable compound",
        }
    ]
    calls = _run(gaps, actionable_tokens=set())

    # It must NOT be retired to vocab_gap.
    assert not _find(calls, "status = 'vocab_gap'"), (
        "a non-actionable gap must not strand the source at vocab_gap"
    )
    # It must be routed through the bounded attempt-count retry path...
    retry = _find(calls, "attempt_count")
    assert retry, "non-actionable gap must keep the source retryable"
    _, kw = retry[0]
    rows = kw["rows"]
    assert rows[0]["sns_id"] == "dd:summary/plasma_duration/value"
    # ...still recording why the composer stumbled.
    assert "physical_base:plasma_pulse_duration" in rows[0]["detail"]


def test_open_segment_gap_ignored():
    gaps = [
        {
            "source_id": "x/y/z",
            "segment": "grammar_ambiguity",
            "token": "whatever",
            "reason": "pseudo",
        }
    ]
    calls = _run(gaps, actionable_tokens=set(), open_segments={"grammar_ambiguity"})
    # Pseudo/open-segment gaps neither retire nor retry the source.
    assert not _find(calls, "status = 'vocab_gap'")
    assert not _find(calls, "attempt_count")
