"""Tests for the code scan progress line built by scan_worker.

A claimed FacilityPath may carry no score (``--min-score 0`` admits unscored
paths), so the progress line must render a ``None`` score as a placeholder
rather than crash on ``str.format``.
"""

from __future__ import annotations

from imas_codex.discovery.code.workers import _render_score, _scan_progress_message


def test_unscored_path_renders_as_placeholder():
    """A None score renders as '-' instead of raising on format."""
    assert _render_score(None) == "-"


def test_scored_path_renders_two_decimals():
    assert _render_score(0.85) == "0.85"


def test_scan_progress_message_accepts_unscored_paths():
    """The message builds when every claimed path is unscored."""
    message = _scan_progress_message(
        [{"path": "/analysis/src/a.c", "score": None} for _ in range(3)]
    )
    assert message == "scanning 3 paths (scores: -, -, -...)"


def test_scan_progress_message_mixes_scored_and_unscored():
    message = _scan_progress_message(
        [
            {"path": "/analysis/src/a.c", "score": 0.9},
            {"path": "/analysis/src/b.c", "score": None},
            {"path": "/analysis/src/c.c", "score": 0.5},
        ]
    )
    assert message == "scanning 3 paths (scores: 0.90, -, 0.50...)"


def test_bool_score_is_not_formatted_as_a_number():
    """A boolean is not a score, so it renders as the placeholder."""
    assert _render_score(True) == "-"
