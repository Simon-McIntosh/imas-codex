"""Tests for the documents pipeline's public pending-work predicate."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from imas_codex.discovery.documents import pipeline


def test_true_when_images_are_pending_to_fetch(monkeypatch):
    monkeypatch.setattr(pipeline, "_has_pending_image_documents", lambda f: True)
    monkeypatch.setattr(pipeline, "_has_pending_image_scores", lambda f: False)
    assert pipeline.has_pending_work("jt-60sa") is True


def test_true_when_images_are_pending_to_score(monkeypatch):
    monkeypatch.setattr(pipeline, "_has_pending_image_documents", lambda f: False)
    monkeypatch.setattr(pipeline, "_has_pending_image_scores", lambda f: True)
    assert pipeline.has_pending_work("jt-60sa") is True


def test_false_when_nothing_is_pending(monkeypatch):
    monkeypatch.setattr(pipeline, "_has_pending_image_documents", lambda f: False)
    monkeypatch.setattr(pipeline, "_has_pending_image_scores", lambda f: False)
    assert pipeline.has_pending_work("jt-60sa") is False


def test_the_score_check_is_not_consulted_when_a_fetch_is_pending(monkeypatch):
    monkeypatch.setattr(pipeline, "_has_pending_image_documents", lambda f: True)
    scores = MagicMock(return_value=False)
    monkeypatch.setattr(pipeline, "_has_pending_image_scores", scores)
    assert pipeline.has_pending_work("jt-60sa") is True
    scores.assert_not_called()


def test_a_failed_query_reaches_the_caller(monkeypatch):
    def boom(facility: str) -> bool:
        raise RuntimeError("graph unavailable")

    monkeypatch.setattr(pipeline, "_has_pending_image_documents", boom)
    with pytest.raises(RuntimeError):
        pipeline.has_pending_work("jt-60sa")
