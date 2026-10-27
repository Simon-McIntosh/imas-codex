"""The review path has one decider, one audit panel and one fallback seat.

Jev decides both review axes. A deterministic fraction of decided items runs the
paid panel as a non-deciding audit, and an item Jev could not judge falls to the
single chat reviewer. These tests pin that shape: each of them fails when a part
of the quorum's deciding path is restored alongside it.
"""

from __future__ import annotations

import ast
import pathlib
from functools import partial
from unittest.mock import patch

import pytest

from imas_codex.standard_names.review import cascade, panel_audit
from imas_codex.standard_names.worker_tasks import worker_review_names

_WORKER_SOURCE = pathlib.Path(worker_review_names.__file__).read_text()
_CASCADE_SOURCE = pathlib.Path(cascade.__file__).read_text()


def _defined_function_names(source: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }


def test_the_persisting_quorum_cycle_runner_is_not_defined():
    """The write-the-whole-chain wrapper is gone, from the module and the API."""
    from imas_codex.standard_names import worker_tasks

    assert not hasattr(worker_review_names, "_persisting_quorum_cycles")
    exported = {
        name for names in worker_tasks._OWNERS_BY_MODULE.values() for name in names
    }
    assert "_persisting_quorum_cycles" not in exported
    assert "_persisting_quorum_cycles" not in _defined_function_names(_WORKER_SOURCE)
    assert "_incumbent_then_unjudged" not in _defined_function_names(_CASCADE_SOURCE)


def test_the_retained_runner_persists_nothing():
    """The cycle runner the audit uses stays free of the canonical write."""
    source = _defined_function_names(_WORKER_SOURCE)
    assert "_run_rd_quorum_cycles" in source
    assert "_persisting_chat_review" in source


@pytest.mark.asyncio
async def test_an_unjudged_item_reaches_the_single_chat_reviewer(monkeypatch):
    """Items Jev could not judge take one chat seat, never the quorum chain."""
    seats_seen: list[list[str]] = []
    stored: list[tuple[str, str]] = []

    async def chat_reviewer(*, models):
        seats_seen.append(list(models))
        return {"winning_score": 0.31, "records": [{"is_canonical": True}]}

    async def store_unjudged(item, axis, model, reason):
        stored.append((str(item["id"]), "unjudged"))

    monkeypatch.setattr(cascade, "_store_unjudged", store_unjudged)
    with patch(
        "imas_codex.discovery.base.judgment.decisions_key_present",
        return_value=False,
    ):
        result = await cascade.adjudicate_review(
            "name",
            {"id": "temperature"},
            ["seat-a", "seat-b", "seat-c"],
            chat_reviewer,
            mgr=object(),
        )

    assert result is not None
    assert seats_seen == [["seat-a"]], (
        "an unjudged item takes one chat seat; the quorum chain is not run"
    )
    assert stored == [("temperature", "unjudged")]


@pytest.mark.asyncio
async def test_the_chat_review_persister_refuses_more_than_one_seat():
    """The persisting fallback is one reviewer, so a chain is refused."""
    resolver = lambda name: worker_review_names._run_rd_quorum_cycles
    with pytest.raises(ValueError, match="one seat"):
        await worker_review_names._persisting_chat_review(
            resolver, models=["seat-a", "seat-b"]
        )


def test_no_review_seat_resolves_outside_the_chat_reviewer_and_audit_panel():
    """No review seat reaches a caller through ``get_model``.

    The chat reviewer takes the first configured seat and the audit panel runs
    the configured chain, so every review seat is one of those two. A review
    seat registered as a model section would be resolvable anywhere, which is
    the shape a restored deciding path needs and this test refuses.
    """
    from imas_codex.settings import MODEL_SECTIONS, get_model_source_models

    review_sections = sorted(
        section
        for section in MODEL_SECTIONS
        if "review" in section or section.startswith("sn-review")
    )
    assert review_sections == [], (
        "review seats resolve through the review accessors, not get_model: "
        f"{review_sections}"
    )
    seats = set(get_model_source_models("sn-review:names")) | set(
        get_model_source_models("sn-review:docs")
    )
    assert seats, "the audit panel and the chat reviewer resolve their seats here"
    for source in ("sn-review:names", "sn-review:docs"):
        assert set(get_model_source_models(source)) <= seats


def test_the_audit_still_recognises_the_persisting_fallback():
    """The panel audit keeps the seam it reads the panel's verdict through."""
    reviewer = partial(
        worker_review_names._persisting_chat_review,
        sn_id="temperature",
        review_axis="name",
    )
    rebound = panel_audit.quorum_without_persisting(reviewer)
    assert rebound.func is worker_review_names._run_rd_quorum_cycles
    assert rebound.keywords == reviewer.keywords