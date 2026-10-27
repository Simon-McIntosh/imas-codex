"""The review path has one decider, one audit panel and one fallback seat.

Jev decides both review axes. A deterministic fraction of decided items runs the
paid panel as a non-deciding audit, and an item Jev could not judge falls to the
single chat reviewer. These tests pin that shape: each of them fails when a part
of the quorum's deciding path is restored alongside it.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib
from functools import partial
from unittest.mock import patch

import pytest

from imas_codex.standard_names.review import cascade, panel_audit
from imas_codex.standard_names.worker_tasks import worker_review_names

_WORKER_SOURCE = pathlib.Path(worker_review_names.__file__).read_text()
_CASCADE_SOURCE = pathlib.Path(cascade.__file__).read_text()

_NAMES_DIMS = ("grammar", "semantic", "convention", "completeness")


def _defined_function_names(source: str) -> set[str]:
    return {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef)
    }


def _seats_declared_in_pyproject() -> dict[str, set[str]]:
    """The review seats this checkout declares, read from the config itself.

    An expectation read from the same accessor it checks would agree with a
    shorted chain, so the declared lists are taken from ``pyproject.toml``
    directly: a section's own ``models`` plus every profile's.
    """
    import tomllib

    root = pathlib.Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text())
    sn_review = config["tool"]["imas-codex"]["sn-review"]
    declared: dict[str, set[str]] = {}
    for axis in ("names", "docs"):
        section = sn_review[axis]
        seats = set(section.get("models") or ())
        for profile in (section.get("profiles") or {}).values():
            seats |= set(profile.get("models") or ())
        declared[f"sn-review:{axis}"] = seats
    return declared


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


class _FakeScores:
    def __init__(self, dims: dict[str, float]) -> None:
        self._dims = dict(dims)

    def model_dump(self) -> dict[str, float]:
        return dict(self._dims)


class _FakeReviewItem:
    def __init__(self, dims: dict[str, float]) -> None:
        self.scores = _FakeScores(dims)
        self.reasoning = "looks fine"
        self.comments = None


class _FakeBatch:
    def __init__(self, reviews: list) -> None:
        self.reviews = reviews


def test_the_retained_runner_persists_nothing(monkeypatch):
    """The cycle runner the audit uses writes no canonical row.

    Driven for real: two configured seats run two cycles, and the runner hands
    the rows back to its caller. The canonical write belongs to the persisting
    wrapper, so the graph writer must not be called from here at all.
    """
    written: list[list[dict]] = []

    def record_writes(records):
        written.append(list(records))

    monkeypatch.setattr(
        "imas_codex.standard_names.graph_store.graph_catalog.write_reviews",
        record_writes,
    )

    async def acall(**_kwargs):
        dims = dict.fromkeys(_NAMES_DIMS, 16.0)
        return (_FakeBatch([_FakeReviewItem(dims)]), 0.01, {})

    result = asyncio.run(
        worker_review_names._run_rd_quorum_cycles(
            sn_id="temperature",
            review_axis="name",
            response_model=None,
            user_prompt="u",
            system_prompt="s",
            models=["seat-a", "seat-b"],
            disagreement_threshold=0.2,
            rubric_dims=tuple(_NAMES_DIMS),
            phase="review",
            acall_llm_structured=acall,
        )
    )

    assert result is not None
    assert len(result["records"]) == 2, "one record per cycle, handed to the caller"
    assert written == [], (
        "the retained runner writes no review row; persistence is the caller's"
    )


@pytest.mark.asyncio
async def test_an_unjudged_item_reaches_the_single_chat_reviewer(monkeypatch):
    """Items Jev could not judge take one chat seat, never the quorum chain.

    The shadow row is asserted on the four values the code under test is
    responsible for producing — the item it fell back for, the axis, the
    unavailable judge's label and the reason it could not judge — rather than on
    anything the stub itself invented.
    """
    item = {"id": "temperature", "description": "the plasma temperature"}
    seats_seen: list[list[str]] = []
    stored: list[tuple[dict, str, str, str]] = []

    async def chat_reviewer(*, models):
        seats_seen.append(list(models))
        return {"winning_score": 0.31, "records": [{"is_canonical": True}]}

    async def store_unjudged(stored_item, axis, model, reason):
        stored.append((stored_item, axis, model, reason))

    monkeypatch.setattr(cascade, "_store_unjudged", store_unjudged)
    monkeypatch.setattr(
        "imas_codex.discovery.base.judgment.decisions_key_absent_reason",
        lambda: "no decision key on this host",
    )
    with patch(
        "imas_codex.discovery.base.judgment.decisions_key_present",
        return_value=False,
    ):
        result = await cascade.adjudicate_review(
            "name",
            item,
            ["seat-a", "seat-b", "seat-c"],
            chat_reviewer,
            mgr=object(),
        )

    assert result is not None
    assert seats_seen == [["seat-a"]], (
        "an unjudged item takes one chat seat; the quorum chain is not run"
    )
    assert stored == [
        (item, "name", "sn-judge:unavailable", "no decision key on this host")
    ], "the shadow row records the item, axis, judge and reason it fell back for"


@pytest.mark.asyncio
async def test_the_chat_review_persister_refuses_more_than_one_seat():
    """The persisting fallback is one reviewer, so a chain is refused."""

    def resolver(_name):
        return worker_review_names._run_rd_quorum_cycles

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
    expected = _seats_declared_in_pyproject()
    assert expected["sn-review:names"], "the names chain is declared in pyproject"
    assert expected["sn-review:docs"], "the docs chain is declared in pyproject"
    for source, declared in expected.items():
        assert set(get_model_source_models(source)) == declared


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
