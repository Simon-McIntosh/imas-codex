"""A challenge judge's answers are stored beside the incumbent's, read by no gate.

A ``StandardNameReview`` record produced by a challenge judge carries the judge's
per-question answers, its distributions, its confidences and the model that
produced them, discriminated by ``judge_kind``. It is never canonical and carries
no scalar ``score``, so the accepting path — which mirrors a score onto the
StandardName and reads no review node — cannot follow it. Every test here is
offline: the graph is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

yaml = pytest.importorskip("yaml")

_REPO = Path(__file__).resolve().parents[2]
_SCHEMA = _REPO / "imas_codex" / "schemas" / "standard_name.yaml"
_GENERATED = _REPO / "imas_codex" / "graph" / "models.py"

_SHADOW_FIELDS = (
    "judge_kind",
    "judgment_model",
    "judge_answers_json",
    "judge_probs_json",
    "judge_confidence_json",
)


def _schema() -> dict:
    return yaml.safe_load(_SCHEMA.read_text())


def _item(sn_id: str = "electron_temperature") -> dict:
    return {
        "id": sn_id,
        "reviewer_score": 0.72,
        "review_tier": "good",
        "reviewer_comments": "fine",
    }


def _judgment() -> dict:
    return {
        "is_single_quantity": {"answer": "yes", "probs": [0.9, 0.1], "confidence": 0.9},
        "convention_sign": {
            "answer": "negative",
            "probs": [0.2, 0.8],
            "confidence": 0.8,
        },
    }


def _build(*args, **kwargs):
    from imas_codex.standard_names.review.pipeline import _build_review_record

    return _build_review_record(*args, **kwargs)


def _write_records(records: list[dict]) -> list:
    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])
    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_reviews

        write_reviews(records)
    for call in mock_gc.query.call_args_list:
        if "batch" in call.kwargs:
            return call.kwargs["batch"]
    raise AssertionError("no batch kwarg reached the write path")


# ---------------------------------------------------------------------------
# The schema declares the fields, and the regenerated models carry them
# ---------------------------------------------------------------------------


def test_schema_declares_the_shadow_fields_on_the_review_record() -> None:
    attributes = _schema()["classes"]["StandardNameReview"]["attributes"]
    for field in _SHADOW_FIELDS:
        assert field in attributes, f"{field} is not declared on StandardNameReview"
    assert attributes["judge_kind"]["range"] == "StandardNameReviewJudgeKind"


def test_schema_declares_the_judge_kinds() -> None:
    kinds = _schema()["enums"]["StandardNameReviewJudgeKind"]["permissible_values"]
    assert set(kinds) == {"incumbent", "jev"}


def test_regenerated_models_carry_the_shadow_fields() -> None:
    text = _GENERATED.read_text()
    assert "StandardNameReviewJudgeKind" in text
    for field in _SHADOW_FIELDS:
        assert field in text, f"{field} is absent from the regenerated models"


# ---------------------------------------------------------------------------
# The record builder
# ---------------------------------------------------------------------------


def test_incumbent_record_is_incumbent_and_keeps_its_score() -> None:
    rec = _build(
        _item(), model="openrouter/x/y", is_canonical=True, reviewed_at="2026-01-01"
    )
    assert rec["judge_kind"] == "incumbent"
    assert rec["is_canonical"] is True
    assert rec["score"] == pytest.approx(0.72)
    assert rec["judge_answers_json"] is None


def test_challenge_judge_record_is_never_canonical_and_carries_no_score() -> None:
    rec = _build(
        _item(),
        model="typesafe/jev-1.13",
        # a caller that asks for canonical still gets a non-canonical record
        is_canonical=True,
        reviewed_at="2026-01-01",
        judgment=_judgment(),
        judgment_model="typesafe/jev-1.13",
    )
    assert rec["judge_kind"] == "jev"
    assert rec["is_canonical"] is False
    assert rec["score"] is None
    assert rec["judgment_model"] == "typesafe/jev-1.13"
    assert json.loads(rec["judge_answers_json"])["is_single_quantity"] == "yes"
    assert json.loads(rec["judge_probs_json"])["convention_sign"] == [0.2, 0.8]
    assert json.loads(rec["judge_confidence_json"])["convention_sign"] == 0.8


def test_shadow_record_round_trips_through_the_record_builder() -> None:
    """The builder round-trips a full shadow payload, keyed by question name."""
    rec = _shadow_record()
    assert rec["judge_kind"] == "jev"
    assert rec["is_canonical"] is False
    assert json.loads(rec["judge_answers_json"])["is_single_quantity"] == "yes"
    assert json.loads(rec["judge_probs_json"])["convention_sign"] == [0.2, 0.8]
    assert json.loads(rec["judge_confidence_json"])["is_single_quantity"] == 0.9


def test_write_reviews_carries_no_shadow_property() -> None:
    """The review write path's fixed property list drops the shadow payload.

    ``write_reviews`` builds its batch and its ``SET`` clause from an explicit
    property list, so a shadow record reaches the graph with its judge fields
    omitted and only its canonical-shaped fields persisted. Extending that list
    is a change to ``imas_codex/standard_names/graph_ops.py``, outside this
    node's write scope; the assertion below trips the moment it happens, so the
    gap cannot be closed silently.
    """
    rec = _shadow_record()
    row = _write_records([rec])[0]
    assert row["id"] == rec["id"]
    assert row["is_canonical"] is False
    for field in _SHADOW_FIELDS:
        assert field not in row, (
            f"{field} now reaches the graph — extend this test to assert the "
            "shadow payload round-trips through the write path"
        )


# ---------------------------------------------------------------------------
# Accept and refine decisions are identical with shadow records present or absent
# ---------------------------------------------------------------------------


def _mirror_batch(entries: list[dict]) -> list:
    """Run the accepting path's name-axis score mirror with a mocked graph."""
    mock_gc = MagicMock()
    mock_gc.query = MagicMock(return_value=[])
    with patch("imas_codex.standard_names.graph_ops.GraphClient") as MockGC:
        MockGC.return_value.__enter__ = MagicMock(return_value=mock_gc)
        MockGC.return_value.__exit__ = MagicMock(return_value=False)
        from imas_codex.standard_names.graph_ops import write_name_review_results

        write_name_review_results([dict(e) for e in entries])
    for call in mock_gc.query.call_args_list:
        if "batch" in call.kwargs:
            return call.kwargs["batch"]
    raise AssertionError("no batch kwarg reached the accept mirror")


def _accept_entries(sn_id: str = "electron_temperature") -> list[dict]:
    """The entries the accepting path mirrors, as the pipeline builds them."""
    return [
        {
            "id": sn_id,
            "reviewer_score": 0.72,
            "reviewer_score_name": 0.72,
            "reviewed_name_at": "2026-01-01",
        }
    ]


def _shadow_record() -> dict:
    return _build(
        _item(),
        model="typesafe/jev-1.13",
        is_canonical=False,
        reviewed_at="2026-01-01",
        judgment=_judgment(),
        judgment_model="typesafe/jev-1.13",
    )


class TestAcceptPathIgnoresShadowRecords:
    def test_mirror_is_identical_with_shadow_records_present(self) -> None:
        """Writing a challenge judge's record does not perturb the accept mirror."""
        alone = _mirror_batch(_accept_entries())
        _write_records([_shadow_record()])  # the shadow channel, in the same run
        beside = _mirror_batch(_accept_entries())
        assert alone == beside

    def test_a_mirror_pointed_at_a_shadow_field_would_change(self) -> None:
        """The identity above is load-bearing, not a property of the fixture.

        The accept mirror reads ``reviewer_score``. Pointing that read at the
        challenge judge's confidence instead writes a different score, so the
        equality asserted beside it can only hold while the accept path reads no
        shadow field.
        """
        entries = _accept_entries()
        answered = _shadow_record()
        leaked = [dict(e) for e in entries]
        leaked[0]["reviewer_score"] = json.loads(answered["judge_confidence_json"])[
            "is_single_quantity"
        ]
        assert leaked[0]["reviewer_score"] != entries[0]["reviewer_score"]
        assert _mirror_batch(entries) != _mirror_batch(leaked)


class TestAcceptScoreSelection:
    """The accept gate's own score selection reads the reviewer's canonical score."""

    @staticmethod
    def _review(*, shadow: bool):
        import logging

        class _Scores:
            score = 0.72
            tier = "good"

            @staticmethod
            def model_dump() -> dict:
                return {"clarity": 18, "physics": 20}

        class _Review:
            source_id = "electron_temperature"
            standard_name = "electron_temperature"
            reasoning = "fine"
            comments = None
            scores = _Scores()

        review = _Review()
        if shadow:
            review.judge_confidence_json = _shadow_record()["judge_confidence_json"]
        return review, logging.LoggerAdapter(logging.getLogger("shadow-test"), {})

    def _scored(self, *, shadow: bool) -> dict:
        from imas_codex.standard_names.review.pipeline import (
            _match_reviews_to_entries,
        )

        review, wlog = self._review(shadow=shadow)
        scored, _, _ = _match_reviews_to_entries([review], [_item()], wlog)
        return scored[0]

    def test_accept_score_ignores_a_shadow_payload_beside_it(self) -> None:
        """A judge's answers beside the reviewer's score do not move the accept
        decision: the entry mirrored for the name axis is identical either way.
        """
        canonical = self._scored(shadow=False)
        beside = self._scored(shadow=True)
        assert canonical["reviewer_score"] == pytest.approx(0.72)
        assert canonical["reviewer_score"] == beside["reviewer_score"]
        assert canonical["review_tier"] == beside["review_tier"]


class TestRefineTriggerIgnoresShadowRecords:
    def test_refine_decision_is_identical_with_shadow_attributes_present(self) -> None:
        """The refining trigger reads no review node, so a shadow record is invisible."""
        from imas_codex.standard_names.pools import standard_name_pool_predicates

        refine = None
        for predicate in standard_name_pool_predicates():
            if predicate.pool == "refine_name":
                refine = predicate.admits
        if refine is None:
            pytest.skip("no refine_name pool predicate is declared")
        state = {
            "name_stage": "reviewed",
            "name_score_state": "below",
            "name_attempts_under_cap": True,
            "rename_resubmit_capped": False,
            "origin": "generated",
            "review_quorum_shortfall": False,
        }
        decision = bool(refine(state))
        assert decision is True
        assert bool(refine({**state, "judge_kind": "jev"})) is decision
        assert (
            bool(
                refine(
                    {
                        **state,
                        "judge_answers_json": _shadow_record()["judge_answers_json"],
                    }
                )
            )
            is decision
        )
