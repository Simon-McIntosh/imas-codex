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
import logging
import re
from pathlib import Path
from unittest.mock import patch

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


class _ReviewGraph:
    """Capture only properties actually assigned by the review write queries."""

    def __init__(self) -> None:
        self.reviews: dict[str, dict] = {}
        self.names: dict[str, dict] = {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def query(self, statement: str, **params) -> list:
        batch = params.get("batch", [])
        if "MERGE (r:StandardNameReview" in statement:
            assignments = re.findall(r"\br\.(\w+)\s*=\s*b\.(\w+)", statement)
            for row in batch:
                self.reviews[row["id"]] = {
                    "id": row["id"],
                    **{
                        property_name: row[parameter_name]
                        for property_name, parameter_name in assignments
                    },
                }
        if "sn.reviewer_score_name = b.reviewer_score_name" in statement:
            for row in batch:
                self.names[row["id"]] = {
                    "reviewer_score_name": row["reviewer_score_name"]
                }
        return []


def _write_records(records: list[dict], graph: _ReviewGraph) -> None:
    from imas_codex.standard_names.graph_store.graph_catalog import write_reviews

    with patch(
        "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
        return_value=graph,
    ):
        write_reviews(records)


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


def test_write_reviews_persists_shadow_properties_and_null_score() -> None:
    """A review survives the batch and the assignments made by the writer."""
    rec = _shadow_record()
    graph = _ReviewGraph()
    _write_records([rec], graph)
    row = graph.reviews[rec["id"]]
    assert row["id"] == rec["id"]
    assert row["is_canonical"] is False
    assert row["score"] is None
    for field in _SHADOW_FIELDS:
        assert row[field] == rec[field]


# ---------------------------------------------------------------------------
# Accept and refine decisions are identical with shadow records present or absent
# ---------------------------------------------------------------------------


def _shadow_record() -> dict:
    return _build(
        _item(),
        model="typesafe/jev-1.13",
        is_canonical=False,
        reviewed_at="2026-01-01",
        judgment=_judgment(),
        judgment_model="typesafe/jev-1.13",
    )


class TestAcceptScoreSelection:
    """The accept gate's own score selection reads the reviewer's canonical score."""

    @staticmethod
    def _review(*, shadow: dict | None):
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
        review.judge_confidence_json = (
            shadow["judge_confidence_json"] if shadow else None
        )
        return review, logging.LoggerAdapter(logging.getLogger("shadow-test"), {})

    def _scored(self, *, shadow: bool) -> tuple[dict, _ReviewGraph]:
        from imas_codex.standard_names.graph_store.graph_catalog import (
            write_name_review_results,
        )
        from imas_codex.standard_names.review.pipeline import (
            _match_reviews_to_entries,
        )

        graph = _ReviewGraph()
        if shadow:
            _write_records([_shadow_record()], graph)
        persisted = next(iter(graph.reviews.values()), None)
        review, wlog = self._review(shadow=persisted)
        scored, _, _ = _match_reviews_to_entries([review], [_item()], wlog)
        with patch(
            "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
            return_value=graph,
        ):
            write_name_review_results(scored)
        return scored[0], graph

    def test_accept_score_ignores_a_shadow_payload_beside_it(self) -> None:
        """A judge's answers beside the reviewer's score do not move the accept
        decision: the entry mirrored for the name axis is identical either way.
        """
        canonical, empty_graph = self._scored(shadow=False)
        beside, shadow_graph = self._scored(shadow=True)
        assert not empty_graph.reviews
        assert len(shadow_graph.reviews) == 1
        stored = next(iter(shadow_graph.reviews.values()))
        assert json.loads(stored["judge_confidence_json"])["is_single_quantity"] == 0.9
        assert canonical["reviewer_score"] == pytest.approx(0.72)
        assert canonical["reviewer_score"] == beside["reviewer_score"]
        assert canonical["review_tier"] == beside["review_tier"]
        assert empty_graph.names == shadow_graph.names


class TestRefineTriggerIgnoresShadowRecords:
    def test_refine_decision_is_identical_with_shadow_attributes_present(self) -> None:
        """The refining trigger reads no review node, so a shadow record is invisible."""
        from imas_codex.standard_names.pools import standard_name_pool_predicates

        refine = None
        for predicate in standard_name_pool_predicates():
            if predicate.pool == "refine_name":
                refine = predicate.admits
        assert refine is not None
        graph = _ReviewGraph()
        _write_records([_shadow_record()], graph)
        assert len(graph.reviews) == 1
        shadow = next(iter(graph.reviews.values()))
        assert shadow["judge_kind"] == "jev"
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
        assert bool(refine({**state, "review_records": [shadow]})) is decision
