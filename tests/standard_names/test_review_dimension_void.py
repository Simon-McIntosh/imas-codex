"""The sanctioned route that voids one review dimension and records why.

A review dimension can be discredited without the rest of the review being
wrong. The route takes no position on WHAT discredits one — the reason is the
caller's free text — so these tests pin the two halves that make it worth
having rather than any particular cause:

* evidence is kept — the voided dimension's original number is still returned
  by the accessor that returned it before, with a void record beside it naming
  the dimension, the actor, the reason, the timestamp and the grammar
  signature;
* the void is CONSULTED — the review score is recomputed over the surviving
  dimensions, so ``update_review_aggregates`` and the canonical-review
  projection honour it without knowing it happened.

The graph double below serves exactly the statements this route issues. Its
aggregate branch reproduces the arithmetic of ``update_review_aggregates``'s
Cypher (``avg(r.score)`` over ``HAS_REVIEW``) so the default tier can measure
the recomputation; ``test_aggregate_mean_follows_the_void_live`` runs the real
statement against Neo4j under the ``graph`` marker.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from imas_codex.standard_names.graph_store.graph_catalog import (
    ReviewDimensionVoidRefused,
    plan_review_dimension_void,
    update_review_aggregates,
    void_review_dimension,
)
from imas_codex.standard_names.review.projection import project_canonical_review

REVIEW_ID = "electron_temperature:name:group-a:0"
SN_ID = "electron_temperature"
# grammar 12, semantic 18, convention 16, completeness 14 -> 60/80 = 0.75
DIMENSIONS = {"grammar": 12, "semantic": 18, "convention": 16, "completeness": 14}
SIGNATURE = "vocab-digest-12b5573"
# The reason is whatever the caller says it is; the route stores it verbatim.
REASON = "review prompt encoded a retired operator convention"


class FakeGraph:
    """In-memory stand-in serving the statements this route issues."""

    def __init__(self, reviews: list[dict], names: dict | None = None) -> None:
        self.reviews = {r["id"]: dict(r) for r in reviews}
        self.names = names or {SN_ID: {"id": SN_ID, "name_stage": "reviewed"}}
        self.queries: list[tuple[str, dict]] = []
        self.changes: list[dict] = []

    # -- helpers -----------------------------------------------------------
    @property
    def writes(self) -> list[tuple[str, dict]]:
        """Every statement that would mutate the graph."""
        return [
            (q, p)
            for q, p in self.queries
            if "SET " in q or "CREATE " in q or "MERGE " in q
        ]

    def close(self) -> None:  # pragma: no cover - parity with GraphClient
        pass

    def __enter__(self) -> FakeGraph:
        return self

    def __exit__(self, *exc) -> None:
        return None

    # -- the query surface -------------------------------------------------
    def query(self, cypher: str, **params):
        self.queries.append((cypher, params))
        if "SET r.score = $score" in cypher:
            review = self.reviews[params["review_id"]]
            review["score"] = params["score"]
            review["tier"] = params["tier"]
            review["voided_dimensions_json"] = params["voided_dimensions_json"]
            return [{"id": review["id"]}]
        if "CREATE (change:StandardNameChange" in cypher:
            self.changes.append(dict(params))
            return []
        if "MATCH (r:StandardNameReview {id: $review_id})" in cypher:
            review = self.reviews.get(params["review_id"])
            return [dict(review)] if review else []
        if "-[:HAS_REVIEW]->(r:StandardNameReview {review_axis: $axis})" in cypher:
            return [
                dict(r)
                for r in self.reviews.values()
                if r["standard_name_id"] == params["sn_id"]
                and r["review_axis"] == params["axis"]
            ]
        if "avg(r.score) AS mean" in cypher:
            # The arithmetic of update_review_aggregates' Cypher.
            out = []
            for sid in params["ids"]:
                name = self.names.get(sid)
                if name is None or name.get("name_stage") in params.get(
                    "frozen_name_stages", []
                ):
                    continue
                scores = [
                    float(r["score"])
                    for r in self.reviews.values()
                    if r["standard_name_id"] == sid and r["score"] is not None
                ]
                name["review_count"] = len(scores)
                name["review_mean_score"] = (
                    sum(scores) / len(scores) if scores else None
                )
                out.append({"id": sid})
            return out
        raise AssertionError(f"unexpected statement: {cypher}")


def _review(
    review_id: str = REVIEW_ID,
    *,
    dims: dict | None = None,
    cycle_index: int = 0,
    resolution_method: str | None = "single_review",
    voided: str | None = None,
) -> dict:
    dims = DIMENSIONS if dims is None else dims
    return {
        "id": review_id,
        "standard_name_id": SN_ID,
        "review_axis": "name",
        "review_group_id": "group-a",
        "cycle_index": cycle_index,
        "resolution_method": resolution_method,
        "score": sum(dims.values()) / (len(dims) * 20.0),
        "scores_json": json.dumps(dims),
        "comments_per_dim_json": None,
        "comments": "reviewed",
        "tier": "good",
        "reviewer_model": "model-a",
        "voided_dimensions_json": voided,
    }


def _void(gc, dimension="grammar", **kwargs):
    """Apply a void through the real route, with *gc* as the whole graph.

    The route refreshes ``review_mean_score`` through its own client, so the
    double stands in for that connection too.
    """
    kwargs.setdefault("actor", "catalog-editor")
    kwargs.setdefault("reason", REASON)
    kwargs.setdefault("grammar_signature", SIGNATURE)
    with patch(
        "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
        lambda *a, **k: gc,
    ):
        return void_review_dimension(REVIEW_ID, dimension, gc=gc, **kwargs)


# ---------------------------------------------------------------------------
# 1. surviving dimension scores are untouched
# ---------------------------------------------------------------------------


def test_surviving_dimension_scores_are_byte_identical():
    gc = FakeGraph([_review()])
    before = gc.reviews[REVIEW_ID]["scores_json"]

    _void(gc)

    after = gc.reviews[REVIEW_ID]["scores_json"]
    assert after == before
    stored = json.loads(after)
    assert stored["semantic"] == 18
    assert stored["convention"] == 16
    assert stored["completeness"] == 14
    # and the voided dimension's own number survives beside the others
    assert stored["grammar"] == 12


# ---------------------------------------------------------------------------
# 2. the derived review score is the mean of the survivors
# ---------------------------------------------------------------------------


def test_review_score_equals_mean_of_surviving_dimensions():
    gc = FakeGraph([_review()])
    assert gc.reviews[REVIEW_ID]["score"] == pytest.approx(60 / 80.0)

    result = _void(gc)

    expected = ((18 + 16 + 14) / 3.0) / 20.0  # 0.8
    assert expected == pytest.approx(0.8)
    assert result["score"] == pytest.approx(expected)
    assert gc.reviews[REVIEW_ID]["score"] == pytest.approx(expected)
    assert gc.reviews[REVIEW_ID]["tier"] == "good"


# ---------------------------------------------------------------------------
# 3. the original number is still returned by the same accessor
# ---------------------------------------------------------------------------


def test_voided_score_survives_in_the_canonical_projection():
    gc = FakeGraph([_review()])
    before = project_canonical_review(SN_ID, "name", gc)
    assert json.loads(before.scores_json)["grammar"] == 12

    _void(gc)

    after = project_canonical_review(SN_ID, "name", gc)
    assert after.scores_json == before.scores_json
    assert json.loads(after.scores_json)["grammar"] == 12
    # the projection's aggregate score DID move — the void is consulted
    assert before.score == pytest.approx(0.75)
    assert after.score == pytest.approx(0.8)

    record = json.loads(gc.reviews[REVIEW_ID]["voided_dimensions_json"])[0]
    assert record["dimension"] == "grammar"
    assert record["void_actor"] == "catalog-editor"
    assert record["void_reason"] == REASON
    assert record["void_grammar_signature"] == SIGNATURE
    assert record["void_active"] is True
    assert record["voided_at"]


def test_void_writes_a_change_ledger_record():
    gc = FakeGraph([_review()])
    _void(gc)
    assert len(gc.changes) == 1
    change = gc.changes[0]
    assert change["operation"] == "void_review_dimension"
    assert change["to_name"] == SN_ID
    assert REASON in change["reason"]


def test_the_reason_is_the_callers_and_the_signature_is_an_observation():
    """The route stores the reason verbatim and infers no cause of its own.

    Two voids with unrelated reasons produce records that differ only in what
    the caller said, and the recorded signature is the vocabulary digest
    observed at void time — an environment fact, not a justification, and not
    read back out of the review being voided.
    """
    stated = [
        "review prompt encoded a retired operator convention",
        "reviewer scored the wrong quantity",
    ]
    records = []
    for reason in stated:
        gc = FakeGraph([_review()])
        _void(gc, reason=reason)
        records.append(json.loads(gc.reviews[REVIEW_ID]["voided_dimensions_json"])[0])

    assert [r["void_reason"] for r in records] == stated
    # nothing else about the record varies with the reason
    for record in records:
        assert record["dimension"] == "grammar"
        assert record["void_grammar_signature"] == SIGNATURE
        assert record["void_active"] is True
    # the signature is not sourced from the review under void
    assert "isn_version" not in json.dumps(records)


def test_signature_is_captured_at_void_time_when_the_caller_omits_it(monkeypatch):
    """An omitted signature is observed from the environment, not left null."""
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_store.graph_audits.isn_vocabulary_signature",
        lambda: "observed-at-void-time",
    )
    gc = FakeGraph([_review()])
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
        lambda *a, **k: gc,
    )
    void_review_dimension(
        REVIEW_ID,
        "grammar",
        actor="catalog-editor",
        reason=REASON,
        gc=gc,
    )
    record = json.loads(gc.reviews[REVIEW_ID]["voided_dimensions_json"])[0]
    assert record["void_grammar_signature"] == "observed-at-void-time"


# ---------------------------------------------------------------------------
# 4. update_review_aggregates follows the recomputed scores
# ---------------------------------------------------------------------------


def test_update_review_aggregates_follows_the_void(monkeypatch):
    second = _review("electron_temperature:name:group-a:1", cycle_index=1)
    gc = FakeGraph([_review(), second])
    monkeypatch.setattr(
        "imas_codex.standard_names.graph_store.graph_foundation.GraphClient",
        lambda *a, **k: gc,
    )

    update_review_aggregates([SN_ID])
    assert gc.names[SN_ID]["review_mean_score"] == pytest.approx(0.75)

    result = _void(gc)

    expected = (0.8 + 0.75) / 2.0  # voided review 0.8, untouched review 0.75
    assert expected == pytest.approx(0.775)
    # the route leaves no window where the aggregate disagrees with the reviews
    assert result["aggregates_refreshed"] is True
    assert gc.names[SN_ID]["review_mean_score"] == pytest.approx(expected)

    # and a later aggregate pass is a no-op rather than a correction
    update_review_aggregates([SN_ID])
    assert gc.names[SN_ID]["review_mean_score"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. a second void leaves exactly one record and changes no score
# ---------------------------------------------------------------------------


def test_second_void_of_the_same_dimension_leaves_one_record():
    gc = FakeGraph([_review()])
    _void(gc)
    score_after_first = gc.reviews[REVIEW_ID]["score"]
    records_after_first = gc.reviews[REVIEW_ID]["voided_dimensions_json"]
    writes_after_first = len(gc.writes)

    with pytest.raises(ReviewDimensionVoidRefused, match="already voided"):
        _void(gc, reason="same call again")

    assert json.loads(gc.reviews[REVIEW_ID]["voided_dimensions_json"]) == json.loads(
        records_after_first
    )
    assert len(json.loads(records_after_first)) == 1
    assert gc.reviews[REVIEW_ID]["score"] == score_after_first
    assert len(gc.writes) - writes_after_first == 0


# ---------------------------------------------------------------------------
# 6. the four refusals, each writing nothing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("dimension", "kwargs", "message"),
    [
        ("physics_accuracy", {}, "not a scored dimension"),
        ("grammar", {"reason": "   "}, "needs a reason"),
    ],
)
def test_refusals_write_nothing(dimension, kwargs, message):
    gc = FakeGraph([_review()])
    with pytest.raises(ReviewDimensionVoidRefused, match=message):
        _void(gc, dimension, **kwargs)
    assert len(gc.writes) == 0
    assert gc.reviews[REVIEW_ID]["voided_dimensions_json"] is None
    assert gc.reviews[REVIEW_ID]["score"] == pytest.approx(0.75)


def test_refuses_the_last_surviving_dimension():
    gc = FakeGraph([_review(dims={"grammar": 12})])
    with pytest.raises(ReviewDimensionVoidRefused, match="last surviving dimension"):
        _void(gc)
    assert len(gc.writes) == 0
    assert gc.reviews[REVIEW_ID]["voided_dimensions_json"] is None


def test_refuses_the_last_surviving_dimension_after_earlier_voids():
    voided = json.dumps(
        [
            {"dimension": d, "void_active": True, "voided_at": "2026-09-06T00:00:00"}
            for d in ("semantic", "convention", "completeness")
        ]
    )
    gc = FakeGraph([_review(voided=voided)])
    with pytest.raises(ReviewDimensionVoidRefused, match="last surviving dimension"):
        _void(gc)
    assert len(gc.writes) == 0
    assert gc.reviews[REVIEW_ID]["voided_dimensions_json"] == voided


def test_second_void_refusal_writes_nothing_from_a_stored_record():
    voided = json.dumps(
        [
            {
                "dimension": "grammar",
                "void_active": True,
                "voided_at": "2026-09-06T00:00:00",
                "void_actor": "catalog-editor",
                "void_reason": "superseded rule",
                "void_grammar_signature": SIGNATURE,
            }
        ]
    )
    gc = FakeGraph([_review(voided=voided)])
    with pytest.raises(ReviewDimensionVoidRefused, match="already voided"):
        _void(gc)
    assert len(gc.writes) == 0
    assert len(json.loads(gc.reviews[REVIEW_ID]["voided_dimensions_json"])) == 1


# ---------------------------------------------------------------------------
# the planner, with no graph at all
# ---------------------------------------------------------------------------


def test_plan_keeps_evidence_and_recomputes():
    plan = plan_review_dimension_void(
        scores=dict(DIMENSIONS),
        void_records=None,
        dimension="grammar",
        actor="catalog-editor",
        reason="superseded rule",
        grammar_signature=SIGNATURE,
        at="2026-09-06T08:00:00+00:00",
    )
    assert plan["surviving_dimensions"] == {
        "completeness": 14.0,
        "convention": 16.0,
        "semantic": 18.0,
    }
    assert plan["score"] == pytest.approx(0.8)
    assert plan["void_record"]["voided_at"] == "2026-09-06T08:00:00+00:00"


def test_plan_refuses_an_unattributed_void():
    with pytest.raises(ReviewDimensionVoidRefused, match="needs an actor"):
        plan_review_dimension_void(
            scores=dict(DIMENSIONS),
            void_records=None,
            dimension="grammar",
            actor="",
            reason="superseded rule",
            grammar_signature=SIGNATURE,
        )


# ---------------------------------------------------------------------------
# the same measurement against a live database
# ---------------------------------------------------------------------------


@pytest.mark.graph
def test_aggregate_mean_follows_the_void_live(graph_client):
    """update_review_aggregates' real Cypher reads the recomputed score."""
    sn_id = "sn_void_route_probe"
    review_id = f"{sn_id}:name:probe:0"
    try:
        graph_client.query(
            """
            MERGE (sn:StandardName {id: $sn_id})
            SET sn.name_stage = 'reviewed', sn.origin = 'derived'
            MERGE (r:StandardNameReview {id: $review_id})
            SET r.standard_name_id = $sn_id, r.review_axis = 'name',
                r.review_group_id = 'probe', r.cycle_index = 0,
                r.resolution_role = 'primary', r.model = 'probe-model',
                r.reviewer_model = 'probe-model',
                r.score = $score, r.scores_json = $scores_json,
                r.voided_dimensions_json = null
            MERGE (sn)-[:HAS_REVIEW]->(r)
            """,
            sn_id=sn_id,
            review_id=review_id,
            score=60 / 80.0,
            scores_json=json.dumps(DIMENSIONS),
        )
        update_review_aggregates([sn_id])
        before = graph_client.query(
            "MATCH (sn:StandardName {id: $id}) RETURN sn.review_mean_score AS m",
            id=sn_id,
        )
        assert before[0]["m"] == pytest.approx(0.75)

        void_review_dimension(
            review_id,
            "grammar",
            actor="catalog-editor",
            reason=REASON,
            grammar_signature=SIGNATURE,
            gc=graph_client,
        )
        after = graph_client.query(
            """
            MATCH (sn:StandardName {id: $id})-[:HAS_REVIEW]->(r:StandardNameReview)
            RETURN sn.review_mean_score AS m, r.score AS score,
                   r.scores_json AS scores_json,
                   r.voided_dimensions_json AS voided
            """,
            id=sn_id,
        )
        row = after[0]
        assert row["m"] == pytest.approx(0.8)
        assert row["score"] == pytest.approx(0.8)
        assert json.loads(row["scores_json"])["grammar"] == 12
        assert json.loads(row["voided"])[0]["void_reason"] == REASON
    finally:
        graph_client.query(
            """
            MATCH (sn:StandardName {id: $sn_id})
            OPTIONAL MATCH (sn)-[:HAS_REVIEW]->(r:StandardNameReview)
            OPTIONAL MATCH (sn)-[:HAS_INTERNAL_CHANGE]->(c:StandardNameChange)
            DETACH DELETE sn, r, c
            """,
            sn_id=sn_id,
        )
