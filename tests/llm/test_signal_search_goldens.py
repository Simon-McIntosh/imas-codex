"""Golden harness for facility signal search over the graded question fixture.

Five ranking arms and a seeded random control are scored on the questions in
``tests/llm/fixtures/signal_search_questions.json``:

``current_tool``  the product ranking (``_rank_signals`` in ``search_tools``):
                  vector on descriptions boosted by a full-text keyword match.
``vector_only``   vector on descriptions alone, pool 100.
``rrf``           reciprocal rank fusion of the 100-candidate vector pool and
                  the 50-candidate full-text pool.
``rrf_rerank``    ``rrf``'s top 60 reordered by a generic Jev relevance score.
``rrf_rerank_role``  ``rrf``'s top 60 reordered by the role-aware Jev question.

The two Jev arms spend money on the discovery-relevance seat, so a default run
does not build them: ``arm_ranking`` returns the fused order for a Jev arm and
records that the rerank permutation was not applied, unless the environment
sets ``SIGNAL_SEARCH_JEV_ARMS=1``. A stub rerank is used only where a test needs
the arm to produce an order without spending.

Every question's candidates that an arm places in its top 10 must be labelled
under the fixture's grading rule before that arm is scored. A candidate with no
label is not the same as one read and judged irrelevant (grade 0): the first is
a gap in the fixture and makes the arm refuse, naming the candidate; the second
is scored. Labelling is pooled across all arms, so an arm is never penalised
for lifting a relevant signal a sibling arm found and no other arm had labelled.

The fixture declares the enrichment cutover its labels were drawn against, and
every labelled signal must lie at or after it: a label on a signal the Jev-era
pipeline has not enriched is a stale seed the population does not contain, and
the eligibility gate fails by name rather than skipping it.

Reports mean nDCG@10, grade-3 precision@5, grade-3 recall@30 and the count of
questions with no grade-3 hit in the top 30, per arm. The control is a seeded
permutation of each question's labelled pool and scores below every arm on mean
nDCG@10; a control that does not means the labels are wrong.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "signal_search_questions.json"

VECTOR_POOL = 100
TEXT_POOL = 50
RERANK_POOL = 60
K = 30
TOP_K = 10
CONTROL_SEED = 20261012

# The cutover the golden population is enumerated against: the first run whose
# signal pipeline reranked enrichment context within its budget. The fixture
# carries this value as data; the gate below pins it so a fixture that declares
# an earlier cutover (and so admits signals enriched before the fix) fails
# rather than silently widening the population.
CANONICAL_ENRICHED_AT_CUTOVER = "2026-10-11T21:08:54Z"

# Set to "1" to build and score the two Jev rerank arms, which spend on the
# discovery-relevance seat. A default run leaves this unset and spends nothing.
JEV_ARMS_ENV = "SIGNAL_SEARCH_JEV_ARMS"

# Arms whose ranking is reordered by a Jev judgement and therefore spends on the
# discovery-relevance seat. A default run does not build them.
JEV_ARMS = ("rrf_rerank", "rrf_rerank_role")

# The role-aware relevance question, written to five relevance levels: a
# measurement of the quantity asked for, through a measurement of a related
# quantity, a commanded or reference value of it, related hardware, to a
# different quantity. Kept here until the rerank ships and it moves into the
# product beside the generic question.
_ROLE_RERANK_LEVELS = [
    "a different quantity, unrelated to the one asked for",
    "related hardware: a status flag, configuration value or neighbouring measurement",
    "a commanded, reference or setpoint value of the quantity",
    "a measurement of a related quantity, or of the quantity asked for at an adjacent point",
    "a measurement of exactly the quantity asked for",
]
_ROLE_RERANK_INSTRUCTIONS = (
    "How relevant is this candidate to answering the query? Judge whether the "
    "candidate MEASURES the quantity asked for. A commanded, reference or "
    "setpoint value of the quantity answers the query far less than a measured "
    "value of it, and a different quantity answers it not at all."
)

json_fixture = json.loads(FIXTURE_PATH.read_text())


@pytest.fixture(autouse=True)
def _use_production_embeddings(monkeypatch):
    """Embed with the production model, not the suite's local test model.

    The session fixture in ``tests/conftest.py`` forces a local
    ``all-MiniLM-L6-v2`` model for all tests. The stored signal vectors in the
    graph come from the production embedding model, so a locally embedded query
    is dimensionally incompatible with the ``facility_signal_desc_embedding``
    index and every vector arm fails before it ranks anything. Deleting the two
    overrides lets ``EncoderConfig`` resolve the production backend.
    """
    monkeypatch.delenv("IMAS_CODEX_EMBEDDING_LOCATION", raising=False)
    monkeypatch.delenv("IMAS_CODEX_EMBEDDING_MODEL", raising=False)


@dataclass
class Question:
    id: str
    query: str
    held_out: bool
    relevant: dict[str, int]
    intent: str = ""
    categories: list[str] = field(default_factory=list)

    @property
    def grade3_total(self) -> int:
        return sum(1 for g in self.relevant.values() if g >= 3)


def _load_questions() -> list[Question]:
    return [
        Question(
            id=q["id"],
            query=q["query"],
            held_out=q["held_out"],
            relevant=q["relevant"],
            intent=q.get("intent", ""),
            categories=q.get("categories", []),
        )
        for q in json_fixture["questions"]
    ]


QUESTIONS = _load_questions()


class UnlabelledCandidates(Exception):
    """An arm surfaced a candidate the fixture does not label.

    The arm is refused rather than scored, because pooling labels across arms is
    the whole point of the fixture: scoring against an incomplete label set is
    what made the prototype's figures a reading rather than a verdict.
    """

    def __init__(self, arm: str, question_id: str, candidates: list[str]):
        self.arm = arm
        self.question_id = question_id
        self.candidates = candidates
        super().__init__(
            f"arm {arm!r} on question {question_id!r} top {TOP_K} holds "
            f"{len(candidates)} unlabelled candidate(s): " + ", ".join(candidates)
        )


# ---------------------------------------------------------------------------
# Metrics (pure)
# ---------------------------------------------------------------------------


def _dcg(grades: list[int], k: int) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades[:k]))


def ndcg_at_k(ranked_grades: list[int], ideal_grades: list[int], k: int) -> float:
    """nDCG@k with gain 2**grade - 1 and log2 discount."""
    ideal = _dcg(sorted(ideal_grades, reverse=True), k)
    return _dcg(ranked_grades, k) / ideal if ideal else 0.0


def grade3_precision_at_k(ranked_grades: list[int], k: int) -> float:
    top = ranked_grades[:k]
    return sum(1 for g in top if g >= 3) / len(top) if top else 0.0


def grade3_recall_at_k(ranked_grades: list[int], total_grade3: int, k: int) -> float:
    if not total_grade3:
        return 0.0
    return sum(1 for g in ranked_grades[:k] if g >= 3) / total_grade3


def has_grade3_hit(ranked_grades: list[int], k: int) -> bool:
    return any(g >= 3 for g in ranked_grades[:k])


def score_ranking(question: Question, ranked_ids: list[str]) -> dict[str, float]:
    """Metric row for one arm on one question."""
    grades = [question.relevant.get(sid, 0) for sid in ranked_ids]
    ideal = list(question.relevant.values())
    return {
        "ndcg@10": ndcg_at_k(grades, ideal, TOP_K),
        "grade3_precision@5": grade3_precision_at_k(grades, 5),
        "grade3_recall@30": grade3_recall_at_k(grades, question.grade3_total, K),
        "grade3_hit@30": float(has_grade3_hit(grades, K)),
    }


def unlabelled_in_top_k(question: Question, ranked_ids: list[str], k: int) -> list[str]:
    """Ordered ids in the top ``k`` that the fixture does not label at all."""
    seen: list[str] = []
    for sid in ranked_ids[:k]:
        if sid not in question.relevant and sid not in seen:
            seen.append(sid)
    return seen


def require_labelled(arm: str, question: Question, ranked_ids: list[str]) -> None:
    """Refuse an arm whose top ``k`` holds an unlabelled candidate, naming it."""
    missing = unlabelled_in_top_k(question, ranked_ids, TOP_K)
    if missing:
        raise UnlabelledCandidates(arm, question.id, missing)


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------


def _embed(gc, encoder, text: str) -> list[float]:
    from imas_codex.llm.search_tools import _embed as embed

    return embed(encoder, text)


def _rrf_fuse(streams):
    from imas_codex.standard_names.search import rrf_fuse

    return rrf_fuse(streams)


def _fusion_streams(gc, embedding, question: Question):
    """The vector and full-text streams the prototype fused (100 and 50)."""
    from imas_codex.llm.search_tools import _text_search_signals, _vector_search_signals

    facility = json_fixture["eligibility"]["facility"]
    vec_ids, _ = _vector_search_signals(
        gc, embedding, facility, VECTOR_POOL, None, None
    )
    text_rows = _text_search_signals(gc, question.query, facility, TEXT_POOL)
    text_ids = [r["id"] for r in text_rows]
    return vec_ids, text_ids


def _rf_ranks(ids: list[str]) -> list[dict]:
    return [{"id": sid} for sid in ids]


def arm_ranking(question: Question, ctx) -> list[str]:
    """Ordered signal ids for one arm on one question, cut at ``K``.

    ``ctx`` carries the open graph client, the encoder and per-question
    embedding so a run reuses one vector per question across arms.
    """
    from imas_codex.llm.search_tools import _rank_signals

    facility = json_fixture["eligibility"]["facility"]
    arm = ctx.arm
    if arm == "current_tool":
        ids, _ = _rank_signals(
            ctx.gc, ctx.embedding, question.query, facility, K, None, None
        )
        return ids
    vec_ids, text_ids = _fusion_streams(ctx.gc, ctx.embedding, question)
    if arm == "vector_only":
        return vec_ids[:K]
    fused = [sid for sid, _ in _rrf_fuse([_rf_ranks(vec_ids), _rf_ranks(text_ids)])]
    if arm == "rrf":
        return fused[:K]
    if arm in JEV_ARMS:
        pool = fused[:RERANK_POOL]
        if not ctx.spend_jev:
            # No Jev spent: return the pool order and record that the rerank
            # permutation was not applied.
            ctx.rerank_applied = False
            return pool[:K]
        ctx.rerank_applied = True
        return _rerank_arm(arm, question, pool, ctx)[:K]
    raise ValueError(f"unknown arm: {arm}")


def _rerank_arm(arm: str, question: Question, pool: list[str], ctx) -> list[str]:
    """Reorder ``pool`` with a Jev relevance score (spends on the Jev seat).

    The role-aware arm uses the five-level role question; the generic arm uses
    the product's existing ``_RERANK_QUESTION`` levels, so the two are measured
    beside each other on the same pool and query.
    """
    from imas_codex.discovery.base.judgment import rerank_pool
    from imas_codex.llm.search_tools import (
        _RERANK_QUESTION,
        _RERANK_SERVICE,
        _rerank_state,
        _run_batch,
    )
    from imas_codex.settings import get_model

    briefs = _signal_texts(ctx.gc, pool)
    candidates = [{"id": sid, "text": briefs.get(sid, ""), "path": sid} for sid in pool]
    if arm == "rrf_rerank_role":
        levels = _ROLE_RERANK_LEVELS
        instructions = _ROLE_RERANK_INSTRUCTIONS
    else:
        levels = _RERANK_QUESTION["relevance_grade"]["criteria"]
        instructions = _RERANK_QUESTION["relevance_grade"]["instructions"]
    ordered, _note, _cost = _run_batch(
        rerank_pool(
            question.query,
            candidates,
            state_for=_rerank_state,
            levels=levels,
            instructions=instructions,
            model=get_model("discovery-relevance"),
            service=_RERANK_SERVICE,
            budget_seconds=5.0,
        )
    )
    return [c["id"] for c in ordered]


def _signal_texts(gc, ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    cypher = """
        UNWIND $ids AS sid
        MATCH (s:FacilitySignal {id: sid})
        RETURN s.id AS id, coalesce(s.name, '') AS name,
               coalesce(s.description, '') AS description,
               coalesce(s.unit, '') AS unit
    """
    rows = gc.query(cypher, ids=ids)
    return {
        r["id"]: " ".join(p for p in (r["name"], r["description"], r["unit"]) if p)
        for r in rows
    }


@dataclass
class ArmContext:
    gc: object
    encoder: object
    arm: str
    spend_jev: bool = False
    rerank_applied: bool = False
    _embeddings: dict[str, list[float]] = field(default_factory=dict)

    def embedding_for(self, question: Question) -> list[float]:
        if question.id not in self._embeddings:
            self._embeddings[question.id] = _embed(
                self.gc, self.encoder, question.query
            )
        return self._embeddings[question.id]


def score_arm(arm: str, ctx, questions=QUESTIONS) -> list[dict]:
    """Score one arm over ``questions``; refuse on any unlabelled top-10."""
    rows: list[dict] = []
    for question in questions:
        ctx.embedding = ctx.embedding_for(question)
        ranked = arm_ranking(question, ctx)
        require_labelled(arm, question, ranked)
        row = {"arm": arm, "question": question.id, "held_out": question.held_out}
        row.update(score_ranking(question, ranked))
        rows.append(row)
    return rows


def control_ranking(question: Question) -> list[str]:
    """A seeded permutation of the question's labelled pool."""
    pool = sorted(question.relevant)
    random.Random(CONTROL_SEED).shuffle(pool)
    return pool


def score_control(questions=QUESTIONS) -> list[dict]:
    rows: list[dict] = []
    for question in questions:
        row = {
            "arm": "random_control",
            "question": question.id,
            "held_out": question.held_out,
        }
        row.update(score_ranking(question, control_ranking(question)))
        rows.append(row)
    return rows


def mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


# ---------------------------------------------------------------------------
# Fixture integrity
# ---------------------------------------------------------------------------


def test_fixture_holds_draft_and_held_out_questions() -> None:
    """At least seventeen questions, at least six held out beside the drafts."""
    assert len(QUESTIONS) >= 17, len(QUESTIONS)
    held = [q for q in QUESTIONS if q.held_out]
    draft = [q for q in QUESTIONS if not q.held_out]
    assert len(held) >= 6, len(held)
    assert len(draft) == 11, len(draft)
    ids = [q.id for q in QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_question_is_phrased_without_facility_or_source_vocabulary() -> None:
    """No facility name and no data-source code in any query."""
    banned = [json_fixture["eligibility"]["facility"]]
    for q in QUESTIONS:
        lowered = q.query.lower()
        for token in banned:
            assert token.lower() not in lowered, (q.id, token)
        # EDDB category codes are upper-case three-to-ten letter tokens.
        assert not any(
            tok.isupper() and tok.isalpha() and len(tok) >= 3 for tok in q.query.split()
        ), (q.id, q.query)


def _as_utc(value: object) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromisoformat(str(value)).astimezone(timezone.utc)
    except ValueError:
        return None


def _label_enrichment(ids: list[str]) -> dict[str, object]:
    """``enriched_at`` per id, or None where the signal is absent or unset."""
    from imas_codex.graph.client import GraphClient

    rows = GraphClient().query(
        "UNWIND $ids AS sid OPTIONAL MATCH (s:FacilitySignal {id: sid}) "
        "RETURN sid AS id, s.enriched_at AS enriched_at",
        ids=ids,
    )
    return {r["id"]: r["enriched_at"] for r in rows}


@pytest.mark.graph
def test_every_labelled_signal_is_eligible() -> None:
    """Every labelled signal lies at or after the Jev-era enrichment cutover.

    A label on a signal the pipeline enriched before the cutover is a seed the
    golden population does not contain, so the fixture fails by name rather than
    skipping it. The check pins the cutover the fixture declares: an earlier
    value readmits the seeds enriched in the window before the fix, so it must
    fail rather than silently widen the population.
    """
    declared = json_fixture["eligibility"]["enriched_at_cutover"]
    assert declared == CANONICAL_ENRICHED_AT_CUTOVER, (
        f"fixture declares enrichment cutover {declared!r}; the canonical "
        f"cutover is {CANONICAL_ENRICHED_AT_CUTOVER!r}"
    )
    cutover = _as_utc(CANONICAL_ENRICHED_AT_CUTOVER)
    labelled = sorted({sid for q in QUESTIONS for sid in q.relevant})
    enrichment = _label_enrichment(labelled)
    ineligible = [
        sid
        for sid in labelled
        if _as_utc(enrichment.get(sid)) is None
        or _as_utc(enrichment[sid]) < cutover
    ]
    assert not ineligible, (
        f"fixture labels {len(ineligible)} signal(s) enriched before the cutover "
        f"{CANONICAL_ENRICHED_AT_CUTOVER}, or absent from the graph: "
        + ", ".join(ineligible)
    )


# ---------------------------------------------------------------------------
# Metrics on a hand-computed case (no graph)
# ---------------------------------------------------------------------------


def test_metrics_on_a_hand_computed_case() -> None:
    """nDCG, precision and recall match values computed by hand."""
    # grades 3, 2, 0, 1 -> gains 7, 3, 0, 1
    # dcg@4 = 7/1 + 3/1.585 + 0 + 1/2.322 = 7 + 1.893 + 0 + 0.431 = 9.324
    # ideal (3,2,1,0) = 7 + 3/1.585 + 1/2 + 0 = 7 + 1.893 + 0.5 = 9.393
    grades = [3, 2, 0, 1]
    ideal = [3, 2, 1, 0]
    assert ndcg_at_k(grades, ideal, 4) == pytest.approx(0.9927, abs=1e-4)
    assert grade3_precision_at_k([3, 3, 1, 0, 2], 5) == pytest.approx(0.4)
    assert grade3_recall_at_k([3, 0, 3, 1], total_grade3=4, k=4) == pytest.approx(0.5)
    assert has_grade3_hit([2, 0, 3], 3) is True
    assert has_grade3_hit([2, 0, 1], 3) is False


def test_control_is_a_seeded_permutation_of_the_labelled_pool() -> None:
    q = Question("t", "q", False, {"a": 3, "b": 2, "c": 0})
    first, second = control_ranking(q), control_ranking(q)
    assert first == second
    assert sorted(first) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# The refusal gate
# ---------------------------------------------------------------------------


def _gate_toy_question() -> Question:
    """A question whose labels cover every candidate an arm could place."""
    return Question(
        "toy",
        "the quantity",
        False,
        {"sig:a": 3, "sig:b": 2, "sig:c": 1, "sig:d": 0},
    )


def test_a_fully_labelled_arm_is_scored_not_refused() -> None:
    """Positive control for the gate: complete labels score, no refusal."""
    q = _gate_toy_question()
    require_labelled("current_tool", q, ["sig:a", "sig:b", "sig:c", "sig:d"])
    row = score_ranking(q, ["sig:a", "sig:b"])
    # grades [3,2] against ideal [3,2,1,0]: dcg 7+3/log2(3) over ideal
    # 7+3/log2(3)+1/2, i.e. 8.893/9.393.
    assert row["ndcg@10"] == pytest.approx(0.9468, abs=1e-4)


def test_an_unlabelled_candidate_refuses_the_arm_and_names_it() -> None:
    """Removing a label makes the arm refuse and name the candidate.

    This is the mutation the node declares: take the fully labelled toy
    question, drop one labelled candidate, and the arm that places it refuses.
    """
    q = _gate_toy_question()
    del q.relevant["sig:c"]
    with pytest.raises(UnlabelledCandidates) as excinfo:
        require_labelled("current_tool", q, ["sig:a", "sig:b", "sig:c", "sig:d"])
    assert "sig:c" in excinfo.value.candidates
    assert "current_tool" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Graph arms (live database)
# ---------------------------------------------------------------------------


def _open_context(
    arm: str, spend_jev: bool = False, gc=None, encoder=None
) -> ArmContext:
    from imas_codex.embeddings import Encoder
    from imas_codex.embeddings.config import EncoderConfig
    from imas_codex.graph.client import GraphClient

    if gc is None:
        gc = GraphClient()
    if encoder is None:
        encoder = Encoder(EncoderConfig())
    return ArmContext(gc=gc, encoder=encoder, arm=arm, spend_jev=spend_jev)


@pytest.mark.graph
def test_five_arm_table_matches_the_control_below_every_arm() -> None:
    """Five-arm table on pooled labels; the control sits below every arm.

    Scores all five arms and the control and asserts the control is below each
    arm on mean nDCG@10. An unlabelled candidate in any arm's top 10 fails the
    test, naming it, rather than skipping: a skipped table would read as a pass.
    """
    spend_jev = os.environ.get(JEV_ARMS_ENV) == "1"
    arms = ["current_tool", "vector_only", "rrf"]
    if spend_jev:
        arms += list(JEV_ARMS)
    rows: list[dict] = []
    for arm in arms:
        rows.extend(score_arm(arm, _open_context(arm, spend_jev=spend_jev)))
    rows.extend(score_control())
    table = {
        a: mean([r for r in rows if r["arm"] == a], "ndcg@10")
        for a in {r["arm"] for r in rows}
    }
    print(json.dumps(table, indent=2))
    control = table["random_control"]
    for arm in arms:
        assert table[arm] > control, (arm, table[arm], control)
