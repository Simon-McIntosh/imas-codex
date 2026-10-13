"""Reciprocal rank fusion: rank consensus beats raw score magnitude.

The signal ranking over ``facility_signal_desc_embedding`` fuses a cosine-ranked
vector stream and a BM25-ranked full-text stream. The two scores are on
incomparable scales, so fusion is by rank: a candidate ranked first by the
vector stream alone outranks one ranked fifth by the full-text stream alone,
whatever the BM25 magnitude of the latter.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from unittest.mock import MagicMock

from imas_codex.ids import slot_context
from imas_codex.llm import search_tools
from imas_codex.search import fusion
from imas_codex.search.fusion import RRF_K, rrf_fuse, rrf_fuse_in_input_order

_FACILITY_DIR = (
    Path(__file__).resolve().parents[2] / "imas_codex" / "config" / "facilities"
)


def _route_query(routes: dict[str, list[dict]]):
    def _query(cypher: str, **params: object) -> list[dict]:
        for marker, rows in routes.items():
            if marker in cypher:
                return rows
        return []

    return _query


def _additive_rank(
    vector_ids: list[str],
    vector_scores: dict[str, float],
    text_hits: list[dict],
    k: int,
) -> list[str]:
    """The retired additive rule, kept here as the comparison the test must fail."""
    scores = dict(vector_scores)
    order = list(vector_ids)
    for hit in text_hits:
        item_id = hit["id"]
        text_score = round(hit["score"], 3)
        if item_id in scores:
            scores[item_id] = round(scores[item_id] * 0.7 + text_score * 0.3 + 0.1, 3)
        else:
            scores[item_id] = text_score
            order.append(item_id)
    return sorted(set(order), key=lambda i: scores[i], reverse=True)[:k]


def test_rank_one_vector_outranks_rank_five_text() -> None:
    """A rank-1 vector hit (cosine 0.9) beats a rank-5 full-text hit (BM25 14).

    Under the rank blend the vector hit's ``1/(k+1)`` exceeds the text hit's
    ``1/(k+5)``; under the additive rule the text hit's BM25 magnitude of 14
    dwarfs the vector hit's cosine of 0.9 and takes the top place. The additive
    order is asserted here as well, so a regression to that rule fails the test
    rather than passing it.
    """
    vector_rows = [{"id": "signal-vector", "score": 0.9}]
    text_rows = [
        {"id": "signal-text-rank-1", "score": 3.0},
        {"id": "signal-text-rank-2", "score": 3.0},
        {"id": "signal-text-rank-3", "score": 3.0},
        {"id": "signal-text-rank-4", "score": 3.0},
        {"id": "signal-text-rank-5", "score": 14.0},
    ]

    gc = MagicMock()
    gc.query = MagicMock(
        side_effect=_route_query(
            {
                "facility_signal_desc_embedding": vector_rows,
                "facility_signal_text": text_rows,
            }
        )
    )

    ids, scores = search_tools._rank_signals(
        gc, [0.1] * 1024, "plasma current", "tcv", 10
    )
    assert ids.index("signal-vector") < ids.index("signal-text-rank-5")
    assert scores["signal-vector"] > scores["signal-text-rank-5"]

    additive = _additive_rank(
        [r["id"] for r in vector_rows],
        {r["id"]: r["score"] for r in vector_rows},
        text_rows,
        10,
    )
    assert additive[0] == "signal-text-rank-5"


def test_rrf_fuse_prefers_the_candidate_both_streams_rank() -> None:
    fused = rrf_fuse([["both", "vector-only"], ["both", "text-only"]])
    assert [item_id for item_id, _ in fused] == ["both", "text-only", "vector-only"]


def test_rrf_fuse_accepts_rows_and_strings() -> None:
    assert rrf_fuse([[{"id": "a"}], ["a"]]) == [("a", 2.0 / (RRF_K + 1))]


def test_in_input_order_variant_breaks_ties_by_the_supplied_order() -> None:
    """Equal-mass candidates keep the caller's order, not id ascending.

    Each stream ranks one candidate first, so the two carry identical fused
    mass and the fusion has nothing but its tie-break to separate them. The
    variant returns the caller's order; ``rrf_fuse`` sorts the same tie by id,
    which is asserted here so a fallback to that rule fails the test.
    """
    streams = [["alpha"], ["beta"]]
    fused = rrf_fuse_in_input_order(streams, ["beta", "alpha"])

    assert [mass for _, mass in fused][0] == [mass for _, mass in fused][1], (
        "the test needs a genuine tie"
    )
    assert [item_id for item_id, _ in fused] == ["beta", "alpha"]
    assert [item_id for item_id, _ in rrf_fuse(streams)] == ["alpha", "beta"]


def test_the_two_fusions_agree_where_no_pair_of_candidates_ties() -> None:
    """The variant shares rrf_fuse's arithmetic and differs only on ties."""
    streams = [["a", "b", "c"], ["a"]]
    assert rrf_fuse_in_input_order(streams, ["a", "b", "c"]) == rrf_fuse(streams)


def test_in_input_order_variant_carries_candidates_no_stream_ranked() -> None:
    """The order names the whole candidate set, so a fused result is total."""
    fused = rrf_fuse_in_input_order([["ranked"]], ["unranked", "ranked"])
    assert [item_id for item_id, _ in fused] == ["ranked", "unranked"]
    assert dict(fused)["unranked"] == 0.0


def test_fusion_and_ranking_carry_no_facility_identifier() -> None:
    """The fusion and the ranking are facility-agnostic.

    A facility identifier in the fusion or ranking code would tie the shared
    rule to one machine's data. The identifiers come from the facility
    configuration, so the test reads them from there rather than naming them.
    """
    facility_ids = sorted(p.stem for p in _FACILITY_DIR.glob("*.yaml"))
    assert facility_ids, "no facility identifiers found to check against"

    sources = {
        "fusion": Path(fusion.__file__).read_text(),
        "rank_signals": inspect.getsource(search_tools._rank_signals),
        "hybrid_ids": inspect.getsource(slot_context._hybrid_ids),
    }
    for source_name, source in sources.items():
        for facility_id in facility_ids:
            assert not re.search(rf"\b{re.escape(facility_id)}\b", source), (
                f"facility identifier {facility_id!r} appears in {source_name}"
            )
