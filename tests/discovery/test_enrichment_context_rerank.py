"""Enrichment context retrieval routes through the Jev relevance rerank.

Both the code-chunk fetch (:func:`parallel._fetch_code_chunks`) and the
wiki-chunk fetch (:func:`wiki.fetch_semantic_wiki_context`) widen their
candidate pool, pass the enrichment query and each candidate's text to
``judgment.rerank_pool``, and keep the reranked top ``k``. When the rerank
returns no ordering they keep the embedding order, and the rerank's decision
spend is added to the enrichment run's cost total.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from imas_codex.discovery.base.progress import WorkerStats
from imas_codex.discovery.signals import parallel as parallel_mod
from imas_codex.discovery.signals.scanners import wiki as wiki_mod

_POOL = parallel_mod.CONTEXT_CANDIDATE_POOL


def _fake_encoder():
    vector = MagicMock()
    vector.tolist.return_value = [0.1, 0.2, 0.3]
    encoder = MagicMock()
    encoder.embed_texts.return_value = [vector]
    return encoder


def _patch_encoder():
    return patch("imas_codex.embeddings.encoder.Encoder", return_value=_fake_encoder())


def _patch_encoder_config():
    return patch("imas_codex.embeddings.config.EncoderConfig", return_value=MagicMock())


def _gc_returning(rows):
    gc = MagicMock()
    gc.__enter__ = MagicMock(return_value=gc)
    gc.__exit__ = MagicMock(return_value=False)
    gc.query = MagicMock(return_value=rows)
    return gc


CODE_ROWS = [
    {"text": f"code {i}", "source_path": f"/p/code_{i}.py", "language": "python"}
    for i in range(_POOL)
]
WIKI_ROWS = [
    {
        "text": f"wiki {i}",
        "page_title": f"Page {i}",
        "conventions": [],
        "units": [],
        "mdsplus_paths": [],
        "score": 0.9 - i * 0.01,
    }
    for i in range(_POOL)
]


def test_code_context_searches_the_wider_pool():
    assert _POOL == 30
    assert "LIMIT 30" in parallel_mod._build_code_context_query()


def test_code_fetch_reranks_with_the_query_and_keeps_jev_order():
    gc = _gc_returning(CODE_ROWS)
    jev_order = list(reversed(CODE_ROWS))
    rerank = AsyncMock(return_value=(jev_order, None, 0.012))

    with (
        patch.object(parallel_mod, "GraphClient", return_value=gc),
        _patch_encoder(),
        _patch_encoder_config(),
        patch("imas_codex.discovery.base.judgment.rerank_pool", rerank),
    ):
        kept = parallel_mod._fetch_code_chunks("jt-60sa", "plasma current")

    # top 3 of the reranked order, not the embedding order
    assert [c["text"] for c in kept] == ["code 29", "code 28", "code 27"]

    args, kwargs = rerank.call_args
    assert args[0] == "plasma current"
    assert [c["text"] for c in args[1]] == [r["text"] for r in CODE_ROWS]

    state = kwargs["state_for"]("plasma current", args[1][0])
    assert state["query"] == "plasma current"
    assert state["candidate"]["text"] == "code 0"
    assert state["candidate"]["locator"] == "/p/code_0.py"


def test_code_fetch_falls_back_to_embedding_order_when_unordered():
    gc = _gc_returning(CODE_ROWS)
    rerank = AsyncMock(
        return_value=(
            list(CODE_ROWS),
            "rerank unavailable (boom); embedding order returned",
            0.0,
        )
    )

    with (
        patch.object(parallel_mod, "GraphClient", return_value=gc),
        _patch_encoder(),
        _patch_encoder_config(),
        patch("imas_codex.discovery.base.judgment.rerank_pool", rerank),
    ):
        kept = parallel_mod._fetch_code_chunks("jt-60sa", "plasma current")

    assert [c["text"] for c in kept] == ["code 0", "code 1", "code 2"]


def test_code_fetch_adds_the_rerank_cost_to_the_run_cost_total():
    gc = _gc_returning(CODE_ROWS)
    rerank = AsyncMock(return_value=(list(CODE_ROWS), None, 0.0123))
    stats = WorkerStats()
    spend: list[float] = []

    with (
        patch.object(parallel_mod, "GraphClient", return_value=gc),
        _patch_encoder(),
        _patch_encoder_config(),
        patch("imas_codex.discovery.base.judgment.rerank_pool", rerank),
    ):
        parallel_mod._fetch_code_chunks("jt-60sa", "plasma current", spend=spend)

    # the fetch reports the cost the rerank returned, not an imputed rate
    assert spend == pytest.approx([0.0123])
    parallel_mod.charge_context_spend(stats, spend)
    assert stats.cost == pytest.approx(0.0123)
    assert spend == []


def test_wiki_fetch_searches_the_wider_pool_and_reranks():
    gc = _gc_returning(WIKI_ROWS)
    jev_order = list(reversed(WIKI_ROWS))
    rerank = AsyncMock(return_value=(jev_order, None, 0.02))
    spend: list[float] = []

    with (
        patch.object(wiki_mod, "GraphClient", return_value=gc),
        _patch_encoder(),
        _patch_encoder_config(),
        patch("imas_codex.discovery.base.judgment.rerank_pool", rerank),
    ):
        kept = wiki_mod.fetch_semantic_wiki_context(
            "jt-60sa", "plasma current", k=3, spend=spend
        )

    assert gc.query.call_args[1]["k"] == _POOL
    assert [c["text"] for c in kept] == ["wiki 29", "wiki 28", "wiki 27"]

    args, kwargs = rerank.call_args
    assert args[0] == "plasma current"
    assert len(args[1]) == _POOL
    state = kwargs["state_for"]("plasma current", args[1][0])
    assert state["candidate"]["locator"] == "Page 0"
    assert spend
