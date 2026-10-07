"""DD-path search rerank wiring and metric.

``search_dd_paths`` can reorder its hybrid hits through the shared
search-time Jev rerank (``imas_codex.llm.search_tools.rerank_candidates``)
when the tool is constructed with ``rerank=True``. The MCP server sets that
switch on for a full or facility server and leaves it off for a read-only or
DD-only server, so an external query never spends a judgement.

These tests pin that contract without a live graph: the switch is a
construction-time decision, the rerank is exactly one call into the shared
owner, the judged order is applied to the hits, and each candidate carries
its path and text so the owner judges the right thing. The nDCG@10 metric
used by the measurement is verified here on a toy ranking, and the labelled
fixture is checked for well-formedness.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imas_codex.search.search_strategy import SearchHit, SearchMode

FIXTURE = Path(__file__).parent / "fixtures" / "dd_mapping_questions.json"


def _hit(path: str, documentation: str = "", description: str = "") -> SearchHit:
    return SearchHit(
        score=1.0,
        rank=1,
        search_mode=SearchMode.HYBRID,
        path=path,
        documentation=documentation,
        description=description,
        ids_name=path.split("/", 1)[0],
    )


# ---------------------------------------------------------------------------
# The labelled fixture
# ---------------------------------------------------------------------------


def _load_questions() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_has_fifteen_labelled_questions():
    data = _load_questions()
    assert data["k"] == 10
    assert len(data["questions"]) == 15


def test_fixture_grades_are_positive_grades_on_full_paths():
    for question in _load_questions()["questions"]:
        assert question["query"]
        assert question["relevant"], question["id"]
        for path, grade in question["relevant"].items():
            assert path.count("/") >= 1, path
            assert grade in (1, 2, 3), (path, grade)


# ---------------------------------------------------------------------------
# nDCG@10 — the metric the measurement reports
# ---------------------------------------------------------------------------


def _ndcg_at_k(ranked_paths: list[str], relevant: dict[str, int], k: int = 10) -> float:
    """Graded nDCG@k for a ranked list of paths against labelled grades."""
    gains = [relevant.get(path, 0) for path in ranked_paths[:k]]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def test_ndcg_is_one_for_the_ideal_ordering():
    relevant = {"a": 3, "b": 2, "c": 1}
    assert _ndcg_at_k(["a", "b", "c"], relevant) == pytest.approx(1.0)


def test_ndcg_penalises_a_worse_ordering():
    relevant = {"a": 3, "b": 2, "c": 1}
    ideal = _ndcg_at_k(["a", "b", "c"], relevant)
    worse = _ndcg_at_k(["c", "b", "a"], relevant)
    assert worse < ideal
    assert 0.0 <= worse <= 1.0


# ---------------------------------------------------------------------------
# The construction-time switch
# ---------------------------------------------------------------------------


def test_tools_leaves_rerank_off_by_default():
    from imas_codex.tools import Tools

    tools = Tools(graph_client=MagicMock())
    assert tools.search_tool._rerank is False


def test_tools_enables_rerank_when_asked():
    from imas_codex.tools import Tools

    tools = Tools(graph_client=MagicMock(), rerank_dd_paths=True)
    assert tools.search_tool._rerank is True


def test_graph_search_tool_defaults_rerank_off():
    from imas_codex.tools.graph_search import GraphSearchTool

    assert GraphSearchTool(MagicMock())._rerank is False


@pytest.fixture
def server_module():
    from imas_codex.llm import server as module

    prior = module._rerank_dd_paths
    yield module
    module._rerank_dd_paths = prior


def _server_search_tool(server_module, **kwargs):
    """Build the search tool the server's mode would construct."""
    from imas_codex.tools import Tools

    server_module.AgentsServer(**kwargs)
    return Tools(
        graph_client=MagicMock(),
        rerank_dd_paths=server_module._rerank_dd_paths,
    ).search_tool


@pytest.mark.asyncio
async def test_full_server_reranks(server_module):
    tool = _server_search_tool(server_module, read_only=False, dd_only=False)
    hits = [_hit("ids/a"), _hit("ids/b")]
    with (
        patch("imas_codex.graph.dd_search.hybrid_dd_search", return_value=list(hits)),
        patch(
            "imas_codex.llm.search_tools.rerank_candidates",
            side_effect=lambda q, c, **kw: (list(reversed(c)), None),
        ) as spy,
    ):
        await tool.search_dd_paths("query", max_results=2)
    assert spy.call_count == 1


@pytest.mark.asyncio
async def test_read_only_server_never_reranks(server_module):
    tool = _server_search_tool(server_module, read_only=True, dd_only=False)
    hits = [_hit("ids/a"), _hit("ids/b")]
    with (
        patch("imas_codex.graph.dd_search.hybrid_dd_search", return_value=list(hits)),
        patch("imas_codex.llm.search_tools.rerank_candidates") as spy,
    ):
        await tool.search_dd_paths("query", max_results=2)
    assert spy.call_count == 0


@pytest.mark.asyncio
async def test_dd_only_server_never_reranks(server_module):
    tool = _server_search_tool(server_module, read_only=True, dd_only=True)
    hits = [_hit("ids/a"), _hit("ids/b")]
    with (
        patch("imas_codex.graph.dd_search.hybrid_dd_search", return_value=list(hits)),
        patch("imas_codex.llm.search_tools.rerank_candidates") as spy,
    ):
        await tool.search_dd_paths("query", max_results=2)
    assert spy.call_count == 0


# ---------------------------------------------------------------------------
# One call into the shared owner, order applied
# ---------------------------------------------------------------------------


def test_rerank_is_one_call_into_the_shared_owner():
    from imas_codex.tools.graph_search import _rerank_dd_hits

    hits = [_hit("ids/a", "alpha"), _hit("ids/b", "beta"), _hit("ids/c", "gamma")]
    with patch(
        "imas_codex.llm.search_tools.rerank_candidates",
        side_effect=lambda q, c, **kw: (list(reversed(c)), None),
    ) as spy:
        ordered, note = _rerank_dd_hits("query", hits)

    assert spy.call_count == 1
    assert [h.path for h in ordered] == ["ids/c", "ids/b", "ids/a"]
    assert note is None


def test_candidate_carries_path_and_text_for_the_owner():
    from imas_codex.tools.graph_search import _rerank_dd_hits

    hits = [_hit("ids/a", "alpha docs", "alpha desc")]
    captured: list[dict] = []

    def _capture(query, candidates, **kw):
        captured.extend(candidates)
        return candidates, None

    with patch("imas_codex.llm.search_tools.rerank_candidates", side_effect=_capture):
        _rerank_dd_hits("query", hits)

    assert captured[0]["path"] == "ids/a"
    assert "alpha docs" in captured[0]["text"]
    assert captured[0]["_hit"] is hits[0]


@pytest.mark.asyncio
async def test_search_reranks_when_enabled():
    from imas_codex.tools.graph_search import GraphSearchTool

    hits = [_hit("ids/a"), _hit("ids/b")]
    tool = GraphSearchTool(MagicMock(), rerank=True)
    with (
        patch("imas_codex.graph.dd_search.hybrid_dd_search", return_value=list(hits)),
        patch(
            "imas_codex.llm.search_tools.rerank_candidates",
            side_effect=lambda q, c, **kw: (list(reversed(c)), None),
        ) as spy,
    ):
        result = await tool.search_dd_paths("query", max_results=2)

    assert spy.call_count == 1
    assert [h.path for h in result.hits] == ["ids/b", "ids/a"]


@pytest.mark.asyncio
async def test_search_does_not_rerank_when_disabled():
    from imas_codex.tools.graph_search import GraphSearchTool

    hits = [_hit("ids/a"), _hit("ids/b")]
    tool = GraphSearchTool(MagicMock(), rerank=False)
    with (
        patch("imas_codex.graph.dd_search.hybrid_dd_search", return_value=list(hits)),
        patch("imas_codex.llm.search_tools.rerank_candidates") as spy,
    ):
        result = await tool.search_dd_paths("query", max_results=2)

    spy.assert_not_called()
    assert [h.path for h in result.hits] == ["ids/a", "ids/b"]


# ---------------------------------------------------------------------------
# The fallback note reaches the summary and the formatted report
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completed_rerank_leaves_no_note():
    from imas_codex.llm.search_formatters import format_search_dd_report
    from imas_codex.tools.graph_search import GraphSearchTool

    hits = [_hit("ids/a", "alpha"), _hit("ids/b", "beta")]
    tool = GraphSearchTool(MagicMock(), rerank=True)
    with (
        patch("imas_codex.graph.dd_search.hybrid_dd_search", return_value=list(hits)),
        patch(
            "imas_codex.llm.search_tools.rerank_candidates",
            side_effect=lambda q, c, **kw: (list(reversed(c)), None),
        ),
    ):
        result = await tool.search_dd_paths("query", max_results=2)

    assert "rerank_note" not in result.summary
    assert "Rerank:" not in format_search_dd_report(result)


@pytest.mark.asyncio
async def test_skipped_rerank_leaves_the_note_in_summary_and_report(monkeypatch):
    """With the decisions key absent, the fallback note is surfaced to the caller.

    Every Jev caller skips cleanly when ``OPENROUTER_API_KEY_IMAS_CODEX`` is
    not set: the judgement cannot run, so the embedding order is returned and
    ``rerank_candidates`` reports the fallback in a note. Without this surface
    the caller cannot tell that fallback from a rerank that ran and reordered
    nothing. The note's wording is owned by the shared rerank owner; this node
    owns only that it reaches the summary and the formatted report.
    """
    from imas_codex.llm.search_formatters import format_search_dd_report
    from imas_codex.tools.graph_search import GraphSearchTool

    monkeypatch.delenv("OPENROUTER_API_KEY_IMAS_CODEX", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY_FACILITY_DISCOVERY", raising=False)

    hits = [_hit("ids/a", "alpha"), _hit("ids/b", "beta")]
    tool = GraphSearchTool(MagicMock(), rerank=True)
    with patch("imas_codex.graph.dd_search.hybrid_dd_search", return_value=list(hits)):
        result = await tool.search_dd_paths("query", max_results=2)

    note = result.summary.get("rerank_note")
    assert isinstance(note, str) and note, (
        "a skipped rerank must leave a note describing the fallback"
    )
    # The order is unchanged where no judgement ran.
    assert [h.path for h in result.hits] == ["ids/a", "ids/b"]

    report = format_search_dd_report(result)
    assert note in report
