"""The comparator label sheet records both orders and writes no graph state.

The sheet builder draws DD sources and, for each, records the comparator block
twice: the embedding-ordered top ten and the reranked top ten, unioned into one
candidate list the coordinator grades. This test drives the builder with a
stubbed embedding search and a stubbed rerank so it exercises the recording
alone — the rerank order must reach the sheet, and no graph client may be
opened.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path

import pytest

_DEFAULT_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docs/evidence/fragments/sn-selection-over-generation/ssog-rerank-label-sheet-run.py"
)
_SCRIPT = Path(os.environ.get("SN_RERANK_LABEL_SHEET_SCRIPT", str(_DEFAULT_SCRIPT)))
pytestmark = pytest.mark.skipif(
    _SCRIPT.resolve() == _DEFAULT_SCRIPT.resolve(),
    reason="archived evidence runner is bound to its recorded revision",
)


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "ssog_rerank_label_sheet_run", _SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ITEM = {
    "path": "summary/local/separatrix/n_i/beryllium/value",
    "description": "beryllium ion density at the plasma boundary",
    "domain": "edge_plasma_physics",
    "family": None,
}

# Twelve comparators, so a rerank can promote a candidate the embedding order put
# below the top ten into the reranked top ten — the union must then carry a
# candidate whose embedding rank is greater than ten.
_POOL = [{"id": f"name_{i:02d}", "description": f"comparator {i}"} for i in range(12)]
_RERANKED = [_POOL[11], _POOL[0], _POOL[1], *[_POOL[i] for i in range(2, 11)]]


def _stub_search(query, k=5):
    return list(_POOL)


def _stub_rerank(seen):
    async def _rerank(items, pool, **kwargs):
        seen.append((items, [c["id"] for c in pool]))
        return list(_RERANKED), 0.25, "sn-judge"

    return _rerank


def test_row_records_both_orders_and_the_union(monkeypatch):
    """The rerank order reaches the sheet; the union spans both top tens."""
    driver = _load_driver()
    seen: list = []
    row = asyncio.run(
        driver.build_row(_ITEM, search=_stub_search, rerank=_stub_rerank(seen))
    )

    assert seen, "the rerank was not called"
    assert row["rerank_cost_usd"] == pytest.approx(0.25)
    assert row["seat"] == "sn-judge"
    assert row["path"] == _ITEM["path"]

    by_name = {c["name"]: c for c in row["candidates"]}

    # Embedding top ten: names 00..09 at embedding ranks 1..10.
    for rank, candidate in enumerate(_POOL[:10], start=1):
        assert by_name[candidate["id"]]["embedding_rank"] == rank
    # Reranked top ten: name_11 promoted to the top, so it carries an embedding
    # rank beyond the embedding top ten and a reranked rank of 1.
    assert by_name["name_11"]["rerank_rank"] == 1
    assert by_name["name_11"]["embedding_rank"] == 12
    assert by_name["name_00"]["rerank_rank"] == 2
    # Every candidate carries an empty grade for the coordinator.
    assert all(c["grade"] is None for c in row["candidates"])
    # The union is larger than either single top ten because of the promotion.
    assert len(row["candidates"]) == 11


def test_build_sheet_writes_no_graph_state(monkeypatch):
    """Building the sheet opens no graph client — it is a read of injected parts."""
    driver = _load_driver()
    import imas_codex.graph.client as graph_client

    opened: list = []

    class _RefusingClient:
        def __init__(self, *args, **kwargs):
            opened.append(1)
            raise AssertionError("the sheet builder opened a graph client")

    monkeypatch.setattr(graph_client, "GraphClient", _RefusingClient)

    seen: list = []
    sheet = asyncio.run(
        driver.build_sheet(
            [_ITEM, {**_ITEM, "path": "other/path"}],
            search=_stub_search,
            rerank=_stub_rerank(seen),
        )
    )

    assert opened == []
    assert sheet["n_items"] == 2
    assert sheet["rerank_cost_usd_total"] == pytest.approx(0.5)
    assert all(row["candidates"] for row in sheet["items"])


def test_draw_is_reproducible_and_includes_the_boundary_family():
    """The draw is deterministic for a seed and reserves the boundary family."""
    driver = _load_driver()
    rows = [
        {"path": f"d{d}/p{i:02d}", "description": f"d{i}", "domain": f"d{d}"}
        for d in range(5)
        for i in range(10)
    ]
    boundary = ["d0/p03", "d1/p07"]
    first = driver.draw_items(rows, boundary, count=12, boundary_reserve=2)
    second = driver.draw_items(rows, boundary, count=12, boundary_reserve=2)
    assert [r["path"] for r in first] == [r["path"] for r in second]
    paths = {r["path"] for r in first}
    assert {"d0/p03", "d1/p07"} <= paths
    assert len(paths) == 12
