"""Typed plant browsing uses bounded pages and respects the server's facility mode."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from fastmcp import Client

from imas_codex.llm import search_tools
from imas_codex.llm.server import AgentsServer
from imas_codex.tools import plant_search


@pytest.fixture(scope="module")
def plant_server():
    return AgentsServer(dd_only=False, no_embed=True)


def _tool_names(server):
    return {
        key.removeprefix("tool:").rstrip("@")
        for key in server.mcp._local_provider._components
        if key.startswith("tool:")
    }


def test_plant_tools_absent_from_dd_only_with_full_mode_positive_control(plant_server):
    facility_tools = {"search_signals", "search_pulses", "fetch_content"}
    assert facility_tools <= _tool_names(plant_server)
    assert not facility_tools & _tool_names(AgentsServer(dd_only=True, no_embed=True))
    assert facility_tools <= _tool_names(
        AgentsServer(dd_only=False, read_only=True, no_embed=True)
    )
    assert facility_tools <= AgentsServer.FACILITY_TOOLS


@pytest.mark.asyncio
async def test_mcp_schema_exposes_typed_plant_filters_without_query_requirement(
    plant_server,
):
    async with Client(plant_server.mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        signal_schema = tools["search_signals"].inputSchema
        assert signal_schema["required"] == ["facility"]
        assert {
            "plant_system",
            "function",
            "component_category",
            "field",
            "offset",
        } <= signal_schema["properties"].keys()
        for name in ("search_signals", "search_pulses", "fetch_content"):
            schema = tools[name].inputSchema
            assert not {"cypher", "endpoint", "host"} & schema["properties"].keys()
            assert tools[name].outputSchema is None


@pytest.mark.asyncio
async def test_missing_designation_is_explicit_empty_page_through_mcp(
    plant_server, monkeypatch
):
    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.query.return_value = [{"total": 0, "items": []}]
    monkeypatch.setattr("imas_codex.graph.client.GraphClient", lambda: graph)
    async with Client(plant_server.mcp) as client:
        response = await client.call_tool(
            "search_pulses", {"facility": "demo", "designation": "absent-coil"}
        )
    result = json.loads(response.content[0].text)
    assert result["total"] == result["count"] == 0
    assert result["items"] == []
    assert result["continuation"] is result["next_offset"] is None
    assert graph.query.call_args.kwargs["designation"] == "absent-coil"


@pytest.mark.asyncio
async def test_catalogue_filter_dispatch_needs_no_embedding(plant_server, monkeypatch):
    graph = MagicMock()
    graph.__enter__.return_value = graph
    graph.query.return_value = [{"total": 3, "items": [{"id": "demo:signal-a"}]}]
    monkeypatch.setattr(search_tools, "GraphClient", lambda: graph)
    monkeypatch.setattr(
        search_tools,
        "_embed",
        lambda *_: pytest.fail("plant listing attempted embedding"),
    )
    async with Client(plant_server.mcp) as client:
        response = await client.call_tool(
            "search_signals",
            {
                "facility": "demo",
                "plant_system": "DEMO-UNIT",
                "component_category": "QAZ",
                "k": 1,
            },
        )
    result = json.loads(response.content[0].text)
    assert result["total"] == 3 and result["count"] == 1
    assert result["continuation"] == {
        "tool": "search_signals",
        "arguments": {
            "facility": "demo",
            "plant_system": "DEMO-UNIT",
            "component_category": "QAZ",
            "k": 1,
            "offset": 1,
        },
    }


@pytest.mark.parametrize(
    ("limit", "offset"), [(0, 0), (101, 0), (1, -1), (True, 0), (1, 0.5)]
)
def test_invalid_pagination_refused_before_graph_read(limit, offset):
    graph = MagicMock()
    with pytest.raises(ValueError):
        plant_search.search_pulse_catalogue(
            graph, "demo", designation="coil-alpha", limit=limit, offset=offset
        )
    graph.query.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [{}, {"designation": " "}, {"designation": "alpha", "category_id": "beta"}],
)
def test_unbounded_or_ambiguous_category_search_refused(arguments):
    graph = MagicMock()
    with pytest.raises(ValueError):
        plant_search.search_pulse_catalogue(graph, "demo", **arguments)
    graph.query.assert_not_called()


@pytest.mark.parametrize("arguments", [{}, {"field": " "}])
def test_unbounded_signal_listing_refused(arguments):
    graph = MagicMock()
    with pytest.raises(ValueError):
        plant_search.list_plant_signals(graph, "demo", **arguments)
    graph.query.assert_not_called()


def test_text_query_and_plant_filters_cannot_silently_ignore_each_other():
    with pytest.raises(ValueError, match="omit query"):
        search_tools._search_signals("current", "demo", field="ABCD", gc=MagicMock())


def test_page_byte_limit_preserves_identities_and_advances_by_delivered_items():
    identity = "unicode-\U0001f9f2" * 500
    items = [
        {"id": f"{identity}-{index}", "description": "\U0001f9f2" * 2048}
        for index in range(100)
    ]
    encoded = plant_search._page_json(
        items,
        120,
        7,
        100,
        tool="search_signals",
        arguments={"facility": "demo", "field": "ABCD", "k": 100},
    )
    result = json.loads(encoded)
    assert len(encoded.encode()) <= plant_search.PAGE_MAX_BYTES < 16_000_000
    assert 0 < result["count"] < 100
    assert result["items"] == items[: result["count"]]
    assert result["next_offset"] == 7 + result["count"]
    assert result["continuation"]["arguments"]["offset"] == result["next_offset"]


def test_oversized_single_identity_is_refused_instead_of_silently_truncated():
    with pytest.raises(ValueError, match="identity was not truncated"):
        plant_search._page_json(
            [{"id": "x" * plant_search.PAGE_MAX_BYTES}],
            1,
            0,
            1,
            tool="fetch_content",
            arguments={"resource": "demo:signal"},
        )
