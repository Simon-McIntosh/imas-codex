"""Plant catalogue queries and MCP traversal against invented, isolated graph data."""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

from imas_codex.discovery.signals.variable_name_graph import persist_name_parses
from imas_codex.graph.client import GraphClient
from imas_codex.llm.server import AgentsServer
from imas_codex.tools.plant_search import (
    fetch_signal_context,
    list_plant_signals,
    search_pulse_catalogue,
)

pytestmark = pytest.mark.graph


@pytest.fixture()
def plant_graph(disposable_graph):
    with GraphClient() as graph:
        for facility in ("demo-alpha", "demo-beta"):
            graph.query("CREATE (:Facility {id: $facility})", facility=facility)
            graph.query(
                """
                MATCH (f:Facility {id: $facility})
                CREATE (c:PulseCategory {id: $facility + ':test:coil-alpha', facility_id: $facility,
                    name: 'Coil Alpha commissioning', description: 'Coil Alpha cold test'})-[:AT_FACILITY]->(f)
                CREATE (:PulseCategory {id: $facility + ':test:coil-alpha-empty', facility_id: $facility,
                    name: 'Coil Alpha empty campaign'})-[:AT_FACILITY]->(f)
                CREATE (:Pulse {id: $facility + ':pulse:closed', facility_id: $facility,
                    name: 'Coil Alpha / closed', start_ns: 1700000000000000001, end_ns: 1700000002000000003,
                    description: 'Invented completed pulse'})-[:IN_PULSE_CATEGORY]->(c)
                CREATE (:Pulse {id: $facility + ':pulse:open', facility_id: $facility,
                    name: 'Coil Alpha / open', start_ns: 1700000003000000007})-[:IN_PULSE_CATEGORY]->(c)
                """,
                facility=facility,
            )
            selectors = [
                "DEMO-UNI-LOOP:QAZ0001-ABCD",
                "DEMO-UNI-LOOP:QAZ0001-EFGH",
                "DEMO-UNI-LOOP:QAZ0001-IJKL",
                "DEMO-UNI-LOOP:QAZ0002-ABCD",
                "DEMO-UNI:STATUS",
                "DEMO-UNIX-LOOP:QAZ0001-ABCD",
                "nonconforming fixture",
            ]
            items = [
                {
                    "id": f"{facility}:signal:{i}",
                    "facility": facility,
                    "selector": selector,
                }
                for i, selector in enumerate(selectors)
            ]
            graph.query(
                """MATCH (f:Facility {id: $facility}) UNWIND $items AS item
                CREATE (s:FacilitySignal {id: item.id, facility_id: $facility, name: item.selector, accessor: item.selector,
                    description: 'Invented signal', unit: 'V'})-[:AT_FACILITY]->(f)""",
                facility=facility,
                items=items,
            )
            persist_name_parses(graph, items)
        yield graph


def test_category_search_empty_control_paging_and_literal_designation(plant_graph):
    categories = json.loads(
        search_pulse_catalogue(
            plant_graph, "demo-alpha", designation="cOiL aLpHa", limit=1
        )
    )
    assert categories["total"] == 2 and categories["count"] == 1
    assert categories["items"][0]["pulse_count"] == 2
    assert categories["next_offset"] == 1
    arguments = categories["continuation"]["arguments"]
    second = json.loads(search_pulse_catalogue(plant_graph, **arguments))
    assert second["items"][0]["pulse_count"] == 0
    assert second["continuation"] is None
    for designation in ("absent-coil", "' OR true //", ".*"):
        empty = json.loads(
            search_pulse_catalogue(plant_graph, "demo-alpha", designation=designation)
        )
        assert empty["items"] == [] and empty["total"] == empty["count"] == 0
        assert empty["continuation"] is None


def test_pulses_preserve_exact_times_open_ends_and_facility_boundary(plant_graph):
    arguments = {
        "facility": "demo-alpha",
        "category_id": "demo-alpha:test:coil-alpha",
        "limit": 1,
    }
    first = json.loads(search_pulse_catalogue(plant_graph, **arguments))
    assert first["total"] == 2 and first["count"] == 1
    assert first["items"][0]["start_ns"] == 1700000000000000001
    assert first["items"][0]["end_ns"] == 1700000002000000003
    assert first["items"][0]["open_ended"] is False
    second = json.loads(
        search_pulse_catalogue(plant_graph, **first["continuation"]["arguments"])
    )
    assert second["items"][0]["start_ns"] == 1700000003000000007
    assert (
        second["items"][0]["end_ns"] is None
        and second["items"][0]["open_ended"] is True
    )
    assert second["continuation"] is None
    assert (
        json.loads(
            search_pulse_catalogue(
                plant_graph, "demo-beta", category_id=arguments["category_id"]
            )
        )["total"]
        == 0
    )


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({"plant_system": "DEMO-UNI", "component_category": "QAZ"}, 4),
        ({"function": "DEMO-UNI-LOOP"}, 4),
        ({"field": "ABCD"}, 3),
    ],
)
def test_signal_filters_return_exact_totals_and_nonoverlapping_continuations(
    plant_graph, filters, expected
):
    result = json.loads(
        list_plant_signals(plant_graph, "demo-alpha", **filters, limit=1)
    )
    assert result["total"] == expected and result["count"] == 1
    assert result["next_offset"] == 1
    identities = {result["items"][0]["id"]}
    while result["continuation"]:
        arguments = result["continuation"]["arguments"].copy()
        arguments["limit"] = arguments.pop("k")
        result = json.loads(list_plant_signals(plant_graph, **arguments))
        assert result["total"] == expected and result["count"] == 1
        assert result["items"][0]["id"] not in identities
        identities.add(result["items"][0]["id"])
    assert len(identities) == expected
    assert all(identity.startswith("demo-alpha:") for identity in identities)


def test_signal_walk_retains_candidate_identity_and_excludes_self_and_other_facility(
    plant_graph,
):
    result = json.loads(
        fetch_signal_context(plant_graph, "demo-alpha:signal:0", limit=1)
    )
    assert result["signal"]["accessor"] == "DEMO-UNI-LOOP:QAZ0001-ABCD"
    assert result["component"]["code"] == "DEMO-UNI-LOOP:QAZ0001"
    assert result["component_identity_status"] == "candidate"
    assert result["system"]["code"] == "DEMO"
    assert [node["code"] for node in result["control_hierarchy"]] == [
        "DEMO",
        "DEMO-UNI",
        "DEMO-UNI-LOOP",
    ]
    assert result["total"] == 2 and result["count"] == 1
    assert result["items"][0]["id"] == "demo-alpha:signal:1"
    second = json.loads(
        fetch_signal_context(plant_graph, **result["continuation"]["arguments"])
    )
    assert second["items"][0]["id"] == "demo-alpha:signal:2"
    assert second["continuation"] is None
    unparsed = json.loads(fetch_signal_context(plant_graph, "demo-alpha:signal:6"))
    assert unparsed["component"] is unparsed["system"] is None
    assert unparsed["total"] == 0 and unparsed["items"] == []
    assert fetch_signal_context(plant_graph, "demo-alpha:missing") is None


@pytest.mark.asyncio
async def test_running_server_walks_categories_pulses_signals_and_siblings_without_queries(
    plant_graph,
):
    server = AgentsServer(dd_only=False, no_embed=True)
    async with Client(server.mcp) as client:
        categories = await client.call_tool(
            "search_pulses",
            {"facility": "demo-alpha", "designation": "Coil Alpha", "limit": 1},
        )
        category = json.loads(categories.content[0].text)
        pulses = await client.call_tool(
            "search_pulses",
            {"facility": "demo-alpha", "category_id": category["items"][0]["id"]},
        )
        assert json.loads(pulses.content[0].text)["total"] == 2
        for filters in (
            {"plant_system": "DEMO-UNI", "component_category": "QAZ"},
            {"function": "DEMO-UNI-LOOP"},
            {"field": "ABCD"},
        ):
            signals = await client.call_tool(
                "search_signals", {"facility": "demo-alpha", **filters, "k": 1}
            )
            page = json.loads(signals.content[0].text)
            assert page["total"] > 1 and page["continuation"] is not None
        context = await client.call_tool(
            "fetch_content", {"resource": "demo-alpha:signal:0", "limit": 1}
        )
        result = json.loads(context.content[0].text)
        assert result["component"] and result["system"] and result["total"] == 2
        assert len(context.content[0].text.encode()) < 256_000
