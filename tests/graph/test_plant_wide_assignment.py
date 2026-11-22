"""The catalogue assignment against invented, isolated graph data.

Every signal, code and identity here is invented. The tests name the facts the
plant-wide run depends on: the catalogue read sees every catalogued signal and
says so, the field-code table decides the role the write stores, a code the
table does not establish stores none, a repeated assignment changes nothing, and
a role search returns only signals that carry the role.
"""

from __future__ import annotations

import json

import pytest
from fastmcp import Client

from iter_codex.discovery.signals.field_codes import FieldCodeEntry, FieldCodeTable
from iter_codex.discovery.signals.vocabulary_graph import (
    CATALOGUE_CONTROL_QUERY,
    CATALOGUE_QUERY,
    apply_assignment,
    assign_cohort,
    snapshot_catalogue,
)
from iter_codex.graph.client import GraphClient
from iter_codex.llm.server import AgentsServer
from iter_codex.tools.plant_search import list_plant_signals

pytestmark = pytest.mark.graph

FACILITY = "demo-gamma"


def _table(*entries) -> FieldCodeTable:
    return FieldCodeTable(
        entries=tuple(
            FieldCodeEntry(
                code=code,
                component_kind=kind,
                role=role,
                meaning="an invented meaning",
                source=source,
                support=1,
            )
            for code, kind, role, source in entries
        )
    )


@pytest.fixture()
def catalogue_graph(disposable_graph):
    with GraphClient() as graph:
        graph.query("CREATE (:Facility {id: $facility})", facility=FACILITY)
        graph.query(
            """MATCH (f:Facility {id: $facility})
            UNWIND $items AS item
            CREATE (s:FacilitySignal {id: item.id, facility_id: $facility,
                name: item.id, accessor: item.id, control_path: item.control_path,
                field_code: item.field_code, component_category: item.category,
                description: item.description, unit: null})-[:AT_FACILITY]->(f)""",
            facility=FACILITY,
            items=[
                {
                    "id": f"{FACILITY}:signal:setpoint",
                    "control_path": "DEMO-UNI-LOOP",
                    "field_code": "QX",
                    "category": "MF",
                    "description": "invented opening threshold",
                },
                {
                    "id": f"{FACILITY}:signal:reading",
                    "control_path": "DEMO-UNI-LOOP",
                    "field_code": "QX",
                    "category": "MT",
                    "description": "invented reading of the flow",
                },
                {
                    "id": f"{FACILITY}:signal:codeless",
                    "control_path": "DEMO-UNI-LOOP",
                    "field_code": "QZ",
                    "category": "MF",
                    "description": "invented command to open",
                },
                {
                    "id": f"{FACILITY}:signal:elsewhere",
                    "control_path": "DEMO-OTHER",
                    "field_code": "QZ",
                    "category": "MF",
                    "description": "invented command to close",
                },
            ],
        )
        yield graph


def test_the_catalogue_read_sees_every_signal_and_matches_its_control(catalogue_graph):
    for statement in (CATALOGUE_QUERY, CATALOGUE_CONTROL_QUERY):
        catalogue_graph.query(f"EXPLAIN {statement}", facility=FACILITY)
    signals, rows = snapshot_catalogue(catalogue_graph, FACILITY)
    control = catalogue_graph.query(CATALOGUE_CONTROL_QUERY, facility=FACILITY)[0]
    assert control["total"] == 4
    assert len(signals) == control["total"]
    assert len(rows) == 0
    assert {signal["id"] for signal in signals} == {
        f"{FACILITY}:signal:setpoint",
        f"{FACILITY}:signal:reading",
        f"{FACILITY}:signal:codeless",
        f"{FACILITY}:signal:elsewhere",
    }


def test_the_table_decides_the_role_the_write_stores_and_an_unestablished_code_stores_none(
    catalogue_graph,
):
    table = _table(("QX", "MF", "setpoint", "signal_list_row"))
    signals, rows = snapshot_catalogue(catalogue_graph, FACILITY)
    result = assign_cohort(signals, rows, table=table)
    roles = {record["id"]: record["vocabulary_role"] for record in result["ledger"]}
    assert roles[f"{FACILITY}:signal:setpoint"] == "setpoint"
    assert roles[f"{FACILITY}:signal:reading"] is None
    assert roles[f"{FACILITY}:signal:codeless"] is None
    apply_assignment(catalogue_graph, FACILITY, result["pending"])
    stored = {
        row["id"]: row["role"]
        for row in catalogue_graph.query(
            """MATCH (s:FacilitySignal {facility_id: $facility})
            RETURN s.id AS id, s.vocabulary_role AS role ORDER BY s.id""",
            facility=FACILITY,
        )
    }
    assert stored[f"{FACILITY}:signal:setpoint"] == "setpoint"
    assert stored[f"{FACILITY}:signal:reading"] is None
    assert stored[f"{FACILITY}:signal:codeless"] is None


def test_a_second_assignment_changes_no_signal(catalogue_graph):
    table = _table(("QX", "MF", "setpoint", "signal_list_row"))
    first_signals, first_rows = snapshot_catalogue(catalogue_graph, FACILITY)
    first = assign_cohort(first_signals, first_rows, table=table)
    apply_assignment(catalogue_graph, FACILITY, first["pending"])
    second_signals, second_rows = snapshot_catalogue(catalogue_graph, FACILITY)
    second = assign_cohort(second_signals, second_rows, table=table)
    assert second["changed"] == 0
    assert second["pending"] == []


def _assign_roles(graph) -> None:
    table = _table(
        ("QX", "MF", "setpoint", "signal_list_row"),
        ("QX", "MT", "measurement", "controls_metadata"),
        ("QZ", "MF", "command", "controls_metadata"),
    )
    signals, rows = snapshot_catalogue(graph, FACILITY)
    result = assign_cohort(signals, rows, table=table)
    apply_assignment(graph, FACILITY, result["pending"])


def test_a_role_search_returns_only_signals_carrying_that_role(catalogue_graph):
    _assign_roles(catalogue_graph)
    page = json.loads(
        list_plant_signals(
            catalogue_graph,
            FACILITY,
            plant_system="DEMO-UNI",
            vocabulary_role="setpoint",
            limit=10,
        )
    )
    assert page["total"] == 1
    assert [item["id"] for item in page["items"]] == [f"{FACILITY}:signal:setpoint"]
    assert all(item["vocabulary_role"] == "setpoint" for item in page["items"])
    assert (
        json.loads(
            list_plant_signals(
                catalogue_graph,
                FACILITY,
                plant_system="DEMO-OTHER",
                vocabulary_role="setpoint",
            )
        )["total"]
        == 0
    )


@pytest.mark.asyncio
async def test_the_running_server_returns_a_setpoint_system_and_only_carriers(
    catalogue_graph,
):
    _assign_roles(catalogue_graph)
    server = AgentsServer(dd_only=False, no_embed=True)
    async with Client(server.mcp) as client:
        response = await client.call_tool(
            "search_signals",
            {
                "facility": FACILITY,
                "plant_system": "DEMO-UNI",
                "vocabulary_role": "setpoint",
                "k": 10,
            },
        )
        page = json.loads(response.content[0].text)
        assert page["total"] == 1
        assert [item["id"] for item in page["items"]] == [f"{FACILITY}:signal:setpoint"]
