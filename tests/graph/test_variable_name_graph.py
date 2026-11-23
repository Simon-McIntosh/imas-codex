"""Real naming-convention writes and refusal on a disposable graph."""

import pytest

from iter_codex.discovery.signals.scanners import codac_uda as scanner
from iter_codex.discovery.signals.variable_name_graph import (
    parse_archive_catalogue,
    persist_name_parses,
)
from iter_codex.graph.client import GraphClient
from iter_codex.graph.models import ArchiveEndpoint

pytestmark = pytest.mark.graph


def test_scanner_and_shared_catalogue_parse_persist_distinct_identities(
    disposable_graph,
):
    with GraphClient() as graph:
        for facility in ("demo-alpha", "demo-beta"):
            graph.query("CREATE (:Facility {id: $id})", id=facility)
            archive = f"{facility}:codac_uda:slow"
            scanner._persist_archives(
                facility,
                [
                    ArchiveEndpoint(
                        id=archive,
                        facility_id=facility,
                        data_system="codac_uda",
                        role="slow",
                        excluded=False,
                    )
                ],
            )
            selectors = [
                "DEMO-UNIT:QAZ0001-ABCD",
                "unparseable fixture",
                "DEMO-UNIT:STATUS",
            ]
            signals = [
                scanner.CodacUDAScanner._signal(
                    facility,
                    archive,
                    scanner._access(facility, "slow").id,
                    {
                        "name": selector,
                        "metadata_state": "available",
                        "archive_metadata": {"type": {"datatypes": []}, "full": []},
                    },
                    f"fixture-{index}",
                )
                for index, selector in enumerate(selectors)
            ]
            for _ in range(2):
                assert scanner._persist_signals(facility, "fixture", signals) == 3
        for _ in range(2):
            assert parse_archive_catalogue(graph, batch_size=2) == {
                "total": 6,
                "written": 6,
                "outcomes": {
                    "component_variable": 2,
                    "function_variable": 2,
                    "not_conforming": 2,
                },
            }
        rows = graph.query("""MATCH (s:FacilitySignal)
            OPTIONAL MATCH (s)-[:IN_ARCHIVE_COMPONENT|CANDIDATE_PLANT_COMPONENT]->(c:PlantComponent)
            OPTIONAL MATCH (s)-[:AT_CONTROL_NODE]->(control:ControlNode)
            RETURN s.accessor AS selector,s.name_parse_outcome AS outcome,
                s.candidate_plant_component AS retired_component_property,c.id AS component,
                s.control_node AS control_property,control.id AS control,control.code AS control_code""")
        assert len(rows) == 6
        for row in rows:
            assert row["retired_component_property"] is None
            assert row["component"] is None
            assert row["control_property"] == row["control"]
            if row["outcome"] == "component_variable":
                assert row["control_code"] == "DEMO-UNIT"
            elif row["outcome"] == "function_variable":
                assert row["control_code"] == "DEMO-UNIT"
            else:
                assert row["control"] is None
        assert graph.query(
            "MATCH (c:PlantComponent) RETURN c.id AS id,c.facility_id AS facility ORDER BY id"
        ) == [
            {
                "id": "demo-alpha:plant_component:DEMO-UNIT:QAZ0001",
                "facility": "demo-alpha",
            },
            {
                "id": "demo-beta:plant_component:DEMO-UNIT:QAZ0001",
                "facility": "demo-beta",
            },
        ]
        hierarchy = graph.query(
            "MATCH (c:ControlNode) OPTIONAL MATCH (c)-[:PARENT_CONTROL_NODE]->(p:ControlNode) RETURN c.parent_control_node AS parent_property,p.id AS parent"
        )
        assert len(hierarchy) == 4
        assert sum(row["parent"] is not None for row in hierarchy) == 2
        assert all(row["parent_property"] == row["parent"] for row in hierarchy)
        assert graph.query("MATCH (c:PlantComponent) RETURN count(c) AS n") == [
            {"n": 2}
        ]


def test_nonconforming_parse_removes_stale_component_and_control_links(
    disposable_graph,
):
    with GraphClient() as graph:
        graph.query("""CREATE (:Facility {id:'demo'})
            CREATE (s:FacilitySignal {id:'demo:signal',facility_id:'demo',accessor:'unparseable fixture',
                candidate_plant_component:'demo:unsupported',control_node:'demo:unsupported-control',
                plant_system_code:'WRONG',control_function:'WRONG',variable_token:'WRONG'})
            CREATE (c:PlantComponent {id:'demo:unsupported'})
            CREATE (control:ControlNode {id:'demo:unsupported-control'})
            CREATE (s)-[:CANDIDATE_PLANT_COMPONENT]->(c)
            CREATE (s)-[:AT_CONTROL_NODE]->(control)""")
        assert persist_name_parses(
            graph,
            [
                {
                    "id": "demo:signal",
                    "facility": "demo",
                    "selector": "unparseable fixture",
                }
            ],
        ) == {"not_conforming": 1}
        assert (
            graph.query("""MATCH (s:FacilitySignal {id:'demo:signal'})
            OPTIONAL MATCH (s)-[:CANDIDATE_PLANT_COMPONENT|AT_CONTROL_NODE]->(c)
            RETURN s.name_parse_outcome AS outcome,s.candidate_plant_component AS component,
                s.control_node AS control,s.control_function AS legacy,count(c) AS links""")
            == [
                {
                    "outcome": "not_conforming",
                    "component": None,
                    "control": None,
                    "legacy": None,
                    "links": 0,
                }
            ]
        )


def test_changed_selector_refuses_a_stale_parse_write(disposable_graph):
    with GraphClient() as graph:
        graph.query(
            "CREATE (:Facility {id:'demo'}), (:FacilitySignal {id:'demo:signal', facility_id:'demo', accessor:'unparseable fixture'})"
        )
        with pytest.raises(RuntimeError, match="not acknowledged"):
            persist_name_parses(
                graph,
                [
                    {
                        "id": "demo:signal",
                        "facility": "demo",
                        "selector": "DEMO-UNIT:QAZ0001-ABCD",
                    }
                ],
            )
        assert graph.query(
            "MATCH (s:FacilitySignal {id:'demo:signal'}) RETURN s.name_parse_outcome AS outcome"
        ) == [{"outcome": None}]
