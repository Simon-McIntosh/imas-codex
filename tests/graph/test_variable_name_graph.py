"""Real parser writes, hierarchy and refusal on a disposable graph."""

import pytest

from imas_codex.discovery.signals.scanners import codac_uda as scanner
from imas_codex.discovery.signals.variable_name_graph import (
    parse_archive_catalogue,
    persist_name_parses,
)
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import ArchiveEndpoint

pytestmark = pytest.mark.graph


def test_scanner_and_all_catalogue_pass_write_only_derived_candidates(disposable_graph):
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
                "DEMO-COOL-LOOP-PUMP:PCV9001-MEAS",
                "arbitrary text",
                "DEMO-COOL-LOOP-PUMP:PCV9001",
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
                    f"example-{index}",
                )
                for index, selector in enumerate(selectors)
            ]
            for _ in range(2):
                assert scanner._persist_signals(facility, "example", signals) == 3
        for _ in range(2):
            assert parse_archive_catalogue(graph, batch_size=2) == {
                "total": 6,
                "written": 6,
                "outcomes": {"parsed": 2, "partly_parsed": 2, "not_conforming": 2},
            }
        rows = graph.query("""
            MATCH (s:FacilitySignal)
            OPTIONAL MATCH (s)-[:CANDIDATE_PLANT_COMPONENT]->(c:PlantComponent)
            RETURN s.accessor AS selector, s.name AS name, s.name_parse_outcome AS outcome,
                   s.candidate_plant_component AS component_property, c.id AS component,
                   c.code AS code, s.name_parse_rule_status AS rule_status
        """)
        assert len(rows) == 6
        for row in rows:
            assert row["selector"] == row["name"]
            assert row["component_property"] == row["component"]
            assert row["rule_status"] == "unconfirmed"
            if row["outcome"] == "not_conforming":
                assert row["component"] is None
            else:
                assert row["code"] == row["selector"].split(":")[0]
        hierarchy = graph.query(
            "MATCH (c:PlantComponent) OPTIONAL MATCH (c)-[:PARENT_COMPONENT]->(p:PlantComponent) RETURN c.code AS code, c.parent_component AS parent_property, p.id AS parent"
        )
        assert len(hierarchy) == 8
        assert sum(row["parent"] is not None for row in hierarchy) == 6
        assert all(row["parent_property"] == row["parent"] for row in hierarchy)


def test_nonconforming_name_removes_stale_candidate_link(disposable_graph):
    with GraphClient() as graph:
        graph.query("""
            CREATE (:Facility {id:'demo'})
            CREATE (s:FacilitySignal {id:'demo:signal', facility_id:'demo', accessor:'arbitrary text',
                                    candidate_plant_component:'demo:unsupported', control_function:'UNSUPPORTED'})
            CREATE (c:PlantComponent {id:'demo:unsupported'})
            CREATE (s)-[:CANDIDATE_PLANT_COMPONENT]->(c)
        """)
        assert persist_name_parses(
            graph,
            [{"id": "demo:signal", "facility": "demo", "selector": "arbitrary text"}],
        ) == {"not_conforming": 1}
        assert graph.query("""
            MATCH (s:FacilitySignal {id:'demo:signal'})
            OPTIONAL MATCH (s)-[:CANDIDATE_PLANT_COMPONENT]->(c)
            RETURN s.name_parse_outcome AS outcome, s.candidate_plant_component AS property,
                   s.control_function AS function, count(c) AS links
        """) == [
            {
                "outcome": "not_conforming",
                "property": None,
                "function": None,
                "links": 0,
            }
        ]


def test_changed_selector_refuses_a_stale_parse_write(disposable_graph):
    with GraphClient() as graph:
        graph.query(
            "CREATE (:Facility {id:'demo'}), (:FacilitySignal {id:'demo:signal', facility_id:'demo', accessor:'arbitrary text'})"
        )
        with pytest.raises(RuntimeError, match="not acknowledged"):
            persist_name_parses(
                graph,
                [
                    {
                        "id": "demo:signal",
                        "facility": "demo",
                        "selector": "DEMO-COOL:PCV9001-MEAS",
                    }
                ],
            )
        assert graph.query(
            "MATCH (s:FacilitySignal {id:'demo:signal'}) RETURN s.name_parse_outcome AS outcome"
        ) == [{"outcome": None}]
