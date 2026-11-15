"""Real transactional grouping projection on a disposable graph, using invented names."""

from contextlib import closing

import pytest

from imas_codex.discovery.signals.archive_grouping import ArchiveGrouping
from imas_codex.discovery.signals.archive_grouping_graph import (
    persist_archive_grouping,
    snapshot_archive_signals,
)
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import (
    ArchiveFieldPattern,
    ArchiveGroupingState,
    FacilitySignal,
)
from imas_codex.graph.schema import GraphSchema

pytestmark = pytest.mark.graph


def test_grouping_schema_declares_facility_and_scalar_relationship_pairs():
    schema = GraphSchema()
    expected = {
        "PlantComponent": {
            (
                "archive_field_pattern",
                "ArchiveFieldPattern",
                "HAS_ARCHIVE_FIELD_PATTERN",
            )
        },
        "ArchiveFieldPattern": {("facility_id", "Facility", "AT_FACILITY")},
        "FacilitySignal": {
            ("archive_component", "PlantComponent", "IN_ARCHIVE_COMPONENT")
        },
    }
    for label, relationships in expected.items():
        actual = {
            (r.slot_name, r.to_class, r.cypher_type)
            for r in schema.get_relationships_from(label)
        }
        assert relationships <= actual
    pattern = ArchiveFieldPattern(
        id="demo:pattern", facility_id="demo", fields=["FCV9001-MEAS"], field_count=1
    )
    assert ArchiveFieldPattern.model_validate_json(pattern.model_dump_json()) == pattern
    signal = FacilitySignal(
        id="demo:signal",
        facility_id="demo",
        accessor="unparseable fixture",
        archive_grouping_state=ArchiveGroupingState.ungrouped,
    )
    assert signal.archive_component is None


def test_project_existing_members_and_private_field_sets_idempotently(
    disposable_graph, tmp_path
):
    names = [
        "DEMO-COOL-PUMP:FCV9001-MEAS",
        "DEMO-COOL-PUMP:FCV9001-TARGET",
        "DEMO-COOL-VALVE:FCV9001-MEAS",
        "DEMO-COOL-VALVE:FCV9001-TARGET",
        "unparseable fixture",
    ]
    with (
        closing(ArchiveGrouping(tmp_path / "groups.sqlite")) as grouping,
        GraphClient() as graph,
    ):
        grouping.rebuild("demo", [{"name": name} for name in names])
        graph.query("CREATE (:Facility {id:'demo'}), (:Facility {id:'other'})")
        graph.query(
            """
            UNWIND $items AS item
            CREATE (:FacilitySignal {id:item.id, facility_id:'demo', accessor:item.selector,
                    metadata_archive:'demo:archive', status:'enriched', description:'retained', claim_token:'retained'})
        """,
            items=[{"id": f"demo:signal:{i}", "selector": names[i]} for i in (0, 3, 4)],
        )
        graph.query(
            "CREATE (:FacilitySignal {id:'other:signal', facility_id:'other', accessor:$name, metadata_archive:'other:archive'})",
            name=names[0],
        )
        snapshot = snapshot_archive_signals(graph, "demo", batch_size=1)
        assert len(snapshot) == 3
        for _ in range(2):
            assert persist_archive_grouping(
                graph, grouping, "demo", snapshot, batch_size=2
            ) == {"variables": 3, "grouped": 2, "ungrouped": 1}
            assert graph.query(
                "MATCH (c:PlantComponent {facility_id:'demo'}) RETURN count(c) AS n"
            ) == [{"n": 2}]
            assert graph.query(
                "MATCH (p:ArchiveFieldPattern {facility_id:'demo'}) RETURN count(p) AS n"
            ) == [{"n": 1}]
        rows = graph.query("""
            MATCH (s:FacilitySignal {facility_id:'demo'})
            OPTIONAL MATCH (s)-[:IN_ARCHIVE_COMPONENT]->(c:PlantComponent)-[:HAS_ARCHIVE_FIELD_PATTERN]->(p:ArchiveFieldPattern)
            RETURN s.accessor AS selector,s.archive_grouping_state AS state,s.archive_component AS component_property,
                   c.id AS component,c.archive_member_count AS members,c.archive_field_pattern AS pattern_property,
                   p.id AS pattern,p.field_count AS fields,p.member_count AS pattern_members,
                   s.status AS status,s.description AS description,s.claim_token AS claim
        """)
        assert len(rows) == 3
        for row in rows:
            assert (row["status"], row["description"], row["claim"]) == (
                "enriched",
                "retained",
                "retained",
            )
            assert row["component_property"] == row["component"]
            if row["selector"] == "unparseable fixture":
                assert row["state"] == "ungrouped"
                assert row["component"] is None
            else:
                assert row["state"] == "grouped"
                assert row["members"] == row["fields"] == 2
                assert row["pattern_members"] == 4
                assert row["pattern_property"] == row["pattern"]
        assert graph.query(
            "MATCH (n) WHERE n:PlantComponent OR n:ArchiveFieldPattern OPTIONAL MATCH (n)-[:AT_FACILITY]->(f:Facility) RETURN count(n) AS total,count(f) AS linked"
        ) == [{"total": 3, "linked": 3}]
        assert graph.query(
            "MATCH (s:FacilitySignal {id:'other:signal'}) RETURN s.archive_grouping_state AS state"
        ) == [{"state": None}]
        # An invalid selector replaces stale membership, rather than inheriting it.
        graph.query(
            "MATCH (s:FacilitySignal {id:'demo:signal:0'}) SET s.accessor='unparseable fixture'"
        )
        snapshot = snapshot_archive_signals(graph, "demo")
        persist_archive_grouping(graph, grouping, "demo", snapshot)
        assert graph.query(
            "MATCH (s:FacilitySignal {id:'demo:signal:0'}) OPTIONAL MATCH (s)-[:IN_ARCHIVE_COMPONENT]->(c) RETURN s.archive_grouping_state AS state,s.archive_component AS component,count(c) AS links"
        ) == [{"state": "ungrouped", "component": None, "links": 0}]


def test_changed_selector_refuses_and_rolls_back_entire_batch(
    disposable_graph, tmp_path
):
    name = "DEMO-COOL:FCV9001-MEAS"
    with (
        closing(ArchiveGrouping(tmp_path / "groups.sqlite")) as grouping,
        GraphClient() as graph,
    ):
        grouping.rebuild("demo", [{"name": name}])
        graph.query(
            "CREATE (:Facility {id:'demo'}), (:FacilitySignal {id:'demo:one', facility_id:'demo', accessor:'changed', metadata_archive:'demo:archive'})"
        )
        with pytest.raises(RuntimeError, match="write not acknowledged"):
            persist_archive_grouping(
                graph, grouping, "demo", [{"id": "demo:one", "selector": name}]
            )
        assert graph.query(
            "MATCH (s:FacilitySignal {id:'demo:one'}) RETURN s.archive_grouping_state AS state"
        ) == [{"state": None}]
        assert graph.query(
            "MATCH (n) WHERE n:PlantComponent OR n:ArchiveFieldPattern RETURN count(n) AS n"
        ) == [{"n": 0}]


def test_snapshot_missing_selector_refuses_with_present_property_control(
    disposable_graph,
):
    with GraphClient() as graph:
        graph.query(
            "CREATE (:FacilitySignal {id:'demo:missing', facility_id:'demo', metadata_archive:'demo:archive'})"
        )
        with pytest.raises(RuntimeError, match="missing selectors"):
            snapshot_archive_signals(graph, "demo")
        assert snapshot_archive_signals(graph, "empty") == []
