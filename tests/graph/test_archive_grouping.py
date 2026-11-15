"""Real projection of component and control groups using invented names."""

from contextlib import closing

import pytest

from imas_codex.discovery.signals.archive_grouping import ArchiveGrouping
from imas_codex.discovery.signals.archive_grouping_graph import (
    graph_group_counts,
    persist_archive_grouping,
)
from imas_codex.discovery.signals.variable_name_graph import snapshot_archive_signals
from imas_codex.graph.client import GraphClient
from imas_codex.graph.models import ArchiveFieldPattern, ControlNode, FacilitySignal
from imas_codex.graph.schema import GraphSchema

pytestmark = pytest.mark.graph


def test_grouping_schema_relationships_and_code_roundtrips():
    schema = GraphSchema()
    expected = {
        "PlantComponent": {
            (
                "archive_field_pattern",
                "ArchiveFieldPattern",
                "HAS_ARCHIVE_FIELD_PATTERN",
            ),
            (
                "archive_category_pattern",
                "ArchiveFieldPattern",
                "HAS_ARCHIVE_CATEGORY_PATTERN",
            ),
            ("control_node", "ControlNode", "AT_CONTROL_NODE"),
        },
        "ControlNode": {
            ("facility_id", "Facility", "AT_FACILITY"),
            ("parent_control_node", "ControlNode", "PARENT_CONTROL_NODE"),
        },
        "FacilitySignal": {
            ("archive_component", "PlantComponent", "IN_ARCHIVE_COMPONENT"),
            ("archive_control_node", "ControlNode", "IN_ARCHIVE_CONTROL_NODE"),
        },
    }
    for label, relationships in expected.items():
        assert relationships <= {
            (r.slot_name, r.to_class, r.cypher_type)
            for r in schema.get_relationships_from(label)
        }
    node = ControlNode(
        id="demo:control_node:DEMO", facility_id="demo", code="DEMO", level=1
    )
    assert node.description is None
    pattern = ArchiveFieldPattern(
        id="demo:pattern",
        facility_id="demo",
        component_category="QAZ",
        key_kind="category_fields",
        fields=["ABCD"],
        field_count=1,
    )
    assert ArchiveFieldPattern.model_validate_json(pattern.model_dump_json()) == pattern
    signal = FacilitySignal(
        id="demo:signal",
        facility_id="demo",
        accessor="unparseable fixture",
        archive_grouping_state="ungrouped",
    )
    assert signal.archive_component is signal.archive_control_node is None


def test_projection_corrects_legacy_membership_and_repeats_without_duplicates(
    disposable_graph, tmp_path
):
    names = [
        "DEMO-UNIT:QAZ0001-ABCD",
        "DEMO-UNIT:QAZ0001-EFGH",
        "DEMO-UNIT:QAZ0002-ABCD",
        "DEMO-UNIT:QAZ0002-EFGH",
        "DEMO-UNIT:STATUS",
        "unparseable fixture",
    ]
    with (
        closing(ArchiveGrouping(tmp_path / "groups.sqlite")) as grouping,
        GraphClient() as graph,
    ):
        grouping.rebuild("demo", [{"name": name} for name in names])
        graph.query("CREATE (:Facility {id:'demo'}), (:Facility {id:'other'})")
        graph.query(
            """UNWIND $items AS item
            CREATE (:FacilitySignal {id:item.id, facility_id:'demo', accessor:item.selector,
                metadata_archive:'demo:archive', status:'enriched', description:'retained', claim_token:'retained',
                archive_grouping_state:'grouped', archive_component:'demo:legacy',
                candidate_plant_component:'demo:legacy', plant_system_code:'DEMO', control_function:'WRONG'})
        """,
            items=[
                {"id": f"demo:signal:{i}", "selector": names[i]} for i in (0, 2, 4, 5)
            ],
        )
        graph.query("""CREATE (c:PlantComponent {id:'demo:legacy', facility_id:'demo', code:'DEMO-UNIT', archive_field_pattern:'demo:legacy-pattern'})
            CREATE (root:PlantComponent {id:'demo:legacy-root', facility_id:'demo', code:'DEMO'})
            CREATE (orphan:PlantComponent {id:'demo:legacy-orphan', facility_id:'demo', code:'DEMO-TEST', parent_component:'demo:legacy-root'})
            SET c.parent_component='demo:legacy-root'
            CREATE (c)-[:PARENT_COMPONENT]->(root)
            CREATE (orphan)-[:PARENT_COMPONENT]->(root)
            CREATE (p:ArchiveFieldPattern {id:'demo:legacy-pattern', facility_id:'demo'})
            CREATE (c)-[:HAS_ARCHIVE_FIELD_PATTERN]->(p)
            WITH c MATCH (s:FacilitySignal {facility_id:'demo'})
            CREATE (s)-[:IN_ARCHIVE_COMPONENT]->(c)
            CREATE (s)-[:CANDIDATE_PLANT_COMPONENT]->(c)""")
        graph.query(
            "CREATE (:FacilitySignal {id:'other:signal', facility_id:'other', accessor:$name, metadata_archive:'other:archive'})",
            name=names[0],
        )
        snapshot = snapshot_archive_signals(graph, "demo", batch_size=1)
        assert len(snapshot) == 4
        for _ in range(2):
            assert persist_archive_grouping(
                graph, grouping, "demo", snapshot, batch_size=2
            ) == {
                "variables": 4,
                "grouped": 3,
                "component_variables": 2,
                "function_variables": 1,
                "ungrouped": 1,
            }
            assert graph.query(
                "MATCH (c:PlantComponent {facility_id:'demo'}) WHERE c.archive_field_pattern IS NOT NULL RETURN count(c) AS n"
            ) == [{"n": 2}]
            assert graph_group_counts(graph, "demo") == {
                "component_groups": 2,
                "control_node_groups": 1,
                "field_patterns": 1,
                "category_patterns": 1,
            }
            assert graph.query("""
                MATCH (c:PlantComponent {facility_id:'demo'})
                OPTIONAL MATCH (c)-[r:PARENT_COMPONENT]->()
                RETURN count(DISTINCT c) AS candidates,count(c.parent_component) AS properties,count(r) AS links
            """) == [{"candidates": 5, "properties": 0, "links": 0}]
            assert graph.query(
                "MATCH (p:ArchiveFieldPattern {facility_id:'demo'}) WHERE p.key_kind IS NOT NULL RETURN count(p) AS n"
            ) == [{"n": 2}]
        rows = graph.query("""MATCH (s:FacilitySignal {facility_id:'demo'})
            OPTIONAL MATCH (s)-[:IN_ARCHIVE_COMPONENT]->(c:PlantComponent)
            OPTIONAL MATCH (s)-[:IN_ARCHIVE_CONTROL_NODE]->(control:ControlNode)
            RETURN s.accessor AS selector,s.archive_grouping_state AS state,s.archive_component AS component_property,
                c.id AS component,c.archive_member_count AS members,s.archive_control_node AS control_property,
                control.id AS control,control.archive_member_count AS control_members,
                s.status AS status,s.description AS description,s.claim_token AS claim,
                s.plant_system_code AS legacy_system,s.control_function AS legacy_function""")
        for row in rows:
            assert (row["status"], row["description"], row["claim"]) == (
                "enriched",
                "retained",
                "retained",
            )
            assert row["component_property"] == row["component"]
            assert row["control_property"] == row["control"]
            assert row["legacy_system"] is row["legacy_function"] is None
            if row["selector"] == "unparseable fixture":
                assert row["state"] == "ungrouped"
                assert row["component"] is row["control"] is None
            elif row["selector"].endswith(":STATUS"):
                assert row["state"] == "control_node"
                assert row["component"] is None
                assert row["control_members"] == 1
            else:
                assert row["state"] == "component"
                assert row["members"] == 2
                assert row["control"] is None
        assert graph.query(
            "MATCH (c:PlantComponent {id:'demo:legacy'}) RETURN c.archive_field_pattern AS pattern"
        ) == [{"pattern": None}]
        assert graph.query(
            "MATCH (s:FacilitySignal {id:'other:signal'}) RETURN s.archive_grouping_state AS state"
        ) == [{"state": None}]
        assert graph.query(
            "MATCH (c:ControlNode {facility_id:'demo'}) RETURN c.level AS level ORDER BY level"
        ) == [{"level": 1}, {"level": 2}]
        pattern = graph.query(
            "MATCH (p:ArchiveFieldPattern {facility_id:'demo', key_kind:'category_fields'}) RETURN p.fields AS fields,p.component_count AS components,p.member_count AS members"
        )
        assert pattern == [{"fields": ["ABCD", "EFGH"], "components": 2, "members": 4}]
        graph.query(
            "MATCH (s:FacilitySignal {id:'demo:signal:0'}) SET s.accessor='unparseable fixture'"
        )
        persist_archive_grouping(
            graph, grouping, "demo", snapshot_archive_signals(graph, "demo")
        )
        assert graph.query(
            "MATCH (s:FacilitySignal {id:'demo:signal:0'}) OPTIONAL MATCH (s)-[:IN_ARCHIVE_COMPONENT|AT_CONTROL_NODE|IN_ARCHIVE_CONTROL_NODE|CANDIDATE_PLANT_COMPONENT]->(c) RETURN s.archive_grouping_state AS state,count(c) AS links"
        ) == [{"state": "ungrouped", "links": 0}]


def test_changed_selector_refuses_and_rolls_back_entire_batch(
    disposable_graph, tmp_path
):
    name = "DEMO-UNIT:QAZ0001-ABCD"
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
            "MATCH (n) WHERE n:PlantComponent OR n:ArchiveFieldPattern OR n:ControlNode RETURN count(n) AS n"
        ) == [{"n": 0}]


def test_snapshot_guard_missing_selector_and_empty_facility(disposable_graph):
    with GraphClient() as graph:
        graph.query(
            "CREATE (:FacilitySignal {id:'demo:missing', facility_id:'demo', metadata_archive:'demo:archive'})"
        )
        with pytest.raises(RuntimeError, match="missing selectors"):
            snapshot_archive_signals(graph, "demo")
        assert snapshot_archive_signals(graph, "empty") == []
