"""Exercise every context statement in a rolled-back synthetic transaction."""

from uuid import uuid4

import pytest

from iter_codex.discovery.signals.plant_context import PlantContextBundle
from iter_codex.discovery.signals.plant_context_graph import (
    context_snapshot,
    context_statements,
    write_context,
)
from iter_codex.graph.client import GraphClient
from iter_codex.graph.models import (
    PlantContextDocument,
    PlantFunctionCategory,
    PlantSignalRow,
    PlantStructure,
)

pytestmark = pytest.mark.graph


@pytest.fixture
def context_transaction():
    facility = "synthetic_context_" + uuid4().hex
    scope = facility + ":context:demo"
    props = {"facility_id": facility, "context_scope": scope}
    plant = PlantStructure(
        id=facility + ":plant:root",
        source_uid="invented-plant",
        code="EX",
        description="Invented plant system",
        source_origin="https://records.example.test",
        source_endpoint="/api/plants/invented",
        **props,
    )
    category = PlantFunctionCategory(
        id=facility + ":category:invented",
        source_uid="invented-category",
        code="ZZ",
        description="Invented sensor category",
        source_origin="https://records.example.test",
        source_endpoint="/api/categories/invented",
        **props,
    )
    doc = PlantContextDocument(
        id=facility + ":document:invented",
        source_uid="invented-document",
        version="draft",
        description="Invented signal list",
        role="signal_list",
        sha256="a" * 64,
        system_code="DEMO-COIL",
        plant_structure=plant.id,
        **props,
    )
    row = PlantSignalRow(
        id=facility + ":row:invented",
        source_document=doc.id,
        sheet="Invented table",
        row_number=2,
        description="Invented source row",
        cells_json="{}",
        control_path="DEMO-COIL-CTRL",
        component_category="ZZ",
        component_sequence="42",
        plant_structure=plant.id,
        function_category=category.id,
        **props,
    )
    bundle = PlantContextBundle(
        facility, scope, "DEMO-COIL", [plant], [category], [doc], [row]
    )
    signal_id = facility + ":signal:invented"
    component_id = facility + ":component:invented"
    bundle.link_signals(
        [
            {
                "id": signal_id,
                "selector": "DEMO-COIL-CTRL:ZZ0042-TT",
                "component_id": component_id,
            }
        ]
    )
    with GraphClient() as graph, graph.session() as session:
        tx = session.begin_transaction()
        try:
            tx.run(
                """
                CREATE (f:Facility {id:$facility,name:'Invented facility'})
                CREATE (s:FacilitySignal {id:$signal,facility_id:$facility,
                    accessor:'DEMO-COIL-CTRL:ZZ0042-TT',metadata_archive:'invented'})
                CREATE (c:PlantComponent {id:$component,facility_id:$facility,code:'DEMO-COIL-CTRL:ZZ0042'})
                CREATE (s)-[:AT_FACILITY]->(f)
                CREATE (c)-[:AT_FACILITY]->(f)
            """,
                facility=facility,
                signal=signal_id,
                component=component_id,
            ).consume()

            class View:
                def query(self, statement, **parameters):
                    return tx.run(statement, **parameters).data()

            yield tx, View(), bundle
        finally:
            tx.rollback()


def test_all_context_statements_parse_against_live_graph(context_transaction):
    tx, _, bundle = context_transaction
    for statement, items in context_statements(bundle):
        tx.run(
            "EXPLAIN " + statement,
            items=items,
            facility=bundle.facility,
            scope=bundle.scope,
        ).consume()


def test_import_replay_adds_zero_and_all_content_reaches_facility(context_transaction):
    tx, graph, bundle = context_transaction
    write_context(tx, bundle)
    before = context_snapshot(graph, bundle.facility, bundle.scope)
    assert before["nodes"] == 5
    assert before["links"] > before["nodes"]
    write_context(tx, bundle)
    assert context_snapshot(graph, bundle.facility, bundle.scope) == before
    state = graph.query(
        "MATCH (s:FacilitySignal {id:$id}) RETURN s.plant_context_state AS state,s.plant_context_links AS links",
        id=bundle.signals[0]["id"],
    )[0]
    assert state["state"] == "derived"
    assert state["links"] == [bundle.links[0].id]


def test_missing_source_reference_cannot_silently_import(context_transaction):
    tx, _, bundle = context_transaction
    bundle.rows[0].source_document = bundle.facility + ":document:absent"
    with pytest.raises(RuntimeError, match="not fully acknowledged"):
        write_context(tx, bundle)


def test_changed_signal_selector_cannot_silently_import(context_transaction):
    tx, _, bundle = context_transaction
    bundle.signals[0]["selector"] = "DEMO-COIL-CTRL:ZZ0043-TT"
    with pytest.raises(RuntimeError, match="not fully acknowledged"):
        write_context(tx, bundle)
