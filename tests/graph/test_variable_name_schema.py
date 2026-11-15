"""Schema relationships keep candidate components separate from verified identity."""

from imas_codex.graph.models import FacilitySignal, PlantComponent
from imas_codex.graph.schema import GraphSchema


def test_candidate_component_schema_relationships_and_round_trip():
    schema = GraphSchema()
    expected = {
        "PlantComponent": {
            ("facility_id", "Facility", "AT_FACILITY"),
            ("parent_component", "PlantComponent", "PARENT_COMPONENT"),
        },
        "FacilitySignal": {
            ("candidate_plant_component", "PlantComponent", "CANDIDATE_PLANT_COMPONENT")
        },
    }
    for label, relationships in expected.items():
        declared = {
            (r.slot_name, r.to_class, r.cypher_type)
            for r in schema.get_relationships_from(label)
        }
        assert relationships <= declared
    component = PlantComponent(
        id="demo:plant_component:DEMO-COOL",
        facility_id="demo",
        code="DEMO-COOL",
        parent_component="demo:plant_component:DEMO",
    )
    assert PlantComponent.model_validate_json(component.model_dump_json()) == component
    signal = FacilitySignal(
        id="demo:example",
        facility_id="demo",
        accessor="arbitrary text",
        name_parse_outcome="not_conforming",
        name_parse_rule_status="unconfirmed",
    )
    assert FacilitySignal.model_validate_json(signal.model_dump_json()) == signal
